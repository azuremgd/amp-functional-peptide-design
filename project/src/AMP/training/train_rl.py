# -*- coding: utf-8 -*-
"""
RL finetune AMP generator (REINFORCE + prior stabilization)
- reward = w_act*act_used - w_struct*struct_err - penalties + w_prior*(logp_ref-logp)
- Conservative "root-fix" reward:
  1) Hard feasibility filters: length + hydrophobic-run hard cap
  2) TM-like penalty with quadratic hinge (steep when out-of-domain)
  3) Amphipathicity shaping: + w_mu * hydrophobic_moment_norm
  4) Hydrophobic fraction centered shaping: - w_hf_center*(hf-hf_target)^2
"""

import os
import math
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from AMP.models.generator_model import AMPGenerator
from AMP.models.classifier import AMPActivity
from AMP.features.features import FEATURE_KEYS, calculate_structural_features
from AMP.data_processing.vocab import BOS_ID, encode_sequence


# ----------------------------- constraints utilities -----------------------------
def sample_constraints_from_stats(stats):
    cons = {}
    for k in FEATURE_KEYS:
        mu, sigma = stats[k]["mean"], stats[k]["std"]
        cons[k] = float(np.random.normal(mu, sigma))
    cons["length"] = max(5.0, cons["length"])
    return cons


def constraints_to_z(cons, stats, device):
    z = [(cons[k] - stats[k]["mean"]) / (stats[k]["std"] + 1e-8) for k in FEATURE_KEYS]
    z = torch.tensor(z, dtype=torch.float32, device=device).unsqueeze(0)
    return z


# ----------------------------- generator logprob -----------------------------
def log_prob_of_seq(model, seq, z_tgt_t, device):
    ids = encode_sequence(seq)
    if len(ids) == 0:
        return None
    inp = torch.tensor([[BOS_ID] + ids[:-1]], dtype=torch.long, device=device)
    tgt = torch.tensor(ids, dtype=torch.long, device=device)
    logits, _ = model(inp, z_tgt_t)
    log_probs = F.log_softmax(logits, dim=-1)[0]
    t_idx = torch.arange(tgt.size(0), device=device)
    lp = log_probs[t_idx, tgt].sum()
    return lp


# ----------------------------- sampling -----------------------------
def generate_seq(model, stats, cons, device, max_length=None, temperature=1.0, top_p=0.9):
    # Import locally to avoid circular import issues
    from AMP.data_processing.vocab import decode_sequence

    if max_length is None:
        max_length = int(round(cons["length"]))

    z_tgt = np.array(
        [(cons[k] - stats[k]["mean"]) / (stats[k]["std"] + 1e-8) for k in FEATURE_KEYS],
        dtype=np.float32,
    )
    z_tgt_t = torch.tensor(z_tgt, dtype=torch.float32, device=device).unsqueeze(0)

    seq_ids = [BOS_ID]
    for _ in range(max_length):
        inp = torch.tensor(seq_ids, dtype=torch.long, device=device).unsqueeze(0)
        logits, _ = model(inp, z_tgt_t)
        next_logits = logits[0, -1] / max(temperature, 1e-6)
        probs = F.softmax(next_logits, dim=-1)

        # nucleus sampling
        if 0 < top_p < 1:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_probs, dim=-1)
            keep = cum <= top_p
            if keep.sum() == 0:
                keep[0] = True
            filtered = torch.zeros_like(probs)
            filtered[sorted_idx[keep]] = probs[sorted_idx[keep]]
            probs = filtered

        probs = probs / probs.sum()
        nxt = torch.multinomial(probs, 1).item()
        seq_ids.append(nxt)

    # keep only 20 AA tokens
    return decode_sequence([i for i in seq_ids if i < 20])


# ----------------------------- simple sequence penalties -----------------------------
def max_run_length(seq: str) -> int:
    if not seq:
        return 0
    m = cur = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            cur += 1
            m = max(m, cur)
        else:
            cur = 1
    return m


def max_hydrophobic_run(seq: str) -> int:
    hyd = set("AVILMFWY")
    m = cur = 0
    for ch in seq:
        if ch in hyd:
            cur += 1
            m = max(m, cur)
        else:
            cur = 0
    return m


def length_penalty(L: int, low: int, high: int) -> float:
    # soft penalty (still useful), but we will also hard-filter length
    if L < low:
        return (low - L) / max(low, 1)
    if L > high:
        return (L - high) / max(high, 1)
    return 0.0


# ----------------------------- TM-like composition features -----------------------------
_CATIONIC = set("KRH")      # H weakly positive around pH7
_HYD = set("AVILMFWY")
_NEG = set("DE")


def cationic_frac(seq: str) -> float:
    L = len(seq)
    if L == 0:
        return 0.0
    return sum(ch in _CATIONIC for ch in seq) / L


def hydrophobic_frac(seq: str) -> float:
    L = len(seq)
    if L == 0:
        return 0.0
    return sum(ch in _HYD for ch in seq) / L


def net_charge_simple(seq: str) -> float:
    return (
        seq.count("K")
        + seq.count("R")
        + 0.1 * seq.count("H")
        - seq.count("D")
        - seq.count("E")
    )


# ----------------------------- Amphipathicity proxy: hydrophobic moment -----------------------------
# Kyte-Doolittle hydrophobicity (rough)
_KD = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8, "G": -0.4,
    "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8, "M": 1.9, "N": -3.5,
    "P": -1.6, "Q": -3.5, "R": -4.5, "S": -0.8, "T": -0.7, "V": 4.2,
    "W": -0.9, "Y": -1.3
}


def hydrophobic_moment(seq: str, angle_deg: float = 100.0) -> float:
    """
    Helical hydrophobic moment proxy (Eisenberg-style).
    angle_deg=100° approximates alpha-helix.
    """
    if not seq:
        return 0.0
    ang = math.radians(angle_deg)
    x = 0.0
    y = 0.0
    for i, aa in enumerate(seq):
        h = _KD.get(aa, 0.0)
        theta = i * ang
        x += h * math.cos(theta)
        y += h * math.sin(theta)
    return math.sqrt(x * x + y * y)


def hydrophobic_moment_norm(seq: str) -> float:
    L = len(seq)
    if L == 0:
        return 0.0
    return hydrophobic_moment(seq) / L


# ----------------------------- Soft gate for activity (feasibility gating) -----------------------------
def _sigmoid(x: float) -> float:
    # numerically stable sigmoid
    if x >= 20.0:
        return 1.0
    if x <= -20.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def soft_gate(cf: float, hf: float, nc: float, args) -> float:
    """
    Smooth feasibility gate g in (0,1], used to down-weight classifier activity when sequence
    drifts into TM-like / low-charge regions.

      g_cf = sigmoid((cf - cf_min)/s_cf)
      g_hf = sigmoid((hf_max - hf)/s_hf)
      g_nc = sigmoid((nc - nc_min)/s_nc)

      g = (g_cf*g_hf*g_nc)^gate_power, then clamped by gate_floor.
    """
    g_cf = _sigmoid((cf - args.gate_cf_min) / max(args.gate_cf_s, 1e-6))
    g_hf = _sigmoid((args.gate_hf_max - hf) / max(args.gate_hf_s, 1e-6))
    g_nc = _sigmoid((nc - args.gate_nc_min) / max(args.gate_nc_s, 1e-6))
    g = (g_cf * g_hf * g_nc) ** max(args.gate_power, 1e-6)
    return max(float(g), float(args.gate_floor))


# ----------------------------- CSV append (optional) -----------------------------
def _ensure_parent_dir(path: str):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def append_row_csv(csv_path: str, header: str, row: str):
    _ensure_parent_dir(csv_path)
    need_header = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)
    with open(csv_path, "a", encoding="utf-8") as f:
        if need_header:
            f.write(header.rstrip("\n") + "\n")
        f.write(row.rstrip("\n") + "\n")


# ----------------------------- main training loop -----------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if not os.path.isfile(args.generator):
        raise FileNotFoundError(f"Generator checkpoint not found: {args.generator}")

    ckpt = torch.load(args.generator, map_location=device)
    if not (isinstance(ckpt, dict) and "model" in ckpt and "stats" in ckpt):
        raise ValueError(f"Bad generator checkpoint format: {args.generator}")
    stats = ckpt["stats"]

    gen = AMPGenerator().to(device)
    gen.load_state_dict(ckpt["model"])
    gen.train()

    gen_ref = AMPGenerator().to(device)
    gen_ref.load_state_dict(ckpt["model"])
    gen_ref.eval()
    for p in gen_ref.parameters():
        p.requires_grad_(False)

    cls = None
    if args.classifier and args.classifier.lower() != "none":
        cls = AMPActivity(device=device, model_dir=args.classifier)
        print("Loaded classifier from", args.classifier)
    else:
        print("No classifier used (classifier=None)")

    opt = AdamW(gen.parameters(), lr=args.lr)

    # optional CSV logging
    samples_csv = args.samples_csv or (args.out + ".samples_train.csv")
    header = (
        "step,seq,len,act,act_used,gate,act_eff,reward,struct_err,rep,hyd_run,len_pen,"
        "tm_pen,tm_hyd2,tm_cat2,tm_net2,cat_frac,hyd_frac,net_charge,mu,logp,logp_ref\n"
    )

    for step in range(1, args.steps + 1):
        cons = sample_constraints_from_stats(stats)
        max_len = int(round(cons["length"]))

        seq = generate_seq(
            gen, stats, cons, device,
            max_length=max_len,
            temperature=args.temperature,
            top_p=args.top_p
        )
        if not seq:
            continue

        # ---------------- HARD FEASIBILITY FILTERS (conservative root fix) ----------------
        L = len(seq)
        if L < args.len_low or L > args.len_high:
            continue
        hyd_run = max_hydrophobic_run(seq)
        if hyd_run > args.hyd_run_hard_max:
            # cut "signal peptide / TM-like long hydrophobic segments" at the sample level
            continue

        # (3) activity reward from classifier
        r_act = 0.0
        if cls is not None:
            r_act = float(cls.predict_prob(seq))
        act_used = min(r_act, args.act_cap)

        # (4) structural error (zscore mean abs diff)
        feats = calculate_structural_features(seq)
        struct_err = float(
            np.mean(
                [
                    abs(feats[k] - cons[k]) / (stats[k]["std"] + 1e-8)
                    for k in FEATURE_KEYS
                ]
            )
        )

        # (5) anti-cheating penalties
        pen_rep = max_run_length(seq) / max(L, 1)
        pen_hyd_run = hyd_run / max(L, 1)
        pen_len = length_penalty(L, args.len_low, args.len_high)

        # (5b) TM-like penalty: quadratic hinge (steeper than linear)
        hf = hydrophobic_frac(seq)
        cf = cationic_frac(seq)
        nc = net_charge_simple(seq)
        mu = hydrophobic_moment_norm(seq)

# (5c) soft gate: down-weight activity when drifting into low-charge / TM-like region
        gate = soft_gate(cf=cf, hf=hf, nc=nc, args=args)
        act_eff = act_used * gate

        tm_hyd2 = max(0.0, hf - args.tm_hyd_thr) ** 2
        tm_cat2 = max(0.0, args.tm_cat_thr - cf) ** 2
        tm_net2 = (max(0.0, args.tm_net_thr - nc) / max(args.tm_net_scale, 1e-6)) ** 2
        tm_pen = args.tm_a * tm_hyd2 + args.tm_b * tm_cat2 + args.tm_c * tm_net2

        # hydrophobic fraction shaping to keep inside AMP "sweet spot"
        p_hf_center = (hf - args.hf_target) ** 2

        # (6) log probabilities
        z_tgt_t = constraints_to_z(cons, stats, device)
        logp = log_prob_of_seq(gen, seq, z_tgt_t, device)
        if logp is None:
            continue
        with torch.no_grad():
            logp_ref = log_prob_of_seq(gen_ref, seq, z_tgt_t, device)
            if logp_ref is None:
                continue

        # (7) final reward
        reward = (
            args.w_act * act_eff
            - args.w_struct * struct_err
            - args.w_rep * pen_rep
            - args.w_hyd * pen_hyd_run
            - args.w_len * pen_len
            - args.w_tm * tm_pen
            - args.w_hf_center * p_hf_center
            + args.w_mu * mu
            + args.w_prior * float(logp_ref - logp)
        )
        reward = float(np.clip(reward, -args.reward_clip, args.reward_clip))

        # (8) REINFORCE loss
        loss = -reward * logp

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=1.0)
        opt.step()

        # optional: turn on if you want dense logging to csv
        # append_row_csv(
        #     samples_csv,
        #     header,
        #     f"{step},{seq},{L},{r_act:.6f},{act_used:.6f},{gate:.6f},{act_eff:.6f},{reward:.6f},{struct_err:.6f},"
        #     f"{pen_rep:.6f},{pen_hyd_run:.6f},{pen_len:.6f},"
        #     f"{tm_pen:.6f},{tm_hyd2:.6f},{tm_cat2:.6f},{tm_net2:.6f},"
        #     f"{cf:.6f},{hf:.6f},{nc:.3f},{mu:.6f},"
        #     f"{float(logp):.6f},{float(logp_ref):.6f}"
        # )

        if step % args.log_every == 0:
            print(
                f"[step {step}] reward={reward:.3f} act={r_act:.3f} act_used={act_used:.3f} gate={gate:.3f} act_eff={act_eff:.3f} "
                f"struct_err={struct_err:.3f} rep={pen_rep:.3f} hyd_run={pen_hyd_run:.3f} len_pen={pen_len:.3f} "
                f"tm_pen={tm_pen:.3f} (hyd2={tm_hyd2:.3f} cat2={tm_cat2:.3f} net2={tm_net2:.3f}) "
                f"cf={cf:.3f} hf={hf:.3f} nc={nc:.2f} mu={mu:.3f} "
                f"loss={loss.item():.4f} seq={seq}"
            )

        if step % args.save_every == 0:
            torch.save({"model": gen.state_dict(), "stats": stats}, args.out)
            print("Saved RL-tuned generator to", args.out)

    torch.save({"model": gen.state_dict(), "stats": stats}, args.out)
    print("Done. Final generator saved to", args.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--generator", type=str, required=True, help="generator checkpoint (dict with model+stats)")
    ap.add_argument("--classifier", type=str, required=True, help="classifier checkpoint dir (HF) or 'none'")

    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)

    # base reward weights
    ap.add_argument("--w_act", type=float, default=1.0)
    ap.add_argument("--w_struct", type=float, default=0.3)
    ap.add_argument("--act_cap", type=float, default=0.95)

    ap.add_argument("--w_rep", type=float, default=1.0)
    ap.add_argument("--w_hyd", type=float, default=1.0)
    ap.add_argument("--w_len", type=float, default=0.5)
    ap.add_argument("--len_low", type=int, default=12)
    ap.add_argument("--len_high", type=int, default=60)

    ap.add_argument("--w_prior", type=float, default=0.02)
    ap.add_argument("--reward_clip", type=float, default=5.0)

    # hard feasibility filter (conservative)
    ap.add_argument("--hyd_run_hard_max", type=int, default=12,
                    help="Hard filter: max hydrophobic run allowed before skipping sample")

    # TM-like penalty params (quadratic hinge)
    ap.add_argument("--w_tm", type=float, default=3.0)
    ap.add_argument("--tm_hyd_thr", type=float, default=0.58, help="hyd_frac threshold (start penalizing)")
    ap.add_argument("--tm_cat_thr", type=float, default=0.15, help="cationic_frac minimum target")
    ap.add_argument("--tm_net_thr", type=float, default=3.0, help="net charge minimum target (rough)")
    ap.add_argument("--tm_net_scale", type=float, default=4.0)

    ap.add_argument("--tm_a", type=float, default=1.0)
    ap.add_argument("--tm_b", type=float, default=1.0)
    ap.add_argument("--tm_c", type=float, default=1.0)

    # amphipathicity + sweet-spot shaping
    ap.add_argument("--w_mu", type=float, default=0.3, help="Reward weight for hydrophobic moment (amphipathicity proxy)")
    ap.add_argument("--w_hf_center", type=float, default=0.2, help="Penalty weight for (hf-hf_target)^2")
    ap.add_argument("--hf_target", type=float, default=0.50, help="hydrophobic fraction target center")

    # soft gate (feasibility gating for activity)
    ap.add_argument("--gate_cf_min", type=float, default=0.12, help="soft gate: min cationic fraction")
    ap.add_argument("--gate_hf_max", type=float, default=0.65, help="soft gate: max hydrophobic fraction")
    ap.add_argument("--gate_nc_min", type=float, default=3.0, help="soft gate: min net charge (rough)")

    ap.add_argument("--gate_cf_s", type=float, default=0.03, help="soft gate steepness for cf (smaller=harder)")
    ap.add_argument("--gate_hf_s", type=float, default=0.03, help="soft gate steepness for hf (smaller=harder)")
    ap.add_argument("--gate_nc_s", type=float, default=0.8, help="soft gate steepness for nc (smaller=harder)")

    ap.add_argument("--gate_power", type=float, default=1.2, help="soft gate power (>1 more conservative)")
    ap.add_argument("--gate_floor", type=float, default=0.02, help="lower bound for gate to avoid zeroing gradients")

    ap.add_argument("--out", type=str, default="checkpoints/generator_rl.pt")
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--save_every", type=int, default=1000)

    ap.add_argument("--samples_csv", type=str, default="", help="optional csv for training samples (disabled by default)")

    args = ap.parse_args()
    main(args)
