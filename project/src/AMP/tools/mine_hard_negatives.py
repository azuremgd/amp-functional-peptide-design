# -*- coding: utf-8 -*-
import os, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from AMP.models.generator_model import AMPGenerator
from AMP.models.classifier import AMPActivity
from AMP.features.features import FEATURE_KEYS
from AMP.data_processing.vocab import BOS_ID, encode_sequence


def sample_constraints_from_stats(stats, len_low, len_high):
    cons = {}
    for k in FEATURE_KEYS:
        mu, sigma = stats[k]["mean"], stats[k]["std"]
        cons[k] = float(np.random.normal(mu, sigma))
    cons["length"] = float(np.clip(cons["length"], len_low, len_high))
    return cons


def constraints_to_z(cons, stats, device):
    z = [(cons[k] - stats[k]["mean"]) / (stats[k]["std"] + 1e-8) for k in FEATURE_KEYS]
    return torch.tensor(z, dtype=torch.float32, device=device).unsqueeze(0)


def generate_seq(model, stats, cons, device, temperature=1.0, top_p=0.95):
    from AMP.data_processing.vocab import decode_sequence

    max_length = int(round(cons["length"]))
    z_tgt_t = constraints_to_z(cons, stats, device)

    seq_ids = [BOS_ID]
    for _ in range(max_length):
        inp = torch.tensor(seq_ids, dtype=torch.long, device=device).unsqueeze(0)
        logits, _ = model(inp, z_tgt_t)
        next_logits = logits[0, -1] / max(temperature, 1e-6)
        probs = F.softmax(next_logits, dim=-1)

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

    return decode_sequence([i for i in seq_ids if i < 20])


def max_run_length(seq: str) -> int:
    if not seq: return 0
    m = cur = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i-1]:
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


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ckpt = torch.load(args.generator, map_location=device)
    stats = ckpt["stats"]

    gen = AMPGenerator().to(device)
    gen.load_state_dict(ckpt["model"])
    gen.eval()

    cls = AMPActivity(device=device, model_dir=args.classifier)

    rng = np.random.default_rng(args.seed)

    rows = []
    seen = set()

    for i in range(args.n_samples):
        cons = sample_constraints_from_stats(stats, args.len_low, args.len_high)
        seq = generate_seq(gen, stats, cons, device, temperature=args.temperature, top_p=args.top_p)
        if not seq:
            continue

        if args.dedup and seq in seen:
            continue

        L = len(seq)
        rep = max_run_length(seq) / max(L, 1)
        hyd = max_hydrophobic_run(seq) / max(L, 1)
        act = float(cls.predict_prob(seq))

        # hard 条件：高 act + (任一漏洞条件)
        bad = False
        if L < args.len_low or L > args.len_high:
            bad = True
        if max_run_length(seq) > args.max_run:
            bad = True
        if max_hydrophobic_run(seq) > args.max_hyd_run:
            bad = True
        # 可选：unique aa 太少（低复杂度）
        if args.min_unique > 0 and len(set(seq)) < args.min_unique:
            bad = True

        if act >= args.act_thr and bad:
            rows.append({
                "seq": seq,
                "amp": 0,
                "act": act,
                "rep": rep,
                "hyd": hyd,
                "length": L
            })
            seen.add(seq)

        if (i + 1) % args.log_every == 0:
            print(f"[mine] {i+1}/{args.n_samples} collected={len(rows)}")

        if args.max_keep > 0 and len(rows) >= args.max_keep:
            break

    df = pd.DataFrame(rows).drop_duplicates(subset=["seq"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[?] saved hard negatives: {len(df)} -> {args.out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", type=str, required=True)
    ap.add_argument("--classifier", type=str, required=True)
    ap.add_argument("--out_csv", type=str, default="data/hard_neg_round1.csv")

    ap.add_argument("--n_samples", type=int, default=50000)
    ap.add_argument("--max_keep", type=int, default=5000)
    ap.add_argument("--act_thr", type=float, default=0.95)

    ap.add_argument("--len_low", type=int, default=12)
    ap.add_argument("--len_high", type=int, default=60)
    ap.add_argument("--max_run", type=int, default=6)
    ap.add_argument("--max_hyd_run", type=int, default=10)
    ap.add_argument("--min_unique", type=int, default=0)

    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--dedup", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_every", type=int, default=2000)
    args = ap.parse_args()
    main(args)
