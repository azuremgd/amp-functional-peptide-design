# AMP/tools/export_candidates.py
import argparse, os
import pandas as pd
import torch

from AMP.models.classifier import AMPActivity
from AMP.models.generator_model import AMPGenerator

# 复用你现成挖矿脚本里的函数（保持一致性）
from AMP.tools.mine_hard_negatives import (
    generate_seq,
    sample_constraints_from_stats,
    max_run_length,
    max_hydrophobic_run,
)

def load_generator(gen_ckpt, device):
    ckpt = torch.load(gen_ckpt, map_location=device)
    if not (isinstance(ckpt, dict) and "model" in ckpt and "stats" in ckpt):
        raise ValueError(f"Bad generator ckpt format: {gen_ckpt}")
    stats = ckpt["stats"]
    gen = AMPGenerator().to(device)
    gen.load_state_dict(ckpt["model"])
    gen.eval()
    return gen, stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", required=True)
    ap.add_argument("--classifier", required=True)
    ap.add_argument("--out_csv", required=True)

    ap.add_argument("--n_samples", type=int, default=20000)
    ap.add_argument("--topk", type=int, default=200)

    ap.add_argument("--act_thr", type=float, default=0.90)
    ap.add_argument("--len_low", type=int, default=12)
    ap.add_argument("--len_high", type=int, default=60)
    ap.add_argument("--max_run", type=int, default=5)
    ap.add_argument("--max_hyd_run", type=int, default=7)

    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dedup", action="store_true")
    ap.add_argument("--log_every", type=int, default=500)

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    gen, stats = load_generator(args.generator, device)

    cls = AMPActivity(device=str(device), model_dir=args.classifier)
    print("Loaded classifier:", args.classifier)

    seen = set()
    kept = []

    with torch.no_grad():
        for i in range(1, args.n_samples + 1):
            # ? 修复点：你的函数需要 (stats, len_low, len_high)
            cons = sample_constraints_from_stats(stats, args.len_low, args.len_high)

            seq = generate_seq(
                gen, stats, cons, device=device,
                temperature=args.temperature, top_p=args.top_p
            )
            if not seq:
                continue

            if args.dedup:
                if seq in seen:
                    continue
                seen.add(seq)

            L = len(seq)
            if L < args.len_low or L > args.len_high:
                continue

            mr = max_run_length(seq)
            mhr = max_hydrophobic_run(seq)
            if mr > args.max_run:
                continue
            if mhr > args.max_hyd_run:
                continue

            act = float(cls.predict_prob(seq))
            if act < args.act_thr:
                continue

            kept.append((act, seq, L, mr, mhr))

            if (i % args.log_every) == 0:
                print(f"[export] {i}/{args.n_samples} kept={len(kept)}")

    kept.sort(key=lambda x: x[0], reverse=True)
    kept = kept[:args.topk]

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    df = pd.DataFrame(kept, columns=["act", "seq", "len", "max_run", "max_hyd_run"])
    df.to_csv(args.out_csv, index=False)
    print(f"[OK] saved top{len(df)} -> {args.out_csv}")

if __name__ == "__main__":
    main()
