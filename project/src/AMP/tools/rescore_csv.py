# AMP/tools/rescore_csv.py
import argparse
import os
import pandas as pd

from AMP.models.classifier import AMPActivity


def guess_seq_col(df: pd.DataFrame) -> str:
    # 常见列名：seq / sequence / Sequence
    for c in ["seq", "sequence", "Sequence", "SEQ"]:
        if c in df.columns:
            return c
    # 兜底：找第一个看起来像序列的 object 列
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    if not obj_cols:
        raise ValueError(f"Cannot find sequence column. Columns={list(df.columns)}")
    return obj_cols[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="Input CSV containing sequences")
    ap.add_argument("--classifier", required=True, help="Classifier checkpoint dir (checkpoint_best)")
    ap.add_argument("--out_csv", required=True, help="Output CSV with rescored act")
    ap.add_argument("--seq_col", default="", help="Sequence column name (auto if empty)")
    ap.add_argument("--batch", type=int, default=64, help="Batch size for scoring")
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    ap.add_argument("--act_col", default="act_rescore", help="Output act column name")
    args = ap.parse_args()

    if not os.path.exists(args.in_csv):
        raise FileNotFoundError(f"in_csv not found: {args.in_csv}")
    if not os.path.exists(args.classifier):
        raise FileNotFoundError(f"classifier path not found: {args.classifier}")

    df = pd.read_csv(args.in_csv)
    seq_col = args.seq_col.strip() or guess_seq_col(df)

    # 清理序列
    seqs = df[seq_col].astype(str).fillna("").tolist()

    # 加载分类器
    cls = AMPActivity(device=args.device, model_dir=args.classifier)
    print(f"[OK] Loaded classifier: {args.classifier}")
    print(f"[INFO] in_csv={args.in_csv} rows={len(df)} seq_col={seq_col}")

    # 打分（尽量批处理；如果 AMPActivity 没有 batch 接口就逐条）
    acts = []
    # 兼容：如果类里提供 predict_prob_batch，就用；否则逐条 predict_prob
    if hasattr(cls, "predict_prob_batch"):
        bs = max(1, int(args.batch))
        for i in range(0, len(seqs), bs):
            acts.extend(cls.predict_prob_batch(seqs[i:i + bs]))
            if (i // bs) % 20 == 0:
                print(f"[score] {min(i+bs, len(seqs))}/{len(seqs)}")
    else:
        for i, s in enumerate(seqs, start=1):
            acts.append(float(cls.predict_prob(s)))
            if i % 200 == 0:
                print(f"[score] {i}/{len(seqs)}")

    df[args.act_col] = acts

    # 统计
    a = pd.Series(acts)
    print(f"[STAT] n={len(a)} mean={a.mean():.6f} median={a.median():.6f}")
    for t in [0.9, 0.95, 0.99]:
        print(f"[STAT] act>={t}: {(a >= t).mean():.4f}")

    # 写文件
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[OK] Saved: {args.out_csv}")


if __name__ == "__main__":
    main()
