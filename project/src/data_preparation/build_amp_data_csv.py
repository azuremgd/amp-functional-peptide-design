from pathlib import Path
import pandas as pd

# ===== 路径（按你工程结构）=====
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

POS = DATA / "positive_all.csv"          # amp=1
NEG = DATA / "uniprot_negative.csv"      # amp=0
OUT = DATA / "AMP-data.csv"              # 最终输出
STATS = DATA / "AMP-data_stats.txt"

# ===== 统一规则（必须和前面一致）=====
MIN_LEN, MAX_LEN = 8, 60
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
SEED = 42

def clean_seq(s: str):
    if not isinstance(s, str):
        return None
    s = s.strip().upper()
    if not s:
        return None
    if not (MIN_LEN <= len(s) <= MAX_LEN):
        return None
    if set(s) - VALID_AA:
        return None
    return s

def load_csv(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}: {path}")
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "seq" not in df.columns:
        raise ValueError(f"{name} missing seq column: {list(df.columns)}")
    if "amp" not in df.columns:
        raise ValueError(f"{name} missing amp column: {list(df.columns)}")
    df = df[["seq", "amp"]].copy()
    df["amp"] = df["amp"].astype(int)
    return df

def main():
    pos = load_csv(POS, "positive_all.csv")
    neg = load_csv(NEG, "uniprot_negative.csv")

    # 合并
    df = pd.concat([pos, neg], ignore_index=True)

    n_raw = len(df)

    # 统一清洗（防止某一步遗漏）
    df["seq_clean"] = df["seq"].apply(clean_seq)
    n_filtered = df["seq_clean"].isna().sum()
    df = df.dropna(subset=["seq_clean"]).copy()

    # 只保留清洗后的序列
    df["seq"] = df["seq_clean"]
    df = df.drop(columns=["seq_clean"])

    # 关键：同一条 seq 如果同时出现在正负中，优先保留正样本（amp=1）
    # 做法：先按 amp 降序排序，再按 seq 去重保留第一条
    df = df.sort_values(by=["seq", "amp"], ascending=[True, False])
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["seq"], keep="first").copy()
    after_dedup = len(df)

    # 打乱（固定随机种子，保证复现）
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    # 统计
    n_pos = int((df["amp"] == 1).sum())
    n_neg = int((df["amp"] == 0).sum())

    # 写出
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    STATS.write_text(
        "\n".join([
            f"raw_rows={n_raw}",
            f"filtered(len/AA)={n_filtered}",
            f"before_dedup={before_dedup}",
            f"after_dedup={after_dedup}",
            f"final_pos={n_pos}",
            f"final_neg={n_neg}",
            f"saved={OUT}"
        ]) + "\n",
        encoding="utf-8"
    )

    print("[DONE] built AMP-data.csv")
    print(f"  -> {OUT} rows={len(df)} pos={n_pos} neg={n_neg}")
    print(f"  -> {STATS}")

if __name__ == "__main__":
    main()
