from pathlib import Path
import pandas as pd

# =========================
# 路径：按你的工程结构固定
# =========================
ROOT = Path(__file__).resolve().parents[1]          # .../AMP
DATA_DIR = ROOT / "data"

APD_CSV = DATA_DIR / "apd_positive.csv"
DBAASP_CSV = DATA_DIR / "dbaasp_positive.csv"

OUT_CSV = DATA_DIR / "positive_all.csv"
OUT_STATS = DATA_DIR / "positive_all_stats.txt"

# =========================
# 统一筛选规则（与你之前一致）
# =========================
MIN_LEN = 8
MAX_LEN = 60
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

def clean_seq(s: str) -> str | None:
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

def load_csv(path: Path, source_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {source_name} file: {path}")
    df = pd.read_csv(path)
    # 兼容列名大小写
    df.columns = [c.strip().lower() for c in df.columns]
    if "seq" not in df.columns:
        raise ValueError(f"{source_name} missing 'seq' column: {path} cols={list(df.columns)}")
    # amp 列可以不存在（我们会强制置为 1）
    return df[["seq"]].copy()

def main():
    apd = load_csv(APD_CSV, "APD")
    dbaasp = load_csv(DBAASP_CSV, "DBAASP")

    n_apd_raw = len(apd)
    n_dbaasp_raw = len(dbaasp)

    # 合并（仅保留 seq）
    df = pd.concat([apd.assign(source="APD"), dbaasp.assign(source="DBAASP")], ignore_index=True)
    n_merged_raw = len(df)

    # 清洗
    df["seq_clean"] = df["seq"].apply(clean_seq)
    n_invalid = df["seq_clean"].isna().sum()
    df = df.dropna(subset=["seq_clean"]).copy()

    # 全局去重（按清洗后的序列）
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["seq_clean"], keep="first").copy()
    after_dedup = len(df)

    # 输出最终正样本
    out = pd.DataFrame({"seq": sorted(df["seq_clean"].tolist()), "amp": 1})
    out.to_csv(OUT_CSV, index=False)

    # 输出统计日志（便于你“工作留痕/组会汇报”）
    lines = []
    lines.append(f"APD raw rows: {n_apd_raw}\n")
    lines.append(f"DBAASP raw rows: {n_dbaasp_raw}\n")
    lines.append(f"Merged raw rows: {n_merged_raw}\n")
    lines.append(f"Invalid/filtered rows (len/AA): {n_invalid}\n")
    lines.append(f"After cleaning: {before_dedup}\n")
    lines.append(f"After global dedup: {after_dedup}\n")
    lines.append(f"Saved: {OUT_CSV}\n")
    OUT_STATS.write_text("".join(lines), encoding="utf-8")

    print("[DONE] positive merge finished")
    print(f"  -> {OUT_CSV} (n={len(out)})")
    print(f"  -> {OUT_STATS}")

if __name__ == "__main__":
    main()
