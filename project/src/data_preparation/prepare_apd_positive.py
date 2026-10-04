from pathlib import Path
import pandas as pd

# 脚本所在目录：AMP/tools
ROOT = Path(__file__).resolve().parent.parent

FASTA_PATH = ROOT / "data" / "naturalAMPs_APD2024a.fasta"
OUT_CSV = ROOT / "data" / "apd_positive.csv"


MIN_LEN = 8
MAX_LEN = 60
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

# =========================
# 1. 读取 FASTA
# =========================
seqs = []

with FASTA_PATH.open() as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            continue
        seqs.append(line)

print(f"[INFO] Loaded {len(seqs)} raw sequences from APD FASTA")

# =========================
# 2. 清洗：字符 + 长度
# =========================
clean_seqs = []

for s in seqs:
    if not (MIN_LEN <= len(s) <= MAX_LEN):
        continue
    if set(s) - VALID_AA:
        continue
    clean_seqs.append(s)

print(f"[INFO] After length & AA filtering: {len(clean_seqs)}")

# =========================
# 3. 去重
# =========================
clean_seqs = sorted(set(clean_seqs))

print(f"[INFO] After deduplication: {len(clean_seqs)} unique sequences")

# =========================
# 4. 生成 CSV（amp = 1）
# =========================
df = pd.DataFrame({
    "seq": clean_seqs,
    "amp": [1] * len(clean_seqs)
})

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_CSV, index=False)

print(f"[DONE] Saved positive AMP CSV to: {OUT_CSV}")
print(df.head())
