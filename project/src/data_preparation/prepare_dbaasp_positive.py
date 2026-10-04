from pathlib import Path
import json
import time
import urllib.request
import urllib.parse

OUT_CSV = Path("data/dbaasp_positive.csv")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

MIN_LEN, MAX_LEN = 8, 60
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

BASE = "https://dbaasp.org/peptides"

def fetch(limit: int, offset: int) -> dict:
    params = {
        "complexity.value": "monomer",
        "sequenceLength.value": f"{MIN_LEN}-{MAX_LEN}",
        "limit": str(limit),
        "offset": str(offset),
    }
    url = BASE + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))

def is_clean(seq: str) -> bool:
    if not (MIN_LEN <= len(seq) <= MAX_LEN):
        return False
    if set(seq) - VALID_AA:
        return False
    return True

def main():
    limit = 1000
    offset = 0

    # 先拿总量
    first = fetch(limit=1, offset=0)
    total = int(first.get("totalCount", 0))
    print(f"[INFO] totalCount = {total}")

    seqs = set()
    while offset < total:
        data = fetch(limit=limit, offset=offset)
        items = data.get("data", [])
        if not items:
            print(f"[WARN] empty page at offset={offset}, stop.")
            break

        for it in items:
            s = it.get("sequence", "")
            if s and is_clean(s):
                seqs.add(s)

        offset += len(items)
        print(f"[INFO] fetched {offset}/{total}, unique_clean={len(seqs)}")
        time.sleep(0.5)  # 轻微限速，避免被封

    # 输出 CSV：seq,amp
    lines = ["seq,amp\n"]
    for s in sorted(seqs):
        lines.append(f"{s},1\n")

    OUT_CSV.write_text("".join(lines), encoding="utf-8")
    print(f"[DONE] saved: {OUT_CSV} (n={len(seqs)})")

if __name__ == "__main__":
    main()
