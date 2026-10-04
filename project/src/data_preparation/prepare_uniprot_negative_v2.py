from pathlib import Path
import time
import re
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

POS_CSV = DATA_DIR / "positive_all.csv"
OUT_NEG = DATA_DIR / "uniprot_negative.csv"
OUT_STATS = DATA_DIR / "uniprot_negative_stats.txt"

MIN_LEN, MAX_LEN = 8, 60
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

BASE = "https://rest.uniprot.org/uniprotkb/search"

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

def extract_next_cursor_from_link_header(link: str | None) -> str | None:
    """
    Link: <...&cursor=XYZ...>; rel="next", <...>; rel="last"
    """
    if not link:
        return None
    # 找 rel="next" 那段
    m = re.search(r'<([^>]+)>\s*;\s*rel="next"', link)
    if not m:
        return None
    next_url = m.group(1)
    m2 = re.search(r'[?&]cursor=([^&]+)', next_url)
    return m2.group(1) if m2 else None

def main():
    pos = pd.read_csv(POS_CSV)
    pos.columns = [c.strip().lower() for c in pos.columns]
    assert "seq" in pos.columns, f"positive_all.csv missing seq column: {pos.columns}"
    target_n = len(pos)
    pos_set = set(pos["seq"].astype(str).str.strip().str.upper().tolist())

    print(f"[INFO] target negatives: {target_n}")

    sess = requests.Session()
    sess.headers.update({"User-Agent": "Mozilla/5.0"})

    query = f'(NOT keyword:KW-0929) AND (length:[{MIN_LEN} TO {MAX_LEN}])'
    params = {
        "query": query,
        "format": "json",
        "fields": "accession,sequence",
        "size": 500,   # 稳一点
    }

    neg_set = set()
    cursor = None
    pages = 0
    filtered = 0
    crossdup = 0

    while len(neg_set) < target_n:
        p = dict(params)
        if cursor:
            p["cursor"] = cursor

        r = sess.get(BASE, params=p, timeout=60)
        r.raise_for_status()
        pages += 1

        data = r.json()
        results = data.get("results", [])
        if not results:
            print("[WARN] empty results, stop.")
            break

        for it in results:
            seq = (it.get("sequence") or {}).get("value", "")
            cs = clean_seq(seq)
            if cs is None:
                filtered += 1
                continue
            if cs in pos_set:
                crossdup += 1
                continue
            neg_set.add(cs)

        cursor = extract_next_cursor_from_link_header(r.headers.get("Link"))
        print(f"[INFO] page={pages} got={len(results)} kept={len(neg_set)}/{target_n} "
              f"filtered={filtered} crossdup={crossdup} next_cursor={'yes' if cursor else 'no'}")

        time.sleep(0.4)

        if not cursor and len(neg_set) < target_n:
            print("[WARN] no next cursor but not enough negatives, stop.")
            break

    neg_list = sorted(list(neg_set))[:target_n]
    df_neg = pd.DataFrame({"seq": neg_list, "amp": [0] * len(neg_list)})
    df_neg.to_csv(OUT_NEG, index=False)

    stats = [
        f"target_n={target_n}\n",
        f"final_neg_n={len(df_neg)}\n",
        f"pages={pages}\n",
        f"filtered(len/AA)={filtered}\n",
        f"cross_dup_with_positive={crossdup}\n",
        f"saved={OUT_NEG}\n"
    ]
    OUT_STATS.write_text("".join(stats), encoding="utf-8")

    print(f"[DONE] saved negatives: {OUT_NEG} (n={len(df_neg)})")
    print(f"[DONE] stats: {OUT_STATS}")

if __name__ == "__main__":
    main()
