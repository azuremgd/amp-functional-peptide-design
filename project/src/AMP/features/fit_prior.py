# fit_prior.py
import argparse, pickle, numpy as np, pandas as pd
from AMP.features.features import FEATURE_KEYS, calculate_structural_features
from AMP.data_processing.data import AMPSeqDataset


def main(args):
    df = pd.read_csv(args.data)
    ds = AMPSeqDataset(df)  # 复用清洗
    feats = []
    for s in ds.seqs:
        f = calculate_structural_features(s)
        feats.append([f[k] for k in FEATURE_KEYS])
    X = np.array(feats, dtype=np.float32)  # [N,5]
    prior = {
        "mean": X.mean(axis=0),
        "cov" : np.cov(X.T),
        "p25" : np.percentile(X, 25, axis=0),
        "p50" : np.percentile(X, 50, axis=0),
        "p75" : np.percentile(X, 75, axis=0),
        "p2_5": np.percentile(X, 2.5, axis=0),
        "p97_5": np.percentile(X, 97.5, axis=0),
        "feature_keys": FEATURE_KEYS,
    }
    with open(args.out, "wb") as f:
        pickle.dump(prior, f)
    print("Saved prior to", args.out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/AMP-data.csv")
    ap.add_argument("--out", type=str, default="checkpoints/prior.pkl")
    args = ap.parse_args()
    main(args)
