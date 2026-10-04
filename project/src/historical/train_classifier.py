# -*- coding: utf-8 -*-
import os, argparse, csv
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from AMP.models.classifier import AMPActivity


# -------------------------- Dataset ------------------------------
class AMPClsDataset:
    def __init__(self, df):
        df = df.copy()
        df.columns = df.columns.str.strip()
        colmap = {c.lower(): c for c in df.columns}
        assert 'seq' in colmap and 'amp' in colmap, f"Need columns: seq, AMP (or amp). got={list(df.columns)}"
        seq_col = colmap['seq']
        amp_col = colmap['amp']

        self.seqs = ["".join(str(s).split()) for s in df[seq_col]]
        self.labels = [int(y) for y in df[amp_col]]

    def __len__(self): return len(self.seqs)
    def __getitem__(self, i): return self.seqs[i], self.labels[i]


# -------------------------- Helpers ------------------------------
def scan_best_threshold_by_f1(y_true: np.ndarray, y_prob: np.ndarray, min_thr=0.05, max_thr=0.95, num=91):
    thresholds = np.linspace(min_thr, max_thr, num)
    best_thr, best_f1 = 0.5, -1.0
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        _, _, f1_thr, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        if f1_thr > best_f1:
            best_f1 = float(f1_thr)
            best_thr = float(thr)
    return best_thr, best_f1


def evaluate_probs(model: AMPActivity, dataloader, device):
    model.model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]
            outputs = model.model(**batch)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1]
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = float("nan")
    return y_true, y_prob, auc


def metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, thr: float):
    y_pred = (y_prob >= thr).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return acc, float(prec), float(rec), float(f1)


def _normalize_seq_col(df: pd.DataFrame) -> pd.Series:
    colmap = {c.lower(): c for c in df.columns}
    if "seq" not in colmap:
        raise ValueError(f"extra_neg csv must contain 'seq' column. got={list(df.columns)}")
    seq_col = colmap["seq"]
    return df[seq_col].astype(str).apply(lambda s: "".join(str(s).split()))


def load_extra_negatives(path: str, min_act: float = None, max_rows: int = None) -> pd.DataFrame:
    """
    读取 hard negatives CSV（至少包含 seq；可选包含 act 列用于进一步筛选）
    返回 df with columns: seq, amp(=0)
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    seqs = _normalize_seq_col(df)
    out = pd.DataFrame({"seq": seqs, "amp": 0})

    # 可选：如果 CSV 里有 act 列，则按 min_act 过滤（避免引入“并不hard”的负例）
    if min_act is not None:
        colmap = {c.lower(): c for c in df.columns}
        if "act" in colmap:
            act_col = colmap["act"]
            mask = df[act_col].astype(float) >= float(min_act)
            out = out[mask.values].copy()

    # 去空与去重
    out = out[out["seq"].astype(str).str.len() > 0]
    out = out.drop_duplicates(subset=["seq"], keep="first")

    if max_rows is not None and len(out) > int(max_rows):
        out = out.sample(n=int(max_rows), random_state=0).reset_index(drop=True)

    return out.reset_index(drop=True)


# -------------------------- Main ----------------------------------
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    # ===================== 读取主数据 ===========================
    if not os.path.isfile(args.data):
        raise FileNotFoundError(f"classifier training csv not found: {args.data}")
    df = pd.read_csv(args.data)
    df.columns = df.columns.str.strip()

    colmap = {c.lower(): c for c in df.columns}
    if 'amp' not in colmap:
        raise ValueError(f"Cannot find label column AMP/amp in columns={list(df.columns)}")
    amp_col = colmap['amp']

    # ===================== 主数据切分 ===========================
    df_train, df_temp = train_test_split(df, test_size=0.3, stratify=df[amp_col], random_state=args.seed)
    df_val, df_test  = train_test_split(df_temp, test_size=0.5, stratify=df_temp[amp_col], random_state=args.seed)

    # ===================== 追加 hard negatives（只进训练集） ===========================
    if args.extra_neg is not None and args.extra_neg.lower() != "none":
        if not os.path.isfile(args.extra_neg):
            raise FileNotFoundError(f"extra_neg csv not found: {args.extra_neg}")

        extra_neg = load_extra_negatives(
            args.extra_neg,
            min_act=args.extra_neg_min_act,
            max_rows=args.extra_neg_max,
        )

        # 防止泄漏：从 extra_neg 中剔除出现在 val/test 的序列
        val_seqs = set(_normalize_seq_col(df_val).tolist())
        test_seqs = set(_normalize_seq_col(df_test).tolist())
        before = len(extra_neg)
        extra_neg = extra_neg[~extra_neg["seq"].isin(val_seqs)]
        extra_neg = extra_neg[~extra_neg["seq"].isin(test_seqs)]
        extra_neg = extra_neg.drop_duplicates(subset=["seq"], keep="first").reset_index(drop=True)
        after = len(extra_neg)

        print(f"[HardNeg] loaded={before}, after_remove_val_test={after}")

        # 训练权重：通过重复拼接实现（无需改 AMPActivity.train_step）
        # repeat=3 相当于 hard neg loss 权重提升 ~3 倍
        if args.extra_neg_repeat > 1 and len(extra_neg) > 0:
            extra_neg = pd.concat([extra_neg] * int(args.extra_neg_repeat), ignore_index=True)

        # ===== 关键修复：把 extra_neg 的列名对齐到主数据的真实列名 =====
        seq_col = colmap["seq"]   # 主数据 seq 列真实名字（可能就是 "seq"）
        amp_col = colmap["amp"]   # 主数据标签列真实名字（可能是 "AMP"）

        # extra_neg 里当前列是标准化的 "seq" 和 "amp"，把它们重命名为主数据列名
        extra_neg = extra_neg.rename(columns={"seq": seq_col, "amp": amp_col})

        # 防御：确保标签列没有 NaN
        extra_neg = extra_neg.dropna(subset=[amp_col]).reset_index(drop=True)


        # 拼到训练集
        df_train = pd.concat([df_train, extra_neg], ignore_index=True)

        # 训练集内部去重（可选）：避免原训练集中已有相同 seq
        if args.extra_neg_dedup_train:
            df_train["__seq_norm__"] = _normalize_seq_col(df_train)
            df_train = df_train.drop_duplicates(subset=["__seq_norm__", amp_col], keep="first")
            df_train = df_train.drop(columns=["__seq_norm__"])

        print(f"[HardNeg] Train after merge: {len(df_train)}")

    print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

    train_ds = AMPClsDataset(df_train)
    val_ds   = AMPClsDataset(df_val)
    test_ds  = AMPClsDataset(df_test)

    # 默认 freeze backbone（更稳）
    model = AMPActivity(device=device, freeze_backbone=(not args.unfreeze), train_pooler=args.train_pooler)

    # 参数分组：backbone(小lr) + head(大lr)
    head_params = []
    backbone_params = []

    if hasattr(model.model, "classifier"):
        head_params = list(model.model.classifier.parameters())

    if hasattr(model.model, "bert") and hasattr(model.model.bert, "encoder"):
        for _, p in model.model.bert.named_parameters():
            if p.requires_grad:
                backbone_params.append(p)

    if hasattr(model.model, "bert") and hasattr(model.model.bert, "pooler"):
        for p in model.model.bert.pooler.parameters():
            if p.requires_grad:
                backbone_params.append(p)

    backbone_params = list({id(p): p for p in backbone_params}.values())
    head_params = list({id(p): p for p in head_params}.values())

    print("Trainable params:",
        sum(p.numel() for p in backbone_params) + sum(p.numel() for p in head_params))

    opt = optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr_backbone, "weight_decay": args.wd},
            {"params": head_params, "lr": args.lr_head, "weight_decay": args.wd},
        ]
    )

    # collate
    def collate_fn(batch):
        seqs, ys = zip(*batch)
        enc = model.encode(seqs)
        enc["labels"] = torch.tensor(ys, dtype=torch.long, device=device)
        return enc

    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, collate_fn=collate_fn)
    val_dl   = DataLoader(val_ds,   batch_size=args.bs, shuffle=False, collate_fn=collate_fn)
    test_dl  = DataLoader(test_ds,  batch_size=args.bs, shuffle=False, collate_fn=collate_fn)

    # paths
    os.makedirs(args.out, exist_ok=True)
    best_model_path = os.path.join(args.out, "checkpoint_best")
    log_csv_path    = os.path.join(args.out, "training_log_classifier.csv")
    best_thr_path   = os.path.join(best_model_path, "best_threshold.txt")

    if not os.path.exists(log_csv_path):
        with open(log_csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_auc", "best_val_f1", "best_thr"])

    best_score = -1.0
    patience_left = args.patience

    # ======================= 训练循环 ============================
    for ep in range(1, args.epochs + 1):
        model.model.train()
        total_loss = 0.0

        for batch in tqdm(train_dl, desc=f"cls ep{ep}"):
            loss = model.train_step(batch, opt)
            total_loss += loss

        train_loss = total_loss / max(len(train_dl), 1)

        # 验证：先拿 y_prob，再在验证集找 best threshold
        y_true, y_prob, val_auc = evaluate_probs(model, val_dl, device)
        best_thr, best_f1 = scan_best_threshold_by_f1(y_true, y_prob)

        print(
            f"[Epoch {ep}] train_loss={train_loss:.4f}, "
            f"val_auc={val_auc:.4f}, best_val_f1={best_f1:.4f} @ thr={best_thr:.3f}"
        )

        with open(log_csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([ep, train_loss, val_auc, best_f1, best_thr])

        score = val_auc if args.monitor == "auc" else best_f1

        if score > best_score + 1e-6:
            best_score = score
            patience_left = args.patience
            print(f"New best ({args.monitor}): {best_score:.4f}  (F1={best_f1:.4f} @ thr={best_thr:.3f}, AUC={val_auc:.4f})")
            model.save(best_model_path)
            with open(best_thr_path, "w") as f:
                f.write(str(best_thr))
        else:
            patience_left -= 1
            if patience_left == 0:
                print("Early stopping triggered.")
                break

    # ====================== 测试 =========================
    print("\n[+] Loading best checkpoint:", best_model_path)
    model.load(best_model_path)

    if not os.path.exists(best_thr_path):
        raise RuntimeError(f"best_threshold.txt not found at {best_thr_path}")
    with open(best_thr_path, "r") as f:
        best_thr = float(f.read().strip())

    y_true, y_prob, test_auc = evaluate_probs(model, test_dl, device)
    acc, prec, rec, f1 = metrics_at_threshold(y_true, y_prob, best_thr)

    print(f"[Test] acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}, auc={test_auc:.4f}")
    print(f"[✓] Best threshold (from val): τ = {best_thr:.3f}  (saved at {best_thr_path})")
    print("[✓] Saved fine-tuned model + tokenizer.\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/AMP-data.csv")
    ap.add_argument("--out", type=str, default="checkpoints/cls")
    ap.add_argument("--bs", type=int, default=16)

    ap.add_argument("--lr_head", type=float, default=3e-4, help="LR for classifier head")
    ap.add_argument("--lr_backbone", type=float, default=2e-5, help="LR for unfrozen backbone layers")
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--monitor", type=str, default="auc", choices=["auc", "f1"], help="Early stopping monitor metric.")

    ap.add_argument("--unfreeze", action="store_true", help="Unfreeze backbone (full finetune).")
    ap.add_argument("--train_pooler", action="store_true", help="If freezing backbone, optionally train pooler.")

    # === hard-negative mining hooks ===
    ap.add_argument("--extra_neg", type=str, default=None, help="CSV of mined hard negatives (contains seq[,act])")
    ap.add_argument("--extra_neg_repeat", type=int, default=3, help="Repeat mined hard negatives in train set for weighting")
    ap.add_argument("--extra_neg_min_act", type=float, default=0.95, help="If extra_neg has 'act' col, keep rows with act>=this")
    ap.add_argument("--extra_neg_max", type=int, default=5000, help="Max unique hard negatives before repeat")
    ap.add_argument("--extra_neg_dedup_train", action="store_true", help="Deduplicate train sequences after merge")

    args = ap.parse_args()
    main(args)
