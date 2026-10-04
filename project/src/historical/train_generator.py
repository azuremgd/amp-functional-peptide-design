# -*- coding: utf-8 -*-
import os, argparse, csv
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from torch import optim, nn
from tqdm import tqdm
from AMP.data_processing.data import AMPSeqDataset
from AMP.features.features import z_normalize, FEATURE_KEYS
from AMP.features.features import compute_feature_stats
from AMP.models.generator_model import AMPGenerator


import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

def collate(batch, stats):
    import torch
    inputs = torch.stack([b['input_ids'] for b in batch])
    labels = torch.stack([b['labels'] for b in batch])
    feats  = torch.stack([b['features'] for b in batch])

    # 统一 z-score
    feats_norm = []
    for row in feats:
        d = {k: float(v) for k, v in zip(FEATURE_KEYS, row.tolist())}
        z = z_normalize(d, stats)
        feats_norm.append([z[k] for k in FEATURE_KEYS])
    feats_norm = torch.tensor(feats_norm, dtype=torch.float32)
    return inputs, labels, feats_norm

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    if not os.path.isfile(args.data):
        raise FileNotFoundError(f"training data csv not found: {args.data}")
    df = pd.read_csv(args.data)
    ds = AMPSeqDataset(df, max_len=args.max_len)

    train_sz = int(0.8 * len(ds))
    val_sz = len(ds) - train_sz
    tr, va = random_split(ds, [train_sz, val_sz])
    # ===== 只用 train 子集计算 feature stats（避免数据泄漏）=====
    train_seqs = [ds.seqs[i] for i in tr.indices]
    train_feature_stats = compute_feature_stats(train_seqs)

    collate_fn = lambda b: collate(b, train_feature_stats)
    dl_tr = DataLoader(tr, batch_size=args.bs, shuffle=True,  collate_fn=collate_fn)
    dl_va = DataLoader(va, batch_size=args.bs, shuffle=False, collate_fn=collate_fn)

    model = AMPGenerator().to(device)
    # ====== 加载 LM 预训练权重（如果提供） ======
    if getattr(args, "lm_ckpt", None) is not None:
        lm_ckpt = args.lm_ckpt
        if os.path.exists(lm_ckpt):
            print(f"[Init] Loading LM pretrained weights from {lm_ckpt} ...")
            state = torch.load(lm_ckpt, map_location=device)

            # 你的 LM 预训练 ckpt 通常只有 "model" 层，没有 feature stats
            if isinstance(state, dict) and "model" in state:
                state = state["model"]  # 取出模型权重

            msg = model.load_state_dict(state, strict=False)
            print("  >> Loaded pretrained weights.")
            print("  >> missing_keys:", msg.missing_keys)
            print("  >> unexpected_keys:", msg.unexpected_keys)
        else:
            print(f"[Warning] lm_ckpt={lm_ckpt} not found. Training from scratch.")
    else:
        print("[Init] No lm_ckpt provided. Training generator from scratch.")

    opt = optim.AdamW(model.parameters(), lr=args.lr)
    ce  = nn.CrossEntropyLoss(ignore_index=-100)
    mse = nn.MSELoss()

    os.makedirs(args.out, exist_ok=True)
    ckpt_path = os.path.join(args.out, args.name)


    # 日志与图
    csv_path  = os.path.join(args.out, "training_log_generator.csv")
    png_curve = os.path.join(args.out, "loss_curve_generator.png")
    png_box   = os.path.join(args.out, "batch_loss_boxplot_generator.png")

    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "val_loss"])

    best = float("inf")
    tr_hist, va_hist = [], []
    all_batch_losses = []

    try:
        for ep in range(args.epochs):
            # ===== 训练 =====
            model.train(); tot = 0.0
            batch_losses = []

            for x, y, f in tqdm(dl_tr, desc=f"train ep{ep+1}"):
                x, y, f = x.to(device), y.to(device), f.to(device)
                logits, c_pred = model(x, f)  # [B, T, 20], [B,5]
                loss_lm = ce(logits.view(-1, logits.size(-1)), y.view(-1))
                loss_c  = mse(c_pred, f)
                loss = loss_lm + args.lambda_c * loss_c

                loss.backward(); opt.step(); opt.zero_grad()
                val = float(loss.item())
                tot += val
                batch_losses.append(val)
                all_batch_losses.append(val)

            tr_loss = tot / max(1, len(dl_tr))
            tr_hist.append(tr_loss)

            # ===== 验证 =====
            model.eval(); val_sum = 0.0
            with torch.no_grad():
                for x, y, f in dl_va:
                    x, y, f = x.to(device), y.to(device), f.to(device)
                    logits, c_pred = model(x, f)
                    loss_lm = ce(logits.view(-1, logits.size(-1)), y.view(-1))
                    loss_c  = mse(c_pred, f)
                    val_sum += float((loss_lm + args.lambda_c * loss_c).item())
            va_loss = val_sum / max(1, len(dl_va))
            va_hist.append(va_loss)

            print(f"Epoch {ep+1}: train {tr_loss:.4f} | val {va_loss:.4f}")

            # 记录 CSV
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f); w.writerow([ep+1, tr_loss, va_loss])

            # 保存最优
            if va_loss < best:
                best = va_loss
                torch.save({'model': model.state_dict(), 'stats': train_feature_stats}, ckpt_path)
                print("Saved best generator to", ckpt_path)

            # —— 可选：每个 epoch 更新一次曲线图 ——
            plt.figure(figsize=(8,5))
            plt.plot(range(1, len(tr_hist)+1), tr_hist, label="Train Loss")
            plt.plot(range(1, len(va_hist)+1), va_hist, label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Generator Training/Validation Loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(png_curve)
            plt.close()

        # 训练结束后输出 batch loss 总体箱线图
        if len(all_batch_losses) >= 2:
            plt.figure(figsize=(6,5))
            plt.boxplot(all_batch_losses, vert=True)
            plt.ylabel("Loss")
            plt.title("Batch Loss Distribution (Generator)")
            plt.tight_layout()
            plt.savefig(png_box)
            plt.close()

    except KeyboardInterrupt:
        # 依然保存当前最好模型
        torch.save({'model': model.state_dict(), 'stats': train_feature_stats}, ckpt_path)
        print("Interrupted. Last best (so far) saved to", ckpt_path)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, default='data/AMP-data.csv')
    ap.add_argument('--out', type=str, default='checkpoints')
    ap.add_argument('--name', type=str, default='generator_supervised.pt',help='output filename under --out')
    ap.add_argument('--max_len', type=int, default=64)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--lambda_c', type=float, default=0.3, help='weight for feature regression loss')
    ap.add_argument('--lm_ckpt', type=str, default=None,help='Path to LM-pretrained checkpoint')
    args = ap.parse_args()

    main(args)
