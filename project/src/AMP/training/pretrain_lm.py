# -*- coding: utf-8 -*-
"""
在大规模无标签蛋白序列（例如 UniRef50 子集）上，对 AMPGenerator 做纯 LM 预训练。

注意：
1. 预训练阶段只优化语言模型 loss（预测下一氨基酸），不使用结构特征监督。
2. FiLM 条件层（feat_gamma, feat_beta）和 reg_head 冻结，不参与更新。
3. 预期输入 FASTA 为已经抽样/过滤过的 UniRef50 子集，例如长度在 [min_len, max_len] 内。
4. 输出是 AMPGenerator 的 state_dict，可在 train_generator.py 中通过 load_state_dict 载入。

使用示例：
    python -m AMP.training.pretrain_lm \
        --fasta data/uniref50_subsample.fasta \
        --out checkpoints/gen_lm_pretrain.pt \
        --max_len 64 \
        --max_seqs 2000000
"""

import os
import argparse
from typing import List, Optional

import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from AMP.data_processing.vocab import encode_sequence, BOS_ID, PAD_ID, VOCAB_SIZE_OUTPUT
from AMP.models.generator_model import AMPGenerator
from AMP.features.features import VALID_AA, FEATURE_KEYS


# ------------------------- 工具函数：随机种子 ------------------------- #

def set_seed(seed: int):
    """设置随机种子，提升实验可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------- 数据集定义 ------------------------- #

class UniRefSeqDataset(Dataset):
    """
    无标签蛋白序列数据集：仅用于 LM 预训练。

    每个样本返回：
        input_ids: [T]  (BOS + seq[:-1], padding 到 max_len)
        labels   : [T]  (seq 本身，padding 为 -100 以便 CE ignore)

    注意：
    - 不包含结构特征；结构特征在训练循环中统一用 0 向量代替。
    - 在 __init__ 中完成长度过滤和非法字符过滤。
    """
    def __init__(
        self,
        seqs: List[str],
        max_len: int = 64,
        min_len: int = 10,
    ):
        super().__init__()
        self.max_len = max_len

        valid = VALID_AA  # 项目中定义的 20AA 集合

        cleaned = []
        for s in seqs:
            s = "".join(str(s).split()).upper()
            if len(s) < min_len:
                continue
            # 这里不强制 <= max_len，因为我们会在 encode 时截断
            if not s:
                continue
            if set(s) - valid:
                # 含有非标准 AA，直接丢弃
                continue
            cleaned.append(s)

        if len(cleaned) == 0:
            raise RuntimeError("UniRefSeqDataset: 过滤后序列数为 0，请检查 FASTA 或过滤条件。")

        self.seqs = cleaned
        print(f"[UniRefSeqDataset] total valid sequences after filtering: {len(self.seqs)}")

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx: int):
        seq = self.seqs[idx]
        ids = encode_sequence(seq)  # List[int] in [0, VOCAB_SIZE_OUTPUT-1]

        # teacher forcing：输入为 BOS + 前 n-1，目标为 前 n
        inp = [BOS_ID] + ids[:-1]
        tgt = ids

        # 截断 / padding 到 max_len
        inp = inp[:self.max_len]
        tgt = tgt[:self.max_len]

        # 输入 padding：多余部分补 BOS_ID（实际不会被用作 label）
        pad_inp = inp + [PAD_ID] * (self.max_len - len(inp))
        # 标签 padding：补 -100，CE loss 会忽略这个位置
        pad_tgt = tgt + [-100] * (self.max_len - len(tgt))

        return {
            "input_ids": torch.tensor(pad_inp, dtype=torch.long),
            "labels": torch.tensor(pad_tgt, dtype=torch.long),
        }


def read_fasta(path: str, max_seqs: Optional[int] = None) -> List[str]:
    """
    极简 FASTA 解析器：按标准规则解析：
    - '>' 开头为新条目；
    - 支持多行序列；
    - 仅返回序列字符串列表。

    max_seqs:
        限制最多读取多少条序列（用于控制内存/时间）。
        为 None 时读取全部。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"FASTA 文件不存在: {path}")

    seqs = []
    buf = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                # flush 之前的序列
                if buf:
                    seqs.append("".join(buf))
                    buf = []
                    if max_seqs is not None and len(seqs) >= max_seqs:
                        break
            else:
                buf.append(line)
        # 文件结尾 flush
        if buf and (max_seqs is None or len(seqs) < max_seqs):
            seqs.append("".join(buf))

    if len(seqs) == 0:
        raise RuntimeError(f"read_fasta: 未从 {path} 读取到任何序列，请检查文件格式。")

    print(f"[read_fasta] loaded {len(seqs)} raw sequences from {path}")
    return seqs


# ------------------------- collate 函数 ------------------------- #

def make_collate_fn(feat_dim: int):
    """
    返回一个用于 DataLoader 的 collate_fn：
    - 拼接 input_ids / labels；
    - 构造全 0 的 norm_feats（仅占位）。
    """
    def collate_fn(batch):
        input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
        labels = torch.stack([b["labels"] for b in batch], dim=0)
        norm_feats = torch.zeros(input_ids.size(0), feat_dim, dtype=torch.float32)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "norm_feats": norm_feats,
        }
    return collate_fn


# ------------------------- 训练主逻辑 ------------------------- #

def main(args):
    # 1. 基本检查与随机种子
    if args.val_frac <= 0 or args.val_frac >= 0.5:
        raise ValueError("val_frac 建议在 (0, 0.5) 范围内，例如 0.01 或 0.02。")
    if args.max_len <= 0:
        raise ValueError("max_len 必须为正整数。")

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Pretrain LM] Device:", device)

    # 2. 读取 FASTA
    raw_seqs = read_fasta(args.fasta, max_seqs=args.max_seqs)

    # 3. 构建 Dataset 并过滤无效序列
    ds_full = UniRefSeqDataset(
        raw_seqs,
        max_len=args.max_len,
        min_len=args.min_len,
    )

    if len(ds_full) < args.bs:
        print(f"[WARN] 数据量 {len(ds_full)} 小于 batch_size {args.bs}，建议减小 batch_size。")

    # 4. 划分 train / val
    val_size = max(1, int(len(ds_full) * args.val_frac))
    train_size = len(ds_full) - val_size
    ds_train, ds_val = random_split(
        ds_full,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"[Split] train={len(ds_train)}, val={len(ds_val)}")

    feat_dim = len(FEATURE_KEYS)
    collate_fn = make_collate_fn(feat_dim)

    train_dl = DataLoader(
        ds_train,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    val_dl = DataLoader(
        ds_val,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    # 5. 构造 AMPGenerator，并冻结 FiLM / reg_head
    model = AMPGenerator(
        emb_dim=args.emb_dim,
        hidden=args.hidden_dim,
        feat_dim=feat_dim,
        num_layers=args.num_layers,
        gru_dropout=args.gru_dropout,
        ffn_hidden=args.ffn_hidden,
        ffn_dropout=args.ffn_dropout,
        use_layernorm=True,
    ).to(device)

    # 冻结 FiLM 参数和 reg_head：预训练只学“序列语法”
    frozen_names = []
    for name, p in model.named_parameters():
        if name.startswith("feat_gamma") or name.startswith("feat_beta") or name.startswith("reg_head"):
            p.requires_grad = False
            frozen_names.append(name)

    print(f"[Pretrain LM] frozen params (FiLM + reg_head): {len(frozen_names)}")
    # 如需查看具体层名，可取消下一行注释
    # print("  " + "\n  ".join(frozen_names))

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[Pretrain LM] trainable_params = {trainable} / total_params = {total}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

    best_val_loss = float("inf")
    patience_left = args.patience

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # ---------- 训练 ----------
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_dl, desc=f"[LM pretrain] epoch {epoch}/{args.epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels    = batch["labels"].to(device, non_blocking=True)
            norm_feats = batch["norm_feats"].to(device, non_blocking=True)
            logits, _ = model(input_ids, norm_feats)  # logits: [B, T, VOCAB_SIZE_OUTPUT]

            B, T, V = logits.shape
            loss = ce_loss(logits.view(B * T, V), labels.view(B * T))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(train_dl)

        # ---------- 验证 ----------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dl:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                norm_feats = batch["norm_feats"].to(device, non_blocking=True)

                logits, _ = model(input_ids, norm_feats)
                B, T, V = logits.shape
                loss = ce_loss(logits.view(B * T, V), labels.view(B * T))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dl)
        # 计算 token-wise perplexity 作为参考（非必须，但常用）
        val_ppl = float(np.exp(avg_val_loss)) if avg_val_loss < 20 else float("inf")

        print(
            f"[Epoch {epoch}] train_loss={avg_train_loss:.4f} "
            f"val_loss={avg_val_loss:.4f} val_ppl={val_ppl:.2f}"
        )

        # ---------- early stopping & checkpoint ----------
        if avg_val_loss < best_val_loss - 1e-4:  # 给一点 margin 防抖
            best_val_loss = avg_val_loss
            patience_left = args.patience
            torch.save(model.state_dict(), args.out)
            print(f"[✓] New best val_loss={best_val_loss:.4f}, checkpoint saved to: {args.out}")
        else:
            patience_left -= 1
            print(f"[EarlyStopping] no improvement, patience_left = {patience_left}")
            if patience_left <= 0:
                print("[!] Early stopping triggered.")
                break

    print(f"[Done] Best val_loss={best_val_loss:.4f}, best checkpoint={args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", type=str, required=True,
                    help="预处理后的 UniRef50 子集 FASTA 路径")
    ap.add_argument("--out", type=str, default="checkpoints/gen_lm_pretrain.pt",
                    help="LM 预训练权重保存路径 (state_dict)")
    ap.add_argument("--max_len", type=int, default=64,
                    help="序列截断 / padding 长度")
    ap.add_argument("--min_len", type=int, default=10,
                    help="最短序列长度过滤")
    ap.add_argument("--max_seqs", type=int, default=None,
                    help="从 FASTA 中最多读取多少条序列 (None 表示全部)")

    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--patience", type=int, default=2,
                    help="val_loss 无提升时允许的最大轮数")
    ap.add_argument("--val_frac", type=float, default=0.01,
                    help="验证集占总数据比例 (0~0.5 建议取 0.01~0.02)")

    ap.add_argument("--emb_dim", type=int, default=256)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--gru_dropout", type=float, default=0.1)
    ap.add_argument("--ffn_hidden", type=int, default=1024)
    ap.add_argument("--ffn_dropout", type=float, default=0.1)

    ap.add_argument("--num_workers", type=int, default=4,
                    help="DataLoader 的 num_workers")
    ap.add_argument("--seed", type=int, default=2025,
                    help="随机种子，用于结果可复现")

    args = ap.parse_args()
    main(args)
