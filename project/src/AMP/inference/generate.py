# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from typing import Dict

# 如果你用了分目录结构：
from AMP.data_processing.vocab import BOS_ID, decode_sequence  # ← 按你实际路径改
from AMP.features.features import FEATURE_KEYS                 # ← 按你实际路径改
# 如果还是老结构，就改成：
# from amp.vocab import BOS_ID, decode_sequence
# from amp.features import FEATURE_KEYS


@torch.no_grad()
def generate_with_constraints(
    model,
    stats: Dict[str, Dict[str, float]],
    constraints: Dict[str, float],
    max_length: int = 32,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.9,
    device: str = "cpu",
):
    """
    纯条件 LM 版本：
    - 只用 “目标结构 → z_tgt” 作为条件输入（norm_feats）
    - 不再根据前缀计算 z_cur / delta，也不再调用 add_bias
    - 模型自己学会如何根据 z_tgt 生成满足约束的序列（通过预训练 + RL）
    """
    model.eval()
    import numpy as np

    # 1) 约束 → z-score 条件向量
    z_tgt = np.array(
        [(constraints[k] - stats[k]["mean"]) / stats[k]["std"] for k in FEATURE_KEYS],
        dtype=np.float32,
    )
    z_tgt_t = torch.tensor(z_tgt, dtype=torch.float32, device=device).unsqueeze(0)  # [1,5]

    # 2) 逐步按 LM 解码
    seq_ids = [BOS_ID]
    for _ in range(max_length):
        inp = torch.tensor(seq_ids, dtype=torch.long, device=device).unsqueeze(0)  # [1, T]
        logits, _ = model(inp, z_tgt_t)  # [1, T, 20], [1,5]
        next_logits = logits[0, -1] / max(temperature, 1e-6)

        # —— 不再手动加 bias，完全交给模型 ——

        # Top-k / Top-p 采样与原来一致
        probs = F.softmax(next_logits, dim=-1)

        if top_k > 0:
            topk = torch.topk(probs, k=min(top_k, probs.numel()))
            mask = torch.ones_like(probs, dtype=torch.bool)
            mask[topk.indices] = False
            probs = probs.masked_fill(mask, 0)

        if 0 < top_p < 1:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_probs, dim=-1)
            keep = cum <= top_p
            if keep.sum() == 0:
                keep[0] = True
            filtered = torch.zeros_like(probs)
            filtered[sorted_idx[keep]] = probs[sorted_idx[keep]]
            probs = filtered

        probs = probs / probs.sum()
        nxt = torch.multinomial(probs, 1).item()
        seq_ids.append(nxt)

    # decode：只把 <20 的 token 当 AA
    return decode_sequence([i for i in seq_ids if i < 20])
