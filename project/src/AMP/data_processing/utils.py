# -*- coding: utf-8 -*-
import torch
from typing import List
from Bio import pairwise2


@torch.no_grad()
def max_identity(seq: str, refs: List[str], limit: int = 100) -> float:
    """
    返回 seq 与 refs 中前 limit 条参考序列的最大 globalxx identity（归一化到 [0,1]）
    加速要点：
    - score_only=True：只算分数，不构造 alignment 列表
    """
    if not seq or not refs:
        return 0.0

    refs = refs[:limit]
    best = 0.0
    len_seq = len(seq)

    for r in refs:
        if not r:
            continue
        denom = max(len_seq, len(r))
        if denom <= 0:
            continue

        # 只取分数，避免生成/存储大量 alignment 结果
        score = pairwise2.align.globalxx(seq, r, score_only=True)

        # globalxx 的 score 是匹配字符数（不含 gap penalty），归一化得到 identity
        s = float(score) / float(denom)
        if s > best:
            best = s

            # 可选早停：已经达到 1.0 不可能更高
            if best >= 1.0:
                break

    return float(best)
