# -*- coding: utf-8 -*-
from typing import List, Dict

# 20 标准氨基酸
AA = "ACDEFGHIKLMNPQRSTVWY"

# special tokens（仅用于输入侧）
PAD = "<PAD>"
BOS = "<BOS>"
EOS = "<EOS>"  # 可选：如需早停

AA_LIST: List[str] = list(AA)
AA2ID: Dict[str, int] = {a: i for i, a in enumerate(AA_LIST)}

# ID 分配（放在 AA 后面）
PAD_ID = len(AA_LIST)        # 20
BOS_ID = len(AA_LIST) + 1    # 21
EOS_ID = len(AA_LIST) + 2    # 22（备用）

# 输入侧：AA + PAD + BOS
VOCAB_SIZE_INPUT = len(AA_LIST) + 2  # = 22

# 输出侧仅预测 20AA
VOCAB_SIZE_OUTPUT = len(AA_LIST)

def encode_sequence(seq: str) -> List[int]:
    return [AA2ID[a] for a in seq]

def decode_sequence(ids: List[int]) -> str:
    return "".join(AA_LIST[i] for i in ids if 0 <= i < len(AA_LIST))
