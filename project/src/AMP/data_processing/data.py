# -*- coding: utf-8 -*-
from typing import Tuple, Dict
import pandas as pd
import torch
from torch.utils.data import Dataset
from .vocab import encode_sequence, BOS_ID, PAD_ID
from AMP.features.features import calculate_structural_features, FEATURE_KEYS



class AMPSeqDataset(Dataset):
    """将 CSV (seq, AMP) 转为：
    - aa_ids: List[int]（0–19）
    - features: 5 维结构特征（原始值 & 训练时再归一化）
    - label: AMP (0/1) 仅供分类器训练时用；生成器不需要
    """
    def __init__(self, df: pd.DataFrame, max_len: int = 64):
        df = df.copy()
        df.columns = df.columns.str.strip().str.lower()
        assert {'seq', 'amp'} <= set(df.columns)
        self.seqs = []
        self.labels = []
        for s, y in zip(df['seq'], df['amp']):
            s = "".join(str(s).split())
            # 过滤非法
            from AMP.features.features import validate_sequence
            ok, _ = validate_sequence(s)
            if ok and len(s) > 0:
                self.seqs.append(s)
                self.labels.append(int(y))
        self.max_len = max_len
        # 预计算特征 & 统计量


    def __len__(self):
        return len(self.seqs)


    def __getitem__(self, idx):
        seq = self.seqs[idx]
        ids = encode_sequence(seq)
        # teacher forcing：输入为 BOS + 前 n-1，目标为 前 n
        inp = [BOS_ID] + ids[:-1]
        tgt = ids
        # padding 到 max_len
        import torch
        pad_inp = inp[:self.max_len]
        pad_tgt = tgt[:self.max_len]
        pad_inp += [PAD_ID] * (self.max_len - len(pad_inp))
        pad_tgt += [-100] * (self.max_len - len(pad_tgt)) # -100 让 CE 忽略


        feats = calculate_structural_features(seq)
        feats_vec = torch.tensor([feats[k] for k in FEATURE_KEYS], dtype=torch.float32)


        return {
            "input_ids": torch.tensor(pad_inp, dtype=torch.long),
            "labels": torch.tensor(pad_tgt, dtype=torch.long),
            "features": feats_vec,
        }