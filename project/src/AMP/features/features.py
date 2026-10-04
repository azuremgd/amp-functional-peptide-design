# -*- coding: utf-8 -*-
from typing import Dict
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis


VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


FEATURE_KEYS = [
    "helix_propensity", # 平均螺旋倾向
    "disulfide_bonds", # 半胱氨酸对数
    "net_charge", # pH 7.0 净电荷
    "hydrophobicity", # GRAVY
    "length", # 长度
]


HELIX_SCORE = {
    'A': 1.42, 'R': 0.98, 'N': 0.67, 'D': 1.01, 'C': 0.70,
    'E': 1.51, 'Q': 1.11, 'G': 0.57, 'H': 1.00, 'I': 1.08,
    'L': 1.21, 'K': 1.16, 'M': 1.45, 'F': 1.13, 'P': 0.57,
    'S': 0.77, 'T': 0.83, 'W': 1.08, 'Y': 0.69, 'V': 1.06
}




def validate_sequence(seq: str) -> (bool, str):
    if not seq:
        return False, "序列为空"
    inv = set(seq) - VALID_AA
    if inv:
        return False, f"包含无效氨基酸: {inv}"
    return True, "OK"




def calculate_structural_features(sequence: str) -> Dict[str, float]:
    if not sequence:
        return {k: 0.0 for k in FEATURE_KEYS}
    analysed = ProteinAnalysis(sequence)
    helix = sum(HELIX_SCORE.get(a, 1.0) for a in sequence) / len(sequence)
    cys = sequence.count('C')
    return {
        "helix_propensity": helix,
        "disulfide_bonds": cys // 2,
        "net_charge": analysed.charge_at_pH(7.0),
        "hydrophobicity": analysed.gravy(),
        "length": float(len(sequence)),
    }




def compute_feature_stats(seqs):
    feats = [calculate_structural_features(s) for s in seqs]
    stats = {}
    for k in FEATURE_KEYS:
        vals = np.array([f[k] for f in feats], dtype=np.float32)
        stats[k] = {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals) + 1e-8),
    }
    return stats




def z_normalize(feat_dict: Dict[str, float], stats: Dict[str, Dict[str, float]]):
    out = {}
    for k in FEATURE_KEYS:
        out[k] = (feat_dict[k] - stats[k]["mean"]) / stats[k]["std"]
    return out