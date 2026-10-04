# -*- coding: utf-8 -*-
import torch
from torch import nn
from AMP.data_processing.vocab import VOCAB_SIZE_INPUT, VOCAB_SIZE_OUTPUT


class AMPGenerator(nn.Module):
    """
    AA-Embedding + 多层 GRU + LayerNorm + FFN + FiLM-style 条件调制
    - 输入: input_ids [B, T], norm_feats [B, feat_dim]
    - 输出: logits [B, T, VOCAB_SIZE_OUTPUT], c_pred [B, feat_dim]
    """

    def __init__(
        self,
        emb_dim: int = 256,
        hidden: int = 512,
        feat_dim: int = 5,
        num_layers: int = 2,
        gru_dropout: float = 0.1,
        ffn_hidden: int = 1024,
        ffn_dropout: float = 0.1,
        use_layernorm: bool = True,
    ):
        super().__init__()

        # 1. token embedding
        self.emb = nn.Embedding(VOCAB_SIZE_INPUT, emb_dim)

        # 2. 多层 GRU，带层间 dropout
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        # 3. 可选 LayerNorm
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.ln = nn.LayerNorm(hidden)

        # 4. FFN + 残差
        self.ffn = nn.Sequential(
            nn.Linear(hidden, ffn_hidden),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(ffn_hidden, hidden),
        )
        self.ffn_dropout = nn.Dropout(ffn_dropout)

        # 5. FiLM-style 条件调制: gamma(f), beta(f)
        #    注意: 初始时 gamma ≈ 1, beta ≈ 0，这样一开始不会破坏 GRU 学到的表征
        self.feat_gamma = nn.Linear(feat_dim, hidden)
        self.feat_beta = nn.Linear(feat_dim, hidden)

        # 6. token 预测头 & 结构特征回归头
        self.head = nn.Linear(hidden, VOCAB_SIZE_OUTPUT)
        self.reg_head = nn.Linear(hidden, feat_dim)

        self._init_weights()

    def _init_weights(self):
        # 保守初始化策略: 线性层 xavier, embedding 正态
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

        # 关键: 让 FiLM 初始时近似于 "gamma ≈ 1, beta ≈ 0"
        # gamma 权重尽量小, 偏置初始化为 1
        nn.init.zeros_(self.feat_gamma.weight)
        nn.init.ones_(self.feat_gamma.bias)
        # beta 权重和偏置初始化为 0
        nn.init.zeros_(self.feat_beta.weight)
        nn.init.zeros_(self.feat_beta.bias)

    def forward(self, input_ids, norm_feats):
        """
        input_ids: [B, T]
        norm_feats: [B, feat_dim]  已经做过 z-score 的约束特征
        """
        # [B, T, E]
        x = self.emb(input_ids)

        # GRU 输出: [B, T, H]
        h, _ = self.gru(x)

        # LayerNorm
        if self.use_layernorm:
            h = self.ln(h)

        # FFN + 残差
        h_ffn = self.ffn(h)
        h = h + self.ffn_dropout(h_ffn)   # [B, T, H]

        # ---------- FiLM-style conditioning ----------
        # norm_feats: [B, feat_dim]
        gamma = self.feat_gamma(norm_feats)  # [B, H]
        beta = self.feat_beta(norm_feats)    # [B, H]

        # 视作每个 time step 共用同一组 (gamma, beta)
        gamma = gamma.unsqueeze(1)  # [B, 1, H]
        beta = beta.unsqueeze(1)    # [B, 1, H]

        # h' = gamma * h + beta
        h = gamma * h + beta        # [B, T, H]
        # -------------------------------------------

        # token logits
        logits = self.head(h)                 # [B, T, VOCAB_SIZE_OUTPUT]

        # 属性回归: 对时间维做平均
        c_pred = self.reg_head(h.mean(dim=1))  # [B, feat_dim]

        return logits, c_pred
