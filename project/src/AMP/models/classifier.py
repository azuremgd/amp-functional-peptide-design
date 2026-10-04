# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from safetensors.torch import load_file as safe_load_file


# 你的本地离线 ProtBERT base 路径（不可访问 HF 时必须本地存在）
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASE_MODEL_DIR = PROJECT_ROOT / "models" / "prot_bert_bfd"
MODEL_NAME = os.environ.get("AMP_BASE_MODEL_DIR", str(DEFAULT_BASE_MODEL_DIR))


class AMPActivity:
    """
    AMP 二分类器封装（离线、本地模型）。
    关键特性：
    1) 支持 model_dir：推理/RL 直接加载 checkpoints/cls/.../checkpoint_best（HF 目录）
    2) 支持非线性 MLP head：冻结 backbone 时提升 probe 容量
    3) freeze_backbone 只在从 base 初始化（model_dir is None）时默认生效；推理时不强制冻结
    """

    def __init__(
        self,
        num_labels: int = 2,
        device: str = "cpu",
        model_dir: str | None = None,
        max_length: int = 200,
        freeze_backbone: bool = True,
        train_pooler: bool = False,
        # head config
        head_type: str = "mlp",     # "linear" or "mlp"
        head_hidden: int = 256,     # mlp hidden size
        head_dropout: float = 0.1,
        head_depth: int = 2,        # 2: hidden+out; 3: hidden1+hidden2+out
    ):
        self.device = device
        self.max_length = max_length

        # Submission inference loads the final checkpoint directly; no base weights are needed.
        if model_dir is not None:
            self.load(model_dir)
            return

        load_dir = MODEL_NAME

        # ---- tokenizer ----
        self.tok = AutoTokenizer.from_pretrained(
            load_dir,
            do_lower_case=False,
            local_files_only=True,
        )

        # ---- model ----
        # 从 base 初始化时需要 num_labels；从 checkpoint_best 加载时 config 里一般已有 num_labels=2
        self.model = AutoModelForSequenceClassification.from_pretrained(
            load_dir,
            num_labels=num_labels,
            local_files_only=True,
        ).to(self.device)

        # ---- head: replace with nonlinear MLP (only when training from base) ----
        # 说明：
        # - 如果 model_dir 指向 checkpoint_best，我们不在这里强行替换 head（避免破坏已训练 head 形状/权重）
        # - 如果 model_dir is None（从 base 初始化训练），则按 head_type 构建 head
        if model_dir is None:
            self._set_head(
                num_labels=num_labels,
                head_type=head_type,
                head_hidden=head_hidden,
                head_dropout=head_dropout,
                head_depth=head_depth,
            )

        # ---- freeze backbone (only meaningful when training from base) ----
        if freeze_backbone and (model_dir is None):
            # 推荐先从 2 层开始（unfreeze_last_n=2）；如果显存/速度压力大，可先试 1
            self.freeze_backbone(train_pooler=train_pooler, unfreeze_last_n=2)


    # ---------------- head ----------------
    def _set_head(
        self,
        num_labels: int,
        head_type: str,
        head_hidden: int,
        head_dropout: float,
        head_depth: int,
    ):
        head_type = (head_type or "linear").lower()
        hidden = int(self.model.config.hidden_size)

        if head_type == "linear":
            self.model.classifier = nn.Linear(hidden, num_labels).to(self.device)
            return

        if head_type != "mlp":
            raise ValueError(f"Unknown head_type={head_type}, expected 'linear' or 'mlp'")

        if head_depth not in (2, 3):
            raise ValueError("head_depth must be 2 or 3")

        # 两层 MLP：Linear(hidden→H) + GELU + Dropout + Linear(H→num_labels)
        if head_depth == 2:
            self.model.classifier = nn.Sequential(
                nn.Linear(hidden, head_hidden),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden, num_labels),
            ).to(self.device)
        else:
            # 三层 MLP：Linear(hidden→H) + GELU + Dropout + Linear(H→H) + GELU + Dropout + Linear(H→num_labels)
            self.model.classifier = nn.Sequential(
                nn.Linear(hidden, head_hidden),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden, head_hidden),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden, num_labels),
            ).to(self.device)

    # ---------------- freezing ----------------
    def freeze_backbone(self, train_pooler: bool = False, unfreeze_last_n: int = 0):
        """
        冻结 ProtBERT backbone，默认只训练分类头（以及可选 pooler）。
        可选：解冻最后 unfreeze_last_n 个 encoder layer（推荐 1~2）。
        """
        # 先全部冻结（包含 backbone + head）
        for p in self.model.parameters():
            p.requires_grad = False

        # 解冻 classifier（无论是 Linear 还是 Sequential）
        if hasattr(self.model, "classifier"):
            for p in self.model.classifier.parameters():
                p.requires_grad = True

        # 可选：解冻 pooler
        if train_pooler and hasattr(self.model, "bert") and hasattr(self.model.bert, "pooler"):
            for p in self.model.bert.pooler.parameters():
                p.requires_grad = True

        # 可选：解冻最后 N 层 encoder（核心改动）
        if unfreeze_last_n and unfreeze_last_n > 0:
            if not (hasattr(self.model, "bert") and hasattr(self.model.bert, "encoder")):
                raise AttributeError("Model has no bert.encoder; cannot unfreeze last layers.")
            layers = self.model.bert.encoder.layer
            n = min(int(unfreeze_last_n), len(layers))
            for layer in layers[-n:]:
                for p in layer.parameters():
                    p.requires_grad = True


    # ---------------- encoding & steps ----------------
    def encode(self, seqs):
        # ProtBERT 常见做法：AA 字符用空格分隔
        texts = [" ".join(str(s).strip()) for s in seqs]
        batch = self.tok(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in batch.items()}

    def train_step(self, batch, optim):
        self.model.train()
        out = self.model(**batch)
        out.loss.backward()
        optim.step()
        optim.zero_grad()
        return float(out.loss.item())

    @torch.no_grad()
    def predict_prob(self, seq: str) -> float:
        self.model.eval()
        inp = self.encode([seq])
        logits = self.model(**inp).logits
        prob = torch.softmax(logits, dim=-1)[0, 1].item()
        return float(prob)

    # ---------------- save/load ----------------
    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tok.save_pretrained(path)
        import json
        with open(os.path.join(path, "head_config.json"), "w") as f:
            json.dump({
                "head_type": "mlp",
                "head_hidden": 256,
                "head_dropout": 0.1,
                "head_depth": 2
            }, f)


    def load(self, path: str):
        """
        覆盖加载训练好的分类器目录（HF 格式目录）。
        关键：支持你自定义的 MLP classifier（classifier.0 / classifier.3 等键）。
        """
        if not os.path.isdir(path):
            raise FileNotFoundError(f"classifier dir not found: {path}")

        # 先加载 tokenizer（推理时也需要）
        try:
            self.tok = AutoTokenizer.from_pretrained(
                path,
                do_lower_case=False,
                local_files_only=True,
            )
        except OSError:
            # tokenizer 缺失则沿用已有 tokenizer
            if not hasattr(self, "tok"):
                raise FileNotFoundError("Final classifier requires tokenizer files")

        # 读取 checkpoint 权重（HF 默认保存为 model.safetensors）
        safepath = os.path.join(path, "model.safetensors")
        if not os.path.isfile(safepath):
            raise FileNotFoundError(f"model.safetensors not found at {safepath}")

        state = safe_load_file(safepath, device="cpu")  # 先放 CPU，后面统一搬设备

        # --------- 根据 state_dict 的 key 判断 head 类型 ---------
        # MLP head 保存后会有 classifier.0.weight / classifier.3.weight 等键
        has_mlp = any(k.startswith("classifier.0.") for k in state.keys())
        has_linear = any(k.startswith("classifier.weight") for k in state.keys())

        # 用 config 构建“空模型结构”（不加载任何预训练权重），避免触发 classifier 新初始化的 warning
        config = AutoConfig.from_pretrained(path, local_files_only=True)
        config.num_labels = 2
        self.model = AutoModelForSequenceClassification.from_config(config)  # 先留在 CPU


        # 如果 checkpoint 表明是 MLP 头，则重建同样结构（默认按你训练时的配置）
        # 你训练时默认：mlp, hidden=256, dropout=0.1, depth=2（即 classifier.0 / classifier.3）
        if has_mlp:
            self._set_head(
                num_labels=2,
                head_type="mlp",
                head_hidden=256,
                head_dropout=0.1,
                head_depth=2,
            )
        else:
            # 否则用线性头
            self._set_head(
                num_labels=2,
                head_type="linear",
                head_hidden=256,
                head_dropout=0.1,
                head_depth=2,
            )

        # 加载权重（strict=True 能帮你及时发现结构不匹配；如你未来改 depth/hidden，可先改为 False 排查）
        ret = self.model.load_state_dict(state, strict=True)
        print("load_state_dict missing_keys:", ret.missing_keys)
        print("load_state_dict unexpected_keys:", ret.unexpected_keys)


        # 最后统一搬到 device
        self.model = self.model.to(self.device)

