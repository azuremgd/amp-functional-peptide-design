# 第三方组件与知识产权

| 名称 | 实际使用 | 来源 | 许可处理 |
|---|---|---|---|
| PyTorch 2.7.1 | 训练/推理 | https://pytorch.org/ | 以安装发行包LICENSE为准（BSD类） |
| Transformers 4.41.2 / tokenizers 0.19.1 | BERT结构与分词 | https://github.com/huggingface/transformers | Apache-2.0软件；不等同模型权重许可 |
| ProtBERT-BFD | 预训练骨干并微调 | https://huggingface.co/Rostlab/prot_bert_bfd | 具体模型revision/权重许可待队伍核验 |
| Biopython 1.84 | 理化性质计算 | https://biopython.org/ | 以该版本Biopython License为准 |
| pandas/numpy/scipy/sklearn | 数据处理、统计 | 各官方发行包 | 以发行包LICENSE为准 |
| APD/DBAASP/UniProt/UniRef50 | 数据 | data/README.md中的官方链接 | 各数据库条款；未核验再分发授权，未打包原数据 |

原项目源代码权属及是否公开开源由队伍决定；本包不擅自为原创代码附加公开开源许可，不发布到外部平台。未包含EBAMP等第三方项目源码。
候选清单/自训练权重由用户授权整理用于提交，公开共享和第三方基础模型衍生权重再分发授权仍需权利人核对。
