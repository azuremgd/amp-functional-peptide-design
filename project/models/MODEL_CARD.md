# Model Card

## 目的与输入输出
性质引导的抗菌肽候选生成和计算排序，适用赛道一。生成器输入氨基酸token和5维标准化目标性质；输出序列/辅助性质。分类器输入标准单字母氨基酸序列（空格分词），输出正标签softmax分数。无MIC、毒性、安全性或菌种选择性的专用预测模型。

## 最终评测模型
- models/generator_round5.pt：仓库实际 gen_rl_round5_tm，state_dict+stats；Embedding256、GRU512×2、LayerNorm、残差FFN1024、FiLM5→512、token/性质输出头。训练数据/预训练见 data/README.md。
- models/classifier/model.safetensors：实际 cls_unfreeze2_hardneg_r2/checkpoint_best，ProtBERT-BFD 30层、hidden1024、16 heads、FFN4096、vocab30、MLP1024→256→2、GELU、Dropout0.1。严格加载全部权重，sha在MANIFEST。
- generator_lm.pt、generator_sft.pt 是既有初始化阶段权重，不是新的最终评测模型。

## 包装修复及证据
本地 hardneg-r2 文件夹只保留 model.safetensors 和 best_threshold；缺失config/tokenizer。为支持离线推理，本包从仓库旧 cls/checkpoint_best 拷贝同源 ProtBERT-BFD配置/tokenizer，去掉机器路径，并按 safetensors classifier.0/3 张量形状确认MLP256→2。没有改动最终权重。配置重建不等于取得原 r2 原文件；应与服务器原目录核对。运行严格权重加载与历史候选重评分可检测结构/tokenizer不一致。
推理加载器直接按最终config构建结构并加载权重，避免先加载基础模型/随机分类头导致警告与冗余显存。统计/随机种子、默认torch.load(weights_only=True)的兼容处理见 PATCHES.md。

## 历史训练与性能证据
logs/historical 保存原训练/反馈/RL/导出记录。hardneg-r2 日志：内部测试acc0.8986、precision0.8801、recall0.9229、F1 0.9010、AUC0.9589；这是原分类器划分上的记录，存在 data/README.md 所披露的跨阶段重叠风险，不宣称外部评测性能。
历史 round5 日志5000步；最终120000采样导出200条候选。历史完整CLI/种子/耗时未留存，末次118000采样时kept40893不是最终通过总量。当前源码含后续算法改动，不能把这些改动归因于round5。
当前服务器环境取自用户2026-10-03 UTC记录；不自动视为历史训练环境。GPU V100S32GB。新运行的环境/时间/模型hash写日志。

## 不确定性与局限
score非校准实验成功率，非MIC/抑菌率；无专门毒性/稳定性预测。规则异常并不证明无活性；螺旋/C配对指标不证明折叠/二硫键。候选与监督集零精确重复不等于全球新颖性。topk为上限，不填充伪候选。GPU/CPU数值及排序可有微小差异，不保证跨平台位级一致。

## 第三方来源与使用方式
ProtBERT-BFD：Rostlab/prot_bert_bfd（https://huggingface.co/Rostlab/prot_bert_bfd），ProtTrans（https://github.com/agemagician/ProtTrans）；官方介绍随包 ProtBERT_BFD_source_card.md。30层结构由本地config验证；具体revision与首次下载时间未知，模型/数据再分发授权需队伍核实，不能将软件许可自动套给权重。
基础权重不需要随推理包重复打包；若重新训练分类器，按官方许可下载config、tokenizer、vocab、pytorch_model.bin至 models/prot_bert_bfd，并记录revision/sha/获取日期。可用 huggingface_hub.snapshot_download(repo_id='Rostlab/prot_bert_bfd', local_dir='models/prot_bert_bfd', allow_patterns=['*.json','vocab.txt','pytorch_model.bin'])，动态默认分支不保证历史同版本。
PyTorch/Transformers/Biopython/sklearn等来源许可见 THIRD_PARTY.md；本项目未调用商业API。原创贡献为性质条件生成、预测和规则反馈、候选计算筛选的整合；本包的提交入口与溯源记录是工程支持，不冒充新科学结果。
