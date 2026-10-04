> **500 MB版本注意：** 最终分类器权重为外部必需文件，下载地址见 MODEL_ACCESS_REQUIRED.md。请先阅读 MODEL_ACCESS_REQUIRED.md；补齐权重后才能执行真实推理及完整demo。以下验证记录来自原完整包。

# 性质引导的抗菌肽设计：代码与模型提交包

赛道一：AI大分子与多肽药物设计。提交运行版本：`amp-submission-1.0`。

本包包含条件 GRU/FiLM、ProtBERT/MLP 分类器、训练/强化学习/筛选源码，实际训练的 round5 生成器和 hardneg-r2 分类器权重、历史日志、200 条历史候选、标准化结果、一个完整小规模设计→优化→推理/筛选示例，以及 Notebook。
原始数据库再分发许可未核验，故不直接打包原始 APD/DBAASP/UniProt/UniRef50 数据；获取及预处理见 `data/README.md`。

## 1. 环境

用户提供的当前运行平台：Ubuntu 22.04.1 LTS、Python 3.10.19、V100S 32GB、驱动 535.274.02、PyTorch 2.7.1+cu118。驱动显示 CUDA 12.2 是兼容上限，PyTorch 自带运行时是 CUDA 11.8，勿混淆。
完整依赖版本在 requirements.txt；用户原环境快照在 logs/server_pip_freeze.txt。当前环境记录不替代历史训练环境证明。
建议至少 16GB 系统内存（32GB 更稳妥）、8GB 可用磁盘；GPU 推理建议显存 ≥8GB，模型训练以原 V100S 32GB 环境为参考。CPU 模式用于复核，速度依硬件而异。CUDA 动态库由 PyTorch 包携带，不要求单独安装 CUDA toolkit；需兼容 NVIDIA 驱动、Linux glibc，Ubuntu 22.04 可用。

```bash
conda create -n amp_submission python=3.10.19 -y
conda activate amp_submission
cd project
python -m pip install -r requirements.txt
python tools/verify_submission.py
```

Windows/CPU 验证改用 `python -m pip install -r requirements-cpu.txt`。

## 2. 主入口：真实重评分并生成最终清单

```bash
python predict.py --mode rescore --input results/historical/round5_top200.csv --out results.csv --device cuda
# Linux 等效入口
bash run.sh
```

此命令对已设计的 200 条历史候选重新执行最终分类器推理，再按分数降序输出 UTF-8 results.csv，并记录输入/模型/输出 SHA256、版本、随机种子、环境和耗时到 logs/results.run.json。
该模式复核候选评分，不假装重新完成历史 120,000 次设计，也不直接复制历史分数。不同硬件/PyTorch 浮点计算可能有细微差异，最大差异会记录在日志中；排名特别接近时可能变化。随包 results.csv 若 remarks 标注 reference conversion，表示文档格式整理示例，必须执行本命令产生真实复核结果。

## 3. 小规模设计/筛选

```bash
python design.py --config configs/demo.json --n-samples 100 --topk 10 --out results/demo_top10.csv --device cuda
```

设置范围为 20–30 aa、预测评分 ≥0.90、单残基连续重复 ≤5、连续疏水残基 ≤7。长度范围之外的性质从训练统计分布采样，不宣称五个性质都已手工固定。输出可能少于 10 条；若零候选通过则报错，不能补造数据或悄悄放宽阈值。
生成器和分类器直接从 models 加载，推理不再需要先载入基础 ProtBERT 权重。

## 4. 完整可运行示例：设计 → 优化 → 推理/筛选

```bash
python demo.py --steps 3 --n-samples 30 --device cuda
```

首先小规模设计候选；从最终生成器开展 3 步 REINFORCE 优化循环，不符合条件的批次会跳过更新；再用更新后权重生成并筛选新候选。结果为 results/demo_before.csv、demo_optimized.pt、demo_results.csv。本包 CPU 验证实际完成 1 次参数更新，权重 SHA256 已发生变化。小规模更新仅证明流程可执行，不代表训练收敛或效果提升。
RL 当前实现优先使用可用 GPU；CPU 运行完整 demo 时用 `CUDA_VISIBLE_DEVICES="" python demo.py --device cpu`（Linux）。Notebook 的对应过程见 notebooks/reproduce.ipynb。

## 5. 训练入口与配置

```bash
# 语言模型预训练：需合法获取/清洗 UniRef50 FASTA
python train.py --stage pretrain --seed 2025 --fasta data/uniref50_15_200_clean.fasta --out results/new_lm.pt --epochs 5 --bs 128 --max_len 64
# 监督微调：需按 data/README.md 放入合法获取的 AMP-data.csv
python train.py --stage generator --seed 42 --data data/AMP-data.csv --lm_ckpt models/generator_lm.pt --out results/new_sft --name generator.pt --epochs 10 --bs 32 --lr 0.0001
# 分类器：需先获取基础 ProtBERT-BFD 到 models/prot_bert_bfd（见 Model Card）
python train.py --stage classifier --seed 42 --data data/AMP-data.csv --out results/new_classifier --epochs 5 --bs 16
# 强化学习（当前源码实现；不是历史 round5 精确重训命令）
python train.py --stage optimize --seed 42 --generator models/generator_sft.pt --classifier models/classifier --steps 5000 --out results/new_rl.pt
```

无数据库权限也可验证生成器训练入口：
```bash
python train.py --stage generator --seed 42 --data data/synthetic_smoke.csv --out results/smoke_training --name smoke.pt --epochs 1 --bs 8
```
合成数据及其人为标签只用于工程验证，不用于生物学结论、正式模型训练或性能报告。
原分类器 CLI 不支持仅解冻最后两层（虽底层类有此能力）；历史 r2 的具体原训练命令未保留，不能把默认配置说成同一训练。
当前 RL 源码含 round5 之后的 soft gate/疏水矩改动；本包保留历史源码快照并明确此差异，当前脚本不能保证逐位重训出 round5 权重。

## 6. 结果定义

默认未提供赛道专用清单模板，采用附件5通用必填字段：candidate_id、track、sequence、predicted_activity、model_version、run_version、remarks，附 length_aa。
sequence 为标准单字母氨基酸序列；score 无量纲，范围 [0,1]，输出8位小数；长度单位 aa；按预测分数降序，缺失值不允许；实际通过规则的候选数量可能少于 topk。
score 不是 MIC、抑菌率、成功概率或已验证安全性。没有三维结构预测任务，因此不输出虚构 PDB/CIF。
历史 Top200 的 IDs 是本提交包排名编号，不保证等于湿实验命名 AMP-1/AMP-5，不能用编号直接对应生物学结果。

## 7. 溯源、局限与核验

数据来源/许可状态、清洗统计、划分方式、泄漏风险见 data/README.md、data/data_manifest.json；模型架构、第三方来源、适用范围、配置修复见 models/MODEL_CARD.md；逐项规范对应见 SUBMISSION_CHECKLIST.md。
SHA256 清单在 MANIFEST.json。下载/调用第三方仅用于离线蛋白模型及公开数据，没有商业 API 或提示词调用。
历史训练耗时与原始数据版本/获取日期若未留存，会标为未知，不以文件修改时间或当前环境冒充历史记录。
在正式上传前，必须核对官网赛道一清单模板、数据/模型再分发授权和补充说明；这些未在所提供附件中给出。不要把本包中的待核验项宣称为已满足。
