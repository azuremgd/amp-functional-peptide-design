# 提交包装层变更

原项目未改动。src/AMP是提交副本；src/historical保留三个训练脚本原始字节。
1. 统一Python源码UTF-8；部分tools原文件为GB18030，转换编码不改计算逻辑。
2. classifier直接加载最终权重目录；不再先加载基础ProtBERT权重。config/tokenizer重建来源明确记录。
3. 新训练固定Python/NumPy/Torch种子；生成器seeded80/20；分类器保存70/15/15索引。历史划分未知仍明确披露，不把新索引说成旧索引。
4. 新入口predict/design/train/demo以及标准结果字段、版本、SHA、环境、运行耗时日志。
5. 原始大规模结果不覆盖；demo输出与正式历史Top200区分。
6. configs/round5_reference.json采用现有导出代码的默认阈值与历史日志120000任务；历史完整CLI不在日志，不宣称这些默认值均为已证实历史配置。
7. 原始辅助脚本保留历史默认目录时，README要求显式传入参数；正式主入口所有路径相对project根解析，可搬迁运行。
8. 提交入口统一终端输出 UTF-8，避免 Windows GBK 重定向日志时特殊符号导致训练退出；不改变模型计算。
