# 附件5逐项对应及提交前核验

| 要求 | 本包对应 | 状态/边界 |
|---|---|---|
| 源代码与核心注释 | src/AMP、predict.py、train.py | 已整理；历史源码快照另存 |
| Python/依赖/硬件/CUDA | requirements.txt、README、logs/server_environment.json | 当前真实服务器记录，历史环境未确认 |
| 最终模型 | models/generator_round5.pt、classifier/model.safetensors | 实际权重；r2原配置缺失有重建说明 |
| 完整设计→优化→推理案例 | demo.py | 实际参数更新，非伪造完整训练 |
| 小规模数据 | data/synthetic_* | 合成工程数据，非生物学标签证据 |
| 主入口与结果 | predict.py、run.sh、results.csv | 默认真实重评分；格式覆盖通用必填字段 |
| 自训练入口/配置/划分 | train.py、configs、data/README | 新训练可执行；历史种子/CLI/生成器索引未留存 |
| Notebook | notebooks/reproduce.ipynb | 可运行关键流程；完整demo可独立执行 |
| 训练/验证日志 | logs/historical、new运行日志 | 原日志保留；硬件/耗时缺失不编造 |
| 数据许可与来源 | data/README、data_manifest | 下载版本/日期及APD/DBAASP授权待核实；原数据未再分发 |
| Model Card | models/MODEL_CARD.md | 架构、局限、第三方调用已说明 |
| 结果排序/不确定性 | README、Model Card、运行差异日志 | score非MIC，无虚构结构文件 |
| 结构PDB/CIF | 不适用 | 本项目未做三维结构输出 |
| 路径/种子/核验 | 相对路径、SHA256、verify_submission | 原项目独立，搬迁可用 |
| 官网赛道模板/IP细则 | 尚未提供 | 最终字段须按官网专用模板再次核对 |

尚不能宣称完全满足所有外部规范：需队伍确认数据/衍生模型再分发许可、补充未留存的历史来源信息，并核对赛道一专用模板。未知项已明确披露，不能用当前时间/文件mtime伪装历史获取时间。


## 500 MB版本补充
最终分类器权重采用外部获取方式，下载链接见获取说明。运行前必须下载权重，平台是否接受外链仍须核对。见 MODEL_ACCESS_REQUIRED.md。
