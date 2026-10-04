# 数据来源、权限与划分

## 数据角色
1. UniRef50（https://www.uniprot.org/help/uniref；原序列来自 UniProt）：生成器语言模型预训练。仓库保留 uniref50_15_200_clean.fasta 和统计，15–200 aa 清洗范围；训练 max_len=64 会截断较长序列。确切 release、下载日期、完整清洗脚本与原始获取日志未保留，不能假定数据与当前在线版本相同。
2. APD：https://aps.unmc.edu/，仓库文件 naturalAMPs_APD2024a.fasta；文件名含2024a，但原数据库发行版本及下载日期待核实。清洗后 apd_positive.csv 2940 条。
3. DBAASP：https://dbaasp.org/；仓库脚本通过 peptides API 分页获取单体、8–60 aa 序列，清洗后14623条。并非逐条筛选了MIC或临床病原体标签；日期/版本/许可待核实。
4. APD/DBAASP 合并去重 positive_all.csv：15479条。
5. UniProt 非 AMP 关键词记录：https://rest.uniprot.org/，查询 NOT keyword:KW-0929、8–60 aa；排除与阳性精确重复，最终15479条。负标签表示未注释抗菌肽属性，不是实验确认无活性。
6. AMP-data.csv：30958条，正负各15479条。生成器 Dataset 忽略标签而学习序列/性质；判别器使用标签。原数据审计摘要与SHA256在 data_manifest.json。
7. 难负样本：生成器产生、规则标记的疑似异常序列，只加入分类器训练集，排除与其验证/测试精确重叠；不是实验确认无活性。反馈样本重评分不能当作独立泛化指标。

## 数据获取/构建
原始数据库再分发许可未核验，故本包不包含数据库原始序列。APD、DBAASP 的访问和再分发必须遵守各自条款；UniProt/UniRef 的许可说明见 https://www.uniprot.org/help/license ，并以具体下载版本条款为准。
队伍可用原本已有的合法快照，将 positive_all.csv、uniprot_negative.csv 放入 data/ 后运行：
```bash
python tools/prepare_data.py --positive data/positive_all.csv --negative data/uniprot_negative.csv --out data/AMP-data.csv
```
原历史获取/清洗脚本在 src/data_preparation/，供方法复核；动态 API 不能保证重建历史字节相同数据。需从数据持有人获取与 data_manifest.json 哈希匹配的原快照，并确认提供给评测方的使用权限。若无合法获取途径，完整历史数据训练不可复现；本包仍能直接用最终模型执行设计/筛选/历史候选重评分。
合成示例 synthetic_smoke.csv、synthetic_pretrain.fasta 随包提供，随机种子42、40条序列，仅用于工程测试，不是抗菌肽标签证据。

## 预处理与划分
统一大写、去空白、标准20种氨基酸、长度8–60 aa；去精确重复，跨正负冲突保留正标签；pandas sample(random_state=42)打乱。五维特征：平均螺旋倾向、C数量//2代理、ProteinAnalysis.charge_at_pH(7)、GRAVY、长度；无三维结构证明。
分类器源码：分层 train_test_split(test_size=0.3, random_state=seed)，再将临时集平分为验证/测试（约70/15/15）；难负样本排除验证/测试精确重复。原训练分割索引未留存；新运行会保存索引。
生成器源码：80/20 随机划分，仅用生成器训练子集计算性质均值/标准差；历史脚本未固定生成器划分种子，本包为新训练增加seed并保存索引，不能逆推出历史索引。
精确去重不能排除同源近似序列泄漏；未证实在生成器预训练与判别器验证/测试间实施联合排重。基础 ProtBERT-BFD 的预训练集合重叠也未核验。因此历史内部随机划分指标不是严格全流程外部泛化保证。评测隐藏集不在本包，不声称与隐藏集零重叠。
