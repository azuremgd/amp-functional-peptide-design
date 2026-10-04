# 性质引导的抗菌肽设计

赛道一：AI大分子与多肽药物设计。项目代码、运行说明及复核材料见 [project/README.md](project/README.md)。

## 模型及参赛压缩包

[下载 v1.0.0 Release](https://github.com/azuremgd/amp-functional-peptide-design/releases/tag/v1.0.0)

- `AMP_code_submission_under500MB.zip`：小于500 MB的参赛代码包。
- `model.safetensors`：实际训练的hardneg-r2最终分类器权重，下载后放入 `project/models/classifier/model.safetensors`。
- 最终及初始化阶段生成器权重已包含在代码包中。

分类器SHA256：`88f7843eff8ccfed67f4e909e0d280ca2598ad4f3cdfb6e48e6cc7da6662e901`。

本项目预测评分用于候选排序，不代表实验抗菌效果或安全性。原始数据库数据未再分发。数据许可、来源及历史复现局限见项目说明；本仓库不额外授予第三方数据或模型权重的许可。是否符合赛事专用模板及外链要求仍需核对。
