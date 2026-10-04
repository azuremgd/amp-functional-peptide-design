# 500 MB 限制版本：分类器模型获取说明

本包保留源码、最终及预训练/微调生成器、分类器配置、日志、结果和Notebook。为满足上传大小限制，最终分类器权重不内嵌；合成训练输出权重也不内嵌，合成数据、训练入口及日志仍保留。

## 必须补充的外部文件

文件：model.safetensors
下载链接：https://github.com/azuremgd/amp-functional-peptide-design/releases/download/v1.0.0/model.safetensors
解压后放置：project/models/classifier/model.safetensors
SHA256：88f7843eff8ccfed67f4e909e0d280ca2598ad4f3cdfb6e48e6cc7da6662e901
大小：1680835136 字节

这是本项目实际训练的hardneg-r2最终分类器权重。不要用基础ProtBERT权重替代，也不要以重新训练替代。取得文件后请核验上述SHA256，再按README运行推理或demo。

本包缺少这份权重时，真实推理和完整demo无法运行。包内MANIFEST校验仅覆盖内嵌文件；external_required_files记录了外部必需权重。运行前需下载外部权重；下载链接及SHA256均在本文提供。

附件5允许模型权重或获取方式；但是否接受外链、访问权限及有效期限，仍需按赛道专用实施细则/平台要求确认。原包里数据许可、历史记录等待核验项继续适用。
