# 项目整理记录

本次整理于 2026-11-22 进行，项目展示名称由 PyTorch_Learning 改为 LLM_From_Scratch。

## 扫描范围与口径

整理前完整遍历得到 **61,325 个文件、8,226 个子目录**：虚拟环境 61,244 个文件，IDE 配置 6 个，根目录缓存 5 个；排除这三类后有 **70 个项目文件**。其中 38 份 Python 源码、4 份 Notebook；完整迁移清单见下表。

所有 Python 与 Notebook 内容均经过阅读；简短数字文件名不作为删除依据。数据、图片和权重均按实际用途分类。本次**没有删除原有源码、Notebook、数据或训练结果**；只移除了已搬空的 sms_spam_collection 目录。缓存和虚拟环境保留本地，由 .gitignore 排除。

## 迁移与命名

根目录 chapter02.py、chapter03.py、chapter04.py、chapter05.py 和 gpt_download.py 保留为兼容入口，原实现迁至章节目录。Notebook 保留输出和交互学习记录，Python 文件作为正式入口。

| 原路径 | 最终路径 | 分类 |
| --- | --- | --- |
| `04_gpt_model_and_text_generation.py` | [`archive/chapter04/gpt_model_and_text_generation_duplicate.py`](../archive/chapter04/gpt_model_and_text_generation_duplicate.py) | 历史版本 / 重复快照 |
| `1.py` | [`archive/generate_spam_classifier_figures.py`](../archive/generate_spam_classifier_figures.py) | 历史版本 / 重复快照 |
| `2_1.py` | [`chapter02_tokenization/vocabulary.py`](../chapter02_tokenization/vocabulary.py) | 正式学习代码 |
| `2_3.py` | [`chapter02_tokenization/simple_tokenizer.py`](../chapter02_tokenization/simple_tokenizer.py) | 正式学习代码 |
| `2_4.py` | [`chapter02_tokenization/tokenizer_with_special_tokens.py`](../chapter02_tokenization/tokenizer_with_special_tokens.py) | 正式学习代码 |
| `3-1.py` | [`chapter03_attention/self_attention.py`](../chapter03_attention/self_attention.py) | 正式学习代码 |
| `3-3.py` | [`chapter03_attention/causal_attention.py`](../chapter03_attention/causal_attention.py) | 正式学习代码 |
| `3-4.py` | [`chapter03_attention/multi_head_attention_wrapper.py`](../chapter03_attention/multi_head_attention_wrapper.py) | 正式学习代码 |
| `3-5.py` | [`archive/chapter03/multi_head_attention_draft.py`](../archive/chapter03/multi_head_attention_draft.py) | 历史版本 / 重复快照 |
| `4-1.py` | [`chapter04_gpt_model/dummy_gpt_and_layer_norm.py`](../chapter04_gpt_model/dummy_gpt_and_layer_norm.py) | 正式学习代码 |
| `4-3.py` | [`chapter04_gpt_model/gelu_activation.py`](../chapter04_gpt_model/gelu_activation.py) | 正式学习代码 |
| `4-4.py` | [`chapter04_gpt_model/feedforward_and_shortcuts.py`](../chapter04_gpt_model/feedforward_and_shortcuts.py) | 正式学习代码 |
| `4-6.py` | [`archive/chapter04/gpt_model_step_by_step.py`](../archive/chapter04/gpt_model_step_by_step.py) | 历史版本 / 重复快照 |
| `4-7.py` | [`archive/chapter04/gpt_model_components.py`](../archive/chapter04/gpt_model_components.py) | 历史版本 / 重复快照 |
| `5-1.py` | [`chapter05_pretraining/loss_and_training.py`](../chapter05_pretraining/loss_and_training.py) | 正式学习代码 |
| `5-3-1.ipynb` | [`notebooks/ch05_sampling_experiments.ipynb`](../notebooks/ch05_sampling_experiments.ipynb) | Notebook |
| `5-3-1.py` | [`chapter05_pretraining/temperature_sampling.py`](../chapter05_pretraining/temperature_sampling.py) | 正式学习代码 |
| `5-3-1_文本生成策略_整理版.ipynb` | [`notebooks/ch05_text_generation_strategies.ipynb`](../notebooks/ch05_text_generation_strategies.ipynb) | Notebook |
| `5.4.py` | [`chapter05_pretraining/save_and_load_checkpoints.py`](../chapter05_pretraining/save_and_load_checkpoints.py) | 正式学习代码 |
| `5.5.ipynb` | [`notebooks/ch05_pretrained_weights.ipynb`](../notebooks/ch05_pretrained_weights.ipynb) | Notebook |
| `5.5.py` | [`chapter05_pretraining/download_gpt2.py`](../chapter05_pretraining/download_gpt2.py) | 正式学习代码 |
| `6-1.ipynb` | [`notebooks/ch06_spam_dataset.ipynb`](../notebooks/ch06_spam_dataset.ipynb) | Notebook |
| `6-1.py` | [`chapter06_classification/download_spam_data.py`](../chapter06_classification/download_spam_data.py) | 正式学习代码 |
| `6-2.py` | [`chapter06_classification/prepare_spam_splits.py`](../chapter06_classification/prepare_spam_splits.py) | 正式学习代码 |
| `6-4.py` | [`chapter06_classification/spam_dataset.py`](../chapter06_classification/spam_dataset.py) | 正式学习代码 |
| `6-6.py` | [`archive/ch06_spam_classifier_walkthrough.py`](../archive/ch06_spam_classifier_walkthrough.py) | 历史版本 / 重复快照 |
| `7-1.py` | [`chapter07_instruction_finetuning/finetuning.py`](../chapter07_instruction_finetuning/finetuning.py) | 正式学习代码 |
| `7-10.py` | [`chapter07_instruction_finetuning/evaluation.py`](../chapter07_instruction_finetuning/evaluation.py) | 正式学习代码 |
| `A11.py` | [`appendix_a_pytorch/device_training.py`](../appendix_a_pytorch/device_training.py) | 正式学习代码 |
| `A9.py` | [`appendix_a_pytorch/neural_network_training.py`](../appendix_a_pytorch/neural_network_training.py) | 正式学习代码 |
| `A9_bufengzhuang.py` | [`archive/appendix_a/neural_network_training_step_by_step.py`](../archive/appendix_a/neural_network_training_step_by_step.py) | 历史版本 / 重复快照 |
| `chapter02.py` | [`chapter02_tokenization/dataset_and_dataloader.py`](../chapter02_tokenization/dataset_and_dataloader.py) | 正式学习代码 |
| `chapter03.py` | [`chapter03_attention/multi_head_attention.py`](../chapter03_attention/multi_head_attention.py) | 正式学习代码 |
| `chapter04.py` | [`chapter04_gpt_model/gpt_model.py`](../chapter04_gpt_model/gpt_model.py) | 正式学习代码 |
| `chapter05.py` | [`chapter05_pretraining/training.py`](../chapter05_pretraining/training.py) | 正式学习代码 |
| `gpt2_spam_classifier_organized.py` | [`chapter06_classification/classifier.py`](../chapter06_classification/classifier.py) | 正式学习代码 |
| `gpt_download.py` | [`chapter05_pretraining/gpt2_weights.py`](../chapter05_pretraining/gpt2_weights.py) | 正式学习代码 |
| `gpt_input_embeddings.py` | [`chapter02_tokenization/input_embeddings.py`](../chapter02_tokenization/input_embeddings.py) | 正式学习代码 |
| `images/PyTorch Batch与样本计算图解.png` | [`images/batch_and_sample_computation.png`](../images/batch_and_sample_computation.png) | 图片 |
| `images/a4f074a6-b50e-4bbf-b33b-548323e9a5ed.png` | [`images/neural_network_forward_pass.png`](../images/neural_network_forward_pass.png) | 图片 |
| `instruction-data.json` | [`data/instruction-data.json`](../data/instruction-data.json) | 公开数据 / 本地生成划分 |
| `main.py` | [`chapter03_attention/multi_head_attention_demo.py`](../chapter03_attention/multi_head_attention_demo.py) | 正式学习代码 |
| `self_attention_from_scratch.py` | [`chapter03_attention/scaled_dot_product_attention.py`](../chapter03_attention/scaled_dot_product_attention.py) | 正式学习代码 |
| `simplified_self_attention.py` | [`chapter03_attention/simplified_self_attention.py`](../chapter03_attention/simplified_self_attention.py) | 正式学习代码 |
| `sms_spam_collection.zip` | `data/downloads/sms_spam_collection.zip`（本地保留，不上传） | 公开数据 / 本地生成划分 |
| `sms_spam_collection/SMSSpamCollection.tsv` | [`data/sms_spam_collection/SMSSpamCollection.tsv`](../data/sms_spam_collection/SMSSpamCollection.tsv) | 公开数据 / 本地生成划分 |
| `sms_spam_collection/readme` | [`data/sms_spam_collection/readme`](../data/sms_spam_collection/readme) | 公开数据 / 本地生成划分 |
| `test.csv` | `data/sms_spam_collection/test.csv`（本地保留，不上传） | 公开数据 / 本地生成划分 |
| `the-verdict.txt` | [`data/the-verdict.txt`](../data/the-verdict.txt) | 公开数据 / 本地生成划分 |
| `train.csv` | `data/sms_spam_collection/train.csv`（本地保留，不上传） | 公开数据 / 本地生成划分 |
| `train_ddp.py` | [`appendix_a_pytorch/distributed_training.py`](../appendix_a_pytorch/distributed_training.py) | 正式学习代码 |
| `validation.csv` | `data/sms_spam_collection/validation.csv`（本地保留，不上传） | 公开数据 / 本地生成划分 |

## 重复代码与归档依据

- `04_gpt_model_and_text_generation.py` 与原 `chapter04.py` 字节完全相同；副本仍保留在 archive/chapter04/，便于追溯。
- `3-5.py` 与原 `chapter03.py` 实现一致、排版不同，归档草稿，正式多头注意力实现保留。
- `4-6.py`、`4-7.py` 是 GPT 组件组合过程中的版本；保留完整内容。
- `A9_bufengzhuang.py` 是未封装的训练过程；`A9.py` 封装版放在附录正式目录。
- `6-6.py` 保留逐步分类实验；正式 classifier.py 补回原脚本已有的短信推理与保存示例。
- `1.py` 实为三页短信分类示意图生成脚本，依赖原 Linux 字体；归档并限制只在主动运行时生成到 outputs/spam_classifier_figures/，避免导入时写 README。
- `chapter05.py` 中重复的三个 GPT-2 下载函数转而导入原 gpt_download.py 的同一实现；不改变下载或权重映射算法。

## 数据、权重与环境

| 类型 | 实际文件 | 处理 |
| --- | --- | --- |
| 公开教学语料 | data/the-verdict.txt、instruction-data.json、短信 TSV 与原 readme | 保留；来源已核对，见 data/README.md |
| 生成数据 | 三个短信划分 CSV | 本地保留、忽略；可按固定种子重建 |
| 下载缓存 | data/downloads/sms_spam_collection.zip | 本地保留、忽略 |
| 图片 | 两份网络数据流 / Batch 图解 | 内容不变，改为英文文件名 |
| GPT-2 权重 | 124M 分片 497,759,232 字节；355M 分片 1,419,292,672 字节 | 本地保留、忽略；1558M 目录仅部分文件，不能视作完整权重 |
| GPT-2 架构配置 | 三个 hparams.json | 小文件，保留提交 |
| GPT-2 分词与检查点元数据 | encoder.json、vocab.bpe、checkpoint、index/meta | 可下载资源，本地保留、忽略 |
| 训练结果 | 根目录 model.pth | 保留、忽略，不加载或重写 |
| IDE / 环境 / 缓存 | .idea/、.venv/、__pycache__/ | 保留、忽略 |
| 无效或临时源文件 | 未确认有可安全删除的学习记录 | 不删除 |

未发现其他本地日志或独立的大型 npy / npz / bin / safetensors 文件。新增输出统一置于忽略的 outputs/；虚拟环境未作为 requirements 的来源，依赖按项目实际 import 整理。

## 限定修改范围

保留原算法和章节顺序，只补充模块说明、必要维度注释、资源路径和导入保护。原 urllib 子模块缺失导入、Notebook 中不存在的 gpt_generate 导入和模型删除后继续使用的问题已修复。Tokenizer V1 演示捕获原本预期的未知词 KeyError，编码器逻辑不变。TensorFlow 改为读取检查点时才导入。

没有运行长时间训练、GPU / DDP、预训练权重下载或 Ollama 评分。Notebook 输出是历史记录，没有伪装成重新执行的结果。

## 最终检查

整理后项目文件 97 个（同样不计 Git、环境、IDE、缓存和本次临时整理工具），其中 78 个纳入 Git；52 个原文件迁移 / 重命名，7 份脚本归档。

- `python -m compileall`：项目代码语法通过。
- `python -m scripts.check_project`：55 个学习模块实际导入通过，兼容入口、Dataset 偏移、微型 CPU GPT 前向 / 因果掩码 / loss 反向传播 / 采样、短信数据及指令 padding 检查通过。
- 四份 Notebook 代码单元语法通过，原有输出全部保留。
- 对照原始备份的函数 / 类 AST 复核未发现实质算法变化。
- 所有 70 个原始项目文件均有最终对应文件。
- 原始检查点、模型文件、IDE、虚拟环境和下载缓存不会提交。

本项目使用独立的 GitHub 仓库 LLM_From_Scratch，未改动原 code-between-pages 笔记网站。

本地文件夹重命名遇到 Windows WinError 32（目录占用）。按用户后续指示延期，当前路径仍为 PyTorch_Learning；项目展示名称及 .idea/.name 已改为 LLM_From_Scratch，虚拟环境路径未修改。
