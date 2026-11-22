# 从零构建大模型｜LLM_From_Scratch

本项目记录我学习 Sebastian Raschka《从零构建大模型》（*Build a Large Language Model (From Scratch)*）的过程：从 PyTorch 基础、分词和注意力，到 GPT-2 实现、预训练、短信分类、指令微调和模型评估。

代码按书中章节组织，保留逐步推导和实验版本。重点是理解模型内部的数据流和训练机制；模型主体使用 PyTorch 手动实现。

## 学习路线

```text
PyTorch 基础 → Tokenizer → Dataset / DataLoader → Token / Position Embedding
    → Self-Attention → Causal / Multi-Head Attention → Transformer Block
    → GPT Model → Loss / Pretraining → Text Generation / Sampling
    → GPT-2 预训练权重 → Classification Fine-tuning
    → Instruction Fine-tuning → Evaluation
```

## 项目结构

下列目录树根据实际文件生成；省略缓存、环境、忽略的权重和目录内部细节。逐文件对应关系见 [整理记录](docs/organization.md)。

<!-- project-tree:start -->
```text
PyTorch_Learning/  # 本地文件夹改名延期；项目名称为 LLM_From_Scratch
├── appendix_a_pytorch/
├── archive/
├── chapter02_tokenization/
├── chapter03_attention/
├── chapter04_gpt_model/
├── chapter05_pretraining/
├── chapter06_classification/
├── chapter07_instruction_finetuning/
├── data/
├── docs/
├── gpt2/
├── images/
├── licenses/
├── notebooks/
├── scripts/
├── .gitattributes
├── .gitignore
├── README.md
├── chapter02.py
├── chapter03.py
├── chapter04.py
├── chapter05.py
├── gpt_download.py
├── project_paths.py
├── requirements-optional.txt
└── requirements.txt
```
<!-- project-tree:end -->

## 章节与核心内容

| 章节 | 入口与内容 |
| --- | --- |
| 附录 A | [neural_network_training.py](appendix_a_pytorch/neural_network_training.py)：Dataset、训练与保存；`device_training.py`：设备迁移；`distributed_training.py`：DDP |
| Chapter 2 | [simple_tokenizer.py](chapter02_tokenization/simple_tokenizer.py)、`tokenizer_with_special_tokens.py`：词表及特殊 token；[dataset_and_dataloader.py](chapter02_tokenization/dataset_and_dataloader.py)：滑动窗口；`input_embeddings.py`：词元与位置嵌入 |
| Chapter 3 | [self_attention.py](chapter03_attention/self_attention.py)：可训练 Q/K/V；`causal_attention.py`：因果掩码；[multi_head_attention.py](chapter03_attention/multi_head_attention.py)：多头注意力；保留简化版与矩阵推导 |
| Chapter 4 | `dummy_gpt_and_layer_norm.py`、`gelu_activation.py`、`feedforward_and_shortcuts.py`：模型组件；[gpt_model.py](chapter04_gpt_model/gpt_model.py)：完整 GPT、参数统计和贪心生成 |
| Chapter 5 | [training.py](chapter05_pretraining/training.py)：交叉熵、训练循环、温度 / top-k 采样与权重映射；`loss_and_training.py`：章节实验；`temperature_sampling.py`：概率分布；`save_and_load_checkpoints.py`：保存与加载；[gpt2_weights.py](chapter05_pretraining/gpt2_weights.py)：下载与读取 GPT-2 |
| Chapter 6 | `download_spam_data.py` → `prepare_spam_splits.py` → `spam_dataset.py` → [classifier.py](chapter06_classification/classifier.py)：公开短信、类别平衡、分割、分类微调和推理 |
| Chapter 7 | [finetuning.py](chapter07_instruction_finetuning/finetuning.py)：指令格式、变长批次、padding / loss mask、微调和生成；[evaluation.py](chapter07_instruction_finetuning/evaluation.py)：通过本地 Ollama 评分 |

`chapter02.py`～`chapter05.py` 和 `gpt_download.py` 是兼容入口，原来的 `from chapter04 import GPTModel` 等导入仍然可用。章节文件保留教学顺序，部分文件用多个 `if __name__ == "__main__"` 隔离演示；导入时不训练、不下载权重、不调用 Ollama。

## 环境与安装

使用 Python 3.10 或更新版本。此次轻量检查使用现有 Python 3.10 环境和 CPU PyTorch；依赖按实际 import 整理，没有导出整个开发环境。

```powershell
# 在项目根目录执行（Windows PowerShell）
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

主要依赖：`torch`、`tiktoken`、`numpy`、`pandas`、`matplotlib`、`requests`、`tqdm`、`psutil`。首次使用 `tiktoken.get_encoding("gpt2")` 可能下载分词资源。

按需要安装可选依赖：

```powershell
python -m pip install tensorflow   # 读取原始 GPT-2 TensorFlow 检查点
python -m pip install jupyterlab   # 打开 Notebook
```

完整可选列表在 [requirements-optional.txt](requirements-optional.txt)。TensorFlow 不参与 PyTorch 模型前向和训练，仅在预训练检查点读取时需要。归档绘图脚本还需要 `fonttools` 及脚本中指定的 Linux 字体。

## 快速开始

从项目根目录以 `python -m 包名.模块名` 运行，便于跨章节导入：

```powershell
# 不下载权重、不运行训练的 CPU 检查
python -m scripts.check_project

# 词表 / 分词 / 因果注意力
python -m chapter02_tokenization.vocabulary
python -m chapter02_tokenization.simple_tokenizer
python -m chapter03_attention.causal_attention
python -m chapter03_attention.multi_head_attention_demo

# 需要 tiktoken 分词资源
python -m chapter02_tokenization.dataset_and_dataloader
```

V1 分词器遇到词表外单词时展示预期的未知词错误；V2 加入 `<|unk|>`。完整 GPT 示例使用 124M 配置，内存需求显著高于轻量检查：

```powershell
python -m chapter04_gpt_model.gpt_model
python -m chapter05_pretraining.temperature_sampling
```

## GPT-2 权重

**GPT-2 预训练权重需要使用者自行下载，仓库不上传模型权重。** 本地已有文件继续保留。`gpt2/<模型尺寸>/hparams.json` 是小型架构配置，保留在 Git 中；检查点、分词资源和下载元数据忽略。

现有下载入口默认读取 / 下载 124M 权重：

```powershell
python -m chapter05_pretraining.download_gpt2
```

也可以使用原有 API，尺寸支持 `124M`、`355M`、`774M`、`1558M`：

```python
from gpt_download import download_and_load_gpt2
from project_paths import MODELS_DIR

settings, params = download_and_load_gpt2("124M", MODELS_DIR)
```

下载器依次尝试代码中配置的 OpenAI、Hugging Face 和 Backblaze 地址，也支持 `source="openai"`、`"hf"`、`"backblaze"`。此过程需要网络、磁盘空间和 TensorFlow；整理时未执行。单独存在 `hparams.json` 不代表权重已经完整下载。

## 数据集与训练入口

小型公开语料保存在 [data/](data/)，来源与许可见 [数据说明](data/README.md)：

- `the-verdict.txt`：书中分词、滑动窗口及预训练示例使用的文本。
- `instruction-data.json`：书籍配套的 1,100 条指令数据。
- `sms_spam_collection/SMSSpamCollection.tsv`：公开短信语料；保留原数据集 `readme`。

短信划分 CSV 是可重建的输出，已保留本地并忽略。首次克隆后先生成划分：

```powershell
python -m chapter06_classification.download_spam_data
python -m chapter06_classification.prepare_spam_splits
python -m chapter06_classification.spam_dataset
```

划分使用原代码的随机种子 123、正负例平衡及 70% / 10% / 剩余比例，生成 1,045 / 149 / 300 行。

以下命令会真正运行训练或模型服务调用，不属于快速检查：

```powershell
python -m chapter05_pretraining.loss_and_training
python -m chapter06_classification.classifier
python -m chapter07_instruction_finetuning.finetuning
python -m chapter07_instruction_finetuning.evaluation
```

分类示例默认 GPT-2 124M；指令微调默认 355M，运行时间和内存需求较高。评估读取 `outputs/instruction-data-with-response.json`，要求本机已启动 Ollama 并准备 `llama3` 模型。训练权重、损失图和生成响应写入 `outputs/`，不提交。

## Notebook 与历史记录

[notebooks/](notebooks/) 中保留四份交互学习记录及原有输出。Python 文件作为正式可导入代码；Notebook 用于按单元格观察中间结果。首个单元格定位项目根目录，请按顺序运行。保留的旧输出不代表整理后重新训练的结果。

```powershell
python -m jupyter lab notebooks
```

[archive/](archive/) 保留旧模型版本、重复快照、逐步训练脚本和历史绘图工具。没有删除原有学习代码或 Notebook；归档原因和全部旧名 → 新路径见 [整理记录](docs/organization.md)。

## 检查与已知边界

```powershell
python -m compileall -q -x "[\\/](\.venv|venv|\.git|\.organization_work|__pycache__)[\\/]" .
python -m scripts.check_project
```

轻量检查覆盖模块导入、旧入口兼容性、微型 GPT 前向 / loss / 反向传播、因果掩码、生成与指令批次；不下载分词器或模型、不运行训练周期。GPU / CUDA、DDP / NCCL、完整预训练、分类 / 指令微调和 Ollama 评分未做端到端验证。

保留书中采样实现的单序列（batch size 1）使用方式；未扩展为通用批量生成框架。附录 DDP 例子需要支持 NCCL 的 CUDA 环境。归档绘图脚本依赖 Linux 字体路径，Windows 上未运行。根目录原有 `model.pth` 保留且忽略；附录保存示例使用该路径，运行前如需保留旧结果请另行备份。

## Reference 与来源说明

- Sebastian Raschka, [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch)
- [书籍配套代码 LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
- [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms%2Bspam%2Bcollection)

保留原文件中的作者和来源注释。书籍配套代码的许可副本见 [Apache 2.0 许可文本](licenses/llms_from_scratch_apache_2.0.txt)；这不把图片、书籍或数据集自动重新许可。数据集按各自来源条款使用。
