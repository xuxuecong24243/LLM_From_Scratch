# 数据来源与本地输出

| 文件 | 内容与来源 | Git 处理 |
| --- | --- | --- |
| `the-verdict.txt` | Edith Wharton 文本；[书籍配套版本](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch02/01_main-chapter-code/the-verdict.txt) | 小型公开教学语料，保留 |
| `instruction-data.json` | 1,100 条指令 / 输入 / 回答；[书籍配套版本](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/01_main-chapter-code/instruction-data.json) | 保留 |
| `sms_spam_collection/SMSSpamCollection.tsv` | [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms%2Bspam%2Bcollection)，本地解析为 5,572 行 | 保留原始公开数据与 `readme` |
| `sms_spam_collection/train.csv`、`validation.csv`、`test.csv` | 原脚本生成的 1,045 / 149 / 300 行划分 | 本地保留，忽略；运行 `python -m chapter06_classification.prepare_spam_splits` 重建 |
| `downloads/sms_spam_collection.zip` | [UCI 原始 ZIP](https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip) | 本地下载缓存，忽略 |

整理时确认两份书籍语料与公开上游内容一致，TSV 与本地下载 ZIP 中的原始文件字节一致，三个 CSV 的全部样本属于该公开 TSV。短信中的电话、地址等是公开研究语料本身的一部分；没有加入用户私人短信或联系人数据。

UCI 页面将数据集标注为 CC BY 4.0；来源作者为 Tiago Almeida、José Hidalgo，DOI：[10.24432/C5CC84](https://doi.org/10.24432/C5CC84)。同时保留下载包的原始说明，不更改其字节或来源声明。

如果原始短信文件缺失，在项目根目录执行 `python -m chapter06_classification.download_spam_data`。指令数据缺失时，第 7 章脚本会从书籍配套仓库下载；分词文本可从上述来源获取。

模型生成的响应、评分输入和训练权重统一放在根目录 `outputs/` 并忽略，避免把运行输出当成原始数据上传。
