# Day13 学习计划：高级数据处理与复杂 PDF 解析

## 1. 今日定位

今天是 Week2 Advanced RAG 的第 6 天，对应路线图中的 Day13：高级数据处理。

前面 Day8-Day12 已经完成了 Query Transformation、Hybrid Search、Rerank、RAGAs 评估和 Milvus 基础入库查询。今天要补上生产级 RAG 很关键的一环：文档进入向量库之前，如何把复杂 PDF 解析成结构化、可追踪、可评估的文本块。

今天优先使用当前项目已有 PDF：

```text
D:\AI_project\AgentGuide\xzkStudyagent\data\2023 - Multi-sensor detection of spring breakup phenology of Canada's lakes.pdf
```

## 2. 今日目标

1. 理解复杂 PDF 解析为什么比普通文本分块更重要。
2. 对比 `PyMuPDF`、`Unstructured`、`Docling/MinerU` 的适用场景。
3. 先用轻量可控的 `PyMuPDF` 建立 PDF 页面级抽取基线。
4. 再预留 `Unstructured` 版本，用于识别标题、段落、表格等元素。
5. 生成可进入 RAG 的结构化 chunk，保留 `source`、`page`、`chunk_id`、`text` 等元数据。
6. 输出一份解析质量检查报告，为 Day14 系统升级做准备。

## 3. 时间安排

| 时间 | 学习内容 | 产出 |
| --- | --- | --- |
| 20 分钟 | 复习复杂 PDF 对 RAG 的影响 | 明确为什么不能只 `load -> split` |
| 30 分钟 | 搭建 PyMuPDF 页面级解析脚本 | `day13_pdf_parse_baseline.py` |
| 40 分钟 | 做 chunk 清洗、长度统计、空页检查 | `parsed_outputs/day13_chunks.jsonl` |
| 30 分钟 | 预留 Unstructured 解析接口 | `day13_unstructured_parse.py` |
| 30 分钟 | 解析质量评估与总结 | 更新 `RAG_Week2_Summary.md` |

## 4. 步骤清单

### Step 1：确认环境与依赖

为什么做：

复杂 PDF 解析依赖较重，尤其是 `unstructured`、`docling`、`mineru` 可能涉及系统库、OCR、模型下载。学习期先用 `PyMuPDF` 建立稳定基线，避免一上来陷入依赖地狱。

怎么做：

你先在终端进入 Week2 目录：

```powershell
cd D:\AI_project\AgentGuide\xzkStudyagent\Week2_Advance_RAG
```

然后检查依赖：

```powershell
python -c "import fitz; print('PyMuPDF OK')"
```

如果报 `ModuleNotFoundError: No module named 'fitz'`，再安装：

```powershell
pip install pymupdf
```

### Step 2：创建 PyMuPDF 基线解析脚本

文件位置：

```text
D:\AI_project\AgentGuide\xzkStudyagent\Week2_Advance_RAG\day13_pdf_parse_baseline.py
```

文件职责：

- 读取 `data` 目录中的湖冰 PDF。
- 按页提取文本。
- 过滤空页和过短文本。
- 按固定字符长度切成 chunk。
- 写出 JSONL，供后续 Milvus/Hybrid RAG 入库。

预计运行命令：

```powershell
python day13_pdf_parse_baseline.py
```

### Step 3：检查解析结果质量

输出文件：

```text
D:\AI_project\AgentGuide\xzkStudyagent\Week2_Advance_RAG\parsed_outputs\day13_chunks.jsonl
D:\AI_project\AgentGuide\xzkStudyagent\Week2_Advance_RAG\parsed_outputs\day13_parse_report.md
```

检查重点：

- PDF 总页数是多少。
- 成功解析了多少页。
- 是否存在空页、乱码页、页眉页脚噪声。
- chunk 数量、平均长度、最大长度、最小长度。
- metadata 是否包含 `source`、`page`、`chunk_id`。

### Step 4：预留 Unstructured 解析版本

为什么做：

`PyMuPDF` 更稳定、更轻，但它主要抽取文本；`Unstructured` 更接近生产级文档解析，能输出 Title、NarrativeText、Table 等元素类型，更适合表格、标题层级和复杂版面。

预计创建文件：

```text
D:\AI_project\AgentGuide\xzkStudyagent\Week2_Advance_RAG\day13_unstructured_parse.py
```

预计运行命令：

```powershell
pip install "unstructured[pdf]"
python day13_unstructured_parse.py
```

如果安装失败，今天不强行卡在这里，先把 PyMuPDF 基线跑通，再记录依赖问题。

### Step 5：更新 Week2 总结

总结要记录：

- 今天完成了什么。
- PyMuPDF 和 Unstructured 的区别。
- 复杂 PDF 对 RAG 召回、Faithfulness、Answer Relevancy 的影响。
- 解析质量指标。
- Day14 如何把解析后的 chunks 接入 Milvus + Hybrid Search + Rerank。

## 5. 验收标准

今天完成后，至少满足：

- 能成功运行 `day13_pdf_parse_baseline.py`。
- 生成 `parsed_outputs/day13_chunks.jsonl`。
- 每条 chunk 是结构化 JSON，至少包含 `chunk_id`、`source`、`page`、`text`、`text_length`。
- 生成 `parsed_outputs/day13_parse_report.md`。
- 能说清楚 PyMuPDF、Unstructured、MinerU/Docling 在 RAG 数据入口中的定位。
- `RAG_Week2_Summary.md` 增加 Day13 总结。

## 6. 预计创建的代码文件

```text
Week2_Advance_RAG/day13_pdf_parse_baseline.py
Week2_Advance_RAG/day13_unstructured_parse.py
Week2_Advance_RAG/parsed_outputs/day13_chunks.jsonl
Week2_Advance_RAG/parsed_outputs/day13_parse_report.md
```

## 7. Git 与数据提醒

当前 `Week2_Advance_RAG/volumes/` 是 Milvus、MinIO、etcd 的本地运行数据，不适合提交。

本次已计划忽略：

```text
**/volumes/
**/parsed_outputs/
```

PDF 原始文件也不适合提交，当前 `.gitignore` 已忽略：

```text
**/*.pdf
```

## 8. 今日最终总结要求

今天结束时，请把运行结果反馈给我，至少包括：

```text
1. PyMuPDF 是否安装成功
2. day13_pdf_parse_baseline.py 是否运行成功
3. PDF 总页数
4. 成功解析页数
5. chunk 数量
6. 是否看到乱码、空页、页眉页脚噪声
7. 是否尝试安装 unstructured，以及报错信息
```

我会根据你的运行结果继续帮你修脚本、解释报错，并更新 Week2 总结。
