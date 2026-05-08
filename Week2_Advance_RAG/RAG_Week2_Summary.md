# Day 8: RAG 检索进阶 —— 查询改写 (Query Transformation)

## 1. Query Transformation 的核心价值
在基础 RAG（Naive RAG）中，检索效果差的根本原因在于**“语义空间错位”**。
* **痛点**：用户的原始提问往往非常简短、随意、口语化（例如：“湖冰物候受啥影响？”），而知识库中的专业文档（如遥感论文）则是长难句且充满学术词汇（例如：“空间异质性”、“海拔”、“封冻期”）。短问题和长原文在向量空间中的距离很远，导致匹配度极低。
* **核心价值**：在将查询送入向量数据库之前，引入一层“大模型思考期”，让 LLM 对用户的原始问题进行**翻译、扩写或伪造**，使其在词汇和句式上无限逼近目标文档的特征，从而实现**降维打击级的检索精度提升**。

## 2. Multi-Query (多路召回扩展) 的实现思路
Multi-Query 旨在通过“群狼战术”提高目标文档的命中率和召回率，解决单一查询视角的局限性。
* **实现流程**：
  1. **Prompt 引导改写**：要求大模型将用户的原始短问题，从不同角度改写并扩展为 N 个（通常为 3-5 个）表述严谨的学术问题。
  2. **并发独立检索**：使用 `map()` 映射或循环机制，拿这 N 个新问题分别去 FAISS 向量库中独立进行相似度搜索。
  3. **合并与去重（核心工程点）**：由于多个相似问题可能会搜回同一段核心材料，必须使用代码（如 Python 中的 `list(set(all_results))`）对收集到的所有文档碎片进行**强行去重**。
* **工程意义**：防止大模型上下文窗口中出现大量重复信息，既节省了 Token 成本（省钱），又避免了大模型的“注意力偏移”。

## 3. HyDE (假设性文档嵌入) 的实践流程
HyDE（Hypothetical Document Embeddings）是一种极具创意的算法，其核心思想是**“用魔法打败魔法，用陈述句匹配陈述句”**。
* **实践流程**：
  1. **盲猜生成（Fake Answer）**：接收用户的短问题，但不去检索。而是先用一段特定的 Prompt，强迫大模型“胡编乱造”写一篇简短的假设性学术论文摘要或回答。
  2. **向量化假文档**：将大模型生成的这段“假论文”进行 Embedding 向量化。
  3. **以假找真**：拿“假论文”的向量去 FAISS 库中搜索最相似的“真论文”片段。
* **数学本质**：因为大模型生成的假文档在**行文风格、专业词汇分布、句式结构**上，与真实的 PDF 论文极其相似，因此“假文档与真文档”在多维向量空间中的距离，远远小于“疑问句与真文档”的距离，从而实现了惊人的检索准确率。


# Day 9: 高阶 RAG 检索 —— 混合检索与大模型重排 (Hybrid Search & Reranking)

## 1. 四大检索方案对比与进化史

在复杂的专业领域（如遥感科学），单一的检索方式往往会漏掉关键论文。业界标准的 RAG 检索经历了以下四个阶段的进化：

* **阶段一：Embedding-only (稠密向量检索 / FAISS)**
    * **原理**：双塔模型（Bi-Encoder）。把问题和文档分别变成高维空间里的向量（坐标点），算它们的空间距离。
    * **形象比喻**：**“懂大意的文科生”**。擅长语义联想和同义词匹配。
    * **致命弱点**：容易脑补过度。遇到极其罕见的专业词汇（如：`MCD43A4` 数据集、`TKFM` 算法），大模型如果在训练时没见过，算出来的向量就会跑偏，导致搜不到原文。
* **阶段二：BM25-only (稀疏关键词检索)**
    * **原理**：纯基于字面统计的传统算法。中文环境必须配合 `jieba` 分词使用。
    * **形象比喻**：**“死抠字眼的理科生”**。完全不懂语义，只拿着放大镜数词频。
    * **核心算分机制**：**TF (词频) × IDF (逆文档频率)**。同一个词在文章里出现次数越多（TF大），且这个词在全库里极其罕见（IDF大），得分就越高。物以稀为贵。
* **阶段三：Hybrid Search (混合检索)**
    * **原理**：把 FAISS 和 BM25 并行运行，然后将两份结果合并。
    * **核心算法：RRF (倒数秩融合)**。因为两者的分数维度不同（一个是空间距离，一个是词频权重），绝对不能直接相加。RRF 只看“名次”。
    * **RRF公式**：`得分 = 1 / (排名 + K)`（K通常取60）。在两路检索里排名都靠前的文档，积分叠加后会实现对单路冠军的绝对碾压。
* **阶段四：Hybrid + Rerank (混合检索 + 交叉编码器重排)**
    * **原理**：引入独立的重排大模型（如 Cohere 或 BGE-Reranker）。
    * **形象比喻**：**“终审大 Boss”**。
    * **底层差异（交叉编码器 Cross-Encoder）**：它不再算向量距离，而是把“问题”和前几步筛选出的“候选文章”首尾相连拼在一起（面对面交流），喂给模型进行极度深度的注意力机制计算。速度最慢，但精度最高。它打出的 `relevance_score` 决定了最终喂给大语言模型的 Top-1 文档。
    * 经常用的重排模型：cohere是联网的一个可以公共的重排模型，BGE是一个可以下载到本地的重排模型

---

## 2. 避坑指南：LangChain 1.x 的“版本地狱”

**🚨 常见报错**：`ModuleNotFoundError: No module named 'langchain.retrievers'`
**🧐 报错场景**：在导入 `EnsembleRetriever`（官方混合检索黑盒）或 `ContextualCompressionRetriever`（官方重排压缩黑盒）时瞬间崩溃。

**💡 官方底层逻辑演进（为什么会报错？）**
网上的教程（甚至是 GPT 的回答）大多停留在 LangChain 0.x 时代。在最新的 1.2+ 时代，LangChain 官方对架构进行了暴力的**“化整为零”**重构：
1.  **反对臃肿的黑盒封装**：官方认为“检索 (Retrieval)”就该老老实实去库里拿数据。至于怎么算 RRF 融合积分，怎么用大模型重新打分，属于数据清洗和深度学习运算，根本不应该套在 `Retriever` 的壳子里。
2.  **物理删除**：这些高级封装类被从核心库中彻底删除了，流放到了 `langchain-classic`（怀旧包）中。

**🛠️ 架构师的终极解法 (解耦思想)**
不再依赖容易过时的 LangChain 官方封装，而是回到纯正的 Python 数据流：
1.  **手写 RRF 算法**：利用 Python 的字典键值对叠加特性，自己写 `for` 循环实现积分融合，透明且永不报错。
2.  **裸调底层重排模型**：直接使用 `sentence-transformers` 库的 `CrossEncoder`。自己把 Query 和 Doc 拼成一对 (Pairs)，调用 `predict` 方法算出浮点数分数。这才是最纯粹、性能最高的工业级流水线写法。

# Day10-11: RAGAs(RAG评估框架)

## 1.评估框架中指标：

`Context Precision` (上下文精度)：搜回来的资料，有用的在不在最前面？

`Context Recall` (上下文召回率)：回答这个问题需要的知识，你搜全了吗？

`Faithfulness` (忠实度)：大模型是不是老老实实看着检索到的资料回答的？有没有“幻觉”（胡编乱造）？

`Answer Relevancy` (回答相关性)：大模型的回答，真的切中用户的问题了吗？

## 2.实现方式：

### (1)上下文精度 (Context Precision) —— 考察“排兵布阵”

**核心目的**：检查真正有用的文档，是不是被排在了检索结果的最前面。

**参与元素**：`question` (问题)、`contexts` (检索资料)、`ground_truth` (标准答案) **大模型执行步骤**：

1. **逐篇审判（二元分类）**：RAGAs 把 `contexts` 里的文章按顺序一篇篇拿出来。问裁判大模型：“结合标准答案来看，这篇文章对回答用户问题有帮助吗？” 大模型只能回答 1（有用）或 0（没用）。
2. **加权算分（位置惩罚）**：如果大模型判定第一篇文章有用，得满分；如果判定第一篇没用，第二篇才有用，分数就会大打折扣。
   - *所以，用 BGE 重排（Rerank）把有用的文章顶到第一名，这个分数就会瞬间飙升。*    

### (2)上下文召回率 (Context Recall) —— 考察“海底捞针”

**核心目的**：检查回答这个问题所必须的知识点，系统是不是全都搜回来了？有没有漏掉关键信息？ 

**参与元素**：`ground_truth` (标准答案)、`contexts` (检索资料) **大模型执行步骤**：

1. **拆解标准答案**：RAGAs 让裁判大模型把你的 `ground_truth` 拆解开。比如你的标准答案有三个要点。
2. **寻找证据**：裁判大模型会拿着这三个要点，去庞大的 `contexts` 堆里翻找：“第一点提到了吗？第二点提到了吗？”
3. **计算得分**：`分数 = (在 contexts 中找到证据的要点数量) / (标准答案的总要点数量)`。如果标准答案有 3 个核心点，你的检索系统只搜回来了 2 个，得分就是 0.66。

### (3). 忠实度 (Faithfulness) —— 专门抓“幻觉”

**核心目的**：检查大模型的回答是不是脱离了检索到的资料，自己在胡编乱造。 

**参与元素**：`question` (问题)、`contexts` (检索资料)、`answer` (生成的回答) **大模型执行步骤**：

1. **拆解陈述（逆向提取）**：RAGAs 首先给裁判大模型下指令，让它把生成的 `answer` 拆解成一条条独立的“陈述句”。
   - *例如回答是“TKFM框架能去云，因为它是深度学习”。大模型会把它拆成：①TKFM能去云；②TKFM是深度学习。*
2. **逐条核对（逻辑推理）**：RAGAs 拿着这些拆解出来的陈述句，去和 `contexts`（检索到的原文）对比。问裁判大模型：“根据原文，这句话能推导出来吗？”
3. **计算得分**：`分数 = (原文能支撑的陈述句数量) / (总陈述句数量)`。如果有两句话，一句原文有，一句原文没有（幻觉），得分就是 0.5。

### (4). 回答相关性 (Answer Relevancy) —— 专治“答非所问”

**核心目的**：检查回答是否直接切中了用户的原始问题。 

**参与元素**：`question` (问题)、`answer` (生成的回答) **大模型执行步骤**：

1. **逆向猜问题（反向生成）**：这是最神奇的一步！RAGAs **不看**用户的原始问题。它直接把生成的 `answer` 扔给裁判大模型，要求它：“只看这个回答，请你倒推、猜测出 3 个可能产生这个回答的用户问题”。
2. **向量相似度对比（计算距离）**：RAGAs 会调用 Embedding 向量模型，把用户**真实的 `question`** 和大模型**猜出来的 3 个假问题**分别转成向量，计算它们的余弦相似度（Cosine Similarity）。
3. **计算得分**：取这 3 个相似度的平均值。如果你答非所问，大模型猜出来的问题肯定和真实问题南辕北辙，相似度就会极低。

# Day12: 生产级向量数据库 Milvus

## 1. 今日目标

Day12 的核心目标是把 Week2 前半段使用的本地向量检索思维，升级为生产级向量数据库思维。

之前的 FAISS 更像是 Python 进程里的本地向量索引；Milvus 则是一个独立运行的向量数据库服务。Python 不再直接“拥有”索引，而是通过 SDK 连接 Milvus，完成 Collection 创建、数据写入、索引构建和向量搜索。

## 2. Docker Compose 启动了什么

本次使用 `docker-compose.milvus.yml` 启动了 3 个核心服务：

| 服务 | 作用 | 通俗理解 |
| --- | --- | --- |
| `milvus-standalone` | Milvus 主服务，负责向量写入、索引、搜索 | 向量数据库本体 |
| `milvus-etcd` | 存储 Collection、Schema、索引状态等元信息 | Milvus 的登记本 |
| `milvus-minio` | 存储向量数据、索引文件等大对象 | 本地版对象存储 |

Python SDK 通过 `localhost:19530` 连接 Milvus。运行 `docker compose -f docker-compose.milvus.yml ps` 后，三个容器均为 `healthy`，说明本地 Milvus Standalone 环境启动成功。

## 3. Collection 与字段设计

本次创建的 Collection 名称：

```text
week2_rag_docs
```

字段设计：

| 字段 | 类型 | 作用 |
| --- | --- | --- |
| `id` | INT64 | 文档 chunk 主键 |
| `text` | VARCHAR | 文档 chunk 原文 |
| `source` | VARCHAR | 文档来源标识 |
| `embedding` | FLOAT_VECTOR, dim=384 | 文档语义向量 |

`dim=384` 的原因是当前使用的 `all-MiniLM-L6-v2` embedding 模型输出 384 维向量。Milvus 的向量字段维度必须与 embedding 模型输出维度一致，否则无法正确插入和搜索。

## 4. 本次完成的两个阶段

### 离线入库阶段

对应脚本：`day12_milvus_basic.py`

完成流程：

```text
连接 Milvus
-> 删除旧测试 Collection
-> 创建 week2_rag_docs
-> 使用 all-MiniLM-L6-v2 生成文档向量
-> 插入 5 条遥感文档
-> 为 embedding 字段创建 IVF_FLAT + COSINE 索引
-> load Collection
-> 执行 Top 3 检索验证
```

其中 `create_index(collection)` 的作用是为 `embedding` 向量字段建立检索索引。可以理解为给向量字段建立“搜索加速目录”，让 Milvus 在大规模数据中更快找到与 query embedding 最相似的文档 embedding。

### 在线查询阶段

对应脚本：`day12_milvus_search_only.py`

完成流程：

```text
连接已有 Milvus
-> 检查 week2_rag_docs 是否存在
-> load Collection
-> 将用户问题转成 query embedding
-> Milvus search Top 3
-> 输出 source、text 和相似度分数
```

这个脚本更接近真实 RAG 在线查询链路。真实项目中不会每次用户提问都删除 Collection、重新插入数据，而是提前完成离线入库，在线阶段只负责查询。

## 5. 检索验证结果

测试问题 1：

```text
青藏高原湖泊的封冻期和消融期主要受哪些因素影响？
```

Top 1 命中：

```text
source: lake_ice
```

说明 Milvus 能够根据问题语义召回湖冰物候相关文档。

测试问题 2：

```text
MCD43A4 数据集为什么要严格处理 QA 波段？
```

Top 1 命中：

```text
source: mcd43a4_qa
```

说明 Milvus 能够针对专业关键词和语义描述召回 QA 波段相关文档。

## 6. FAISS、Chroma、Milvus 对比

| 向量库 | 适合阶段 | 优点 | 局限 |
| --- | --- | --- | --- |
| FAISS | 本地原型、算法实验 | 简单、快、适合单机实验 | 服务化、权限、运维和数据管理能力弱 |
| Chroma | 学习期、轻量 RAG 项目 | 上手简单，支持持久化 | 大规模生产能力和集群能力有限 |
| Milvus | 生产级、大规模向量检索 | 服务化、索引丰富、可扩展、适合工程部署 | 部署组件更多，概念和运维成本更高 |

## 7. 今日关键理解

Milvus 在 RAG 系统中的位置：

```text
离线：文档 -> chunk -> embedding -> Milvus
在线：query -> embedding -> Milvus search -> context -> LLM
```

FAISS 到 Milvus 的升级，本质不是改 Prompt，也不是改 LLM，而是把“向量召回层”从本地库替换为独立的数据库服务。这样系统才更接近可部署、可维护、可扩展的生产级 RAG 架构。


# Day13: 高级数据处理与复杂 PDF 解析

## 1. 今日目标

Day13 的核心目标是补齐 RAG 系统最前置的数据入口能力：把复杂 PDF 从“不可直接检索的版面文件”解析成“可入库、可追踪、可评估的结构化 chunks”。

前面 Day8-Day12 已经完成了查询改写、混合检索、重排、RAGAs 评估和 Milvus 入门。今天的重点不是继续调 Prompt，而是处理 RAG 系统的原材料质量。真实工程中，如果 PDF 解析出来的文本存在乱码、空页、表格错乱、页眉页脚噪声或元数据缺失，后面的 Embedding、Hybrid Search、Rerank 和 Faithfulness 评估都会被污染。

## 2. 本次实现的解析流程

对应脚本：

```text
day13_pdf_parse_baseline.py
```

处理流程：

```text
读取 PDF
-> 按页提取文本
-> 清洗空字符、多余空格和连续换行
-> 过滤空页或过短页
-> 按固定窗口切分 chunk
-> 为每个 chunk 添加 source、page、chunk_id、text_length 元数据
-> 输出 JSONL 和 Markdown 解析质量报告
```

输出文件：

```text
parsed_outputs/day13_chunks.jsonl
parsed_outputs/day13_parse_report.md
```

其中 JSONL 面向后续程序入库，每一行是一条结构化 chunk；Markdown 报告面向人工检查，用于确认解析质量。

## 3. 运行结果

本次测试 PDF：

```text
2023 - Multi-sensor detection of spring breakup phenology of Canada's lakes.pdf
```

解析指标：

| 指标 | 结果 |
| --- | --- |
| PDF 总页数 | 19 |
| 成功解析页数 | 19 |
| 空页或过短页 | 0 |
| Chunk 数量 | 148 |
| 最短 Chunk 长度 | 140 |
| 最长 Chunk 长度 | 900 |
| 平均 Chunk 长度 | 848.53 |

这个结果说明该 PDF 是文本型 PDF，不是纯扫描件，因此 PyMuPDF 可以直接提取文本，不需要先做 OCR。

## 4. 解析质量观察

优点：

- 没有明显乱码。
- 19 页全部成功解析。
- Chunk 带有 `source`、`page`、`chunk_id` 和 `text_length`，后续可以追踪答案引用来源。
- 平均 chunk 长度接近 900 字符，适合作为第一版 RAG 入库粒度。

存在的问题：

- 第 1 页包含期刊名、版权信息、作者单位等页眉和元信息，会带来少量检索噪声。
- 固定字符窗口会切断句子，例如英文单词可能跨 chunk 断开。
- 当前版本没有识别标题、表格、图注等版面结构。

## 5. PyMuPDF、Unstructured、MinerU/Docling 的定位

| 工具 | 适合场景 | 优点 | 局限 |
| --- | --- | --- | --- |
| PyMuPDF | 文本型 PDF 的快速解析基线 | 轻量、稳定、安装简单、速度快 | 不理解复杂版面结构，表格和标题层级较弱 |
| Unstructured | 需要识别标题、段落、表格等文档元素 | 更接近生产级文档解析，输出元素类型 | 依赖更重，PDF 解析环境更容易出问题 |
| MinerU / Docling | 论文、报告、复杂版面、OCR 场景 | 对复杂 PDF 和结构化抽取更强 | 部署和模型依赖更重，学习期成本更高 |

学习期的合理路线是先用 PyMuPDF 建立稳定基线，再在需要处理表格、扫描件或复杂论文版面时引入 Unstructured、MinerU 或 Docling。

## 6. 对 RAG 效果的影响

文档解析质量会直接影响：

- **Context Recall**：如果页面漏解析或表格丢失，关键证据根本进不了知识库。
- **Context Precision**：如果页眉页脚、版权信息、参考文献噪声太多，检索结果会被无关文本挤占。
- **Faithfulness**：如果 chunk 断裂严重或上下文不完整，LLM 更容易补全不存在的信息。
- **Answer Relevancy**：如果 chunk 粒度不合适，回答容易抓到局部词而偏离用户问题。

因此，生产级 RAG 不能只关注模型和向量库，也必须重视文档解析、清洗、chunk 策略和 metadata 设计。

## 7. 下一步计划

Day14 可以把 `parsed_outputs/day13_chunks.jsonl` 接入 Milvus 入库流程，将 Day12 的 5 条手写测试文档升级为真实 PDF chunks。

建议升级方向：

```text
day13_chunks.jsonl
-> embedding
-> Milvus Collection
-> Milvus search
-> Hybrid Search
-> Rerank
-> RAGAs 评估
```

同时可以继续优化 Day13 脚本：

- 增加按段落或句子边界切分，减少固定字符切断问题。
- 增加页眉页脚清洗规则。
- 增加 Unstructured 解析版本，对比元素级解析效果。
- 为每条 chunk 增加 `section_title`、`doc_type` 等更丰富 metadata。

## 8. Unstructured 元素级解析补充

为补齐路线图中“使用 Unstructured/MinerU 解析包含表格、图片的复杂 PDF”的要求，本次额外实现了 Unstructured 解析脚本：

```text
day13_unstructured_parse.py
```

运行前需要安装：

```powershell
pip install "unstructured[pdf]"
pip install unstructured-inference
```

安装过程中出现了依赖冲突提示，主要是 `unstructured` 升级了 `pydantic`、`aiofiles`、`beautifulsoup4`、`protobuf` 等包，与当前环境中的 `crewai` 版本要求不一致。这说明 Agent/RAG 工具链依赖较重，后续学习时最好拆分环境，例如：

```text
rag_study_env
agent_study_env
```

本次 Unstructured 使用 `strategy="fast"` 成功解析 PDF，输出文件：

```text
parsed_outputs/day13_unstructured_elements.jsonl
parsed_outputs/day13_unstructured_report.md
```

解析结果：

| 指标 | 结果 |
| --- | --- |
| 元素数量 | 425 |
| 覆盖页数 | 19 |
| 最短元素长度 | 1 |
| 最长元素长度 | 2512 |
| 平均元素长度 | 254.68 |

元素类型统计：

| 元素类型 | 数量 |
| --- | ---: |
| NarrativeText | 223 |
| Title | 108 |
| UncategorizedText | 63 |
| Header | 18 |
| ListItem | 12 |
| Footer | 1 |

终端中出现的警告：

```text
Cannot set non-stroke color because expected 4 components but got [1]
No languages specified, defaulting to English.
```

这些属于 PDF 版面颜色和语言默认配置相关的解析警告，不影响最终输出。

## 9. PyMuPDF 与 Unstructured 实测对比

| 维度 | PyMuPDF baseline | Unstructured |
| --- | --- | --- |
| 输出粒度 | 页面文本再切 chunk | 文档元素 |
| 输出数量 | 148 chunks | 425 elements |
| 元数据 | source、page、chunk_id、text_length | source、page、type、coordinates、links、filetype 等 |
| 结构识别 | 弱 | 较强，可识别 Title、NarrativeText、Header、ListItem |
| 安装复杂度 | 低 | 高，依赖多，容易与 Agent 框架冲突 |
| 文本质量 | 连续性较好，但有页眉页脚噪声 | 结构更丰富，但可能出现空格丢失和类型误判 |

本次样例中，Unstructured 成功识别了 `NarrativeText`、`Title`、`Header`、`ListItem` 等元素类型，也保留了坐标、页码、链接等 metadata。但它也存在一些问题：

- 部分期刊页眉和版权信息被识别为 `Title`。
- 个别文本出现空格丢失，例如 `RemoteSensingofEnvironment295(2023)113656`。
- 当前 `strategy="fast"` 没有识别出 `Table`，说明如果要处理复杂表格，可能需要尝试 `strategy="hi_res"` 或改用 MinerU/Docling。

因此，今天的核心结论是：

```text
PyMuPDF 适合作为稳定 baseline；
Unstructured 适合补充文档结构 metadata；
但任何解析工具的输出都必须经过质量检查、清洗和评估，不能直接无脑入库。
```


# Day14: Week2 系统升级 —— 真实 PDF Chunks 接入 Milvus

## 1. 今日目标

Day14 的目标是完成 Week2 Advanced RAG 的闭环：把 Day13 从真实 PDF 中解析出的 chunks 接入 Milvus，将 Day12 的 5 条手写测试文档升级为真实论文知识库。

Day14 的工程链路：

```text
PDF
-> PyMuPDF 解析
-> Cleaning
-> Sentence-aware chunking
-> JSONL chunks
-> Embedding
-> Milvus Collection
-> Vector Search
-> 人工检查 TopK 召回质量
```

## 2. 本次新增脚本

### 入库脚本

```text
day14_milvus_ingest_pdf_chunks.py
```

职责：

- 读取 `parsed_outputs/day13_chunks.jsonl`。
- 创建新的 Milvus Collection：`week2_pdf_chunks`。
- 使用 `all-MiniLM-L6-v2` 生成 384 维向量。
- 分批写入 PDF chunks。
- 创建 `IVF_FLAT + COSINE` 向量索引。

### 检索脚本

```text
day14_milvus_search_pdf_chunks.py
```

职责：

- 连接 `week2_pdf_chunks`。
- 使用英文测试问题生成 query embedding。
- 执行 Milvus TopK 检索。
- 输出 `score`、`chunk_id`、`page`、`source`、`text_length` 和文本预览。

## 3. Milvus 字段设计与 Metadata 的价值

本次 Collection 字段：

| 字段 | 类型 | 作用 |
| --- | --- | --- |
| `id` | INT64 | Milvus 主键 |
| `chunk_id` | VARCHAR | 追踪 chunk 来源与页内编号 |
| `source` | VARCHAR | 原始 PDF 文件名 |
| `page` | INT64 | 原文页码 |
| `text` | VARCHAR | chunk 文本 |
| `text_length` | INT64 | chunk 长度，用于质检 |
| `embedding` | FLOAT_VECTOR | 语义向量，用于相似度检索 |

这里的 `chunk_id`、`source`、`page`、`text_length` 就是 metadata。metadata 插入 Milvus 后很有用：

- **溯源**：回答可以标明来自哪篇 PDF、第几页、哪个 chunk。
- **过滤**：后续可以只检索某个文档、某个页码范围、某类章节。
- **调试**：检索结果差时，可以根据 page/chunk_id 回到原文检查。
- **评估**：可以统计 TopK 是否来自正确页、正确章节、正确文档。
- **权限控制**：生产系统中可以按用户权限过滤不同 source/doc_type。

向量字段负责“语义相似度”，metadata 字段负责“工程可控性”。生产级 RAG 不能只存 text 和 embedding。

## 4. Chunking 优化记录

Day13 初版使用固定字符窗口切分，暴露出两个问题：

- 单词被切断，例如 `breakup` 可能变成 `b` + `reakup`。
- PDF 换行和软连字符会造成 `con- structed`、`highresolution` 等清洗问题。

Day14 对 `day13_pdf_parse_baseline.py` 做了轻量优化：

```text
清理 PDF 软连字符
清理换行断词
统一换行为空格
切分时优先找句子边界
找不到句子边界时再找单词边界
overlap 起点尽量对齐到空格
```

重新生成后的 chunk 指标：

| 指标 | Day13 初版 | Day14 优化后 |
| --- | ---: | ---: |
| Chunk 数量 | 148 | 162 |
| 最短 Chunk 长度 | 140 | 134 |
| 最长 Chunk 长度 | 900 | 899 |
| 平均 Chunk 长度 | 848.53 | 770.83 |

优化后，`breakup` 被切成 `reakup` 的问题已明显减少。仍然存在一些从半句开头的 chunk，这是 overlap 的正常副作用；后续可以进一步升级为“按完整句子 overlap”。

## 5. 检索质量观察

本次检索脚本把测试问题改成英文，因为当前 embedding 模型 `all-MiniLM-L6-v2` 主要适合英文语义检索。中文问题直接检索英文论文 chunks，会出现 language mismatch，导致分数偏低、召回不稳定。

测试问题覆盖：

```text
OPEN-ICE algorithm steps
satellite sensors
Canadian Ice Service comparison
spatial/temporal resolution motivation
4000 lakes analysis
```

本次人工检查发现：

- 英文 query 更适合当前英文 embedding 模型。
- TopK 能召回 OPEN-ICE 流程、传感器、CIS 对比、4000 lakes 等相关上下文。
- 部分 chunk 仍包含作者名、期刊信息、图注、引用等噪声。
- 单纯向量检索还不够稳定，后续需要结合 BM25、Rerank 和 metadata filter。

## 6. Week2 完整技术链路

Week2 从基础 RAG 升级到了更接近生产的 Advanced RAG：

```text
Query Transformation
-> Hybrid Search
-> Rerank
-> RAGAs Evaluation
-> Milvus Vector DB
-> Complex PDF Parsing
-> Real PDF Chunk Ingestion
-> Milvus Search Validation
```

本周真正形成的工程认知是：

```text
RAG 效果差，不一定是模型差；
可能是 parser、cleaning、chunking、embedding language mismatch、metadata 设计的问题。
```

各环节定位：

| 环节 | 作用 | 常见问题 |
| --- | --- | --- |
| Parser | 从 PDF/网页/Word 中抽取内容 | 乱码、漏页、表格丢失、版面顺序错乱 |
| Cleaning | 清洗抽取出的脏文本 | 页眉页脚、断词、引用、版权信息噪声 |
| Chunking | 切成适合检索的语义块 | 太短、太长、切断单词、上下文不完整 |
| Embedding | 把 query 和 chunk 转成向量 | 中英文不匹配、模型维度不一致 |
| Metadata | 让检索可过滤、可追踪、可评估 | 无页码、无来源、无法权限过滤 |
| Vector DB | 服务化存储和搜索向量 | 索引参数、字段设计、数据更新策略 |
| Rerank/Eval | 提升排序质量并量化效果 | 评估集不足、成本和延迟增加 |

## 7. 后续学习建议

进入 Week3 Agent 之前，建议保留 Week2 的工程资产：

- `day13_chunks.jsonl` 作为真实 PDF chunks 样例，但不要提交到 Git。
- `week2_pdf_chunks` 作为 Milvus 中的真实论文 Collection。
- `day14_milvus_ingest_pdf_chunks.py` 作为离线入库脚本模板。
- `day14_milvus_search_pdf_chunks.py` 作为在线检索脚本模板。

后续优化方向：

1. **Chunking 升级**：用段落/句子级 chunking，过滤 References 和 Header/Footer。
2. **多语言检索**：如果希望中文问英文论文，换多语言 embedding 或做 query translation。
3. **Hybrid + Rerank 接入 Milvus**：先 Milvus 召回 TopK，再用 BM25/RRF/Reranker 优化排序。
4. **Metadata Filter**：增加 `section_title`、`doc_type`、`is_reference` 等字段，让检索更可控。
5. **RAGAs 复测**：用真实 PDF chunks 对比优化前后的 Context Precision、Context Recall、Faithfulness。
6. **Week3 Agent 衔接**：把这个 RAG 检索能力封装成 Agent Tool，让 Agent 能调用论文知识库回答问题。

Week3 开始进入 Agent 开发时，不建议从零开始。更好的路线是：

```text
先把 Week2 的 Milvus PDF 检索封装成一个 tool
-> 再学习 ReAct / Tool Calling
-> 最后做 RAG + Tool Calling 的研究助手 Agent
```

