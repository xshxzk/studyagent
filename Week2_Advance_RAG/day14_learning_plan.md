# Day14 学习计划：Week2 系统升级与真实 PDF Chunks 入库

## 1. 今日定位

今天是 Week2 Advanced RAG 的收束日。前面已经完成：

- Day8：Query Transformation
- Day9：Hybrid Search + Rerank
- Day10-11：RAGAs 评估
- Day12：Milvus 生产级向量数据库基础
- Day13：复杂 PDF 解析，生成真实 chunks 与 Unstructured elements

Day14 的目标是把这些能力串起来，完成一次 RAG 系统升级：从 Day12 的 5 条手写测试文档，升级为 Day13 解析出的真实 PDF chunks，并用 Milvus 做真实检索验证。

## 2. 今日目标

1. 复习 RAG 离线入库与在线查询的工程分层。
2. 使用 `parsed_outputs/day13_chunks.jsonl` 作为真实知识库数据源。
3. 创建新的 Milvus Collection，避免覆盖 Day12 的手写测试 Collection。
4. 将 148 条 PDF chunks 生成 embedding 并写入 Milvus。
5. 编写搜索脚本，从 Milvus 检索真实 PDF 内容。
6. 对比 Day12 手写数据和 Day14 真实 PDF chunks 的区别。
7. 更新 Week2 总结，完成本周 Advanced RAG 闭环。

## 3. 今日核心链路

```text
Day13 PDF chunks
-> 读取 JSONL
-> Embedding
-> Milvus Collection
-> Vector Search
-> TopK contexts
-> 人工检查召回质量
-> Week2 总结
```

## 4. 时间安排

| 时间 | 学习内容 | 产出 |
| --- | --- | --- |
| 20 分钟 | 复习离线入库与在线查询分层 | 理解 ingestion/search 分离 |
| 40 分钟 | 编写真实 chunks 入库脚本 | `day14_milvus_ingest_pdf_chunks.py` |
| 30 分钟 | 编写真实 chunks 检索脚本 | `day14_milvus_search_pdf_chunks.py` |
| 30 分钟 | 运行检索并人工评估 TopK | 记录命中情况 |
| 30 分钟 | 更新 Week2 总结 | Week2 Advanced RAG 闭环总结 |

## 5. 步骤清单

### Step 1：确认 Milvus 服务状态

为什么做：

Day14 要把真实 PDF chunks 写入 Milvus。Milvus 是独立服务，Python 脚本只是客户端。如果 Milvus 容器没有启动，入库和检索都会失败。

你先运行：

```powershell
docker compose -f docker-compose.milvus.yml ps
```

如果服务没有启动，再运行：

```powershell
docker compose -f docker-compose.milvus.yml up -d
```

验收标准：

```text
milvus-standalone
milvus-etcd
milvus-minio
```

都处于 `healthy` 或正常运行状态。

### Step 2：创建 PDF chunks 入库脚本

文件位置：

```text
D:\AI_project\AgentGuide\xzkStudyagent\Week2_Advance_RAG\day14_milvus_ingest_pdf_chunks.py
```

文件职责：

- 读取 `parsed_outputs/day13_chunks.jsonl`
- 创建新的 Milvus Collection：`week2_pdf_chunks`
- 字段包含 `id`、`chunk_id`、`source`、`page`、`text`、`text_length`、`embedding`
- 使用 `all-MiniLM-L6-v2` 生成 384 维 embedding
- 批量插入 PDF chunks
- 创建 Milvus 向量索引

预计运行命令：

```powershell
python day14_milvus_ingest_pdf_chunks.py
```

### Step 3：创建 PDF chunks 检索脚本

文件位置：

```text
D:\AI_project\AgentGuide\xzkStudyagent\Week2_Advance_RAG\day14_milvus_search_pdf_chunks.py
```

文件职责：

- 连接 `week2_pdf_chunks`
- 对 3-5 个真实问题做向量检索
- 输出 TopK 的 `score`、`page`、`chunk_id`、`source`、`text preview`
- 人工判断是否召回到正确上下文

预计运行命令：

```powershell
python day14_milvus_search_pdf_chunks.py
```

### Step 4：人工检查召回质量

建议测试问题：

```text
1. OPEN-ICE algorithm 的核心流程是什么？
2. 这篇论文使用了哪些卫星传感器？
3. OPEN-ICE 与 Canadian Ice Service 的观测结果相比，误差是多少？
4. 为什么高时空分辨率对湖冰 breakup phenology 监测重要？
5. 这篇论文研究了加拿大多少个湖泊？
```

检查标准：

- Top1 或 Top3 是否包含相关原文。
- 返回 chunk 是否有页码和来源。
- 是否出现明显页眉、版权、作者单位等噪声。
- 分数最高的结果是否和问题语义匹配。

### Step 5：更新 Week2 总结

总结要包含：

- Day14 完成了什么。
- 从手写文档到真实 PDF chunks 的工程升级。
- Milvus Collection 字段设计。
- 入库数量、检索问题、TopK 命中观察。
- Week2 的完整技术链路。
- 下一周 Day15 进入 Agent 核心概念之前，需要保留哪些工程资产。

## 6. 验收标准

Day14 完成后，至少满足：

- `day14_milvus_ingest_pdf_chunks.py` 可运行。
- `week2_pdf_chunks` Collection 创建成功。
- 成功写入 148 条 PDF chunks。
- `day14_milvus_search_pdf_chunks.py` 可运行。
- 至少完成 3 个问题的 TopK 检索验证。
- `RAG_Week2_Summary.md` 更新 Day14 总结。
- `volumes/`、`parsed_outputs/`、PDF 原文件不进入 Git 提交。

## 7. 今日注意事项

- 当前 `AGENTS.md` 仍有本地未提交改动，今天不主动处理它，避免混入 Day14 学习提交。
- `parsed_outputs/` 是运行产物，已经被 `.gitignore` 忽略。
- `volumes/` 是 Milvus/MinIO/etcd 运行数据，已经从 Git 跟踪中移除并被忽略。
- 今天代码优先保持清晰，不急着引入 Hybrid/Rerank；先让真实 PDF chunks 的 Milvus 入库和搜索跑通。

## 8. 最终总结要求

今天结束时，请把以下运行结果反馈给我：

```text
1. Milvus 容器是否 healthy
2. 入库脚本是否成功
3. Collection 名称
4. 成功插入 chunk 数量
5. 检索脚本是否成功
6. 3 个测试问题的 TopK 是否命中相关内容
7. 是否出现连接、embedding、依赖或字段长度报错
```

我会根据你的运行结果继续帮你修脚本、解释报错，并完成 Week2 总结。
