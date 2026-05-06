from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from langchain_huggingface import HuggingFaceEmbeddings


COLLECTION_NAME = "week2_rag_docs"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


DOCS = [
    {
        "id": 1,
        "source": "lake_ice",
        "text": "青藏高原地区的湖泊冰情物候特征在空间异质性上表现为：海拔越高、纬度越高的区域，其封冻期显著延长，而消融期相应推迟。此外，湖泊面积和水深等几何形态也会对冰情产生非线性影响。",
    },
    {
        "id": 2,
        "source": "mcd43a4_qa",
        "text": "采用时空融合算法处理 MCD43A4 数据集时，必须严格通过其内置的 QA 波段精准剔除云层、冰雪及气溶胶污染像元，以保证融合后数据的地表真实反射率。",
    },
    {
        "id": 3,
        "source": "transformer_remote_sensing",
        "text": "传统深度学习在处理遥感小样本图像时存在过拟合风险，引入注意力机制的 Transformer 架构能有效提取长距离上下文依赖。",
    },
    {
        "id": 4,
        "source": "gee_water_extraction",
        "text": "在利用 Google Earth Engine 云平台进行大规模水体提取与湖冰监测时，常用算法包括基于阈值法的 NDWI 和基于机器学习的随机森林。",
    },
    {
        "id": 5,
        "source": "cv_time_window",
        "text": "针对地表反射率时间序列的变异系数分析，30 天滑动窗口对短期突发性气象扰动更敏感，而 60 天滑动窗口能更平滑地揭示大尺度季节性演变趋势。",
    },
]


def connect_milvus():
    connections.connect(
        alias="default",
        host="localhost",
        port="19530",
    )
    print("Milvus 连接成功")


def recreate_collection():
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
        print(f"已删除旧 Collection: {COLLECTION_NAME}")

    fields = [
        FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=False,
        ),
        FieldSchema(
            name="text",
            dtype=DataType.VARCHAR,
            max_length=2000,
        ),
        FieldSchema(
            name="source",
            dtype=DataType.VARCHAR,
            max_length=200,
        ),
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=384,
        ),
    ]

    schema = CollectionSchema(
        fields=fields,
        description="Week2 Advanced RAG documents",
    )

    collection = Collection(
        name=COLLECTION_NAME,
        schema=schema,
    )

    print(f"Collection 创建成功: {COLLECTION_NAME}")
    print("字段列表:")
    for field in collection.schema.fields:
        print(f"- {field.name}: {field.dtype}")

    return collection


def insert_documents(collection):
    print(f"正在加载 Embedding 模型: {EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    ids = [doc["id"] for doc in DOCS]
    texts = [doc["text"] for doc in DOCS]
    sources = [doc["source"] for doc in DOCS]
    vectors = embeddings.embed_documents(texts)

    collection.insert([ids, texts, sources, vectors])
    collection.flush()

    print(f"已插入文档数量: {collection.num_entities}")


def create_index(collection):
    index_params = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128},
    }
    collection.create_index(
        field_name="embedding",
        index_params=index_params,
    )
    collection.load()
    print("向量索引创建并加载成功")


def search_documents(collection):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    search_params = {
        "metric_type": "COSINE",
        "params": {"nprobe": 10},
    }

    queries = [
        "青藏高原湖泊的封冻期和消融期主要受哪些因素影响？",
        "MCD43A4 数据集为什么要严格处理 QA 波段？",
    ]

    for query in queries:
        query_vector = embeddings.embed_query(query)
        results = collection.search(
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=3,
            output_fields=["text", "source"],
        )

        print(f"\n查询问题: {query}")
        print("Top 3 检索结果:")
        for rank, hit in enumerate(results[0], start=1):
            print(f"\n第 {rank} 名")
            print(f"相似度分数: {hit.score:.4f}")
            print(f"来源: {hit.entity.get('source')}")
            print(f"文本: {hit.entity.get('text')}")


if __name__ == "__main__":
    connect_milvus()
    collection = recreate_collection()
    insert_documents(collection)
    create_index(collection)
    search_documents(collection)
    print("当前已有 Collections:", utility.list_collections())
    connections.disconnect("default")
    print("Milvus 连接已关闭")
