from pymilvus import Collection, connections, utility
from langchain_huggingface import HuggingFaceEmbeddings


COLLECTION_NAME = "week2_rag_docs"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def connect_milvus():
    connections.connect(
        alias="default",
        host="localhost",
        port="19530",
    )
    print("Milvus 连接成功")


def search_existing_collection():
    if not utility.has_collection(COLLECTION_NAME):
        raise RuntimeError(
            f"Collection 不存在: {COLLECTION_NAME}，请先运行 day12_milvus_basic.py 初始化数据。"
        )

    collection = Collection(COLLECTION_NAME)
    collection.load()

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    queries = [
        "青藏高原湖泊的封冻期和消融期主要受哪些因素影响？",
        "MCD43A4 数据集为什么要严格处理 QA 波段？",
    ]

    search_params = {
        "metric_type": "COSINE",
        "params": {"nprobe": 10},
    }

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
    search_existing_collection()
    connections.disconnect("default")
    print("\nMilvus 连接已关闭")
