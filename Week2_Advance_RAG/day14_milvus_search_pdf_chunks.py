from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import Collection, connections, utility


COLLECTION_NAME = "week2_pdf_chunks"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5


QUERIES = [
    "What are the main steps of the OPEN-ICE algorithm?",
    "Which satellite sensors were used in this study?",
    "What was the mean bias error compared with Canadian Ice Service breakup dates?",
    "Why is high spatial and temporal resolution important for monitoring lake ice breakup phenology?",
    "How many lakes across Canada were analyzed in this study?",
]


def connect_milvus() -> None:
    connections.connect(
        alias="default",
        host="localhost",
        port="19530",
    )
    print("Milvus 连接成功")


def preview(text: str, max_length: int = 500) -> str:
    text = " ".join(text.split())
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def search_pdf_chunks() -> None:
    if not utility.has_collection(COLLECTION_NAME):
        raise RuntimeError(
            f"Collection 不存在: {COLLECTION_NAME}。请先运行 day14_milvus_ingest_pdf_chunks.py。"
        )

    collection = Collection(COLLECTION_NAME)
    collection.load()

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    search_params = {
        "metric_type": "COSINE",
        "params": {"nprobe": 10},
    }

    print(f"Collection: {COLLECTION_NAME}")
    print(f"实体数量: {collection.num_entities}")

    for query in QUERIES:
        query_vector = embeddings.embed_query(query)
        results = collection.search(
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=TOP_K,
            output_fields=["chunk_id", "source", "page", "text", "text_length"],
        )

        print("\n" + "=" * 80)
        print(f"查询问题: {query}")
        print(f"Top {TOP_K} 检索结果:")

        for rank, hit in enumerate(results[0], start=1):
            entity = hit.entity
            print(f"\n第 {rank} 名")
            print(f"相似度分数: {hit.score:.4f}")
            print(f"chunk_id: {entity.get('chunk_id')}")
            print(f"page: {entity.get('page')}")
            print(f"text_length: {entity.get('text_length')}")
            print(f"source: {entity.get('source')}")
            print("文本预览:")
            print(preview(entity.get("text")))


def main() -> None:
    connect_milvus()
    search_pdf_chunks()
    connections.disconnect("default")
    print("\nMilvus 连接已关闭")


if __name__ == "__main__":
    main()
