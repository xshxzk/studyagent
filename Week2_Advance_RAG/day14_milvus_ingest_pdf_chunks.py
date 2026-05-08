import json
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)


BASE_DIR = Path(__file__).resolve().parent
CHUNKS_PATH = BASE_DIR / "parsed_outputs" / "day13_chunks.jsonl"

COLLECTION_NAME = "week2_pdf_chunks"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
BATCH_SIZE = 32


def connect_milvus() -> None:
    connections.connect(
        alias="default",
        host="localhost",
        port="19530",
    )
    print("Milvus 连接成功")


def load_chunks() -> list[dict]:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"找不到 chunks 文件: {CHUNKS_PATH}。请先运行 day13_pdf_parse_baseline.py。"
        )

    chunks = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))

    if not chunks:
        raise ValueError(f"chunks 文件为空: {CHUNKS_PATH}")

    print(f"已读取 chunks 数量: {len(chunks)}")
    return chunks


def recreate_collection() -> Collection:
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
            name="chunk_id",
            dtype=DataType.VARCHAR,
            max_length=100,
        ),
        FieldSchema(
            name="source",
            dtype=DataType.VARCHAR,
            max_length=500,
        ),
        FieldSchema(
            name="page",
            dtype=DataType.INT64,
        ),
        FieldSchema(
            name="text",
            dtype=DataType.VARCHAR,
            max_length=2000,
        ),
        FieldSchema(
            name="text_length",
            dtype=DataType.INT64,
        ),
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=EMBEDDING_DIM,
        ),
    ]

    schema = CollectionSchema(
        fields=fields,
        description="Week2 Day14 real PDF chunks from Day13 parser",
    )
    collection = Collection(
        name=COLLECTION_NAME,
        schema=schema,
    )

    print(f"Collection 创建成功: {COLLECTION_NAME}")
    return collection


def batched(items: list[dict], batch_size: int) -> list[list[dict]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def insert_chunks(collection: Collection, chunks: list[dict]) -> None:
    print(f"正在加载 Embedding 模型: {EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    next_id = 1
    for batch_index, batch in enumerate(batched(chunks, BATCH_SIZE), start=1):
        texts = [item["text"] for item in batch]
        vectors = embeddings.embed_documents(texts)

        ids = list(range(next_id, next_id + len(batch)))
        next_id += len(batch)

        collection.insert(
            [
                ids,
                [item["chunk_id"] for item in batch],
                [item["source"] for item in batch],
                [int(item["page"]) for item in batch],
                texts,
                [int(item["text_length"]) for item in batch],
                vectors,
            ]
        )
        print(f"已插入 batch {batch_index}: {len(batch)} 条")

    collection.flush()
    print(f"已写入 Milvus 实体数量: {collection.num_entities}")


def create_index(collection: Collection) -> None:
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


def main() -> None:
    connect_milvus()
    chunks = load_chunks()
    collection = recreate_collection()
    insert_chunks(collection, chunks)
    create_index(collection)
    print("当前已有 Collections:", utility.list_collections())
    connections.disconnect("default")
    print("Milvus 连接已关闭")


if __name__ == "__main__":
    main()
