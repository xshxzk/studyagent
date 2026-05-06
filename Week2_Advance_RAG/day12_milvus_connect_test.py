from pymilvus import connections, utility

connections.connect(
    alias="default",
    host="localhost",
    port="19530",
)

print("Milvus 连接成功")

collections = utility.list_collections()
print("当前已有 collections:", collections)

connections.disconnect("default")
print("Milvus 连接已关闭")
