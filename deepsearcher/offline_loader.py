from vector_db.milvus import Milvus


def load_from_local_files(collection_name):

    embedding_model = "text-embedding-ada-002"
    vector_db = Milvus()
    vector_db.init_collection(
        dim=embedding_model.dimension,
        collection=collection_name,
    )




