def load_from_local_files(collection_name):

    vector_db = "Milvus"
    embedding_model = "text-embedding-ada-002"

    vector_db.init_collection(
        dim=embedding_model.dimension,
        collection=collection_name,
    )




