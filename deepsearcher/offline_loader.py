from deepsearcher.vector_db.milvus import Milvus
from deepsearcher import configuration
from tqdm import tqdm
import os
from typing import List, Union
from deepsearcher.loader.splitter import split_docs_to_chunks
def load_from_local_files(

        paths_or_directory: Union[str, List[str]],
        collection_name: str = None,
        collection_description: str = None,
        force_new_collection: bool = False,
        chunk_size: int = 1500,
        chunk_overlap: int = 100,
        batch_size: int = 256
):
    """
    Load knowledge from local files or directories into the vector database
    It processes files from a path, splits them into chunks, embeds the chunks,
    and stores them in a vector db
    :param paths_or_directory:
    :param collection_name:
    :param collection_description:
    :param force_new_collection:
    :param chunk_size:
    :param chunk_overlap:
    :param batch_size:
    :return:
    """

    embedding_model = configuration.embedding_model
    vector_db = configuration.vector_db
    if collection_name is None:
        collection_name = vector_db.default_collection

    collection_name = collection_name.replace(" ", "_").replace("-", "_")
    file_loader = configuration.file_loader
    vector_db.init_collection(
        dim=embedding_model.dimension,
        collection=collection_name,
        description=collection_description,
        force_new_collection=force_new_collection
    )
    if isinstance(paths_or_directory, str):
        paths_or_directory = [paths_or_directory]

    all_docs = []
    for path in tqdm(paths_or_directory, desc="Loading Files"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Error: File or directory '{path}' does not exist.")
        if os.path.isdir(path):
            docs = file_loader.load_directory(path)
        else:
            docs = file_loader.load_file(path)
        all_docs.extend(docs)

    chunks = split_docs_to_chunks(
        all_docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = embedding_model.embed_chunks(chunks, batch_size=batch_size)
    vector_db.insert_data(collection=collection_name, chunks=chunks)





