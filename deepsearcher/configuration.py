import os
from typing import Literal
import yaml

from deepsearcher.llm.base import BaseLLM
from deepsearcher.vector_db.base import BaseVectorDB
from deepsearcher.embedding.base import BaseEmbedding
from deepsearcher.agent.rag_router import RAGRouter
from deepsearcher.agent.naive_rag import NaiveRAG
from deepsearcher.loader.base import BaseLoader

from deepsearcher.llm.openai_llm import OpenAISearch
from deepsearcher.embedding.openai_embedding import OpenAIEmbedding
from deepsearcher.loader.pdf_loader import PDFLoader
from deepsearcher.vector_db.milvus import Milvus
from deepsearcher.agent.deep_search import DeepSearch
from deepsearcher.agent.chain_of_rag import ChainOfRAG

current_dir = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_YAML_PATH = os.path.join(current_dir, "..", "config.yaml")
print(DEFAULT_CONFIG_YAML_PATH)

class Configuration:

    def __init__(self, config_path: str = DEFAULT_CONFIG_YAML_PATH):
        config_data = self.load_config_data_from_yaml(config_path)
        self.provide_settings = config_data["provide_settings"]
        self.query_settings = config_data["query_settings"]
        self.load_settings = config_data["load_settings"]

    def load_config_data_from_yaml(self, config_path: str):

        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def set_provider_config(self, feature, provider, provider_configs):

        if feature not in self.provide_settings:
            raise ValueError(f"Unsupported feature: {feature}")
        self.provide_settings[feature]["provider"] = provider
        self.provide_settings[feature]["config"] = provider_configs



config = Configuration()
print(config.provide_settings)

llm: BaseLLM = None
embedding_model: BaseEmbedding = None
file_loader: BaseLoader = None
vector_db: BaseVectorDB = None
default_searcher: RAGRouter = None
naive_rag: NaiveRAG = None


def init_config(config: Configuration):

    global llm, embedding_model, file_loader, vector_db, default_searcher, naive_rag
    llm_config = config.provide_settings["llm"]["config"]
    embedding_config = config.provide_settings["embedding"]["config"]
    vector_db_config = config.provide_settings["vector_db"]["config"]
    print(llm_config)

    llm = OpenAISearch(**llm_config)
    embedding_model = OpenAIEmbedding(**embedding_config)
    file_loader = PDFLoader()
    vector_db = Milvus(**vector_db_config)
    default_searcher = RAGRouter(
        llm=llm,
        rag_agents=[
            DeepSearch(
                llm=llm,
                embedding_model=embedding_model,
                vector_db=vector_db,
                max_iter=config.query_settings['max_iter'],
                route_collection=True,
                text_window_splitter=True
            ),
            ChainOfRAG(
                llm=llm,
                embedding_model=embedding_model,
                vector_db=vector_db,
                max_iter=config.query_settings["max_iter"],
                route_collection=True,
                text_window_splitter=True
            )
        ]
    )

    naive_rag = NaiveRAG(
        llm=llm,
        embedding_model=embedding_model,
        vector_db=vector_db,
        top_k=10,
        route_collection=True,
        text_window_splitter=True
    )


