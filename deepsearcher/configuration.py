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

llm: BaseLLM = None
embedding_model: BaseEmbedding = None
file_loader: BaseLoader = None
vector_db: BaseVectorDB = None
default_searcher: RAGRouter = None
naive_rag: NaiveRAG = None


def init_config(config):

    global llm, embedding_model, file_loader, vector_db, default_searcher, naive_rag
    llm = OpenAISearch(config["model"], )
    embedding_model = OpenAIEmbedding(config)
    file_loader = PDFLoader(config)
    vector_db = Milvus(config)
    default_searcher = RAGRouter(
        llm = llm,
        rag_agents=[
            DeepSearch(
                llm=llm,
                embedding_model=embedding_model,
                vector_db=vector_db,
                max_iter=config['max_iter'],
                route_collection=True,
                text_window_splitter=True
            ),
            ChainOfRAG(
                llm=llm,
                embedding_model=embedding_model,
                vector_db=vector_db,
                max_iter=config["max_iter"],
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


