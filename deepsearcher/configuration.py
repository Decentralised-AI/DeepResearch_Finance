from deepsearcher.llm.base import BaseLLM
from deepsearcher.vector_db.base import BaseVectorDB
from deepsearcher.embedding.base import BaseEmbedding
from deepsearcher.agent.rag_router import RAGRouter

llm: BaseLLM = None
embedding_model: BaseEmbedding = None
vector_db: BaseVectorDB = None
default_searcher: RAGRouter = None
naive_rag: NaiveRAG = None
