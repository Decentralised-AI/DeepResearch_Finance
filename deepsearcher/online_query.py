from typing import List, Tuple
from deepsearcher.vector_db.base import RetrievalResult
from openai import OpenAI
from agent.rag_router import RAGRouter
from agent.deep_search import DeepSearch
from embedding.openai_embedding import OpenAIEmbedding
from deepsearcher import configuration

def query(original_query: str, max_iter: int = 3) -> Tuple[str, List[RetrievalResult]]
    """
    Query the knowlwdge base with a question to get an answer.
    :param original_query: 
    :param max_iter: 
    :return: A Tuple containing:
        - The generated answer as a string
        - A list of retrieval results that were used to generat the answer
        - The number of tokens consumed during the process
    """
    client = OpenAI(api_key=api_key, base_url=base_url, **kwargs)
    embed_config = {
        "model_name": "text-embedding-ada-002",
        "api_key": api_key,
        "base_url": base_url,
        "dimension": 1536}
    embedding_model = OpenAIEmbedding(embed_config)


    default_searcher = RAGRouter(llm = client, rag_agents=[
        DeepSearch(llm=client,
                   embedding_model = embedding_model)
    ])
    return RAGRouter.query(original_query, max_iter=max_iter)

def retrieve(original_query: str, max_iter: int = 3) -> Tuple[List[RetrievalResult], List[str], int]:
    """
    Retrieve relevant information from the knowlwdge base without generating
    an answer.

    This function uses the default searcher to retrieve information from the
    knowlwdge base.
    :param original_query:
    :param max_iter:
    :return:
       A Tuple containing:
          - A list of retrieval results
          - An empty list (placeholder for future use)
          - The number of tokens consumed during the process
    """
    retrieved_results, consume_tokens, metadata = RAGRouter.retrieve(original_query, max_iter=max_iter)
    return retrieved_results, [], consume_tokens