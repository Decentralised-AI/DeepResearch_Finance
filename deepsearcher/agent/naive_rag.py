from typing import List, Tuple
from deepsearcher.llm.base import BaseLLM
from deepsearcher.embedding.base import BaseEmbedding
from deepsearcher.vector_db.base import BaseVectorDB
from deepsearcher.agent.collection_router import CollectionRouter
from deepsearcher.vector_db.base import RetrievalResult
from deepsearcher.vector_db.base import deduplicate_results

SUMMARY_PROMPT = """You are a AI content analysis expert, good at summarizing content. Please summarize a specific and 
detailed answer or report based on the previous queries and the retrieved document chunks.

Original Query: {query}

Related Chunks: 
{mini_chunk_str}
"""

class NaiveRAG:
    """
    This agent implements the vanilla RAG approach.
    """

    def __init__(
            self,
            llm: BaseLLM,
            embedding_model: BaseEmbedding,
            vector_db: BaseVectorDB,
            top_k: int = 10,
            route_collection: bool = True,
            text_window_splitter: bool = True,
            **kwargs
    ):
        """
        Initialize the naive RAG agent
        :param llm:
        :param embedding_model:
        :param vector_db:
        :param top_k:
        :param route_collection:
        :param text_window_splitter:
        :param kwargs:
        """
        self.llm = llm
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.top_k = top_k
        self.route_collection = route_collection
        if self.route_collection:
            self.collection_router = CollectionRouter(
                llm=self.llm,
                vector_db=self.vector_db,
                dim=embedding_model.dimension
            )
        self.text_window_splitter = text_window_splitter

    def retrieve(self, query: str, **kwargs) -> Tuple[List[RetrievalResult], int, dict]:
        """
        Retrieve relevant docs from the knowledge base for the given query.
        :param query:
        :param kwargs:
        :return:
        """
        consume_tokens = 0
        if self.route_collection:
            selected_collections, n_token_route = self.route_collection.invoke(
                query=query, dim=self.embedding_model.dimension
            )
        else:
            selected_collections = self.collection_router.all_collections
            n_token_route = 0
        consume_tokens += n_token_route
        all_retrieved_results = []
        for collection in selected_collections:
            retrieval_res = self.vector_db.search_data(
                collection=collection,
                vector=self.embedding_model.embed_query(query),
                top_k=max(self.top_k // len(selected_collections), 1)
            )
            all_retrieved_results.extend(retrieval_res)
        all_retrieved_results = deduplicate_results(all_retrieved_results)
        return all_retrieved_results, consume_tokens, {}

    def query(self, query: str, **kwargs) -> Tuple[str, List[RetrievalResult], int]:
        """
        Query the agent and generate an answer based on retrieved documents.
        :param query:
        :param kwargs:
        :return:
        """
        all_retrieved_results, n_token_retrieval, _ = self.retrieve(query)
        chunk_texts = []
        for chunk in all_retrieved_results:
            if self.text_window_splitter and "wider_text" in chunk.metadata:
                chunk_texts.append(chunk.metadata["wider_text"])
            else:
                chunk_texts.append(chunk.text)
        mini_chunk_str = ""
        for i, chunk in enumerate(chunk_texts):
            mini_chunk_str += f"""<chunk_{i}>\n{chunk}\n</chunk_{i}>\n"""

        summary_prompt = SUMMARY_PROMPT.format(query=query, mini_chunk_str=mini_chunk_str)
        chat_response = self.llm.chat([{"role": "user", "content": summary_prompt}])
        final_answer = chat_response.content
        log.color_print("\n==== FINAL ANSWER====\n")
        log.color_print(final_answer)
        return final_answer, all_retrieved_results, n_token_retrieval + chat_response.total_tokens





