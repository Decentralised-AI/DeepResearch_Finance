from typing import List
from openai import OpenAI
from deepsearcher.embedding.base import BaseEmbedding
import os

OPENAI_MODEL_DIM_MAP = {
    "text-embedding-ada-002": 1536,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}


class OpenAIEmbedding(BaseEmbedding):

    def __init__(self, model: str = "text-embedding-ada-002", **kwargs):
        api_key = kwargs.pop("api_key")

        if "dimension" in kwargs:
            dimension = kwargs.pop("dimension")
        else:
            dimension = OPENAI_MODEL_DIM_MAP[model]
        if "model_name" in kwargs and (not model or model == "text-embedding-ada-002"):
            model = kwargs.pop("model_name")
        if "base_url" in kwargs:
            base_url = kwargs.pop("base_url")
        else:
            base_url = os.getenv("OPENAI_BASE_URL")
        self.dim = dimension
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url, **kwargs)

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text
        :param text: The query text to embed
        :return: List[float]: A list of floats representing the embedding vector
        """
        return (
            self.client.embeddings.create(input=[text], model=self.model, dimensions=self.dim).data[0].embedding
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of document texts
        :param texts: A list of document texts to embed
        :return: A list of embedding vectors, one for each input text
        """
        res = self.client.embeddings.create(input=[texts], model=self.model, dimensions=self.dim)
        res = [r.embedding for r in res.data]
        return res

    @property
    def dimension(self) -> int:
        return self.dim