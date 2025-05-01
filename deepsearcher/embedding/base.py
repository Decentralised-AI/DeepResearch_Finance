from typing import List
from deepsearcher.loader.splitter import Chunk
from tqdm import tqdm

class BaseEmbedding:
    """
    Abstract base class for embedding model implementations.
    """

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query
        :param text:
        :return:
        """
        pass

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of document texts.
        :param texts:
        :return:
        """
        return [self.embed_query(text) for text in texts]

    def embed_chunks(self, chunks: List[Chunk], batch_size: int = 256) -> List[Chunk]:
        """
        Embed a list of Chunk objects.
        This method extracts the text from each chunk, embeds it in batches,
        updates the chunks with their embeddings.
        :param chunks:
        :param batch_size:
        :return:
        """
        texts = [chunk.text for chunk in chunks]
        batch_texts = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
        embeddings = []
        for batch_text in tqdm(batch_texts, desc="Embedding chunks"):
            batch_embeddings = self.embed_documents(batch_text)
            embeddings.extend(batch_embeddings)
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        return chunks

    @property
    def dimension(self):
        pass



