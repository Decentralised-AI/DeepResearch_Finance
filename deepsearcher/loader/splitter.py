from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class Chunk:

    """
    Represents a chunk of text with associated metadata and embeddings.
    A chunk is a segment of text extracted from a document, along with
    its reference information, metadata, and optional embedding vector.
    :param text:
    :param reference:
    :param metadata:
    :param embedding:
    """
    def __init__(self,
                 text: str,
                 reference: str,
                 metadata: dict,
                 embedding: List[float] = None):

        self.text = text
        self.reference = reference
        self.metadata = metadata or {}
        self.embedding = embedding or None




