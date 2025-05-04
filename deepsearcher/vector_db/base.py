from abc import ABC, abstractmethod
import numpy as np
from typing import List, Union
from deepsearcher.loader.splitter import Chunk

class RetrievalResult:
    """
    Represents a result retrieved from the vector database.

    This class encapsulates the information about a retrieved document,
    including its embedding, text content, reference, metadata, and similarity score.

    Attributes:
        embedding: The vector embedding of the document.
        text: The text content of the document.
        reference: A reference to the source of the document.
        metadata: Additional metadata associated with the document.
        score: The similarity score of the document to the query.
    """

    def __init__(
            self,
            embedding: np.array,
            text: str,
            reference: str,
            metadata: dict,
            score: float = 0.0,
    ):
        """
        Initialize a RetrievalResult object.

        Args:
            embedding: The vector embedding of the document.
            text: The text content of the document.
            reference: A reference to the source of the document.
            metadata: Additional metadata associated with the document.
            score: The similarity score of the document to the query. Defaults to 0.0.
        """
        self.embedding = embedding
        self.text = text
        self.reference = reference
        self.metadata = metadata
        self.score: float = score

    def __repr__(self):
        """
        Return a string representation of the RetrievalResult.

        Returns:
            A string representation of the RetrievalResult object.
        """
        return f"RetrievalResult(score={self.score}, embedding={self.embedding}, text={self.text}, reference={self.reference}), metadata={self.metadata}"


class CollectionInfo:
    """
    Represents information about a collection in the vector database.

    This class encapsulates the name and description of a collection.

    Attributes:
        collection_name: The name of the collection.
        description: The description of the collection.
    """

    def __init__(self, collection_name: str, description: str):
        """
        Initialize a CollectionInfo object.

        Args:
            collection_name: The name of the collection.
            description: The description of the collection.
        """
        self.collection_name = collection_name
        self.description = description

class BaseVectorDB:
    """
    Abstract class for vector database implementations
    """

    def __init__(
            self,
            default_collection: str = "deepsearcher",
            *args,
            **kwargs):

        self.default_collections = default_collection

    @abstractmethod
    def init_collection(
            self,
            dim: int,
            collection: str,
            description: str,
            force_new_collection=False,
            *args,
            **kwargs
    ):
        """
        Initialize a collection in the vector db
        :param dim:
        :param collection:
        :param description:
        :param force_new_collection:
        :param args:
        :param kwargs:
        :return:
        """
        pass


    @abstractmethod
    def insert_data(
            self,
            collection: str,
            chunks: List[Chunk],
            *args,
            **kwargs
    ):
        """
        Initialize a collection in the vector db
        :param collection:
        :param chunks:
        :param args:
        :param kwargs:
        :return:
        """
        pass

    @abstractmethod
    def search_data(
            self, collection: str, vector: Union[np.array, List[float]], *args, **kwargs
    ) -> List[RetrievalResult]:
        """
        Search for similar vectors in a collection
        :param collection:
        :param vector:
        :param args:
        :param kwargs:
        :return:
        """
        pass

    def list_collections(selfself, *args, **kwargs) -> List[CollectionInfo]:
        pass

    @abstractmethod
    def clear_db(self, *args, **kwargs):
        pass









def deduplicate_results(results: List[RetrievalResult]) -> List[RetrievalResult]:
    """
    Remove duplicate results based on text content.

    This function removes duplicate results from a list of RetrievalResult objects
    by keeping only the first occurrence of each unique text content.

    Args:
        results: A list of RetrievalResult objects to deduplicate.

    Returns:
        A list of deduplicated RetrievalResult objects.
    """
    all_text_set = set()
    deduplicated_results = []
    for result in results:
        if result.text not in all_text_set:
            all_text_set.add(result.text)
            deduplicated_results.append(result)
    return deduplicated_results
