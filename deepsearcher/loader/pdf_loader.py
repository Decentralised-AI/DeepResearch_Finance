from typing import List
from langchain_core.documents import Document
from deepsearcher.loader.base import BaseLoader
import pdfplumber


class PDFLoader(BaseLoader):

    def __init__(self):

        pass

    def load_file(self, file_path: str) -> List[Document]:
        """
        Load a PDF file and convert it to a Document object
        :param file_path:
        :return:
        """
        if file_path.endswith(".pdf"):
            with pdfplumber.open(file_path) as file:
                page_content = "\n\n".join([page.extract_text() for page in file.pages])
                return [Document(page_content=page_content, metadata={"reference": file_path})]

        elif file_path.endswith(".txt") or file_path.endswith(".md"):
            with open(file_path, "r") as file:
                page_content = file.read()
                return [Document(page_content=page_content, metadata={"reference": file_path})]


    @property
    def supported_file_types(self) -> List[str]:
        return ["pdf", "md", 'txt']