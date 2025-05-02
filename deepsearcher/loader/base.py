import os
from typing import List

class BaseLoader:

    def __init__(self, **kwargs):
        pass

    def load_file(self, file_path: str) -> List[Document]:

        pass

    def load_directory(self, directory: str) -> List[Document]:
        """
        Load all supported files from a directory
        :param directory:
        :return:
        """
        documents = []
        for file in os.listdir(directory):
            for suffix in self.supported_file_types:
                if file.endswith(suffix):
                    documents.extend(self.load_file(os.path.join(directory, file)))
        return documents

    @property
    def supported_file_types(self) -> List[str]:
        pass