## Deep Research - Finance

A project for Deep Research in Finance

First, we initialize the configuration using init_config.

init_config initializes the following:

1. llm: Initializes the OpenAI client and provides the api_key, base_url. It also create 
a function called `chat` which is using the chat.completion.create finctionality.
   
2. In the same way it initializes the embedding model using the OpenAI SDK.

3. The file loader: The file loader supports multiple file types such as pdf, json, txt, 
and unstructured data. The pdf_loader loads pdfs. It parses pdf using the pdfplumber, reads
   the content from each page using the `extract_text()` function, and returns 
   a Document class (from langchain_core.documents):
   ```python

       import pdfplumber

       if file_path.endswith(".pdf"):
            with pdfplumber.open(file_path) as file:
                page_content = "\n\n".join([page.extract_text() for page in file.pages])
                return [Document(page_content=page_content, metadata={"reference": file_path})]
   ```
   
4. The vectordb:  We initialize the [milvus db](https://milvus.io/docs/quickstart.md). Milvus is an open-source vector db build for GenAI apps.
   ##### Quick Example:

   





