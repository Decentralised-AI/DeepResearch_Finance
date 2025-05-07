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
   
5. initialize a web crawler. One option is [firecrawl](https://www.firecrawl.dev/)  or  [Crawl4AI](https://apify.com/janbuchar/crawl4ai)

   
6. Then it initializes the `default_searcher` using the class `RAGRouter`. The class `RAGRouter` is initialized with the
llm, a list of `RAGAgents`. These RAGAgents are the classes: `DeepSearch` and `ChainOfRAG`.  The role of the RAGRouter is
   to route queries to the most appropriate RAG agent implementation. The `DeepSearch`  RAGAgent performs a thorough 
   search through the knowledge base, analyzing multiple aspectecs of the query to provide comprehensive and detailed 
   answers. The `ChainOfRAG` RAGAgent implements a multi-step RAG process ([Chain-of-Retrieval Augmented Generation](https://arxiv.org/pdf/2501.14342)) where each step can refine the query and 
   retrieval processs based on previous results, creating a chain of increasingly focused and relevant information 
   retrieval and generation.
   

How the DeepSearch Agent works in details:



* First, we use the original query to generate up to 4 sub-queries. The idea is that for complex questions, 
sub-questions can be generated to help search for more info to generate a more comprehensive answer. For example,
  the query: Explain dep learning can be broken down in a python list of questions
  ["What is deep learning?",
  "What is the difference between deep learning and machine learning?",
  "What is the history of deep learning?"]
  
* Then for each one of the sub-queries, we search to identify the data collection(s) that has 
  the information to answer the question. Then for each collection, we identify and extract the 
  chunks that are relevant to the sub-query, all the relevant chunk 
in the Milvus DB using the embeddings to perform search (hybrid approach might be more accurate here). 
  Then for each extracted chunk, we use llm to check if it is useful to address the suq-query. 
  The LLM responses with a yes or no (This can help us reduce noise). All retrieved relevant content/chunks are 
  added in a list. The above process runs for all the sub queries. Finally, deduplication is applied in the 
  identified content. Finally, the original query, all the sub-queries and the deduplicated content 
  is passed as input to a final LLM that is trying to identify additional sub-gap-queries in case additional 
  research and content is required to answer the query. this sub_gap_queries are also used to extract relevant content. 
  This process iterates for three times, where new sub-queries and the corresponding context is identified, 
  sub-gap queries and content is also identified and added to the rest of usinque relevant content. The final 
  relevant contents along with the original query and the relenat sub-queries are passed in an LLM which is then
  trying to summarize the content and generate an answer.
  



How the `Chain of RAG` agent works:

The idea of this agent is to decompose complex queries and gradually
find the fact information of sub-queries. It is very suitable
for handling concrete factual queries and multi-hop questions
The agent operates in a few iterations. In the first iteration, 
the agent is using previous intermediate queries and answers 
(not available to the 1st iteration) and tries to generate a new follow-up question
that can help answer the main query when previous answers are not helpful. 
The tool at this stage asks only simple follow-up question as it might be
difficult to address complex questions. The followup query is passed as input to
another LLM agent which is using this query to first identify the data collections that
potentially contain the relevant information. Then it iterates through the identified data
collections and retrieved the relevant chunks. All relevant chunks are gathered in a list.
Then the deduplicated relevant chunks and the follow up questions are passed into 
another LLM agent which is trying to generate and precise answer without hallucinating. 
Then the retrieved relevent to the follow-up question chunks, the follow-up query and 
the intermediate answer are passed as input to another LLM agent which selectes the 
relevant chunks that support the Q-A pair. By Q-A pair, we mean the follow-up query 
the intermediate answer. This selected chunks are stored in a list. 
Also, the follow-up query and the intermediate answer are stored in a list so they 
can be used in the second iteration. Therefore, in the 2nd iteration, the agent 
will try to come up with another follow-up question to help answer parts of the 
initial query that were not covered in the first intermediate answer. 
In this way the model ensure that is asking the question from multiple angles and 
extract the relevant content to fully answer the intial query. An early 
stopping mechanism is developed to stop the iterations when the question has been 
answered thoroughly.






When all the above have been initialized we do the following:

1) We load the data from local files/directories to our milvus db:

2) we load data from a website (TODOs)

3) Then we run the query




