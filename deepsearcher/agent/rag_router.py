from deepsearcher.tools import log
from deepsearcher.vector_db.base import RetrievalResult

RAG_ROUTER_PROMPT = """Given a list of agent indexes and corresponding descriptions, each agent has a specific function. 
Given a query, select only one agent that best matches the agent handling the query, and return the index without any other information.

## Question
{query}

## Agent Indexes and Descriptions
{description_str}

Only return one agent index number that best matches the agent handling the query:
"""

class RAGRouter:

    def __init__(self, llm, rag_agents, agent_descriptions):

        """
        Initializes the RAGRouter
        :param llm:
        :param rag_agents:
        :param agent_descriptions:
        """
        self.llm = llm
        self.rag_agents = rag_agents
        self.agent_descriptions = agent_descriptions

    def _route(self, query: str):
        description_str = "\n".join(
            [f"[{i + 1}]: {description}" for i, description in enumerate(self.agent_descriptions)]
        )
        prompt = RAG_ROUTER_PROMPT.format(query=query, description_str = description_str)
        chat_response = self.llm.chat(messages=[{"role": "user", "content": prompt}])
        try:
            selected_agent_index = int(chat_response.content) - 1
        except ValueError:
            print("Parse int failed in RAGRouter, but will try to find the last digit as fallback.")
            selected_agent_index = int(self.find_last_digit(chat_response.content)) - 1

        selected_agent = self.rag_agents[selected_agent_index]
        log.color_print(
            f"<think> Select agent [{selected_agent.__class__.__name__}] to answer the query [{query}] </think>\n"
        )
        return self.rag_agents[selected_agent_index], chat_response.total_tokens


    def retrieve(self, query: str, **kwargs) -> Tuple[List[RetrievalResult], int, dict]:












