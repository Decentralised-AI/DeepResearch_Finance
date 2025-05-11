from langchain_openai import ChatOpenAI
from deepsearcher.agent.base import describe_class

from typing import Dict, List

@describe_class(
    "This can perform calculations on structured data that are available on excel files"
    "It is very suitable for performing analytics on companies fundamental such as: the 'price' of the share, the"
    " market cap, enterprice values, PEG Ratio, Price/Book, earnings, p/e ratios and etc."
)


class StructureDBAgent:

    def __init__(
            self,
            llm):

        self.tools =
        self.llm = ChatOpenAI(model="gpt-4o-mini")

    def bind_tools(self, tools: List):
        if tools:
            self.tools = {t.name: t for t in tools}
            self.llm = self.llm.bind_tools(tools)








