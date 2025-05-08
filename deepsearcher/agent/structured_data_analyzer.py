
from deepsearcher.agent.base import describe_class

@describe_class(
    "This can perform calculations on structured data that are available on excel files"
    "It is very suitable for performing analytics on companies fundamental such as: the 'price' of the share, the"
    " market cap, enterprice values, PEG Ratio, Price/Book, earnings, p/e ratios and etc."
)
class StructureDBAgent:

    def __init__(
            self,
            llm):

        self.llm = llm



