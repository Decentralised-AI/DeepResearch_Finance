from langchain_openai import ChatOpenAI
from deepsearcher.agent.base import describe_class
import re
import sqlite3
from typing import Dict, List

SYSTEM_TEMPLATE = """
    Answer the user's questions based on the below context. 
    If the context doesn't contain any relevant information to the questions, don't make someting up
    and just reply information cannot be fount:
    <context>
    {context}
    </context>
    """

sql_prompt = """You are an SQL expert coder and you need to convert user queries into relevant SQL code in order
                to retrieve the required information from a structured SQLite database. \n

                Below are the columns from the table called fundamentals in the sqlite database along with their description \n:

                date - The date which the company (Ticker) earnings where reported \n
                ticker - The ticker of the company in question. The tickers are always in lowercase strings such as nvda, vz \n
                price - The price of the stock for the particular ticker the specified date \n
                sp500 - The S&P 500 index price for the specified date \n
                market_cap - The market cap of the company \n
                enterprice_value - The enterprice value of the company \n
                trailing_pe - The trailing P/E ratio \n
                forward_pe - The forward P/E ratio \n
                peg_ratio - The PEG Ratio \n
                price_sales - It is the financial evaluation metric that compares a company's market cap (or stock price) to its total revenue \n
                price_book - This is the price to book ratio. It compares a company's market price per share to its book value per share (BVPS) \n
                enterprice_value_revenue - is a ratio that compares a company's total enterprice value (EV) to its revenue \n
                quarter - the quarter and the year which the financial data were reported. For example, 2012Q4
                """

@describe_class(
    "This can perform calculations on structured data that are available on sqlite database"
    "It is very suitable for performing analytics on companies fundamental such as: the 'price' "
    "of the share, the"
    " market cap, enterprice values, PEG Ratio, Price/Book, earnings, p/e ratios and etc."
)
class StructureDBAgent:

    def __init__(
            self,
            model,
            db_path):
        self.llm = ChatOpenAI(model=model)
        self.db_path = db_path

    def texttosqlqa(self, query: str) -> str:
        """
        It allows users to answer questions related to historical financial fundamentals from s&p 500 companies.
        For example, it allows you to retrieve information ragarding the price in a particular quarter, market cap,
        enterprise value, P/E and etc.

        Parameters:
        -statsquestions: the question we need to answer from the historical financial fundamentals of the s&p 500 companies

        Returns:
        - A string with the right answer
        """
        messages = [
            {"role": "system", "content": sql_prompt},
            {"role": "user", "content": query}
        ]
        response = self.llm(messages)
        output = response.content
        pattern = r"```sql\n(.*?)\n```"
        match = re.search(pattern, output, re.DOTALL)
        if match:
            sql_code = match.group(1)
            conn = sqlite3.connect("sp500_financials_database.db")
            cursor = conn.cursor()
            cursor.execute(sql_code)
            results = cursor.fetchall()
            return results
        else:
            return "No SQL code found"

    def query(self, query: str) -> str:
        """
        Return the retrieved information from the SQL DB.
        :param query:
        :return:
        """
        retrieved_results = self.texttosqlqa(query)
        return retrieved_results









