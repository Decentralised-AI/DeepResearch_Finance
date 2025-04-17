from typing import Dict, List

class ChatResponse:

    def __init__(self, content: str, total_tokens: int) -> None:
        self.content = content
        self.total_tokens = total_tokens

    def __repr__(self) -> str:
        """
        Returns a string representation of the chat response
        :return:
           A String representation of the ChatResponse object
        """
        return f"ChatResponse(content={self.content}, total_tokens={self.total_tokens})"

