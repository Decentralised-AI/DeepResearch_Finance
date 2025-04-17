from openai import OpenAI
from deepsearcher.llm.base import ChatResponse
from typing import Dict, List

class OpenAISearch:


    def __init__(self, model: str = "o1-mini", **kwargs):

        """
        Initializes an OpenAI language model client.
        :param model:
        :param kwargs:
        """
        self.model = model
        api_key = kwargs.pop("api_key")
        base_url = kwargs.pop("base_url")
        self.client = OpenAI(api_key=api_key, base_url=base_url, **kwargs)

    def chat(self, messages: List[Dict]) -> ChatResponse:
        completion = self.client.chat.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return ChatResponse(
            content=completion.choices[0].message.content,
            total_tokens=completion.usage.total_tokens
        )


