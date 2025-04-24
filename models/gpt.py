from openai import OpenAI
import os
from models.base import Model


class GPT(Model):

    def __init__(self, gpt_model_id: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_id = gpt_model_id
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def get_chat_template(self, messages: list[object]):
        return messages

    def get_response(self, messages: list[object]):
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content
