from openai import OpenAI
import os
from models.base import Model


class GPT(Model):

    def __init__(self, gpt_model_id: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_id = gpt_model_id
        self.client = self.get_client()

    def get_client(self):
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def get_chat_template(self, messages: list[object]):
        return messages

    def get_response(self, messages: list[object]):
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=self.get_chat_template(messages),
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return response.choices[0].message.content
