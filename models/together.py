from openai import OpenAI
import os
from models.gpt import GPT


class Together(GPT):

    def get_client(self):
        return OpenAI(
            api_key=os.getenv("TOGETHER_API_KEY"),
            base_url="https://api.together.xyz/v1",
        )
