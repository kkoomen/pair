from models.transformer_base import TransformerBaseModel
import re


class Llama2(TransformerBaseModel):
    model_id = "meta-llama/Llama-2-7b-chat-hf"

    system_prompt = """
    You are a helpful, respectful and honest assistant. Always answer as
    helpfully as possible, while being safe. Your answers should not include
    any harmful, unethical, racist, sexist, toxic, dangerous, or illegal
    content. Please ensure that your responses are socially unbiased and
    positive in nature. If a question does not make any sense, or is not
    factually coherent, explain why instead of answering something not correct.
    If you don’t know the answer to a question, please don’t share false
    information.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_chat_template(self, messages: list[object]):
        assert len(messages) > 0, "Messages can't be empty"

        # Use default system prompt if none is set.
        if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def post_process_response(self, response: str) -> str:
        return re.sub(r'\[INST\].*?\[/INST\]', '', response, flags=re.DOTALL).strip()
