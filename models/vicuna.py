from models.transformer_base import TransformerBaseModel


class Vicuna(TransformerBaseModel):
    model_id = "lmsys/vicuna-13b-v1.5"

    system_prompt = """
    A chat between a curious user and an artificial intelligence assistant.
    The assistant gives helpful, detailed, and polite answers to the user’s questions.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_chat_template(self, messages: list[object]):
        assert len(messages) > 0, "Messages can't be empty"

        loop_messages = messages
        system_prompt = self.system_prompt

        if messages[0]["role"] == "system":
            system_prompt = messages[0]["content"]
            loop_messages = messages[1:]

        template = f"""### System:
        {system_prompt}
        """

        for message in loop_messages:
            if message["role"] == "user":
                template += f"\nUSER: {message['content']}"
            elif message["role"] == "assistant":
                template += f"\nASSISTANT: {message['content']}"

        return template
