from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, BitsAndBytesConfig
import torch
from threading import Thread
from logger import setup_logger
from models.base import Model


class TransformerBaseModel(Model):
    model_id: str
    system_prompt: str

    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        self.load_model()

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
        )

    def get_streamer(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        # Start generation in a thread
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=self.max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        return streamer

    def get_stream_output(self, streamer):
        response = ""
        for token in streamer:
            response += token
        return response

    def get_response(self, messages: list[object]):
        prompt = self.get_chat_template(messages)
        streamer = self.get_streamer(prompt)
        return self.get_stream_output(streamer)
