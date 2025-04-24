from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, BitsAndBytesConfig
import torch
from threading import Thread
from logger import setup_logger
from models.base import Model


class TransformerBaseModel(Model):
    model_id: str
    system_prompt: str

    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = setup_logger(self.__class__.__name__)
        self.tokenizer = self.load_tokenizer()
        self.model = self.load_model()

    def load_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_id)

    def load_model(self):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=quant_config,
            device_map="auto"
        )

        return model

    def get_response(self, messages: list[object]):
        prompt = self.get_chat_template(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=self.max_tokens)
        response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return ''.join(response)
