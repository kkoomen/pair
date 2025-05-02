from abc import ABC, abstractmethod
import torch


class Model(ABC):

    def __init__(self, max_tokens: int, temperature: float, top_p: float):
        self.max_tokens = max_tokens
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.temperature = temperature
        self.top_p = top_p

    @property
    def name(self):
        return self.__class__.__name__.lower()

    @property
    def model_name(self):
        return self.model_id.split("/")[-1] if "/" in self.model_id else self.model_id

    @abstractmethod
    def get_response(self, messages: list[object]):
        pass

    @abstractmethod
    def get_chat_template(self, messages: list[object]):
        pass
