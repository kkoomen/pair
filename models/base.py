from abc import ABC, abstractmethod


class Model(ABC):

    def __init__(self, max_tokens: int):
        print('max_tokens', max_tokens)
        self.max_tokens = max_tokens

    @property
    def name(self):
        return self.__class__.__name__.lower()

    @abstractmethod
    def get_response(self, messages: list[object]):
        pass

    @abstractmethod
    def get_chat_template(self, messages: list[object]):
        pass
