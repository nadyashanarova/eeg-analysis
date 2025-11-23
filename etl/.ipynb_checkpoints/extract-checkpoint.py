import attr
from abc import ABC, abstractmethod
from dataset.dataset import Signal

@attr.s(auto_attribs=True, kw_only=True)
class Extractor(ABC):
    @abstractmethod
    def extract(self, signal: Signal) -> None:
        pass