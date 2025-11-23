from abc import ABC, abstractmethod
import attr
import numpy as np
import pandas as pd
from scipy import signal
from dataset.dataset import Signal

@attr.s(auto_attribs=True, kw_only=True)
class Transformer(ABC):
    @abstractmethod
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def transform(self, input_signal: Signal) -> None:
        input_signal.df = self.process(input_signal.df)

@attr.s(auto_attribs=True, kw_only=True)
class Filter(Transformer, ABC):
    sample_freq: int
    order: int

    def __attrs_post_init__(self) -> None:
        self.sos = self.design_filter()

    @abstractmethod
    def design_filter(self) -> np.ndarray:
        pass

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        filtered = signal.sosfiltfilt(self.sos, df, axis=0)
        return pd.DataFrame(filtered, columns=df.columns, index=df.index)