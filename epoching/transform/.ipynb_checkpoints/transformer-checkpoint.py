from abc import ABC, abstractmethod
import attr
import pandas as pd
from dataset.dataset import ProcessingSignal

@attr.s(auto_attribs=True, kw_only=True)
class EpochTransformer(ABC):
    @abstractmethod
    def process_epoch(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    def transform(self, input_signal: ProcessingSignal) -> None:
        for epoch in input_signal.epochs:
            epoch.samples = self.process_epoch(epoch.samples)