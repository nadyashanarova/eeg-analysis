import abc
import typing
import attr
from common import utils
from dataset.dataset import ProcessingSignal, Signal
from epoching.model.epoch import Epoch
from epoching.model.markers import Markers

@attr.s(auto_attribs=True)
class EpochCutter(abc.ABC):
    def cut(self, signal: ProcessingSignal, markers: Markers) -> None:
        artifacts = utils.get_artifacts_samples(signal.signal.df)
        signal.epochs = self.get_epochs(signal.signal, markers, artifacts)

    @abc.abstractmethod
    def get_epochs(
        self, signal: Signal, markers: Markers, artifacts_samples: typing.List[int]
    ) -> typing.List[Epoch]:
        pass