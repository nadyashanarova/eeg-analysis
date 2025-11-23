import attr
import typing
from dataset.dataset import Dataset, ProcessingSignal
from epoching.cut.epoch_cutter import EpochCutter
from epoching.model.markers import MarkersExtractor
from etl.extract import Extractor
from etl.transform.transform import Transformer
from epoching.transform.transformer import EpochTransformer

@attr.s(auto_attribs=True, kw_only=True)
class EpochingPipeline:
    extractor: Extractor
    markers_extractor: MarkersExtractor
    transformers: typing.List[Transformer] = attr.ib(factory=list)
    epoch_cutter: EpochCutter
    epoch_transformers: typing.List[EpochTransformer] = attr.ib(factory=list)
    processed_signals: typing.List[ProcessingSignal] = attr.ib(init=False, factory=list)
    persist_intermediate: bool = False

    def process(self, dataset: Dataset) -> None:
        for signal in dataset.signals:
            self.extractor.extract(signal)
            processing_signal = ProcessingSignal(signal=signal)
            
            for transformer in self.transformers:
                transformer.transform(processing_signal.signal)

            markers = self.markers_extractor.create(processing_signal.signal.markers_path)
            self.epoch_cutter.cut(processing_signal, markers)

            for epoch_transformer in self.epoch_transformers:
                epoch_transformer.transform(processing_signal)

            self.processed_signals.append(processing_signal)