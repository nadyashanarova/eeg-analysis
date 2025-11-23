import typing

import attr

from dataset.dataset import Signal
from etl.extract import Extractor
from etl.load import Loader
from etl.transform.transform import Transformer


@attr.s(auto_attribs=True, kw_only=True)
class EtlPipeline:
    extractor: Extractor
    transformers: typing.List[Transformer]
    loader: Loader

    def process(self, signal: Signal, output_path: str) -> None:
        self.extractor.extract(signal)
        for transformer in self.transformers:
            transformer.transform(signal)
        self.loader.save(signal, output_path)
