import attr
import pandas as pd
import typing

from epoching.transform.transformer import EpochTransformer
from etl.transform.preprocessing import FirBandPassFilter


@attr.s(auto_attribs=True)
class DesynchronizationFiltersTransformer(EpochTransformer):
    sample_freq: int = 500
    input_bands: typing.List[tuple] = attr.ib(factory=list)
    input_channels: typing.List[str] = attr.ib(factory=list)
    order: int = 101

    def filter_for_band(self, samples: pd.DataFrame, low_cut: int, high_cut: int):
        filter_band = FirBandPassFilter(sample_freq=self.sample_freq, order=self.order,
                                        low_cut=low_cut, high_cut=high_cut, filter_back=True)

        for input_channel in self.input_channels:
            channel = samples[[input_channel]]
            filtered = filter_band.process(channel)
            filtered_squared = filtered * filtered
            samples[input_channel + "_filter_band_" + str(low_cut) + "_" + str(high_cut)] = filtered_squared

    def process_epoch(self, data: pd.DataFrame) -> None:
        for low_cut, high_cut in self.input_bands:
            self.filter_for_band(data, low_cut, high_cut)

        return data
