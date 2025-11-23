import pywt
import attr
import pandas as pd
import numpy as np
import typing

from epoching.transform.transformer import EpochTransformer


@attr.s(auto_attribs=True)
class WaveletTransformer(EpochTransformer):
    high_freq_band: int
    low_freq_band: int
    sample_freq: int = 500
    input_channels: typing.List[str] = attr.ib(factory=list)
    level_decomposition: int = 5
    name_wavelet: str = 'db4'

    def __attrs_post_init__(self):
        self.wavelet = pywt.Wavelet(self.name_wavelet)

    def create_band(self,
                    lenght_epoch: int,
                    coeffs: typing.List[np.ndarray],
                    name_band: str,
                    order_coeff: int,
                    level: int,
                    band_names: typing.List[str],
                    band_columns: typing.List[np.ndarray]):

        coeffs_for_band = [coeffs[order_coeff]] + [None] * (level)

        # When converting back, we may not get the exact number of samples.
        # Therefore, we will cut length of the epoch
        signal_for_band = pywt.waverec(coeffs_for_band, self.wavelet)[:lenght_epoch]

        band_columns.append(signal_for_band)
        band_names.append(name_band)

    def transform_for_channel(self, channel: pd.Series, input_channel: str) -> pd.DataFrame:
        index = channel.index
        data = channel.to_numpy()
        lenght_epoch = len(data)

        coeffs = pywt.wavedec(data, self.wavelet, level=self.level_decomposition)

        right_border_freq_band = self.sample_freq // 2

        band_columns = []
        band_names = []
        for i in range(self.level_decomposition, 0, -1):
            middle_freq_band = right_border_freq_band // 2

            if middle_freq_band < self.high_freq_band and right_border_freq_band > self.low_freq_band:
                name_band = f"{input_channel}_wavelet_band_{middle_freq_band}_{right_border_freq_band}"

                self.create_band(lenght_epoch, coeffs, name_band, i,
                                 self.level_decomposition - i + 1, band_names, band_columns)

            right_border_freq_band = middle_freq_band

        name_band = f"{input_channel}_wavelet_band_0_{right_border_freq_band}"

        self.create_band(lenght_epoch, coeffs, name_band, 0,
                         self.level_decomposition, band_names, band_columns)

        data_band = pd.DataFrame(band_columns).T.set_index(index)
        data_band.columns = band_names

        return data_band

    def process_epoch(self, data: pd.DataFrame) -> pd.DataFrame:
        all_channel_bands = []
        for channel_name in self.input_channels:
            all_channel_bands.append(self.transform_for_channel(data[channel_name], channel_name))
        combined = [data] + all_channel_bands
        return pd.concat(combined, axis=1)
