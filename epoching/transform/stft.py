import attr
import pandas as pd
import numpy as np
import typing
from scipy.fftpack import fft, fftfreq
from common.utils import ms_to_sample, samples_to_ms
from math import ceil
from scipy.interpolate import interp1d

from epoching.transform.transformer import EpochTransformer


@attr.s(auto_attribs=True)
class STFT(EpochTransformer):
    sample_freq: int = 500
    window_width_ms: int = 500
    window_out_of_bounds: bool = False
    windows_shift_ms: int = 50
    input_bands: typing.List[tuple] = attr.ib(factory=list)
    input_channels: typing.List[str] = attr.ib(factory=list)
    normalization_interval_size_ms: int = 500

    def __attrs_post_init__(self):
        self.window_width_sample = ms_to_sample(self.window_width_ms, self.sample_freq)
        self.window_middle = self.window_width_sample // 2
        self.step = ms_to_sample(self.windows_shift_ms, self.sample_freq)

    def transform_for_channel(self, channel: pd.Series, input_channel: str) -> pd.DataFrame:
        bands_powers = [[] for _ in self.input_bands]

        index = channel.index + 1
        time_axis = samples_to_ms(index, self.sample_freq)

        left_border = 0
        right_border = self.window_width_sample
        lenght_time_axis = len(time_axis)

        extended_channel = np.zeros(lenght_time_axis + self.window_width_sample
                                    - self.step + (lenght_time_axis % self.step))
        extended_channel[self.window_middle:self.window_middle+len(time_axis)] = channel.to_numpy()

        while right_border <= len(extended_channel):
            segment_channel = extended_channel[left_border: right_border + 1]

            w = np.hanning(len(segment_channel))

            segment_fft = fft(segment_channel * w)

            power = np.abs(segment_fft) ** 2
            power /= len(segment_channel)
            signal_freq = fftfreq(len(segment_channel), 1 / self.sample_freq)

            positive_signal_freq = signal_freq[signal_freq > 0]
            positive_power = power[signal_freq > 0]

            band_func = interp1d(positive_signal_freq, positive_power)
            signal_freq_hz = np.array(
                [x for x in range(ceil(positive_signal_freq[0]), len(positive_signal_freq))])
            power_hz = np.array([band_func(x) for x in signal_freq_hz])

            for band_ind, band in enumerate(self.input_bands):
                band_indices = np.where((signal_freq_hz >= band[0]) & (signal_freq_hz <= band[1]))
                band_power = np.sum(power_hz[band_indices])
                bands_powers[band_ind] += [band_power] * self.step

            left_border += self.step
            right_border += self.step

        bands_powers = np.array(bands_powers)

        band_columns = np.zeros((lenght_time_axis, len(self.input_bands)))
        band_names = []
        for band_ind, band in enumerate(self.input_bands):
            band_columns[:, band_ind] = bands_powers[band_ind][:lenght_time_axis]
            normalization_interval_size_sample = ms_to_sample(self.normalization_interval_size_ms, self.sample_freq)
            normalization_interval_mean = \
                band_columns[self.window_middle: self.window_middle + normalization_interval_size_sample, band_ind].mean()
            band_columns[:, band_ind] = (band_columns[:, band_ind] - normalization_interval_mean) / normalization_interval_mean

            band_names.append(input_channel + "_fft_band_" + str(band[0]) + "_" + str(band[1]))
        return pd.DataFrame(band_columns, columns=band_names, index=index-1)

    def process_epoch(self, data: pd.DataFrame) -> pd.DataFrame:
        all_channel_bands = []
        for channel_name in self.input_channels:
            all_channel_bands.append(self.transform_for_channel(data[channel_name], channel_name))
        combined = [data] + all_channel_bands
        return pd.concat(combined, axis=1)