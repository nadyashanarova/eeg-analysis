import attr
import numpy as np
from scipy import signal
from etl.transform.transform import Filter

@attr.s(auto_attribs=True, kw_only=True)
class BandStopFilter(Filter):
    low_cut: float
    high_cut: float

    def design_filter(self) -> np.ndarray:
        nyq = 0.5 * self.sample_freq
        low = self.low_cut / nyq
        high = self.high_cut / nyq
        return signal.butter(self.order, [low, high], btype='bandstop', output='sos')

@attr.s(auto_attribs=True, kw_only=True)
class BandPassFilter(Filter):
    low_cut: float
    high_cut: float

    def design_filter(self) -> np.ndarray:
        nyq = 0.5 * self.sample_freq
        low = self.low_cut / nyq
        high = self.high_cut / nyq
        return signal.butter(self.order, [low, high], btype='bandpass', output='sos')