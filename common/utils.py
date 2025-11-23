import typing
import numpy as np
import pandas as pd

def ms_to_sample(time_ms: typing.Union[int, float], sample_freq: int) -> int:
    return round(time_ms * sample_freq / 1000)

def get_artifacts_samples(df: pd.DataFrame) -> typing.List[int]:
    return np.unique(np.where(df.isna())[0]).tolist()