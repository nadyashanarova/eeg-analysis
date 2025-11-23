import attr
import pandas as pd
from etl.transform.transform import Transformer
from common.utils import ms_to_sample

@attr.s(auto_attribs=True)
class DCRemover(Transformer):
    sample_freq: int = 500
    start_time: float = 0.0

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        count_rows_start = ms_to_sample(self.start_time, self.sample_freq)
        return df - df[count_rows_start:].mean()