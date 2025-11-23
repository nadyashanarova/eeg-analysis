import attr
import pandas as pd
import typing

from etl.transform.transform import Transformer


@attr.s(auto_attribs=True)
class LargeLaplacianFilter(Transformer):
    target_channel: str = 'Cz'
    input_channels: typing.List[str] = attr.ib(factory=list)
    output_channel: str = None

    def __attrs_post_init__(self):
        if not self.input_channels:
            self.input_channels = ['F3', 'Fz', 'F4', 'C3', 'C4', 'Pz']
        if not self.output_channel:
            self.output_channel = f"{self.target_channel}_laplacian"

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.output_channel] = \
            df[self.target_channel] - df[self.input_channels].sum(axis=1) / len(self.input_channels)
        print(f"Create a new surrogate channel: {self.output_channel} from target channel {self.target_channel} "
              f"using channels: {self.input_channels}")
        return df


@attr.s(auto_attribs=True)
class ChannelsSubtractor(Transformer):
    subtrahend_channel: str
    minuend_channels: str
    output_channel: str = None

    def __attrs_post_init__(self):
        if not self.output_channel:
            self.output_channel = f"{self.subtrahend_channel}-{self.minuend_channels}"

    def process(self, df: pd.DataFrame) -> None:
        df[self.output_channel] = \
            df[self.subtrahend_channel] - df[self.minuend_channels]
        print(f"Create a new surrogate channel: {self.output_channel} from difference {self.subtrahend_channel}"
              f" and {self.minuend_channels}")
        return df
