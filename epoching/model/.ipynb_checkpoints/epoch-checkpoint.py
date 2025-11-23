import pandas as pd
import attr
import typing

@attr.s(auto_attribs=True)
class Epoch:
    samples: pd.DataFrame
    noise: typing.Any
    marker_sample: typing.Optional[int] = None