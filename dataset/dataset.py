import attr
import typing
import pandas as pd

@attr.s(auto_attribs=True, kw_only=True)
class Signal:
    name: str
    edf_path: str
    markers_path: typing.Optional[str] = None
    df: pd.DataFrame = attr.ib(init=False)

@attr.s(auto_attribs=True, kw_only=True)
class ProcessingSignal:
    signal: Signal
    epochs: typing.List['Epoch'] = attr.ib(init=False, factory=list)

@attr.s(auto_attribs=True)
class Dataset:
    signal_to_edf_path: typing.Dict[str, str]
    signal_to_markers_path: typing.Dict[str, str]

    @property
    def signals(self) -> typing.List[Signal]:
        signals = []
        for signal_name, edf_path in self.signal_to_edf_path.items():
            markers_path = self.signal_to_markers_path.get(signal_name)
            signals.append(Signal(name=signal_name, edf_path=edf_path, markers_path=markers_path))
        return signals