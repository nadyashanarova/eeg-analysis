import attr
import typing
from abc import ABC, abstractmethod

@attr.s(auto_attribs=True)
class Markers:
    markers: typing.Dict[int, typing.List[float]]

    def get_events(self, query_marker_type: int):
        return self.markers[query_marker_type]

@attr.s(auto_attribs=True)
class MarkersExtractor(ABC):
    @abstractmethod
    def create(self, path: str) -> Markers:
        pass