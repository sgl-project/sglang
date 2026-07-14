from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Layout(ABC):
    initial_shape: list[int]

    @abstractmethod
    def swizzle_data(self, data):
        pass

    @abstractmethod
    def unswizzle_data(self, data):
        pass

    @abstractmethod
    def swizzle_block_shape(self, block_shape):
        pass
