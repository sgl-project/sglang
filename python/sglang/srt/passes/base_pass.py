from abc import ABC, abstractmethod
import torch.fx as fx


class BaseFXPass(ABC):
    def __init__(self):
        self.name = self.__class__.__name__
    
    @abstractmethod
    def is_applicable(self, graph: fx.Graph) -> bool:
        pass
    
    @abstractmethod
    def apply(self, graph: fx.Graph) -> bool:
        pass
