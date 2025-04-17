from abc import ABC


class ModelWeightUpdater:
    def __init__(self, init_pin_memory: bool):
        self._manager_transfer_manager = TODO
        self._model_weight_source = _ModelWeightSourcePinnedMemory() if init_pin_memory else _ModelWeightSourceVanilla()


class _ModelWeightSourceBase(ABC):
    pass


class _ModelWeightSourceVanilla(ABC):
    TODO


class _ModelWeightSourcePinnedMemory(ABC):
    TODO
