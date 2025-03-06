from enum import IntEnum, auto


class SpeculativeAlgorithm(IntEnum):
    NONE = auto()
    EAGLE = auto()

    def is_none(self):
        return self == SpeculativeAlgorithm.NONE

    def is_eagle(self):
        return self == SpeculativeAlgorithm.EAGLE

    @staticmethod
    def from_string(name: str):
        name_map = {
            "EAGLE": SpeculativeAlgorithm.EAGLE,
            None: SpeculativeAlgorithm.NONE,
        }
        if name is not None:
            name = name.upper()
        return name_map[name]
