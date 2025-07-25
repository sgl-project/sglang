from enum import IntEnum, auto


class SpeculativeAlgorithm(IntEnum):
    NONE = auto()
    EAGLE = auto()
    EAGLE3 = auto()
    SIMPLE_EAGLE = auto()

    def is_none(self):
        return self == SpeculativeAlgorithm.NONE

    def is_eagle(self):
        return self in [
            SpeculativeAlgorithm.EAGLE,
            SpeculativeAlgorithm.EAGLE3,
            SpeculativeAlgorithm.SIMPLE_EAGLE,
        ]

    def is_eagle3(self):
        return self == SpeculativeAlgorithm.EAGLE3

    def is_simple_eagle(self):
        return self == SpeculativeAlgorithm.SIMPLE_EAGLE

    @staticmethod
    def from_string(name: str):
        name_map = {
            "EAGLE": SpeculativeAlgorithm.EAGLE,
            "EAGLE3": SpeculativeAlgorithm.EAGLE3,
            "SIMPLE_EAGLE": SpeculativeAlgorithm.SIMPLE_EAGLE,
            None: SpeculativeAlgorithm.NONE,
        }
        if name is not None:
            name = name.upper()
        return name_map[name]
