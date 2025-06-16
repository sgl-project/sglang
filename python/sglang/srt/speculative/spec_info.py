from enum import IntEnum, auto


class SpeculativeAlgorithm(IntEnum):
    NONE = auto()
    EAGLE = auto()
    EAGLE3 = auto()
    NAIVE_EAGLE = auto()

    def is_none(self):
        return self == SpeculativeAlgorithm.NONE

    def is_eagle(self):
        return self in [
            SpeculativeAlgorithm.EAGLE,
            SpeculativeAlgorithm.EAGLE3,
            SpeculativeAlgorithm.NAIVE_EAGLE,
        ]

    def is_eagle3(self):
        return self == SpeculativeAlgorithm.EAGLE3

    def is_naive_eagle(self):
        return self == SpeculativeAlgorithm.NAIVE_EAGLE

    @staticmethod
    def from_string(name: str):
        name_map = {
            "EAGLE": SpeculativeAlgorithm.EAGLE,
            "EAGLE3": SpeculativeAlgorithm.EAGLE3,
            "NAIVE_EAGLE": SpeculativeAlgorithm.NAIVE_EAGLE,
            None: SpeculativeAlgorithm.NONE,
        }
        if name is not None:
            name = name.upper()
        return name_map[name]
