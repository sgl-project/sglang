from enum import IntEnum, auto


class SpeculativeAlgorithm(IntEnum):
    EAGLE = auto()

    def is_eagle(self):
        return self == SpeculativeAlgorithm.EAGLE

    @staticmethod
    def from_string(name: str):
        name_map = {
            "EAGLE": SpeculativeAlgorithm.EAGLE,
        }
        return name_map[name]


class SpecInfo:
    pass
