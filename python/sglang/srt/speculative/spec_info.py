from enum import IntEnum, auto


class SpeculativeAlgorithm(IntEnum):
    NONE = auto()
    EAGLE = auto()
    LOOKAHEAD = auto()

    def is_none(self):
        return self == SpeculativeAlgorithm.NONE

    def is_eagle(self):
        return self == SpeculativeAlgorithm.EAGLE

    def is_lookahead(self):
        return self == SpeculativeAlgorithm.LOOKAHEAD

    @staticmethod
    def from_string(name: str):
        name_map = {
            "EAGLE": SpeculativeAlgorithm.EAGLE,
            "LOOKAHEAD": SpeculativeAlgorithm.LOOKAHEAD,
            None: SpeculativeAlgorithm.NONE,
        }
        return name_map[name]


class SpecInfo:
    pass
