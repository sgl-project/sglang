from enum import IntEnum, auto


class SpeculativeAlgorithm(IntEnum):
    NONE = auto()
    EAGLE = auto()
    NGRAM = auto()

    def is_none(self):
        return self == SpeculativeAlgorithm.NONE

    def is_eagle(self):
        return self == SpeculativeAlgorithm.EAGLE

    def is_ngram(self):
        return self == SpeculativeAlgorithm.NGRAM

    @staticmethod
    def from_string(name: str):
        name_map = {
            "EAGLE": SpeculativeAlgorithm.EAGLE,
            "NGRAM": SpeculativeAlgorithm.NGRAM,
            None: SpeculativeAlgorithm.NONE,
        }
        return name_map[name]


class SpecInfo:
    pass
