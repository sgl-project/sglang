from enum import IntEnum, auto


class SpeculativeAlgorithm(IntEnum):
    NONE = auto()
    EAGLE = auto()

    # NEXTN spec decoding is for DeepSeek V3/R1
    # currently it's implemented based on EAGLE
    NEXTN = auto()

    def is_none(self):
        return self == SpeculativeAlgorithm.NONE

    def is_eagle(self):
        return self == SpeculativeAlgorithm.EAGLE or self == SpeculativeAlgorithm.NEXTN

    def is_nextn(self):
        return self == SpeculativeAlgorithm.NEXTN

    @staticmethod
    def from_string(name: str):
        name_map = {
            "EAGLE": SpeculativeAlgorithm.EAGLE,
            "NEXTN": SpeculativeAlgorithm.NEXTN,
            None: SpeculativeAlgorithm.NONE,
        }
        if name is not None:
            name = name.upper()
        return name_map[name]


class SpecInfo:
    pass
