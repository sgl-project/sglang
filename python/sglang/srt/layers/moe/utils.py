from enum import Enum


class MoeA2ABackend(Enum):

    STANDARD = ("standard", "none")
    DEEPEP = "deepep"

    @classmethod
    def _missing_(cls, value):
        if value is None:
            return cls.STANDARD
        for member in cls:
            if value in member.value:
                return member
        raise ValueError(f"No {cls.__name__} member for value {value}")

    def is_deepep(self):
        return self == MoeA2ABackend.DEEPEP

    def is_standard(self):
        return self == MoeA2ABackend.STANDARD


class DeepEPMode(Enum):
    NORMAL = "normal"
    LOW_LATENCY = "low_latency"
    AUTO = "auto"

    def enable_normal(self):
        return self in [DeepEPMode.NORMAL, DeepEPMode.AUTO]

    def enable_low_latency(self):
        return self in [DeepEPMode.LOW_LATENCY, DeepEPMode.AUTO]

    def resolve(self, is_extend_in_batch: bool):
        if self != DeepEPMode.AUTO:
            return self

        if is_extend_in_batch:
            return DeepEPMode.NORMAL
        else:
            return DeepEPMode.LOW_LATENCY
