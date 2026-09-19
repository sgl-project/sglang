from enum import Enum, auto


class Modality(Enum):
    IMAGE = auto()
    VIDEO = auto()
    AUDIO = auto()

    @staticmethod
    def from_str(modality_str: str):
        try:
            return Modality[modality_str.upper()]
        except KeyError:
            raise ValueError(
                f"Invalid modality string: {modality_str}. Valid modalities are: {[m.name for m in Modality]}"
            )

    @staticmethod
    def all():
        return [Modality.IMAGE, Modality.VIDEO, Modality.AUDIO]


class MultimodalInputFormat(Enum):
    NORMAL = auto()
    PROCESSOR_OUTPUT = auto()
    PRECOMPUTED_EMBEDDING = auto()
