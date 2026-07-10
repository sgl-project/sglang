from enum import IntEnum, unique


@unique
class PpP2PTag(IntEnum):
    """
    Tags reserved for PP point-to-point communication protocols.

    PP syncs introduced outside the scheduler loop need explicit tags to avoid
    being consumed by scheduler PP send/recv paths.
    """

    DEFAULT = 0
    HIRADIX_PP_SYNC = int.from_bytes(b"PpHi", byteorder="big")
    GRAMMAR_PP_SYNC = int.from_bytes(b"PpGr", byteorder="big")
