from enum import IntEnum, unique


@unique
class P2PTag(IntEnum):
    """
    Tags reserved for point-to-point communication protocols.

    Communications introduced outside existing scheduler loops need explicit
    tags to avoid being consumed by unrelated send/recv paths.
    """

    DEFAULT = 0
    HIRADIX_PP_SYNC = int.from_bytes(b"PpHi", byteorder="big")
    GRAMMAR_PP_SYNC = int.from_bytes(b"PpGr", byteorder="big")
