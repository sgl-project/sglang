"""Device-independent integer alignment helpers."""


def align_up(value: int, alignment: int) -> int:
    """Round ``value`` up to a positive byte ``alignment``."""
    return (int(value) + alignment - 1) // alignment * alignment


def align_down(value: int, alignment: int) -> int:
    """Round ``value`` down to a positive byte ``alignment``."""
    return int(value) // alignment * alignment
