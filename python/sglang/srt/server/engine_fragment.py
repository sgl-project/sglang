# TODO rename this class
class EngineFragment:
    """
    Similar to `Engine`. The difference is that, `Engine` handles TP internally, thus users only need
    to have one single `Engine`. Contrary to that, users need to have one `EngineFragment` per TP rank.
    """
