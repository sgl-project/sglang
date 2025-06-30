# connector for NVIDIA Dynamo KVBM integration

try:
    from dynamo.llm.sglang_integration import KvbmCacheManager  # noqa: F401
except ModuleNotFoundError:
    KvbmCacheManager = None  # type: ignore


_kvbm_manager = None  # type: ignore


def get_kvbm_manager():
    """Return a singleton instance of KvbmCacheManager.

    Raises
    ------
    ImportError
        If `ai-dynamo` python binding is not installed when requested.
    """
    global _kvbm_manager

    if KvbmCacheManager is None:
        raise ImportError(
            "ai-dynamo python bindings not installed. Install with `pip install \"ai-dynamo[all]\"`."
        )

    if _kvbm_manager is None:
        _kvbm_manager = KvbmCacheManager()
    return _kvbm_manager
