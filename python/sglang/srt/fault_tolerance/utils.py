"""Shared fault-tolerance configuration queries."""


def is_fault_tolerance_enabled() -> bool:
    """Return False until exec configuration is published, then its FT flag."""
    from sglang.srt.runtime_context import get_context, get_exec

    if not get_context().is_config_namespace_published("exec"):
        return False
    return get_exec().features.enable_fault_tolerance
