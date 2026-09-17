# SPDX-License-Identifier: Apache-2.0
"""Placement policy selection for the KVCR linker's local DRAM tier."""

from __future__ import annotations

from sglang.srt.utils import dynamic_import

# The core's default policy has changed between revisions, so the linker always
# names one and records the name with any benchmark.
_BUILTIN_POLICIES = ("fifo", "lru")


def resolve_policy(name: str):
    from kvcr.policy import FIFOPolicy, KVCachePolicy, LRUPolicy

    builtin = {"fifo": FIFOPolicy, "lru": LRUPolicy}
    policy_type = builtin.get(name)
    if policy_type is None:
        if "." not in name:
            raise ValueError(
                f"KVCR linker policy {name!r} is unknown; use one of "
                f"{list(_BUILTIN_POLICIES)} or a module.Class path. The G3 policies "
                "need a file tier this linker does not configure."
            )
        policy_type = dynamic_import(name)
        if not isinstance(policy_type, type) or not issubclass(
            policy_type, KVCachePolicy
        ):
            raise TypeError(
                f"KVCR linker policy {name} is not a KVCachePolicy subclass."
            )
    return policy_type()
