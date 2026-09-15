# SPDX-License-Identifier: Apache-2.0
"""Reject supported mutation APIs before they touch shared cache allocations."""


def reject_cached_weight_mutation(pipeline, operation):
    args = getattr(pipeline, "server_args", None)
    if getattr(args, "weight_cache_mode", "off") != "off" or any(
        hasattr(module, "_weight_cache_importer")
        for module in getattr(pipeline, "modules", {}).values()
    ):
        raise ValueError(
            f"{operation} is unavailable while weight-cache allocations are shared; restart with weight_cache_mode=off"
        )
