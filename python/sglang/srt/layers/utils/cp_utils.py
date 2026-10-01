"""Import-only shims for deprecated platform backends awaiting CP refactoring.

The legacy CP algorithms have been removed. These names keep the retained
NPU/MUSA attention backends importable for non-CP inference; calling them fails.
"""


def _deprecated_platform_cp():
    raise ValueError(
        "Prefill CP on HIP/NPU/MUSA is deprecated; CP support will be refactored soon."
    )


def cp_all_gather_rerange_output(input_tensor, cp_size, forward_batch, stream):
    _deprecated_platform_cp()


def cp_all_gather_rerange_kv_cache(input_tensor, cp_size, forward_batch, stream):
    _deprecated_platform_cp()


def cp_allgather_and_save_kv_cache(forward_batch, layer, k, v, cp_size, swa_loc=None):
    _deprecated_platform_cp()
