# SPDX-License-Identifier: Apache-2.0
"""Process group options for Ascend."""

import os

from sglang.srt.runtime_context import get_context, get_disagg, get_exec


def eplb_reuses_world_group() -> bool:
    # Disaggregated serving can use the WORLD device group for KV transfers.
    return (
        get_context().is_config_namespace_published("exec")
        and get_exec().moe.eplb_async_rebalance
        and get_disagg().disaggregation_mode == "null"
    )


def _hccl_options(buffer_mib: int):
    import torch_npu

    options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
    options.hccl_config = {"hccl_buffer_size": buffer_mib}
    return options


def eplb_hccl_options():
    return _hccl_options(min(int(os.environ.get("HCCL_BUFFSIZE") or 64), 64))


def get_process_group_options(group_name):
    if group_name == "world" and eplb_reuses_world_group():
        return eplb_hccl_options()

    if group_name is not None and "moe" not in group_name:
        return None

    return _hccl_options(
        int(
            os.environ.get("DEEPEP_HCCL_BUFFSIZE")
            or os.environ.get("HCCL_BUFFSIZE")
            or 200
        )
    )
