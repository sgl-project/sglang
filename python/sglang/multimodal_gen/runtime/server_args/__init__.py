# SPDX-License-Identifier: Apache-2.0

from sglang.multimodal_gen.runtime.server_args import base as _base
from sglang.multimodal_gen.runtime.server_args.base import (
    LORA_MERGE_MODES,
    Backend,
    PortArgs,
    ServerArgs,
    get_global_server_args,
    is_ltx2_two_stage_pipeline_name,
    prepare_server_args,
    set_global_server_args,
)

__all__ = [
    "Backend",
    "LORA_MERGE_MODES",
    "PortArgs",
    "ServerArgs",
    "get_global_server_args",
    "is_ltx2_two_stage_pipeline_name",
    "prepare_server_args",
    "set_global_server_args",
]


def __getattr__(name: str):
    if name == "_global_server_args":
        return _base._global_server_args
    raise AttributeError(name)
