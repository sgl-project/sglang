"""Distributed execution context for exported callsite runtimes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class DistributedExportContext:
    tp_rank: int = 0
    tp_size: int = 1
    pp_rank: int = 0
    pp_size: int = 1
    dp_rank: int = 0
    dp_size: int = 1
    ep_rank: int = 0
    ep_size: int = 1
    device_type: str = "cpu"
    device_index: int | None = None

    @classmethod
    def current(cls, args: tuple[Any, ...] = ()) -> "DistributedExportContext":
        device = _first_tensor_device(args)
        server_args = _safe_server_args()
        return cls(
            tp_rank=_safe_parallel_value("get_tensor_model_parallel_rank", 0),
            tp_size=_safe_parallel_value(
                "get_tensor_model_parallel_world_size",
                _server_arg(server_args, "tp_size", 1),
            ),
            pp_rank=_safe_parallel_value("get_pipeline_model_parallel_rank", 0),
            pp_size=_safe_parallel_value(
                "get_pipeline_model_parallel_world_size",
                _server_arg(server_args, "pp_size", 1),
            ),
            dp_rank=0,
            dp_size=_server_arg(server_args, "dp_size", 1),
            ep_rank=_safe_parallel_value("get_moe_expert_parallel_rank", 0),
            ep_size=_safe_parallel_value(
                "get_moe_expert_parallel_world_size",
                _server_arg(server_args, "ep_size", 1),
            ),
            device_type=device.type,
            device_index=device.index,
        )

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


def _first_tensor_device(args: tuple[Any, ...]) -> torch.device:
    for arg in args:
        if isinstance(arg, torch.Tensor):
            return arg.device
    return torch.device("cpu")


def _safe_parallel_value(fn_name: str, default: int) -> int:
    try:
        from sglang.srt.distributed import parallel_state

        fn = getattr(parallel_state, fn_name)
        return int(fn())
    except Exception:
        return default


def _safe_server_args() -> Any | None:
    try:
        from sglang.srt.server_args import get_global_server_args

        return get_global_server_args()
    except Exception:
        return None


def _server_arg(server_args: Any | None, name: str, default: int) -> int:
    if server_args is None:
        return default
    return int(getattr(server_args, name, default))


__all__ = ["DistributedExportContext"]
