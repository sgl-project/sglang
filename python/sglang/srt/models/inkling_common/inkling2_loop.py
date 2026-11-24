from typing import Literal

import torch
from torch import nn

from sglang.kernels.ops.elementwise.inkling2_loop import message_controllers

CONTROLLER_MODULES = ("attn", "k", "v", "mlp")
_SIDE_ROWS = {"attn": slice(0, 3), "mlp": slice(3, 4)}


class Inkling2MessageController(nn.Module):
    def __init__(self, width: int, rank: int):
        super().__init__()
        self.down_weight = nn.Parameter(
            torch.empty(rank, 2 * width, dtype=torch.bfloat16)
        )
        self.gain_weight = nn.Parameter(torch.empty(width, rank, dtype=torch.bfloat16))
        self.forget_weight = nn.Parameter(
            torch.empty(width, rank, dtype=torch.bfloat16)
        )


def stack_controller_weights(controllers: nn.ModuleDict) -> tuple[torch.Tensor, ...]:
    """Stack each weight of the CONTROLLER_MODULES controllers into one tensor and
    point the parameters at its rows, so reloads update the stacked copy."""
    stacked = []
    for name in ("down_weight", "gain_weight", "forget_weight"):
        params = [getattr(controllers[m], name) for m in CONTROLLER_MODULES]
        weight = torch.stack([p.data for p in params])
        for param, row in zip(params, weight):
            param.data = row
        stacked.append(weight)
    return tuple(stacked)


def scaled_loop_queries(queries: torch.Tensor, kv_queries: torch.Tensor):
    """(core layers, CONTROLLER_MODULES, width) message queries times width**-0.5."""
    stacked = torch.stack(
        (queries[0::2], kv_queries[0::2], kv_queries[1::2], queries[1::2]), dim=1
    )
    return stacked.float() * (stacked.shape[-1] ** -0.5)


class LoopMessageBank:
    def __init__(self, blocks: tuple[torch.Tensor, ...], scaled_queries: torch.Tensor):
        assert blocks and scaled_queries.dtype == torch.float32
        self.blocks = [x.contiguous() for x in blocks]
        self.scaled_queries = scaled_queries
        self._all_messages = None
        if scaled_queries.ndim == 3:
            from sglang.kernels.ops.attention.inkling2_attn_res import attn_res

            self._total_rows = scaled_queries.shape[1]
            self._all_messages = attn_res(
                self.blocks,
                None,
                scaled_queries.reshape(-1, scaled_queries.shape[-1]).contiguous(),
                normalize_values=True,
            )

    def mix(
        self,
        layer: int,
        destinations: tuple[torch.Tensor, ...],
        weights: tuple[torch.Tensor, ...],
        side: Literal["attn", "mlp"],
        packed_weights: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        rows = _SIDE_ROWS[side]
        if self._all_messages is not None:
            base = layer * self._total_rows
            start, stop, step = rows.indices(self._total_rows)
            messages = tuple(
                self._all_messages[base + idx] for idx in range(start, stop, step)
            )
        else:
            from sglang.kernels.ops.attention.inkling2_attn_res import attn_res

            messages = attn_res(
                self.blocks,
                None,
                self.scaled_queries[layer, rows],
                normalize_values=True,
            )
        packed = (
            (packed_weights[0][rows], packed_weights[1][rows])
            if packed_weights is not None
            else None
        )
        return message_controllers(
            tuple(d.float() for d in destinations),
            messages,
            *(w[rows] for w in weights),
            packed_weights=packed,
        )
