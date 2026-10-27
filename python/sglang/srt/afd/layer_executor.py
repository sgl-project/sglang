"""Shared layer/stage order for AFD attention and ordinary FFN backends."""

from contextlib import nullcontext
from functools import partial
from typing import Any

from sglang.srt.batch_overlap.operations import (
    YieldOperation,
    execute_overlapped_operations,
)


def execute_layer_operations(num_layers: int, operation: Any) -> None:
    execute_overlapped_operations(
        inputs_arr=[{}, {}],
        operations_arr=[
            [
                op
                for layer in range(num_layers)
                for op in (
                    partial(operation, layer=layer, index=index),
                    YieldOperation(),
                )
            ]
            for index in range(2)
        ],
        delta_stages=[0, 0],
        stage_contexts=[nullcontext, nullcontext],
    )
