# SPDX-License-Identifier: Apache-2.0
"""Collect immutable model-specific derived weight state for IPC export."""

from __future__ import annotations

from typing import TypeAlias

import torch

_PlainValue: TypeAlias = bool | int | float

_PLAIN_TENSOR_ATTRS = frozenset(
    {"w_kc", "w_vc", "w_scale", "w_scale_k", "w_scale_v"}
)
_PLAIN_VALUE_ATTRS = frozenset({"use_deep_gemm_bmm"})


def collect_derived_weight_state(
    model: torch.nn.Module,
) -> tuple[dict[str, torch.Tensor], dict[str, _PlainValue]]:
    """Return exact post-load tensors and scalar flags not registered on modules."""
    tensors: dict[str, torch.Tensor] = {}
    values: dict[str, _PlainValue] = {}
    supported_attrs = _PLAIN_TENSOR_ATTRS | _PLAIN_VALUE_ATTRS

    for module_name, module in model.named_modules():
        registered_names = set(module._parameters) | set(module._buffers)
        for attr_name in sorted(supported_attrs - registered_names):
            if attr_name not in vars(module):
                continue
            name = f"{module_name}.{attr_name}" if module_name else attr_name
            value = vars(module)[attr_name]
            if isinstance(value, torch.Tensor):
                if attr_name in _PLAIN_VALUE_ATTRS:
                    raise RuntimeError(
                        "weight-cache scalar derived attribute became a Tensor: "
                        f"{name}"
                    )
                tensors[name] = value
            elif value is None:
                continue
            elif isinstance(value, (bool, int, float)):
                values[name] = value
            else:
                raise RuntimeError(
                    "weight-cache derived attribute has unsupported type: "
                    f"{name}: {type(value).__name__}"
                )
    return tensors, values
