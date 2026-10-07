# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""ROCm-only linear-capability probes for FP8 producer fusion.

A producer fusion (RMSNorm/gate kernel that also quantizes) is only worth
taking when the consuming linear accepts the quantized activation as is. These
probes answer that for the two AITER layouts; CUDA never takes these paths, so
the widened detection lives here instead of in the shared ``utils`` module.
"""

import torch

__all__ = ["accepts_group128_fp8_tuple", "accepts_ptpc_fp8_tuple"]


# gfx942 stores fp8 weights as e4m3fnuz; gfx95x as e4m3fn.
_FP8_ACTIVATION_DTYPES = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)


def accepts_group128_fp8_tuple(proj: torch.nn.Module) -> bool:
    """Whether ``proj`` consumes ``(fp8, per-group-128 scale)`` without re-quant.

    Wider than the shared ``_is_block_scale_fp8``: ROCm checkpoints also land as
    fnuz weights, as ``weight_scale_inv``, or with the block size only on the
    quant method. Kept separate so CUDA weight-name derivation is unaffected.
    """
    if proj.weight.dtype not in _FP8_ACTIVATION_DTYPES:
        return False
    for name in ("weight_scale", "weight_scale_inv"):
        weight_scale = getattr(proj, name, None)
        if (
            weight_scale is not None
            and weight_scale.dim() == 2
            and weight_scale.shape[-1] > 1
        ):
            return True
    weight_block_size = getattr(proj.quant_method, "weight_block_size", None)
    return list(weight_block_size or []) == [128, 128]


def accepts_ptpc_fp8_tuple(module: torch.nn.Module) -> bool:
    """Whether ``apply_fp8_linear`` can consume ``(fp8, per_token_scale)``.

    Quark W8A8-FP8 uses ``per_token`` + ``per_channel``; compressed-tensors
    uses ``strategy=channel``.
    """
    scheme = getattr(module, "scheme", None)
    if scheme is None or getattr(module, "weight_scale", None) is None:
        return False
    if getattr(scheme, "per_token", False) and getattr(
        scheme, "weight_qscheme", None
    ) in (None, "per_channel"):
        return True
    strategy = getattr(scheme, "strategy", None)
    if strategy is None:
        return False
    value = getattr(strategy, "value", strategy)
    return str(value).lower() == "channel"
