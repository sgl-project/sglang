"""Routed experts of Inkling-2 on the packed tensors of a Quark export, in checkpoint
layout. A sibling of Inkling2Experts, not a subclass, so native code that takes uint8
weights for NVFP4 never sees the MXFP4 ones."""

import logging
from functools import partialmethod

import torch
from torch import nn

from sglang.kernels.ops.moe.inkling2_quark_gemm_hip import expert_gemm_quark
from sglang.srt.layers.quantization.inkling2_quark import (
    MXFP4_BLOCK_SIZE,
    UINT4_GROUP_SIZE,
)
from sglang.srt.layers.quantization.kvfp4_tensor import E2M1_VALUES
from sglang.srt.layers.utils.common import set_weight_attrs
from sglang.srt.models.inkling_common.inkling2_moe import Inkling2Experts, RoutedOutput
from sglang.srt.runtime_context import get_parallel, get_platform

logger = logging.getLogger(__name__)

# Rows of the identity src2dst for the decode epilogue, as on the native HIP path.
IDENTITY_ROWS = 1 << 17


def packed_parameter_specs(
    expert_format: str, num_experts: int, hidden: int, intermediate: int
) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """Name -> (shape, dtype) in checkpoint layout, ``num_experts`` local experts."""
    experts, h, i = num_experts, hidden, intermediate
    if expert_format == "mxfp4":
        if h % MXFP4_BLOCK_SIZE or i % MXFP4_BLOCK_SIZE:
            raise ValueError(
                "Quark MXFP4 experts need hidden and intermediate sizes divisible by "
                f"{MXFP4_BLOCK_SIZE}, got {h} and {i}"
            )
        return {
            "w13_weight": ((experts, 2 * i, h // 2), torch.uint8),
            "w13_scale": ((experts, 2 * i, h // MXFP4_BLOCK_SIZE), torch.uint8),
            "w2_weight": ((experts, h, i // 2), torch.uint8),
            "w2_scale": ((experts, h, i // MXFP4_BLOCK_SIZE), torch.uint8),
        }
    if expert_format != "uint4_awq":
        raise ValueError(f"Unknown Inkling2 Quark expert format: {expert_format}")
    if h % UINT4_GROUP_SIZE or i % UINT4_GROUP_SIZE:
        raise ValueError(
            "Quark UINT4 experts need hidden and intermediate sizes divisible by the "
            f"quantization group {UINT4_GROUP_SIZE}, got {h} and {i}"
        )
    group = UINT4_GROUP_SIZE
    # zero points keep their checkpoint names; load_weights renames only *_weight_scale
    return {
        "w13_weight": ((experts, h, 2 * i // 8), torch.int32),
        "w13_scale": ((experts, h // group, 2 * i), torch.float32),
        "w13_weight_zero_point": ((experts, h // group, 2 * i // 8), torch.int32),
        "w2_weight": ((experts, i, h // 8), torch.int32),
        "w2_scale": ((experts, i // group, h), torch.float32),
        "w2_weight_zero_point": ((experts, i // group, h // 8), torch.int32),
    }


def mxfp4_pair_table(device) -> torch.Tensor:
    """BF16 [256, 2]: packed byte -> its two E2M1 values, low (even) nibble first."""
    values = torch.tensor(E2M1_VALUES, dtype=torch.bfloat16, device=device)
    byte = torch.arange(256, device=device)
    return torch.stack((values[byte & 0xF], values[byte >> 4]), dim=-1)


class Inkling2QuarkExperts(nn.Module):
    def __init__(self, config, num_experts, checkpoint_prefix, quant_config):
        super().__init__()
        parallel = get_parallel()
        self.tp_size = parallel.moe_tp_size
        self.ep_rank, self.ep_size = parallel.moe_ep_rank, parallel.moe_ep_size
        if self.tp_size != 1:
            raise ValueError(
                "Quark MXFP4/UINT4 experts need expert parallelism: omit --ep-size "
                "(Inkling defaults EP to TP), pass --ep-size equal to --tp, or use "
                "--tp 1"
            )
        expert_format = quant_config.expert_format
        self.expert_format = expert_format
        self.num_global_experts = num_experts
        if num_experts % self.ep_size:
            raise ValueError("The expert count must be divisible by expert parallelism")
        self.num_local_experts = num_experts // self.ep_size
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self._specs = packed_parameter_specs(
            expert_format,
            self.num_local_experts,
            self.hidden_size,
            self.intermediate_size,
        )
        for name, (shape, dtype) in self._specs.items():
            param = nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
            self.register_parameter(name, param)
            set_weight_attrs(param, {"weight_loader": self._load_shard})
        self.gate_cap, self.up_cap = config.mlp_gate_softcap, config.mlp_up_softcap
        if expert_format == "mxfp4" and (self.gate_cap <= 0 or self.up_cap <= 0):
            # situ_and_mul divides by the caps; elsewhere a cap of 0 means no cap
            raise ValueError(
                "Quark MXFP4 experts need positive softcaps, got "
                f"mlp_gate_softcap={self.gate_cap} and mlp_up_softcap={self.up_cap}"
            )
        self.output_sizes = {"w13": 2 * self.intermediate_size, "w2": self.hidden_size}
        self._identity = None
        self.pair_table = None
        self._platform_checked = False

    # The native loader without a TP dimension: copies this rank's experts in place, as
    # CUDA graphs hold the parameter addresses.
    _load_shard = partialmethod(Inkling2Experts._load_shard, dim=None)

    def packed_bytes(self) -> int:
        return sum(param.numel() * param.element_size() for param in self.parameters())

    def finalize_load(self) -> None:
        """Checks the platform and that the parameters are still packed, then allocates
        the forward's buffers once, outside CUDA graph capture."""
        self._check_platform()
        for name, (shape, dtype) in self._specs.items():
            param = getattr(self, name)
            if param.dtype != dtype or tuple(param.shape) != shape:
                raise ValueError(
                    f"Quark experts must stay packed: {name} is {param.dtype} "
                    f"{tuple(param.shape)}, expected {dtype} {shape}"
                )
        if self._identity is None:
            self._identity = torch.arange(
                IDENTITY_ROWS, dtype=torch.int32, device=self.w13_weight.device
            )
        if self.expert_format == "mxfp4" and self.pair_table is None:
            self.pair_table = mxfp4_pair_table(self.w13_weight.device)

    def _check_platform(self) -> None:
        """Weights on a GPU need ROCm gfx950; CPU/meta weights (tests) are allowed."""
        if self._platform_checked or not self.w13_weight.is_cuda:
            return
        platform = get_platform()
        if not platform.is_gfx95:
            raise ValueError(
                "Quark MXFP4/UINT4 experts of Inkling-2 run on ROCm gfx950 only"
            )
        self._platform_checked = True

    @staticmethod
    def log_load_summary(model: nn.Module) -> None:
        experts = [m for m in model.modules() if isinstance(m, Inkling2QuarkExperts)]
        if not experts:
            return
        logger.info(
            "Inkling2 Quark experts: format=%s layers=%d packed=%.2f GiB kernel=%s",
            experts[0].expert_format,
            len(experts),
            sum(module.packed_bytes() for module in experts) / (1 << 30),
            expert_gemm_quark.__module__,
        )

    def finish_forward(self, sums, combine, out_dtype):
        """All-reduces the FP32 per-token sums over the EP group. Returns the combined
        output, or for combine=False a RoutedOutput for the decode epilogue."""
        if self.ep_size > 1:
            # In FP32; the native HIP path reduces BF16 partials at prefill.
            sums = get_parallel().moe_ep_group.all_reduce(sums)
        if combine:
            return sums.to(out_dtype)
        if self._identity is None:
            raise RuntimeError("Inkling2QuarkExperts.finalize_load() has not run")
        if sums.shape[0] > IDENTITY_ROWS:
            # a shorter identity would be sliced silently and corrupt the combine
            raise ValueError(
                f"{sums.shape[0]} rows exceed the {IDENTITY_ROWS} identity rows of "
                "the decode epilogue"
            )
        return RoutedOutput(sums, self._identity[: sums.shape[0]], 1)

    def uses_hip_path(self) -> bool:
        # The native HIP MoE infers NVFP4/MXFP8 from the weight dtype.
        return False

    def mega_local_input_rows(self, tokens: int) -> None:
        # No MegaMoE exchange: every rank needs every input row.
        return None

    def forward(self, x, weights, ids, combine=True):
        """``weights, ids`` were routed from the unquantized ``x`` by ``Inkling2MoE``."""
        self._check_platform()
        if self.expert_format == "mxfp4":
            from sglang.srt.models.inkling_common.inkling2_moe_quark_mxfp4 import (
                forward_experts,
            )
        else:
            from sglang.srt.models.inkling_common.inkling2_moe_quark_uint4 import (
                forward_experts,
            )
        return forward_experts(self, x, weights, ids, combine)
