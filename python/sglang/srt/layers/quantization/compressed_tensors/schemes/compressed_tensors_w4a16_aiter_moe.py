from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.moe.utils import get_moe_a2a_backend, get_moe_runner_backend
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16_moe import (
    CompressedTensorsWNA16MoE,
)
from sglang.srt.utils import get_bool_env_var, is_hip, set_weight_attrs

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )

logger = logging.getLogger(__name__)

__all__ = [
    "CompressedTensorsW4A16AiterMoE",
    "unpack_kimi_int4",
    "dequant_kimi_int4",
    "transcode_to_aiter_w4a16",
]

_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

# Bound the bf16 working set of the one-time load transcode (experts per chunk).
_TRANSCODE_CHUNK_ELEMS = 256 * 1024 * 1024


def unpack_kimi_int4(packed: torch.Tensor, num_bits: int) -> torch.Tensor:
    """Unpack Kimi's mxint4 pack_quantized weights to signed int8 values in [-8, 7].

    No dequant -- returns the raw int4 levels (int8 container). This is the part WE
    author; it is validated directly by the round-trip script.

    Args:
        packed: ``[E, N, K // pack_factor]`` int32, int4 packed along K (output-major).

    Returns:
        ``[E, N, K]`` int8 in ``[-8, 7]``.
    """
    pf = 32 // num_bits
    E, N, Kp = packed.shape
    K = Kp * pf
    shifts = torch.arange(pf, device=packed.device, dtype=torch.int32) * num_bits
    # nibble i of packed col j -> K position j*pf + i (matches unpack_from_int32).
    vals = ((packed.unsqueeze(-1) >> shifts) & 0xF).to(torch.int16) - 8
    return vals.reshape(E, N, K).to(torch.int8)


def dequant_kimi_int4(
    packed: torch.Tensor,
    group_scale: torch.Tensor,
    num_bits: int,
    group_size: int,
) -> torch.Tensor:
    """Reference bf16 dequant (== q4 * group_scale). Used only for validation."""
    E, N, K = packed.shape[0], packed.shape[1], packed.shape[2] * (32 // num_bits)
    q4 = unpack_kimi_int4(packed, num_bits).to(torch.bfloat16)
    w = q4.reshape(E, N, K // group_size, group_size) * group_scale.to(
        torch.bfloat16
    ).unsqueeze(-1)
    return w.reshape(E, N, K)


def transcode_to_aiter_w4a16(
    q4: torch.Tensor, group_scale: torch.Tensor, group_size: int = 32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack our int4 weights into aiter's a16wi4 layout -- **lossless** (no requant).

    The FlyDSL a16wi4 kernel decodes a signed ``[-8,7]`` nibble and dequantizes as
    ``q4 * scale`` (see ``aiter.fused_moe.torch_moe_stage1`` reference and the
    ``pack_int8_to_packed_int4`` "[-8,7]" contract), so we pack our existing int4
    levels and our own group scale directly -- no ``per_1x32_i4_quant`` rescale, no
    ``-8``-clamp, exact reproduction of the checkpoint's int4 weights.

    Args:
        q4: ``[E, N, K]`` int8 in ``[-8,7]`` (from ``unpack_kimi_int4``).
        group_scale: ``[E, N, K // group_size]`` bf16 group scales (output-major).

    Returns:
        ``(weight_i4x2 [E, N, K//2], scale_flat 1-D bf16)``; weight is 16x16-preshuffled
        and tagged ``is_shuffled=True``.
    """
    from aiter import dtypes
    from aiter.ops.shuffle import (
        pack_int8_to_packed_int4,
        shuffle_scale_for_int4,
        shuffle_weight,
    )

    E, N, K = q4.shape
    qt_shuf = shuffle_weight(q4.view(dtypes.i8), (16, 16))
    weight = pack_int8_to_packed_int4(qt_shuf).view(E, N, K // 2).view(dtypes.i4x2)
    weight.is_shuffled = True
    # Kernel/reference want scale as [E, K//gs, N]; our checkpoint stores [E, N, K//gs].
    scale_gn = group_scale.transpose(-1, -2).contiguous().to(torch.bfloat16)
    scale_flat = (
        shuffle_scale_for_int4(scale_gn, group_size=group_size).view(-1).contiguous()
    )
    return weight, scale_flat


class CompressedTensorsW4A16AiterMoE(CompressedTensorsWNA16MoE):
    """Native W4A16 int4 MoE on AMD via aiter's a16wi4 (per_1x32 + i4x2) kernel.

    Keeps weights int4 (memory + bandwidth win), activations bf16 (no activation
    quant). The checkpoint's mxint4 pack_quantized layout is transcoded **once at
    load** into aiter's preshuffled i4x2 format (see ``transcode_to_aiter_w4a16``);
    the forward then calls ``aiter.fused_moe`` with the already-prepared resident
    weights -- no per-call repacking.

    Gated by ``SGLANG_MOE_W4A16_AITER`` (and requires ``SGLANG_USE_AITER`` + the
    ``flydsl`` package for the FlyDSL int4 kernels).
    """

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # mxint4 output-major pack_quantized layout (mirrors CompressedTensorsMxInt4MoE).
        extra_weight_attrs.update({"quant_method": self.strategy})

        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.packed_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.packed_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.group_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_scale)
        set_weight_attrs(w13_scale, extra_weight_attrs)

        w2_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.group_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_scale)
        set_weight_attrs(w2_scale, extra_weight_attrs)

        for name in ("w13_weight_shape", "w2_weight_shape"):
            p = torch.nn.Parameter(torch.empty(num_experts, 2), requires_grad=False)
            layer.register_parameter(name, p)
            set_weight_attrs(p, extra_weight_attrs)

        layer.a13_scale = None
        layer.a2_scale = None

    def _transcode_layer(
        self, packed: torch.Tensor, group_scale: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One-time int4 -> aiter a16wi4 transcode for one weight, chunked over experts.

        ``shuffle_scale_for_int4`` is per-expert (dim 0 is outermost throughout), so the
        per-chunk flattened scales concatenate in expert order into the full flat scale.
        """
        from aiter import dtypes

        E, N, Kp = packed.shape
        K = Kp * self.packed_factor
        weight = torch.empty(
            (E, N, K // 2), dtype=torch.uint8, device=packed.device
        ).view(dtypes.i4x2)
        scale_chunks = []

        step = max(1, _TRANSCODE_CHUNK_ELEMS // max(1, N * K))
        for s in range(0, E, step):
            e = slice(s, min(s + step, E))
            q4 = unpack_kimi_int4(packed[e], self.num_bits)  # int8 [-8,7], no dequant
            w_i4x2, sc_flat = transcode_to_aiter_w4a16(
                q4, group_scale[e], self.group_size
            )
            weight[e] = w_i4x2
            scale_chunks.append(sc_flat)

        weight.is_shuffled = True
        return weight, torch.cat(scale_chunks)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "is_w4a16_aiter_converted", False):
            return
        assert (
            not self.actorder
        ), "W4A16->aiter does not support activation reordering (actorder)"
        assert _use_aiter, "CompressedTensorsW4A16AiterMoE requires SGLANG_USE_AITER on HIP"

        w13, w13_scale = self._transcode_layer(
            layer.w13_weight_packed.data, layer.w13_weight_scale.data
        )
        w2, w2_scale = self._transcode_layer(
            layer.w2_weight_packed.data, layer.w2_weight_scale.data
        )

        layer.w13_weight = torch.nn.Parameter(w13, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(w2, requires_grad=False)
        layer.w13_weight.is_shuffled = True
        layer.w2_weight.is_shuffled = True
        layer.w13_weight_scale = torch.nn.Parameter(w13_scale, requires_grad=False)
        layer.w2_weight_scale = torch.nn.Parameter(w2_scale, requires_grad=False)
        layer.w13_input_scale = None
        layer.w2_input_scale = None

        for name in (
            "w13_weight_packed",
            "w2_weight_packed",
            "w13_weight_shape",
            "w2_weight_shape",
        ):
            if hasattr(layer, name):
                delattr(layer, name)

        layer.is_w4a16_aiter_converted = True

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        self.runner = MoeRunner(MoeRunnerBackend.AITER, moe_runner_config)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> "CombineInput":
        from sglang.srt.layers.moe.moe_runner.aiter import (
            AiterMoeQuantInfo,
            AiterQuantType,
        )

        assert (
            self.moe_runner_config.activation == "silu"
        ), "Only SiLU activation is supported."
        assert not self.moe_runner_config.no_combine, "unsupported"

        quant_info = AiterMoeQuantInfo(
            w13_weight=layer.w13_weight,
            w2_weight=layer.w2_weight,
            quant_type=AiterQuantType.PER_1X32,
            w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a13_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
        )
        return self.runner.run(dispatch_output, quant_info)
