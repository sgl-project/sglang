from __future__ import annotations

import logging
from typing import Callable, Optional

import torch

from sglang.srt.layers.quantization.marlin_utils import (
    USE_FP32_REDUCE_DEFAULT,
    _sm70_marlin_v100_gemm_op,
    _sm70_marlin_v100_repack_ops,
    marlin_make_workspace,
    marlin_permute_bias,
    marlin_permute_scales,
    should_use_atomic_add_reduce,
    sm70_nvfp4_marlin_process_global_scale,
    sm70_nvfp4_marlin_process_scales,
)
from sglang.srt.layers.quantization.utils import get_scalar_types
from sglang.srt.utils import is_cuda
from sglang.srt.utils.custom_op import register_custom_op

_is_cuda = is_cuda()

if _is_cuda:
    from sglang.kernels.ops.gemm.gptq_marlin import gptq_marlin_gemm
    from sglang.kernels.ops.quantization.gptq_marlin_repack import gptq_marlin_repack

ScalarType, scalar_types = get_scalar_types()
logger = logging.getLogger(__name__)


def nvfp4_marlin_process_scales(marlin_scales: torch.Tensor) -> torch.Tensor:
    if not (marlin_scales >= 0).all():
        # NVFP4 ModelOpt scales are expected to be non-negative. Keep this as
        # a warning so unusual checkpoints can still load for diagnosis.
        logger.warning_once(
            "NVFP4 Marlin assumes non-negative scales, but negative scales "
            "were found. Accuracy may be degraded."
        )

    marlin_scales = marlin_scales.to(torch.half)
    marlin_scales = marlin_scales.view(-1, 4)[:, [0, 2, 1, 3]].view(
        marlin_scales.size(0), -1
    )
    marlin_scales = (marlin_scales * (2**7)).view(torch.int16) << 1
    marlin_scales = marlin_scales.view(torch.float8_e4m3fn)
    return marlin_scales[:, 1::2].contiguous()


def nvfp4_marlin_process_global_scale(global_scale: torch.Tensor) -> torch.Tensor:
    assert global_scale.dtype in [torch.half, torch.bfloat16]
    global_scale_shape = global_scale.shape
    fp4_exponent = 2
    if global_scale.dtype == torch.half:
        target_exponent = 5
    elif global_scale.dtype == torch.bfloat16:
        target_exponent = 8
    exponent_bias = 2 ** (target_exponent - 1) - 2 ** (fp4_exponent - 1)
    global_scale = global_scale * (2.0 ** (exponent_bias - 7))
    if global_scale_shape == torch.Size([]):
        global_scale = global_scale.reshape(1)
    return global_scale


def fake_apply_fp4_marlin_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor | None,
    workspace: torch.Tensor,
    size_n: int,
    size_k: int,
    bias: torch.Tensor | None = None,
    use_fp32_reduce: bool = USE_FP32_REDUCE_DEFAULT,
) -> torch.Tensor:
    del weight, weight_scale, weight_global_scale, workspace, size_k, bias
    out_shape = input.shape[:-1] + (size_n,)
    return input.new_empty(out_shape)


@register_custom_op(fake_impl=fake_apply_fp4_marlin_linear)
def apply_fp4_marlin_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor | None,
    workspace: torch.Tensor,
    size_n: int,
    size_k: int,
    bias: torch.Tensor | None = None,
    use_fp32_reduce: bool = USE_FP32_REDUCE_DEFAULT,
) -> torch.Tensor:
    if input.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("FP4 Marlin requires FP16 or BF16 activations.")

    reshaped_x = input.reshape(-1, input.shape[-1])
    out_shape = input.shape[:-1] + (size_n,)
    # Recover the physical Marlin tile dimensions from the repacked weight.
    # They can exceed the logical TP shard dimensions when preparation padded a
    # misaligned shard (e.g. N=928 -> 960 for TP=4).
    padded_size_k = weight.size(0) * 16
    padded_size_n = weight.size(1) * 8 // 16
    if padded_size_k != size_k:
        reshaped_x = torch.nn.functional.pad(reshaped_x, (0, padded_size_k - size_k))

    use_atomic_add = should_use_atomic_add_reduce(
        m=reshaped_x.size(0),
        n=padded_size_n,
        k=padded_size_k,
        device=input.device,
        dtype=input.dtype,
    )

    sm70_gemm = _sm70_marlin_v100_gemm_op()
    if sm70_gemm is not None:
        # M=1 Marlin on these tiles is launch-bound. Stream the same packed
        # weights with a GEMV when N is a multiple of 256 (the four-tile
        # interleave). Larger prefill batches stay on Marlin.
        if (
            reshaped_x.dtype == torch.float16
            and 1 <= reshaped_x.shape[0] <= 4
            and padded_size_n % 256 == 0
            and padded_size_k % 16 == 0
            and weight.dtype == torch.int32
            and (bias is None or bias.numel() == 0)
        ):
            from sglang.kernels.ops.gemm.sm70_glm_nvfp4_gemv import (
                sm70_glm_nvfp4_gemv,
                sm70_glm_nvfp4_gemv_available,
            )

            if sm70_glm_nvfp4_gemv_available():
                output = sm70_glm_nvfp4_gemv(
                    reshaped_x,
                    weight,
                    weight_scale,
                    weight_global_scale,
                )
                if output is not None:
                    if padded_size_n != size_n:
                        output = output[:, :size_n].contiguous()
                    return output.reshape(out_shape)

        # The in-tree Marlin GEMM is an empty stub below SM80. marlin_v100
        # consumes the logical scale layout prepared above and applies bias.
        output = sm70_gemm(
            reshaped_x,
            None,
            weight,
            bias,
            weight_scale,
            None,
            weight_global_scale,
            None,
            None,
            None,
            workspace,
            scalar_types.float4_e2m1f.id,
            reshaped_x.size(0),
            padded_size_n,
            padded_size_k,
            True,
            use_atomic_add,
            use_fp32_reduce,
            False,
        )
        if padded_size_n != size_n:
            output = output[:, :size_n].contiguous()
        return output.reshape(out_shape)

    output = gptq_marlin_gemm(
        a=reshaped_x,
        c=None,
        b_q_weight=weight,
        b_scales=weight_scale,
        global_scale=weight_global_scale,
        b_zeros=None,
        g_idx=None,
        perm=None,
        workspace=workspace,
        b_q_type=scalar_types.float4_e2m1f,
        size_m=reshaped_x.size(0),
        size_n=padded_size_n,
        size_k=padded_size_k,
        is_k_full=True,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=use_fp32_reduce,
    )

    # A narrowed N dimension has the padded row stride, so materialize it
    # before reshaping. This is only needed for a TP shard that was padded.
    if padded_size_n != size_n:
        output = output[:, :size_n].contiguous()

    # This GEMM takes no bias, so it is added here, to the output in logical
    # channel order: `bias` is the layer's unpermuted, unpadded bias.
    if bias is not None:
        output.add_(bias)
    return output.reshape(out_shape)


def prepare_nvfp4_layer_for_marlin(layer: torch.nn.Module) -> None:
    if getattr(layer, "quant_config", None) is not None:
        group_size = layer.quant_config.group_size
        if group_size != 16:
            raise ValueError(f"NVFP4 Marlin requires group_size=16, got {group_size}.")

    part_size_n = layer.output_size_per_partition
    part_size_k = layer.input_size_per_partition
    param_dtype = getattr(layer, "params_dtype", getattr(layer, "orig_dtype", None))
    if param_dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("NVFP4 Marlin requires FP16 or BF16 activation dtype.")

    assert layer.weight.shape == (part_size_n, part_size_k // 2)

    # Marlin accepts either N%64/K%128 or N%128/K%64. Select the smaller
    # padded shape, matching vLLM's marlin_padded_nk helper.
    padded_size_n, padded_size_k = min(
        (
            ((part_size_n + 63) // 64 * 64, (part_size_k + 127) // 128 * 128),
            ((part_size_n + 127) // 128 * 128, (part_size_k + 63) // 64 * 64),
        ),
        key=lambda nk: (nk[0] * nk[1], nk[0] + nk[1]),
    )

    if (padded_size_n, padded_size_k) != (part_size_n, part_size_k):
        pad_rows = padded_size_n - part_size_n
        pad_cols = (padded_size_k - part_size_k) // 2
        scale_pad_cols = (padded_size_k - part_size_k) // 16
        layer.weight = torch.nn.Parameter(
            torch.nn.functional.pad(layer.weight, (0, pad_cols, 0, pad_rows)),
            requires_grad=False,
        )
        layer.weight_scale = torch.nn.Parameter(
            torch.nn.functional.pad(
                layer.weight_scale, (0, scale_pad_cols, 0, pad_rows)
            ),
            requires_grad=False,
        )

    device = layer.weight.device
    layer.workspace = marlin_make_workspace(device)

    perm = torch.empty(0, dtype=torch.int, device=device)
    qweight = layer.weight.view(torch.int32).T.contiguous()
    sm70_repack, _ = _sm70_marlin_v100_repack_ops()
    if sm70_repack is not None:
        # The in-tree repack kernel returns without writing below SM80.
        marlin_qweight = sm70_repack(qweight, perm, padded_size_k, padded_size_n, 4)
    else:
        marlin_qweight = gptq_marlin_repack(
            b_q_weight=qweight,
            perm=perm,
            size_k=padded_size_k,
            size_n=padded_size_n,
            num_bits=4,
        )
    layer.weight = torch.nn.Parameter(marlin_qweight, requires_grad=False)

    if sm70_repack is not None:
        # Logical [K/16, N], then the S0E5M3 encoding the V100 iterator reads.
        logical_scales = layer.weight_scale.T.unsqueeze(0).contiguous()
        encoded, scale_factor = sm70_nvfp4_marlin_process_scales(
            logical_scales, param_dtype
        )
        layer.weight_scale = torch.nn.Parameter(
            encoded[0].contiguous(), requires_grad=False
        )
        global_scale = sm70_nvfp4_marlin_process_global_scale(
            layer.weight_global_scale, param_dtype
        )
        if scale_factor != 1.0:
            global_scale = global_scale / scale_factor
        layer.weight_global_scale = torch.nn.Parameter(
            global_scale.reshape(-1).contiguous(), requires_grad=False
        )
    else:
        weight_scale = layer.weight_scale.T.contiguous().to(param_dtype)
        weight_scale = marlin_permute_scales(
            s=weight_scale,
            size_k=padded_size_k,
            size_n=padded_size_n,
            group_size=16,
        )
        weight_scale = nvfp4_marlin_process_scales(weight_scale)
        layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)

        weight_global_scale = layer.weight_global_scale.to(param_dtype)
        weight_global_scale = nvfp4_marlin_process_global_scale(weight_global_scale)
        layer.weight_global_scale = torch.nn.Parameter(
            weight_global_scale, requires_grad=False
        )

    if sm70_repack is not None and getattr(layer, "bias", None) is not None:
        # marlin_v100 adds bias in-kernel over the padded N tile, so the SM70
        # path still pads it. The stock path leaves bias logical and unpadded:
        # apply_fp4_marlin_linear adds it after narrowing back to size_n.
        assert layer.bias.shape == (part_size_n,)
        bias = torch.nn.functional.pad(layer.bias, (0, padded_size_n - part_size_n))
        layer.bias = torch.nn.Parameter(bias, requires_grad=False)
    # layer.bias is left in logical order and length: apply_fp4_marlin_linear
    # adds it after the GEMM, and with skip_bias_add the linear layer returns it
    # to its caller. Only the MoE kernel, which applies bias in-kernel, needs it
    # permuted.

    if (
        sm70_repack is not None
        and padded_size_n % 256 == 0
        and padded_size_k % 16 == 0
        and not (
            hasattr(layer, "bias")
            and layer.bias is not None
            and layer.bias.numel() > 0
        )
    ):
        from sglang.kernels.ops.gemm.sm70_glm_nvfp4_gemv import (
            prepack_glm_nvfp4_weight,
            sm70_glm_nvfp4_gemv_available,
        )

        if sm70_glm_nvfp4_gemv_available():
            prepack_glm_nvfp4_weight(layer.weight)


def prepare_mxfp4_layer_for_marlin(layer: torch.nn.Module) -> None:
    group_size = 32
    if getattr(layer, "quant_config", None) is not None:
        configured = layer.quant_config.group_size
        if configured != group_size:
            raise ValueError(
                f"MXFP4 Marlin requires group_size={group_size}, got {configured}."
            )

    part_size_n = layer.output_size_per_partition
    part_size_k = layer.input_size_per_partition
    param_dtype = getattr(layer, "params_dtype", getattr(layer, "orig_dtype", None))
    # The E8M0 scale decoder is only instantiated for bf16; see marlin/dequant.h.
    if param_dtype != torch.bfloat16:
        raise RuntimeError("MXFP4 Marlin requires BF16 activation dtype.")

    assert layer.weight.shape == (part_size_n, part_size_k // 2)

    # Marlin accepts either N%64/K%128 or N%128/K%64. Select the smaller
    # padded shape, matching vLLM's marlin_padded_nk helper.
    padded_size_n, padded_size_k = min(
        (
            ((part_size_n + 63) // 64 * 64, (part_size_k + 127) // 128 * 128),
            ((part_size_n + 127) // 128 * 128, (part_size_k + 63) // 64 * 64),
        ),
        key=lambda nk: (nk[0] * nk[1], nk[0] + nk[1]),
    )

    if (padded_size_n, padded_size_k) != (part_size_n, part_size_k):
        pad_rows = padded_size_n - part_size_n
        pad_cols = (padded_size_k - part_size_k) // 2
        scale_pad_cols = (padded_size_k - part_size_k) // group_size
        layer.weight = torch.nn.Parameter(
            torch.nn.functional.pad(layer.weight, (0, pad_cols, 0, pad_rows)),
            requires_grad=False,
        )
        layer.weight_scale = torch.nn.Parameter(
            torch.nn.functional.pad(
                layer.weight_scale, (0, scale_pad_cols, 0, pad_rows)
            ),
            requires_grad=False,
        )

    device = layer.weight.device
    layer.workspace = marlin_make_workspace(device)

    perm = torch.empty(0, dtype=torch.int, device=device)
    qweight = layer.weight.view(torch.int32).T.contiguous()
    marlin_qweight = gptq_marlin_repack(
        b_q_weight=qweight,
        perm=perm,
        size_k=padded_size_k,
        size_n=padded_size_n,
        num_bits=4,
    )
    layer.weight = torch.nn.Parameter(marlin_qweight, requires_grad=False)

    # Scales are raw E8M0 exponent bytes: reinterpret rather than convert, then
    # permute and repack in the same order as prepare_moe_mxfp4_layer_for_marlin.
    weight_scale = _normalize_scale_tensor(layer.weight_scale, param_dtype)
    weight_scale = weight_scale.T.contiguous()
    weight_scale = marlin_permute_scales(
        s=weight_scale,
        size_k=padded_size_k,
        size_n=padded_size_n,
        group_size=group_size,
    )
    weight_scale = mxfp4_marlin_process_scales(weight_scale, input_dtype=param_dtype)
    layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
    # layer.bias stays in logical order and length; see
    # prepare_nvfp4_layer_for_marlin.


def mxfp4_marlin_process_scales(
    marlin_scales: torch.Tensor,
    input_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if input_dtype is None or input_dtype.itemsize == 2:
        marlin_scales = marlin_scales.view(-1, 4)[:, [0, 2, 1, 3]].view(
            marlin_scales.size(0), -1
        )
    marlin_scales = marlin_scales.to(torch.float8_e8m0fnu)
    if input_dtype == torch.float8_e4m3fn:
        marlin_scales = marlin_scales.view(torch.uint8)
        assert marlin_scales.max() <= 249
        # exponent_bias (fp4->fp8) = 2 ** 3 - 2 ** 1 = 6
        marlin_scales = marlin_scales + 6
        marlin_scales = marlin_scales.view(torch.float8_e8m0fnu)
    return marlin_scales


def _normalize_scale_tensor(
    scales: torch.Tensor, target_dtype: torch.dtype
) -> torch.Tensor:
    # The kernel consumes E8M0 exponents. Regardless of the placeholder dtype
    # the loader used, we want the *numerical* value 2**e in ``target_dtype``.
    # float32/bfloat16/float16 containers hold the numerical 2**e directly
    # (they were filled via a dtype-promoting copy from uint8/e8m0).
    # uint8/int8 containers hold the raw E8M0 byte and must be reinterpreted.
    if scales.dtype == torch.float8_e8m0fnu:
        return scales.to(target_dtype)
    if scales.dtype == torch.uint8:
        return scales.view(torch.float8_e8m0fnu).to(target_dtype)
    if scales.dtype == torch.int8:
        return scales.view(torch.uint8).view(torch.float8_e8m0fnu).to(target_dtype)
    if scales.dtype in (torch.float32, torch.bfloat16, torch.float16):
        return scales.to(target_dtype)
    raise TypeError(f"Unsupported MXFP4 scale dtype for Marlin: {scales.dtype}")


def _get_optional_param(layer: torch.nn.Module, *names: str) -> torch.Tensor | None:
    for name in names:
        value = getattr(layer, name, None)
        if value is not None:
            return value
    return None


def deinterleave_moe_mxfp4_w13_for_marlin(layer: torch.nn.Module) -> None:
    """Convert GPT-OSS interleaved w13 rows to Marlin's contiguous halves.

    GPT-OSS stores gate/up rows as [gate0, up0, gate1, up1, ...]. The Marlin
    fused activation consumes [all_gate_rows, all_up_rows].
    """

    w13 = layer.w13_weight.data
    w13_scale = _get_optional_param(layer, "w13_weight_scale", "w13_weight_scale_inv")
    w13_bias = _get_optional_param(layer, "w13_weight_bias", "w13_bias")

    if w13.shape[1] % 2 != 0:
        raise ValueError(f"Expected even w13 row dimension, got {w13.shape}.")

    e, n, k = w13.shape
    layer.w13_weight.data = (
        w13.view(e, n // 2, 2, k).permute(0, 2, 1, 3).contiguous().view(e, n, k)
    )

    if w13_scale is not None:
        scale = w13_scale.data
        if scale.shape[1] != n:
            raise ValueError(
                f"Expected w13 scale row dimension {n}, got {scale.shape}."
            )
        w13_scale.data = (
            scale.view(e, n // 2, 2, scale.shape[-1])
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(e, n, scale.shape[-1])
        )

    if w13_bias is not None:
        bias = w13_bias.data
        if bias.shape[1] != n:
            raise ValueError(f"Expected w13 bias row dimension {n}, got {bias.shape}.")
        w13_bias.data = bias.view(e, n // 2, 2).permute(0, 2, 1).contiguous().view(e, n)


def _repack_moe_fp4_weight_for_marlin(
    weight: torch.Tensor,
    *,
    num_experts: int,
    size_n: int,
    size_k: int,
    perm: torch.Tensor,
) -> torch.Tensor:
    assert weight.shape == (num_experts, size_n, size_k // 2)

    tensor_list = []
    for i in range(num_experts):
        qweight = weight[i].view(torch.int32).T.contiguous()
        marlin_qweight = gptq_marlin_repack(
            b_q_weight=qweight,
            perm=perm,
            size_k=size_k,
            size_n=size_n,
            num_bits=4,
        )
        tensor_list.append(marlin_qweight)
    return torch.stack(tensor_list)


def _permute_moe_fp4_scales_for_marlin(
    scales: torch.Tensor,
    *,
    num_experts: int,
    size_n: int,
    size_k: int,
    group_size: int,
    process_scales: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    tensor_list = []
    for i in range(num_experts):
        scale = scales[i].T.contiguous()
        marlin_scales = marlin_permute_scales(
            s=scale,
            size_k=size_k,
            size_n=size_n,
            group_size=group_size,
        )
        tensor_list.append(process_scales(marlin_scales))
    return torch.stack(tensor_list)


def _use_sm70_mxfp4_marlin_pack() -> bool:
    """True on SM70 when marlin_v100 can repack; loud fail if the stub would run."""
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability()[0] != 7:
        return False
    from sglang.srt.layers.quantization.marlin_utils import (
        _sm70_marlin_v100_available,
    )

    if not _sm70_marlin_v100_available():
        raise RuntimeError(
            "MXFP4 Marlin on SM70 requires marlin_v100 "
            "(scripts/setup_v100_marlin.sh). The stock JIT gptq_marlin_repack "
            "is a zero-output stub below SM80; refusing to pack zeros."
        )
    return True


def _repack_moe_mxfp4_weight_for_sm70_marlin(
    weight: torch.Tensor,
    *,
    num_experts: int,
    size_n: int,
    size_k: int,
    pack_device: torch.device,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """GPTQ-layout view + marlin_v100 SM70 repack, not the Hopper JIT stub.

    Stream one expert at a time. A full-E ``transpose.contiguous()`` of the
    packed weight is a second copy of the same buffer and OOMs when the
    device is already near capacity (e.g. after MXFP8 dense unpacks to FP16).
    ``weight`` should already be on CPU so the original device allocation is
    free. Pass preallocated ``out`` so the expert-stream scratch cannot
    fragment the block that the packed tensor is meant to reuse.
    """
    from sglang.srt.hardware_backend.gpu.quantization.gptq_kernels import (
        gptq_marlin_moe_repack,
    )

    assert weight.shape == (num_experts, size_n, size_k // 2), (
        f"MXFP4 packed weight must be [E, N, K/2], got {tuple(weight.shape)} "
        f"vs E={num_experts} N={size_n} K/2={size_k // 2}"
    )
    packed_shape = (num_experts, size_k // 16, size_n * 2)
    if out is not None:
        if tuple(out.shape) != packed_shape:
            raise RuntimeError(
                f"SM70 MXFP4 packed out shape {tuple(out.shape)} != {packed_shape}"
            )
        packed = out
    else:
        packed = torch.empty(packed_shape, dtype=torch.int32, device="cpu")
    weight_cpu = weight if weight.device.type == "cpu" else weight.detach().to("cpu")
    empty_perm = torch.empty((1, 0), dtype=torch.int32, device=pack_device)
    for e in range(num_experts):
        w_e = weight_cpu[e : e + 1].to(pack_device)
        gptq_layout = w_e.contiguous().view(torch.int32).transpose(1, 2).contiguous()
        del w_e
        packed_e = gptq_marlin_moe_repack(
            gptq_layout, empty_perm, size_k, size_n, 4
        )
        del gptq_layout
        packed[e].copy_(packed_e[0] if packed.is_cuda else packed_e[0].cpu())
        del packed_e
    del empty_perm
    if packed.shape[0] != num_experts:
        raise RuntimeError(
            "SM70 MXFP4 Marlin repack dropped experts: "
            f"in={num_experts} out={packed.shape[0]}. Refusing silent drop."
        )
    return packed


def _prepare_moe_mxfp4_layer_for_sm70_marlin(layer: torch.nn.Module) -> None:
    """MXFP4 e2m1 + UE8M0 g32 → marlin_v100 W4A16 (logical E8M0 scales).

    Checkpoint:
      w13/w2 uint8 ``[E, N, K/2]`` (low nibble even-K, high nibble odd-K)
      scales UE8M0 ``[E, N, K/32]`` (127 = 1.0 = 2**(e-127))
    Packed:
      qweight int32 ``[E, K/16, N * 2]`` (u4 packed macro-N)
      scales  e8m0  ``[E, K/32, N]`` raw bytes, no Hopper permute / pair-swap
    No global_scale. Activations FP16 only.
    """
    from sglang.srt.layers.quantization.marlin_utils import (
        DSV41_FLASH_MXFP4_GROUP_SIZE,
        sm70_mxfp4_logical_ue8m0_scales,
        sm70_mxfp4_refuse_nvfp4_metadata,
        sm70_mxfp4_ue8m0_to_uint8,
        sm70_mxfp4_validate_ue8m0_scales,
    )

    sm70_mxfp4_refuse_nvfp4_metadata(layer)

    group_size = DSV41_FLASH_MXFP4_GROUP_SIZE
    w13 = layer.w13_weight.data
    w2 = layer.w2_weight.data
    w13_scale = _get_optional_param(layer, "w13_weight_scale", "w13_weight_scale_inv")
    w2_scale = _get_optional_param(layer, "w2_weight_scale", "w2_weight_scale_inv")
    w13_bias = _get_optional_param(layer, "w13_weight_bias", "w13_bias")
    w2_bias = _get_optional_param(layer, "w2_weight_bias", "w2_bias")

    if w13_scale is None or w2_scale is None:
        raise ValueError("MXFP4 Marlin requires w13/w2 weight scales.")

    w13_scale_data = w13_scale.data if hasattr(w13_scale, "data") else w13_scale
    w2_scale_data = w2_scale.data if hasattr(w2_scale, "data") else w2_scale
    w13_bias_data = w13_bias.data if hasattr(w13_bias, "data") else w13_bias
    w2_bias_data = w2_bias.data if hasattr(w2_bias, "data") else w2_bias

    if w13_scale_data.dtype == torch.float8_e4m3fn or (
        w2_scale_data.dtype == torch.float8_e4m3fn
    ):
        raise RuntimeError(
            "MXFP4 experts are UE8M0 g32, not NVFP4 E4M3 g16. "
            "Refusing to approximately pack."
        )

    num_experts = w13.shape[0]
    intermediate_size = w13.shape[1] // 2
    hidden_size = w13.shape[2] * 2
    if hidden_size % 128 == 0:
        padded_intermediate_size = ((intermediate_size + 63) // 64) * 64
    else:
        if hidden_size % 64 != 0:
            raise ValueError(
                f"MXFP4 Marlin requires hidden_size to be divisible by 64, "
                f"got {hidden_size}."
            )
        padded_intermediate_size = ((intermediate_size + 127) // 128) * 128

    param_dtype = getattr(
        layer,
        "orig_dtype",
        w13_bias_data.dtype if w13_bias_data is not None else torch.float16,
    )
    if param_dtype != torch.float16:
        raise RuntimeError(
            "SM70 MXFP4 Marlin is W4A16 FP16 (V100 has no BF16 tensor cores). "
            f"got orig_dtype={param_dtype}."
        )

    src_was_cuda = bool(w13.is_cuda)
    pack_device = (
        w13.device
        if src_was_cuda
        else torch.device(f"cuda:{torch.cuda.current_device()}")
    )

    def _pad_w13(x: torch.Tensor) -> torch.Tensor:
        if padded_intermediate_size == intermediate_size:
            return x
        x = x.view(num_experts, 2, intermediate_size, x.shape[-1])
        x = torch.nn.functional.pad(
            x, (0, 0, 0, padded_intermediate_size - intermediate_size)
        )
        return x.reshape(num_experts, 2 * padded_intermediate_size, -1)

    def _pad_w2(x: torch.Tensor, packing: int) -> torch.Tensor:
        if padded_intermediate_size == intermediate_size:
            return x
        return torch.nn.functional.pad(
            x, (0, (padded_intermediate_size - intermediate_size) // packing)
        )

    w13_size_n, w13_size_k = padded_intermediate_size * 2, hidden_size
    w2_size_n, w2_size_k = hidden_size, padded_intermediate_size

    import gc

    def _pack_attr(attr: str, pad_fn, size_n: int, size_k: int) -> torch.Tensor:
        param = getattr(layer, attr)
        cpu = param.detach().to("cpu").contiguous()
        cpu = pad_fn(cpu)
        packed_shape = (num_experts, size_k // 16, size_n * 2)
        packed_nbytes = packed_shape[0] * packed_shape[1] * packed_shape[2] * 4
        packed_out = None
        if (
            src_was_cuda
            and param.is_cuda
            and param.untyped_storage().nbytes() == packed_nbytes
        ):
            # Same-sized rewrite: packed int32 overwrites the checkpoint
            # bytes in place. Source is the CPU copy.
            packed_out = param.data.contiguous().view(torch.int32).reshape(
                *packed_shape
            )
        else:
            delattr(layer, attr)
            del param
            gc.collect()
        if getattr(layer, "workspace", None) is None:
            layer.workspace = marlin_make_workspace(pack_device, 4)
        packed = _repack_moe_mxfp4_weight_for_sm70_marlin(
            cpu,
            num_experts=num_experts,
            size_n=size_n,
            size_k=size_k,
            pack_device=pack_device,
            out=packed_out,
        )
        del cpu
        gc.collect()
        setattr(layer, attr, torch.nn.Parameter(packed, requires_grad=False))
        return packed

    # Drop locals that alias GPU storage before packing so the hole is real.
    w13 = None
    w2_hold = w2
    w2 = None
    w13_marlin = _pack_attr("w13_weight", _pad_w13, w13_size_n, w13_size_k)
    del w2_hold
    w2_marlin = _pack_attr(
        "w2_weight", lambda t: _pad_w2(t, packing=2), w2_size_n, w2_size_k
    )

    w13_scale_cpu = w13_scale_data.detach().to("cpu").contiguous()
    w2_scale_cpu = w2_scale_data.detach().to("cpu").contiguous()
    w13_bias_cpu = (
        w13_bias_data.detach().to("cpu").contiguous()
        if w13_bias_data is not None
        else None
    )
    w2_bias_cpu = (
        w2_bias_data.detach().to("cpu").contiguous()
        if w2_bias_data is not None
        else None
    )
    w13_scale_u8 = _pad_w13(sm70_mxfp4_ue8m0_to_uint8(w13_scale_cpu))
    w2_scale_u8 = _pad_w2(
        sm70_mxfp4_ue8m0_to_uint8(w2_scale_cpu), packing=group_size
    )
    del w13_scale_cpu, w2_scale_cpu
    sm70_mxfp4_validate_ue8m0_scales(w13_scale_u8)
    sm70_mxfp4_validate_ue8m0_scales(w2_scale_u8)
    if w13_bias_cpu is not None:
        w13_bias_cpu = _pad_w13(w13_bias_cpu.unsqueeze(-1)).squeeze(-1)

    w13_scale_marlin = sm70_mxfp4_logical_ue8m0_scales(
        w13_scale_u8, size_k=w13_size_k, size_n=w13_size_n, group_size=group_size
    )
    w2_scale_marlin = sm70_mxfp4_logical_ue8m0_scales(
        w2_scale_u8, size_k=w2_size_k, size_n=w2_size_n, group_size=group_size
    )
    if src_was_cuda:
        w13_scale_marlin = w13_scale_marlin.to(pack_device)
        w2_scale_marlin = w2_scale_marlin.to(pack_device)

    if w13_marlin.shape[0] != num_experts or w2_marlin.shape[0] != num_experts:
        raise RuntimeError(
            "SM70 MXFP4 Marlin would drop experts "
            f"(in={num_experts}, w13={w13_marlin.shape[0]}, "
            f"w2={w2_marlin.shape[0]})."
        )

    layer.w13_weight_scale = torch.nn.Parameter(w13_scale_marlin, requires_grad=False)
    layer.w2_weight_scale = torch.nn.Parameter(w2_scale_marlin, requires_grad=False)

    # SM70 kernels consume logical N-contiguous bias, matching FP8 Marlin.
    if w13_bias_cpu is not None:
        layer.w13_weight_bias = torch.nn.Parameter(
            w13_bias_cpu.to(
                device=pack_device if src_was_cuda else "cpu", dtype=param_dtype
            ),
            requires_grad=False,
        )
    if w2_bias_cpu is not None:
        layer.w2_weight_bias = torch.nn.Parameter(
            w2_bias_cpu.to(
                device=pack_device if src_was_cuda else "cpu", dtype=param_dtype
            ),
            requires_grad=False,
        )

    for stale in ("w13_weight_scale_inv", "w2_weight_scale_inv"):
        if hasattr(layer, stale):
            delattr(layer, stale)


def prepare_moe_mxfp4_layer_for_marlin(layer: torch.nn.Module) -> None:
    if _use_sm70_mxfp4_marlin_pack():
        _prepare_moe_mxfp4_layer_for_sm70_marlin(layer)
        return

    group_size = 32
    w13 = layer.w13_weight.data
    w2 = layer.w2_weight.data
    w13_scale = _get_optional_param(layer, "w13_weight_scale", "w13_weight_scale_inv")
    w2_scale = _get_optional_param(layer, "w2_weight_scale", "w2_weight_scale_inv")
    w13_bias = _get_optional_param(layer, "w13_weight_bias", "w13_bias")
    w2_bias = _get_optional_param(layer, "w2_weight_bias", "w2_bias")

    if w13_scale is None or w2_scale is None:
        raise ValueError("MXFP4 Marlin requires w13/w2 weight scales.")

    w13_scale_data = w13_scale.data if hasattr(w13_scale, "data") else w13_scale
    w2_scale_data = w2_scale.data if hasattr(w2_scale, "data") else w2_scale
    w13_bias_data = w13_bias.data if hasattr(w13_bias, "data") else w13_bias
    w2_bias_data = w2_bias.data if hasattr(w2_bias, "data") else w2_bias

    # Gated experts fuse gate and up into w13; non-gated experts have only up.
    # Layers built without a runner config (the fused GPT-OSS paths) are gated.
    moe_runner_config = getattr(layer, "moe_runner_config", None)
    num_shards = (
        1 if moe_runner_config is not None and not moe_runner_config.is_gated else 2
    )

    num_experts = w13.shape[0]
    intermediate_size = w13.shape[1] // num_shards
    hidden_size = w13.shape[2] * 2
    if hidden_size % 128 == 0:
        padded_intermediate_size = ((intermediate_size + 63) // 64) * 64
    else:
        if hidden_size % 64 != 0:
            raise ValueError(
                f"MXFP4 Marlin requires hidden_size to be divisible by 64, "
                f"got {hidden_size}."
            )
        padded_intermediate_size = ((intermediate_size + 127) // 128) * 128
    param_dtype = getattr(
        layer,
        "orig_dtype",
        w13_bias_data.dtype if w13_bias_data is not None else torch.bfloat16,
    )

    device = w13.device
    layer.workspace = marlin_make_workspace(device, 4)
    perm = torch.empty(0, dtype=torch.int, device=device)

    def _pad_w13(x: torch.Tensor) -> torch.Tensor:
        if padded_intermediate_size == intermediate_size:
            return x
        x = x.view(num_experts, num_shards, intermediate_size, x.shape[-1])
        x = torch.nn.functional.pad(
            x, (0, 0, 0, padded_intermediate_size - intermediate_size)
        )
        return x.reshape(num_experts, num_shards * padded_intermediate_size, -1)

    def _pad_w2(x: torch.Tensor, packing: int) -> torch.Tensor:
        if padded_intermediate_size == intermediate_size:
            return x
        return torch.nn.functional.pad(
            x, (0, (padded_intermediate_size - intermediate_size) // packing)
        )

    w13 = _pad_w13(w13)
    w2 = _pad_w2(w2, packing=2)
    w13_scale_data = _pad_w13(_normalize_scale_tensor(w13_scale_data, param_dtype))
    w2_scale_data = _pad_w2(
        _normalize_scale_tensor(w2_scale_data, param_dtype),
        packing=group_size,
    )
    if w13_bias_data is not None:
        w13_bias_data = _pad_w13(w13_bias_data.unsqueeze(-1)).squeeze(-1)

    w13_size_n, w13_size_k = padded_intermediate_size * num_shards, hidden_size
    w2_size_n, w2_size_k = hidden_size, padded_intermediate_size

    def _process_scales(marlin_scales: torch.Tensor) -> torch.Tensor:
        return mxfp4_marlin_process_scales(marlin_scales, input_dtype=param_dtype)

    def _permute_bias(bias: torch.Tensor | None) -> torch.Tensor | None:
        if bias is None:
            return None
        tensor_list = []
        for i in range(num_experts):
            tensor_list.append(marlin_permute_bias(bias[i].to(param_dtype)))
        return torch.stack(tensor_list)

    w13_marlin = _repack_moe_fp4_weight_for_marlin(
        w13, num_experts=num_experts, size_n=w13_size_n, size_k=w13_size_k, perm=perm
    )
    w2_marlin = _repack_moe_fp4_weight_for_marlin(
        w2, num_experts=num_experts, size_n=w2_size_n, size_k=w2_size_k, perm=perm
    )
    w13_scale_marlin = _permute_moe_fp4_scales_for_marlin(
        w13_scale_data,
        num_experts=num_experts,
        size_n=w13_size_n,
        size_k=w13_size_k,
        group_size=group_size,
        process_scales=_process_scales,
    )
    w2_scale_marlin = _permute_moe_fp4_scales_for_marlin(
        w2_scale_data,
        num_experts=num_experts,
        size_n=w2_size_n,
        size_k=w2_size_k,
        group_size=group_size,
        process_scales=_process_scales,
    )

    layer.w13_weight = torch.nn.Parameter(w13_marlin, requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(w2_marlin, requires_grad=False)
    layer.w13_weight_scale = torch.nn.Parameter(w13_scale_marlin, requires_grad=False)
    layer.w2_weight_scale = torch.nn.Parameter(w2_scale_marlin, requires_grad=False)

    if w13_bias_data is not None:
        layer.w13_weight_bias = torch.nn.Parameter(
            _permute_bias(w13_bias_data), requires_grad=False
        )
    if w2_bias_data is not None:
        layer.w2_weight_bias = torch.nn.Parameter(
            _permute_bias(w2_bias_data), requires_grad=False
        )

    # Marlin uses the repacked scales; release the loader-format parameters.
    for stale in ("w13_weight_scale_inv", "w2_weight_scale_inv"):
        if hasattr(layer, stale):
            delattr(layer, stale)


def prepare_moe_nvfp4_layer_for_marlin(
    layer: torch.nn.Module, group_size: Optional[int] = None
) -> None:
    # compressed-tensors carries the group size on the per-layer scheme rather
    # than on layer.quant_config (which the MoE weight loader owns), so callers
    # outside ModelOpt pass it in.
    if group_size is None:
        group_size = layer.quant_config.group_size
    if group_size != 16:
        raise ValueError(f"NVFP4 Marlin MoE requires group_size=16, got {group_size}.")

    w13 = layer.w13_weight.data
    w2 = layer.w2_weight.data
    w13_scale = layer.w13_weight_scale.data
    w2_scale = layer.w2_weight_scale.data
    w13_global_scale = layer.w13_weight_scale_2.data
    w2_global_scale = layer.w2_weight_scale_2.data
    w13_bias = getattr(layer, "w13_bias", None)
    w2_bias = getattr(layer, "w2_bias", None)

    num_experts = w13.shape[0]
    num_shards = 2 if layer.moe_runner_config.is_gated else 1
    intermediate_size = layer.intermediate_size_per_partition
    hidden_size = w13.shape[2] * 2
    param_dtype = layer.params_dtype
    if param_dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("NVFP4 Marlin MoE requires FP16 or BF16 activations.")

    device = w13.device
    layer.workspace = marlin_make_workspace(device, 4)
    perm = torch.empty(0, dtype=torch.int, device=device)

    if not layer.moe_runner_config.is_gated:
        padded_intermediate_size = ((intermediate_size + 127) // 128) * 128
        intermediate_size_pad = padded_intermediate_size - intermediate_size
        if intermediate_size_pad:
            w13 = torch.nn.functional.pad(w13, (0, 0, 0, intermediate_size_pad))
            w13_scale = torch.nn.functional.pad(
                w13_scale, (0, 0, 0, intermediate_size_pad)
            )
            w2 = torch.nn.functional.pad(w2, (0, intermediate_size_pad // 2, 0, 0))
            w2_scale = torch.nn.functional.pad(
                w2_scale, (0, intermediate_size_pad // 16)
            )
            if w13_bias is not None:
                w13_bias = torch.nn.functional.pad(w13_bias, (0, intermediate_size_pad))
            intermediate_size = padded_intermediate_size

    w13_size_n, w13_size_k = intermediate_size * num_shards, hidden_size
    w2_size_n, w2_size_k = hidden_size, intermediate_size

    def _process_global_scale(global_scale: torch.Tensor) -> torch.Tensor:
        return nvfp4_marlin_process_global_scale(global_scale.to(param_dtype))

    def _permute_bias(bias: torch.Tensor | None) -> torch.Tensor | None:
        if bias is None:
            return None
        tensor_list = []
        for i in range(num_experts):
            tensor_list.append(marlin_permute_bias(bias[i].to(param_dtype)))
        return torch.stack(tensor_list)

    layer.w13_weight = torch.nn.Parameter(
        _repack_moe_fp4_weight_for_marlin(
            w13,
            num_experts=num_experts,
            size_n=w13_size_n,
            size_k=w13_size_k,
            perm=perm,
        ),
        requires_grad=False,
    )
    layer.w2_weight = torch.nn.Parameter(
        _repack_moe_fp4_weight_for_marlin(
            w2, num_experts=num_experts, size_n=w2_size_n, size_k=w2_size_k, perm=perm
        ),
        requires_grad=False,
    )
    layer.w13_weight_scale = torch.nn.Parameter(
        _permute_moe_fp4_scales_for_marlin(
            w13_scale.to(param_dtype),
            num_experts=num_experts,
            size_n=w13_size_n,
            size_k=w13_size_k,
            group_size=16,
            process_scales=nvfp4_marlin_process_scales,
        ),
        requires_grad=False,
    )
    layer.w2_weight_scale = torch.nn.Parameter(
        _permute_moe_fp4_scales_for_marlin(
            w2_scale.to(param_dtype),
            num_experts=num_experts,
            size_n=w2_size_n,
            size_k=w2_size_k,
            group_size=16,
            process_scales=nvfp4_marlin_process_scales,
        ),
        requires_grad=False,
    )
    layer.w13_weight_scale_2 = torch.nn.Parameter(
        _process_global_scale(w13_global_scale), requires_grad=False
    )
    layer.w2_weight_scale_2 = torch.nn.Parameter(
        _process_global_scale(w2_global_scale), requires_grad=False
    )

    if w13_bias is not None:
        layer.w13_bias = torch.nn.Parameter(
            _permute_bias(w13_bias), requires_grad=False
        )
    if w2_bias is not None:
        layer.w2_bias = torch.nn.Parameter(_permute_bias(w2_bias), requires_grad=False)
