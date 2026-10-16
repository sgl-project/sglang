# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.quantization.marlin_utils import (
    USE_FP32_REDUCE_DEFAULT,
    marlin_make_workspace,
    marlin_permute_scales,
    should_use_atomic_add_reduce,
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


def _is_experimental_sm80_fp8_marlin(device: torch.device) -> bool:
    return (
        envs.SGLANG_DSV41_SM80_EXPERIMENTAL.get() == "1"
        and torch.version.cuda is not None
        and torch.version.hip is None
        and device.type == "cuda"
        and torch.cuda.get_device_capability(device) == (8, 0)
    )


# Opt-in SM80 single-token fast path; the kernel lives in
# sglang.kernels.ops.gemm.dsv4_m1_gemv. It reads the raw row-major FP8 weight
# and the plain (unfolded) bf16 group-32 scales, both stashed next to the Marlin
# repack at load time. Every gate below is default-off.
#
# The flag name and the K group are returned by functions rather than module
# constants because the CPU-side regression test execs these function bodies on
# their own (see test/registered/kernels/ops/gemm/test_fp8_marlin_group32_sm80.py).
def _dsv41_m1_gemv_flag() -> str:
    """Env var that opts into the SM80 M=1 GEMV fast path."""
    return "SGLANG_DSV41_SM80_M1_GEMV"


def _dsv41_m1_gemv_group() -> int:
    """K group width of the block-32 FP8 weight this kernel reads."""
    return 32


def _dsv41_m1_gemv_enabled(device: torch.device) -> bool:
    """Whether the opt-in SM80 M=1 GEMV fast path may run on this device.

    Deliberately cheap and sync-free: one env lookup plus the cached SM80 check
    shared with the experimental Marlin paths. With the flag unset every caller
    keeps the unchanged Marlin behaviour.
    """
    return (
        envs.SGLANG_DSV41_SM80_M1_GEMV.get() == "1"
        and _is_experimental_sm80_fp8_marlin(device)
    )


def _dsv41_m1_gemv_impl():
    """Lazily import the optional M=1 GEMV kernel module.

    Imported on first use only, so with the flag off the import graph and the
    CPU-side AST tests of this module never touch the Triton kernel.
    """
    impl = globals().get("_DSV41_M1_GEMV_IMPL")
    if impl is None:
        from sglang.kernels.ops.gemm import dsv4_m1_gemv

        impl = dsv4_m1_gemv
        globals()["_DSV41_M1_GEMV_IMPL"] = impl
    return impl


def _pad_fp8_marlin_k(weight: torch.Tensor, size_k_first: bool) -> torch.Tensor:
    """Copy raw E4M3 bytes and append zero K rows/columns, never dequantize."""
    k_axis = 0 if size_k_first else 1
    physical_shape = list(weight.shape)
    physical_shape[k_axis] = (physical_shape[k_axis] + 63) // 64 * 64
    padded_bytes = torch.zeros(physical_shape, dtype=torch.uint8, device=weight.device)
    if size_k_first:
        padded_bytes[: weight.shape[0], :].copy_(weight.view(torch.uint8))
    else:
        padded_bytes[:, : weight.shape[1]].copy_(weight.view(torch.uint8))
    return padded_bytes.view(weight.dtype)


def fp8_fused_exponent_bias_into_scales(scales):
    fp8_exponent = 4
    if scales.dtype == torch.half:
        target_exponent = 5
    elif scales.dtype == torch.bfloat16:
        target_exponent = 8
    # exponent_bias_fp16 = 2 ** 4 - 2 ** 3 = 8
    # exponent_bias_bf16 = 2 ** 7 - 2 ** 3 = 120
    exponent_bias = 2 ** (target_exponent - 1) - 2 ** (fp8_exponent - 1)
    s = torch.ones_like(scales) * 2
    s = s**exponent_bias
    return scales * s


def fake_apply_fp8_marlin_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    workspace: torch.Tensor,
    size_n: int,
    size_k: int,
    bias: Optional[torch.Tensor],
    use_fp32_reduce: bool = USE_FP32_REDUCE_DEFAULT,
) -> torch.Tensor:
    out_shape = input.shape[:-1] + (size_n,)
    fake_output = torch.empty(out_shape, dtype=input.dtype, device=input.device)
    return fake_output


def _dsv41_m1_gemv_forward(
    input: torch.Tensor,
    weight: torch.Tensor,
    size_n: int,
    size_k: int,
    bias: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Run the stashed-tensor M=1 GEMV, or return None to fall back to Marlin.

    ``weight`` is the Marlin-repacked parameter; the raw source tensors travel
    with it as attributes (see _dsv41_m1_attach_source). Every rejection here is
    a cheap shape/device check, so the caller only ever sees a full result or a
    fall-through.
    """
    stash_w = getattr(weight, "dsv41_m1_w", None)
    stash_s = getattr(weight, "dsv41_m1_s", None)
    if stash_w is None or stash_s is None:
        return None
    if stash_w.device != input.device or stash_s.device != input.device:
        return None
    group = _dsv41_m1_gemv_group()
    stash_n = int(stash_w.shape[0])
    stash_k = int(stash_w.shape[1])
    if stash_n != size_n or stash_k < size_k or stash_k % group:
        return None
    if tuple(stash_s.shape) != (stash_n, stash_k // group):
        return None
    if input.shape[-1] != size_k:
        return None
    impl = _dsv41_m1_gemv_impl()
    if not impl.dsv41_m1_gemv_eligible(size_n, stash_k):
        return None

    x = input.reshape(-1)
    if stash_k != size_k:
        # Only a late-padded K tail differs; those raw weights are FP8 zero, so
        # zero-padded activations reproduce the Marlin path exactly.
        x = torch.nn.functional.pad(x, (0, stash_k - size_k))
    output = impl.dsv41_m1_gemv(stash_w, stash_s, x)
    if bias is not None:
        # Same place as the Marlin wrapper: after the kernel's bf16 result.
        output.add_(bias)
    return output.reshape(input.shape[:-1] + (size_n,))


@register_custom_op(fake_impl=fake_apply_fp8_marlin_linear)
def apply_fp8_marlin_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    workspace: torch.Tensor,
    size_n: int,
    size_k: int,
    bias: Optional[torch.Tensor],
    use_fp32_reduce: bool = USE_FP32_REDUCE_DEFAULT,
) -> torch.Tensor:
    # For GPUs that lack FP8 hardware support, we can leverage the
    # Marlin kernel for fast weight-only FP8 quantization

    # Opt-in SM80 M=1 fast path over the stashed raw weight. One env lookup when
    # the flag is off, no sync and no .item() when it is on.
    if (
        input.dtype is torch.bfloat16
        and input.numel() == input.shape[-1]
        and _dsv41_m1_gemv_enabled(input.device)
    ):
        fast_output = _dsv41_m1_gemv_forward(input, weight, size_n, size_k, bias)
        if fast_output is not None:
            return fast_output

    reshaped_x = input.reshape(-1, input.shape[-1])
    out_shape = input.shape[:-1] + (size_n,)

    # Marlin packs 16 physical K elements per row. Only late-padded weights
    # differ from the public logical K; leave ordinary calls untouched.
    physical_k = weight.shape[0] * 16
    if physical_k != size_k:
        assert physical_k > size_k
        reshaped_x = torch.nn.functional.pad(
            reshaped_x, (0, physical_k - size_k)
        ).contiguous()
        size_k = physical_k

    use_atomic_add = should_use_atomic_add_reduce(
        m=reshaped_x.size(0), n=size_n, k=size_k, device=input.device, dtype=input.dtype
    )

    output = gptq_marlin_gemm(
        a=reshaped_x,
        c=None,
        b_q_weight=weight,
        b_scales=weight_scale,
        global_scale=None,
        b_zeros=None,
        g_idx=None,
        perm=None,
        workspace=workspace,
        b_q_type=scalar_types.float8_e4m3fn,
        size_m=reshaped_x.size(0),
        size_n=size_n,
        size_k=size_k,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=use_fp32_reduce,
    )

    if bias is not None:
        output.add_(bias)

    return output.reshape(out_shape)


def _dsv41_m1_stash_source(
    layer: torch.nn.Module,
    weight: torch.Tensor,
    scales: Optional[torch.Tensor],
    size_k_first: bool,
    part_size_k: int,
    part_size_n: int,
    logical_size_k: int,
    pad_k: bool,
) -> None:
    """Stash the raw row-major FP8 weight and its plain bf16 group-32 scales.

    ``weight`` is the very tensor Marlin is about to repack (already K-padded
    when ``pad_k``), and ``scales`` is the checkpoint-layout
    (ceil(N / block_n), ceil(K / block_k)) scale. This mirrors the Marlin scale
    expansion below but keeps the (N, K // group) orientation the M=1 GEMV
    reads and the UNFOLDED scales: Marlin's 2**120 exponent-bias fold is applied
    later, to its own copy only, and marlin_permute_scales is never used here.

    A one-time load-time cost, and a strict no-op unless the new flag is on. The
    stash is a snapshot: an in-place update of the raw weight or of the model
    scales after loading is not reflected here (Marlin keeps reading the live
    tensors), so reloading weights requires re-running this function.
    """
    if envs.SGLANG_DSV41_SM80_M1_GEMV.get() != "1":
        return
    if scales is None or not _is_experimental_sm80_fp8_marlin(weight.device):
        return
    weight_block_size = getattr(layer, "weight_block_size", None)
    if weight_block_size is None or len(weight_block_size) != 2:
        return
    group = _dsv41_m1_gemv_group()
    block_n, block_k = int(weight_block_size[0]), int(weight_block_size[1])
    assert block_k == group, (block_k, group)
    assert weight.dtype == torch.float8_e4m3fn, weight.dtype

    if size_k_first:
        # (K, N) weight and (K // group, ceil(N / block_n)) scales: transpose
        # both to the (N, K) orientation the kernel reads.
        weight = weight.T.contiguous()
        scales = scales.T.contiguous()
    assert tuple(weight.shape) == (part_size_n, part_size_k), (
        tuple(weight.shape),
        (part_size_n, part_size_k),
    )

    n_blocks = (part_size_n + block_n - 1) // block_n
    logical_groups = (logical_size_k + block_k - 1) // block_k
    assert tuple(scales.shape) == (n_blocks, logical_groups), (
        tuple(scales.shape),
        (n_blocks, logical_groups),
    )
    # Same bf16 conversion the Marlin path applies, before the 2**120 fold.
    s = scales.to(torch.bfloat16)
    if pad_k:
        # Newly appended K groups only ever pair with FP8 zero bytes, so the
        # neutral scale 1 is the same choice the Marlin padding above makes.
        padded = s.new_ones((s.shape[0], part_size_k // block_k))
        padded[:, : s.shape[1]].copy_(s)
        s = padded
    s = s.repeat_interleave(block_n, dim=0)[:part_size_n].contiguous()
    k_groups = part_size_k // block_k
    assert tuple(s.shape) == (part_size_n, k_groups), (
        tuple(s.shape),
        (part_size_n, k_groups),
    )
    if not _dsv41_m1_gemv_impl().dsv41_m1_gemv_eligible(part_size_n, part_size_k):
        return

    w_bytes = weight.contiguous().view(torch.uint8)
    assert tuple(w_bytes.shape) == (part_size_n, part_size_k), tuple(w_bytes.shape)
    layer.dsv41_m1_w = w_bytes
    layer.dsv41_m1_s = s
    logger.info(
        "SM80 M=1 GEMV source stashed: N=%d K=%d groups=%d",
        part_size_n,
        part_size_k,
        k_groups,
    )


def _dsv41_m1_attach_source(layer: torch.nn.Module) -> None:
    """Hand the stashed M=1 GEMV source tensors to the repacked weight.

    apply_fp8_marlin_linear only receives layer.weight, so the stash has to
    travel with that tensor. Absent stash means the fast path declines.
    """
    w_bytes = getattr(layer, "dsv41_m1_w", None)
    s = getattr(layer, "dsv41_m1_s", None)
    if w_bytes is None or s is None:
        return
    layer.weight.dsv41_m1_w = w_bytes
    layer.weight.dsv41_m1_s = s


def prepare_fp8_layer_for_marlin(
    layer: torch.nn.Module, size_k_first: bool = True
) -> None:
    logger.warning_once(
        "Your GPU does not have native support for FP8 computation but "
        "FP8 quantization is being used. Weight-only FP8 compression will "
        "be used leveraging the Marlin kernel. This may degrade "
        "performance for compute-heavy workloads."
    )

    part_size_n = layer.output_size_per_partition
    part_size_k = layer.input_size_per_partition
    weight_block_size = getattr(layer, "weight_block_size", None)

    if size_k_first:
        assert layer.weight.shape == (part_size_k, part_size_n)
    else:
        assert layer.weight.shape == (part_size_n, part_size_k)

    device = layer.weight.device
    logical_size_k = part_size_k
    pad_k = (
        weight_block_size is not None
        and tuple(weight_block_size) == (32, 32)
        and layer.orig_dtype == torch.bfloat16
        and part_size_k % 64 != 0
        and _is_experimental_sm80_fp8_marlin(device)
    )
    weight = layer.weight
    if pad_k:
        # Checkpoint allocation/loading stays logical. Pad only raw FP8 bytes
        # here, just before repacking; do not materialize a dense BF16 weight.
        part_size_k = (part_size_k + 63) // 64 * 64
        assert part_size_n % 128 == 0 or (
            part_size_k % 128 == 0 and part_size_n % 64 == 0
        ), "FP8 Marlin K-tail padding requires a legal N tile; N padding is unsupported"
        weight = _pad_fp8_marlin_k(weight, size_k_first)

    # Opt-in SM80 M=1 GEMV source tensors. Stashed here, before Marlin replaces
    # the weight and deletes weight_scale_inv, and from the same (possibly
    # K-padded) tensor Marlin repacks so both paths see identical bytes.
    _dsv41_m1_stash_source(
        layer,
        weight,
        getattr(layer, "weight_scale_inv", None),
        size_k_first,
        part_size_k,
        part_size_n,
        logical_size_k,
        pad_k,
    )

    # WORKSPACE
    layer.workspace = marlin_make_workspace(device)

    # WEIGHT
    # Repack weights to marlin format
    perm = torch.empty(0, dtype=torch.int, device=device)
    qweight = pack_fp8_to_int32(weight, size_k_first)
    if not size_k_first:
        qweight = qweight.T.contiguous()

    marlin_qweight = gptq_marlin_repack(
        b_q_weight=qweight,
        perm=perm,
        size_k=part_size_k,
        size_n=part_size_n,
        num_bits=8,
    )
    layer.weight = torch.nn.Parameter(marlin_qweight, requires_grad=False)
    _dsv41_m1_attach_source(layer)

    # WEIGHT SCALES
    # Permute scales
    if "weight_scale" in dir(layer):
        scales = layer.weight_scale.to(layer.orig_dtype)
    elif "weight_scale_inv" in dir(layer):
        scales = layer.weight_scale_inv.to(layer.orig_dtype)
        del layer.weight_scale_inv

    group_size = -1 if weight_block_size is None else weight_block_size[1]

    # marlin kernel only support channel-wise and group-wise quantization
    # we need to convert the scales
    if weight_block_size is None:
        if scales.nelement() == 1:
            # tensor-wise quantization -> channel-wise quantization
            # (1, 1) =>(repeat)=> (1, size_n)
            scales = scales.view(1, 1).repeat_interleave(part_size_n, 1)
        elif scales.nelement() > 1 and scales.nelement() != part_size_n:
            assert part_size_n % scales.nelement() == 0
            s_size = scales.nelement()
            # tensor-wise quantization (for gate-up proj)
            #     -> channel-wise quantization
            # (1, s_size) =>(repeat)=> (1, size_n)
            scales = scales.view(1, s_size)
            scales = scales.repeat_interleave(part_size_n // s_size, 1)
        else:
            # channel-wise quantization
            # (1, size_n)
            scales = scales.view(1, part_size_n)
    else:
        # block-wise quantization -> group-wise quantization
        # (size_k // block_size[1], ceil(size_n / block_size[0]))
        #  =>(repeat)=> (size_k // block_size[1], size_n)
        if not size_k_first:
            scales = scales.T.contiguous()
        block_n = weight_block_size[0]
        if pad_k:
            # Loaded scales cover ceil(logical K / 32) groups, not physical K.
            # A partially occupied last group keeps its checkpoint scale; only
            # newly appended groups get the neutral multiplicative scale 1.
            assert scales.shape == (
                (logical_size_k + group_size - 1) // group_size,
                (part_size_n + block_n - 1) // block_n,
            )
            padded_scales = scales.new_ones(
                (part_size_k // group_size, scales.shape[1])
            )
            padded_scales[: scales.shape[0]].copy_(scales)
            scales = padded_scales
        scales = scales.repeat_interleave(block_n, 1)
        # size_n may not divisible by block_size[0]
        scales = scales[:, :part_size_n]

    marlin_scales = marlin_permute_scales(
        s=scales, size_k=part_size_k, size_n=part_size_n, group_size=group_size
    )
    marlin_scales = fp8_fused_exponent_bias_into_scales(marlin_scales)
    layer.weight_scale = torch.nn.Parameter(marlin_scales, requires_grad=False)

    # The dense FP8 Marlin wrapper adds bias after the kernel returns, so the
    # bias must remain in logical output-channel order. Only scales need the
    # Marlin tile permutation.
    if hasattr(layer, "bias") and layer.bias is not None:
        assert layer.bias.shape == (part_size_n,)
        layer.bias = torch.nn.Parameter(layer.bias.detach(), requires_grad=False)


def pack_fp8_to_int32(
    fp8_tensor: torch.Tensor, size_k_first: bool = True
) -> torch.Tensor:
    """
    Repack FP8 weights to gptq format (packed int32 elements)
    """
    assert fp8_tensor.dtype == torch.float8_e4m3fn
    assert fp8_tensor.ndim == 2

    fp8_tensor = fp8_tensor.T if size_k_first else fp8_tensor
    fp8_tensor = fp8_tensor.contiguous()
    # fp8_tensor is contiguous and have shape (N, K) now
    # with `.view(torch.int32)`, it become (N, K // 4)
    int32_tensor = fp8_tensor.view(torch.int32)
    return int32_tensor.T.contiguous() if size_k_first else int32_tensor
