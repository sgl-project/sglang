# SPDX-License-Identifier: Apache-2.0
"""CUDA entry for Step5 BF16 indexer normalization and partial NeoX RoPE."""

import math

import torch


def _check_tensor(value, *, shape, dtype, device, name, contiguous=True):
    if (
        value.layout != torch.strided
        or tuple(value.shape) != shape
        or value.dtype != dtype
        or value.device != device
        or (contiguous and not value.is_contiguous())
    ):
        raise ValueError(f"Step5 indexer {name} requires {shape}/{dtype} on {device}")


def _overlaps(a, b):
    # Check addressed intervals, allowing disjoint views of a caller workspace.
    if not a.numel() or not b.numel():
        return False

    def interval(t):
        end = sum((n - 1) * s for n, s in zip(t.shape, t.stride())) + 1
        return t.data_ptr(), t.data_ptr() + end * t.element_size()

    a0, a1 = interval(a)
    b0, b1 = interval(b)
    return max(a0, b0) < min(a1, b1)


def step5_indexer_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    k_bias: torch.Tensor,
    cache: torch.Tensor,
    positions: torch.Tensor,
    q_out: torch.Tensor | None,
    k_out: torch.Tensor,
    eps: float,
    zero_centered: bool,
    fp8_saturate: bool = False,
    use_pdl: bool = False,
    packed_workspace=None,
    scoring_weights: torch.Tensor | None = None,
    row_to_req: torch.Tensor | None = None,
    weight_scale: float = 1.0,
    packed_q_e4m3: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Normalize Q/K then rotate channels 0..31; outputs are caller-owned.

    Q[T,H,256] and K[T,1,256] may have padded row strides; other tensors must
    be contiguous. H is 2 or 4. Parameters/cache are BF16, statistics are FP32,
    normalized Q/K round to BF16 before FP32 rotation and BF16 output.
    Positions[T] must lie in [0, cache.shape[0]); the serving owner validates
    this bound. Inputs must be finite and output storage disjoint from inputs.
    fp8_saturate selects the CUDA Torch cast policy established by the caller.
    With packed_workspace, q_out must be None: Q/W write directly into the
    scoring ABI and the first return value is None. packed_q_e4m3 preserves
    the selected original cast policy. Metadata must be prepared separately
    for this forward; this function does not own it.
    Warm up before Graph capture. No allocation or host tensor-value reads.
    """
    if q.device.type != "cuda" or torch.version.hip is not None:
        raise ValueError("Step5 indexer requires NVIDIA CUDA")
    if q.ndim != 3 or q.shape[1] not in (2, 4) or q.shape[2] != 256:
        raise ValueError("Step5 index Q requires [tokens, 2 or 4, 256]")
    if not math.isfinite(eps) or eps <= 0 or not isinstance(zero_centered, bool):
        raise ValueError(
            "Step5 indexer requires positive finite eps and bool zero_centered"
        )
    tokens, heads, _ = q.shape
    for name, value, shape in (
        ("q", q, (tokens, heads, 256)),
        ("k", k, (tokens, 1, 256)),
    ):
        _check_tensor(
            value,
            shape=shape,
            dtype=torch.bfloat16,
            device=q.device,
            name=name,
            contiguous=False,
        )
        if (
            value.stride(2) != 1
            or value.stride(1) != 256
            or value.stride(0) < shape[1] * 256
        ):
            raise ValueError(
                "Step5 indexer requires contiguous heads and nonoverlapping rows"
            )
    for name, value in (
        ("q_weight", q_weight),
        ("k_weight", k_weight),
        ("k_bias", k_bias),
    ):
        _check_tensor(
            value, shape=(256,), dtype=torch.bfloat16, device=q.device, name=name
        )
    if cache.ndim != 2 or cache.shape[1] != 32 or cache.shape[0] < 1:
        raise ValueError("Step5 indexer cache requires [positions, 32]")
    _check_tensor(
        cache,
        shape=tuple(cache.shape),
        dtype=torch.bfloat16,
        device=q.device,
        name="cache",
    )
    if positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("Step5 indexer positions must be int32 or int64")
    _check_tensor(
        positions,
        shape=(tokens,),
        dtype=positions.dtype,
        device=q.device,
        name="positions",
    )
    packed = packed_workspace is not None
    if packed:
        if q_out is not None:
            raise ValueError(
                "Packed norm writes workspace Q directly; q_out must be None"
            )
        if not math.isfinite(weight_scale):
            raise ValueError("Packed norm weight_scale must be finite")
        w = packed_workspace
        if w.rows_per_group not in (1, 4):
            raise ValueError("Packed norm supports groups of 1 or 4")
        groups = (tokens + w.rows_per_group - 1) // w.rows_per_group
        for name, value, shape, dtype in (
            ("packed_q", w.q_runtime, (groups * 16, 256), torch.float8_e4m3fn),
            ("packed_w", w.kernel_weights, (groups, 32, 4), torch.float32),
        ):
            _check_tensor(value, shape=shape, dtype=dtype, device=q.device, name=name)
        if scoring_weights is None or scoring_weights.dtype not in (
            torch.bfloat16,
            torch.float32,
        ):
            raise ValueError("Packed norm requires BF16/FP32 scoring weights")
        _check_tensor(
            scoring_weights,
            shape=(tokens, heads),
            dtype=scoring_weights.dtype,
            device=q.device,
            name="scoring_weights",
        )
        if row_to_req is None or row_to_req.dtype not in (torch.int32, torch.int64):
            raise ValueError("Packed norm requires integer request slots")
        _check_tensor(
            row_to_req,
            shape=(tokens,),
            dtype=row_to_req.dtype,
            device=q.device,
            name="row_to_req",
        )
        outputs = (w.q_runtime, w.kernel_weights, k_out)
        inputs = (
            q,
            k,
            q_weight,
            k_weight,
            k_bias,
            cache,
            positions,
            scoring_weights,
            row_to_req,
        )
        if any(_overlaps(a, b) for a in outputs for b in inputs) or any(
            _overlaps(a, b) for i, a in enumerate(outputs) for b in outputs[i + 1 :]
        ):
            raise ValueError("Packed norm outputs must not alias inputs or each other")
    elif q_out is None:
        raise ValueError("Unpacked norm requires q_out")
    q_e4m3 = packed_q_e4m3 if packed else q_out.dtype == torch.float8_e4m3fn
    q_out_dtype = torch.float8_e4m3fn if q_e4m3 else torch.bfloat16
    for name, out, shape, dtype in (
        ("q_out", q_out, tuple(q.shape), q_out_dtype),
        ("k_out", k_out, tuple(k.shape), torch.bfloat16),
    ):
        if packed and name == "q_out":
            continue
        _check_tensor(out, shape=shape, dtype=dtype, device=q.device, name=name)
        if any(
            _overlaps(out, x)
            for x in (q, k, q_weight, k_weight, k_bias, cache, positions)
        ):
            raise ValueError("Step5 indexer outputs must not alias inputs")
    if not packed and _overlaps(q_out, k_out):
        raise ValueError("Step5 indexer outputs must not alias each other")
    if tokens:
        from sglang.kernels.ops.attention.step5.indexer_norm_triton import launch

        launch(
            q,
            k,
            q_weight,
            k_weight,
            k_bias,
            cache,
            positions,
            q_out,
            k_out,
            eps=eps,
            zero_centered=zero_centered,
            q_e4m3=q_e4m3,
            fp8_saturate=fp8_saturate,
            use_pdl=use_pdl,
            packed_workspace=packed_workspace,
            weights=scoring_weights,
            requests=row_to_req,
            weight_scale=weight_scale,
        )
    return q_out, k_out
