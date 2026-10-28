# SPDX-License-Identifier: Apache-2.0
"""Step5 region-score entry; all state and output storage belong to callers."""

import torch

from sglang.kernels.ops.attention.step5 import options as step5_options


def can_use_step5_scoring_cute(device: torch.device) -> bool:
    return (
        # TODO(yhyang201): frozen SGLANG_OPT_STEP5_SCORING_CUTE; prune dead paths.
        step5_options.SCORING_CUTE
        and device.type == "cuda"
        and torch.version.hip is None
        and torch.cuda.get_device_capability(device)[0] == 10
    )


def _check(value, *, shape, dtype, device, name):
    if (
        value.layout != torch.strided
        or tuple(value.shape) != shape
        or value.dtype != dtype
        or value.device != device
        or not value.is_contiguous()
    ):
        raise ValueError(
            f"Step5 scoring {name} requires contiguous {shape}/{dtype} on {device}"
        )


def _assert_group32_ownership(row_to_req: torch.Tensor) -> None:
    groups = row_to_req.view(-1, 32)
    owner = groups[:, :1]
    all_padding = (groups == 0).all(dim=1)
    owned_or_padding = ((groups == owner) | (groups == 0)).all(dim=1)
    torch._assert_async(
        (all_padding | ((owner[:, 0] > 0) & owned_or_padding)).all(),
        "Step5 group-32 rows must share one request slot or be padding",
    )


def step5_paged_scores(
    q: torch.Tensor,
    weights: torch.Tensor,
    *,
    compressed_index_keys: torch.Tensor,
    req_to_token: torch.Tensor,
    row_to_req: torch.Tensor,
    positions: torch.Tensor,
    out: torch.Tensor,
    clear_future: bool = True,
    group_rows: int = 1,
    workspace=None,
    weight_scale: float | None = None,
) -> torch.Tensor:
    """Score completed historical regions for flattened ragged query rows.

    Q is BF16 or prequantized unscaled E4M3. FP32 weights are already scaled by the
    index head count (``weight_scale=1``). BF16 weights are the unscaled GEMM
    output and default to ``weight_scale=0.5`` (exact: ``4**-0.5``). An explicit
    scale overrides the dtype default. CompressedIndexKeys is
    the existing SW128 byte backing.
    Request slot zero is padding. Valid slots/positions and compressed_index_keys
    publication are the backend's responsibility. Each valid req_to_token entry
    at logical offset 8*r must be an aligned physical start of a complete region.
    Native 64-token pages retain their contiguous logical offsets. Groups of
    4 or 32 rows share one request owner, with optional padding rows.
    Output width must cover all positions//8, including Graph replay inputs.
    With ``clear_future=True``, invalid/future output columns are always -inf.
    The SM10x live-length selector may set it false because it never reads those
    columns. Inputs are immutable; output must not share storage with them.
    Warm up before Graph capture.
    """
    if q.device.type != "cuda" or torch.version.hip is not None:
        raise ValueError("Step5 scoring requires NVIDIA CUDA")
    if type(clear_future) is not bool:
        raise ValueError("Step5 scoring clear_future must be a bool")
    if q.ndim != 3 or q.shape[1] not in (2, 4) or q.shape[2] != 256:
        raise ValueError("Step5 scoring Q requires [tokens, 2 or 4, 256]")
    tokens, heads, _ = q.shape
    if group_rows not in (1, 4, 32) or (group_rows > 1 and tokens % group_rows):
        raise ValueError("Step5 scoring group_rows must be 1, 4, or 32 and divide rows")
    if q.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise ValueError("Step5 scoring Q must be BF16 or E4M3")
    _check(q, shape=tuple(q.shape), dtype=q.dtype, device=q.device, name="q")
    if weights.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("Step5 scoring weights must be BF16 or FP32")
    explicit_weight_scale = weight_scale is not None
    if weight_scale is None:
        weight_scale = 0.5 if weights.dtype == torch.bfloat16 else 1.0
    _check(
        weights,
        shape=(tokens, heads),
        dtype=weights.dtype,
        device=q.device,
        name="weights",
    )
    if (
        compressed_index_keys.ndim != 4
        or compressed_index_keys.shape[1:] != (8, 1, 256)
        or compressed_index_keys.shape[0] < 1
    ):
        raise ValueError(
            "Step5 compressed_index_keys requires SW128 [pages, 8, 1, 256]"
        )
    _check(
        compressed_index_keys,
        shape=tuple(compressed_index_keys.shape),
        dtype=torch.uint8,
        device=q.device,
        name="compressed_index_keys",
    )
    if req_to_token.ndim != 2 or req_to_token.shape[0] < 1:
        raise ValueError("Step5 request map requires [request slots, token capacity]")
    for name, value, shape in (
        ("req_to_token", req_to_token, tuple(req_to_token.shape)),
        ("row_to_req", row_to_req, (tokens,)),
        ("positions", positions, (tokens,)),
    ):
        if value.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"Step5 scoring {name} must be int32 or int64")
        _check(value, shape=shape, dtype=value.dtype, device=q.device, name=name)
    if group_rows == 32:
        _assert_group32_ownership(row_to_req)
    if (
        out.ndim != 2
        or not 1 <= out.shape[1] < 2**31
        or out.shape[1] > (req_to_token.shape[1] // 8 + 255) // 256 * 256
    ):
        raise ValueError(
            "Step5 score width must fit native request capacity and int32 indexing"
        )
    _check(
        out,
        shape=(tokens, out.shape[1]),
        dtype=torch.float32,
        device=q.device,
        name="out",
    )
    if out.numel():
        a, b = out.data_ptr(), out.data_ptr() + out.numel() * out.element_size()
        for value in (
            q,
            weights,
            compressed_index_keys,
            req_to_token,
            row_to_req,
            positions,
        ):
            lo = value.data_ptr()
            if max(a, lo) < min(b, lo + value.numel() * value.element_size()):
                raise ValueError("Step5 scoring output must not alias inputs")
    if tokens:
        if can_use_step5_scoring_cute(q.device):
            from sglang.kernels.ops.attention.step5.scoring_cute import launch

            launch(
                q,
                weights,
                compressed_index_keys,
                req_to_token,
                row_to_req,
                positions,
                out,
                clear_future=clear_future,
                group_rows=group_rows,
                weight_scale=weight_scale,
                workspace=workspace,
            )
            return out
        from sglang.kernels.ops.attention.step5.scoring_triton import launch

        if explicit_weight_scale:
            weights = weights.float() * weight_scale
            weight_scale = 1.0
        launch(
            q.to(torch.float8_e4m3fn),
            weights,
            compressed_index_keys,
            req_to_token,
            row_to_req,
            positions,
            out,
            clear_future=clear_future,
            group_rows=group_rows,
            weight_scale=weight_scale,
        )
    return out
