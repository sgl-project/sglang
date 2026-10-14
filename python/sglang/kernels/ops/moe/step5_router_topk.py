# SPDX-License-Identifier: Apache-2.0
"""Step5 routing with unbiased sigmoid weights and ordered FP32 normalization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sglang.kernels.fused_op import BaseFusedOp
from sglang.kernels.spec import CapabilityRequirement, FormatSignature, KernelBackend

if TYPE_CHECKING:
    import torch


def _storage_overlaps(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.device != right.device or not left.numel() or not right.numel():
        return False
    left_storage, right_storage = left.untyped_storage(), right.untyped_storage()
    left_start, right_start = left_storage.data_ptr(), right_storage.data_ptr()
    return max(left_start, right_start) < min(
        left_start + left_storage.nbytes(), right_start + right_storage.nbytes()
    )


def _prepare_outputs(
    *,
    logits: torch.Tensor,
    bias: torch.Tensor,
    out_weights: Optional[torch.Tensor],
    out_ids: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    import torch

    if (
        logits.layout != torch.strided
        or logits.device.type not in ("cpu", "cuda")
        or logits.dtype != torch.float32
        or logits.ndim != 2
        or logits.shape[1] != 352
        or logits.stride(1) != 1
        or (logits.shape[0] > 1 and logits.stride(0) < 352)
        or logits.requires_grad
    ):
        raise ValueError(
            "logits must be inference-only FP32 [tokens, 352] on CPU/CUDA, "
            "with contiguous columns and nonoverlapping rows"
        )
    if (
        bias.layout != torch.strided
        or bias.device != logits.device
        or bias.dtype != torch.float32
        or bias.shape != (352,)
        or bias.stride(0) != 1
        or bias.requires_grad
    ):
        raise ValueError(
            "bias must be inference-only contiguous FP32 [352] on logits.device"
        )
    if (out_weights is None) != (out_ids is None):
        raise ValueError(
            "out_weights and out_ids must both be supplied or both omitted"
        )
    shape = (logits.shape[0], 8)
    if out_weights is None:
        return (
            torch.empty(shape, dtype=torch.float32, device=logits.device),
            torch.empty(shape, dtype=torch.int32, device=logits.device),
        )
    assert out_ids is not None
    for out, dtype in ((out_weights, torch.float32), (out_ids, torch.int32)):
        if (
            out.layout != torch.strided
            or out.device != logits.device
            or out.dtype != dtype
            or out.shape != shape
            or out.stride(1) != 1
            or (out.shape[0] > 1 and out.stride(0) < 8)
            or out.requires_grad
        ):
            raise ValueError(
                "outputs must be inference-only FP32 weights/int32 IDs [tokens, 8], "
                "on logits.device with contiguous columns and nonoverlapping rows"
            )
        if _storage_overlaps(out, logits) or _storage_overlaps(out, bias):
            raise ValueError(
                "outputs must not share storage memory with logits or bias"
            )
    if _storage_overlaps(out_weights, out_ids):
        raise ValueError("weights and IDs must not share storage memory")
    return out_weights, out_ids


class Step5RouterTopKOp(BaseFusedOp):
    """FP32 sigmoid + correction bias, top-8, unbiased probabilities, then scale 3.

    The supported input geometry is FP32 logits [tokens, 352] and bias [352].
    Bias must be finite, verified by the loader before device execution; this
    operator validates metadata only and does not synchronize to inspect values.
    Logit NaNs produce positive-zero weights and ID 0 for the entire row;
    infinities follow sigmoid. Ties choose the lower logical expert ID. Outputs
    retain descending score order and use FP32 weights / int32 IDs [tokens, 8].

    Each selected weight is sigmoid of its original logit. The denominator adds those
    eight FP32 values in selection order, then adds 1e-20. Division precedes a
    separate multiplication by 3. Bias affects selection only; it is never subtracted from a rounded score.
    There is no tree sum, grouped routing, shared expert, or physical remapping.

    Supplied output buffers may have padded rows/nonzero offsets. They must use
    storage address ranges distinct from both inputs and each other, including
    external-memory wrappers. Both outputs must be supplied together. Empty
    token rows return without importing or launching the Triton leaf. After
    warmup, supplied buffers support CUDA Graph capture without scratch allocation
    or data-dependent host reads. Only logical output elements are written.

    Native torch is a cast-graph diagnostic, not a device sigmoid/division oracle;
    inherited torch.compile and the Triton candidate need independent source
    device qualification. This operator does not compute the BF16-to-FP32 GEMM.
    """

    op = "moe.step5_router_topk"
    priority = (KernelBackend.TRITON, KernelBackend.TORCH)
    capabilities = {KernelBackend.TRITON: frozenset({CapabilityRequirement.CUDA})}
    format_signature = FormatSignature(
        supported_dtypes=("float32",),
        description="FP32 logits/bias; FP32 weights and int32 IDs; padded rows allowed",
    )
    descriptions = {
        KernelBackend.TRITON: "Step5 top-8 with ordered FP32 unbiased probability normalization.",
        KernelBackend.TORCH: "CPU cast-graph diagnostic; not a device arithmetic oracle.",
    }

    def __init__(
        self,
        *,
        num_experts: int = 352,
        top_k: int = 8,
        routed_scaling_factor: float = 3.0,
    ) -> None:
        super().__init__()
        if type(num_experts) is not int or num_experts != 352:
            raise ValueError("Step5 routing supports num_experts=352 only")
        if type(top_k) is not int or top_k != 8:
            raise ValueError("Step5 routing supports top_k=8 only")
        if (
            isinstance(routed_scaling_factor, bool)
            or not isinstance(routed_scaling_factor, (int, float))
            or routed_scaling_factor != 3.0
        ):
            raise ValueError("Step5 routing supports routed_scaling_factor=3.0 only")

    def forward_native(
        self,
        logits: torch.Tensor,
        bias: torch.Tensor,
        *,
        num_token_non_padded: Optional[torch.Tensor] = None,
        out_weights: Optional[torch.Tensor] = None,
        out_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        import torch

        weights, ids = _prepare_outputs(
            logits=logits, bias=bias, out_weights=out_weights, out_ids=out_ids
        )
        if num_token_non_padded is not None:
            if (
                num_token_non_padded.device != logits.device
                or num_token_non_padded.dtype != torch.int32
                or num_token_non_padded.numel() != 1
            ):
                raise ValueError("padding count must be a same-device int32 scalar")
        if logits.shape[0] == 0:
            return weights, ids
        probabilities = torch.sigmoid(logits)
        scores = probabilities + bias
        total = torch.zeros(logits.shape[0], dtype=torch.float32, device=logits.device)
        selected, selected_ids = [], []
        for _ in range(8):
            score, expert_id = scores.max(dim=-1)
            probability = probabilities.gather(1, expert_id[:, None]).squeeze(1)
            selected.append(probability)
            selected_ids.append(expert_id)
            total = total + probability
            scores = scores.scatter(1, expert_id[:, None], -float("inf"))
        normalized = torch.stack(selected, dim=-1) / (total[:, None] + 1e-20)
        scaled = normalized * 3.0
        bad = torch.isnan(logits).any(dim=-1, keepdim=True)
        weights.copy_(scaled.masked_fill(bad, 0.0))
        ids.copy_(torch.stack(selected_ids, dim=-1).masked_fill(bad, 0))
        if num_token_non_padded is not None:
            rows = torch.arange(logits.shape[0], device=logits.device)
            ids.masked_fill_(rows[:, None] >= num_token_non_padded, -1)
        return weights, ids

    def forward_triton(
        self,
        logits: torch.Tensor,
        bias: torch.Tensor,
        *,
        num_token_non_padded: Optional[torch.Tensor] = None,
        out_weights: Optional[torch.Tensor] = None,
        out_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        import torch

        if logits.device.type != "cuda" or torch.version.hip is not None:
            raise ValueError("the Triton backend requires NVIDIA CUDA logits")
        weights, ids = _prepare_outputs(
            logits=logits, bias=bias, out_weights=out_weights, out_ids=out_ids
        )
        if num_token_non_padded is not None:
            if (
                num_token_non_padded.device != logits.device
                or num_token_non_padded.dtype != torch.int32
                or num_token_non_padded.numel() != 1
            ):
                raise ValueError("padding count must be a same-device int32 scalar")
        if logits.shape[0] == 0:
            return weights, ids
        from sglang.kernels.ops.moe.step5_router_topk_triton import _step5_router_topk

        _step5_router_topk(
            logits=logits,
            bias=bias,
            weights=weights,
            ids=ids,
            num_token_non_padded=num_token_non_padded,
        )
        return weights, ids
