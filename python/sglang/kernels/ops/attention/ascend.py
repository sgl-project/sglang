from collections.abc import Sequence

import torch


def _packed_boundaries(
    cu_seqlens: torch.Tensor,
    cu_seqlens_host: Sequence[int] | None,
    total_tokens: int,
    name: str,
) -> tuple[int, ...]:
    if cu_seqlens is None:
        raise ValueError(f"{name} is required for NPU packed attention")
    if cu_seqlens.ndim != 1 or cu_seqlens.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError(f"{name} must be a 1D int32 or int64 tensor")
    if cu_seqlens_host is not None and len(cu_seqlens_host) != cu_seqlens.numel():
        raise ValueError(f"{name} and its host copy must have the same length")

    boundaries = tuple(
        int(value)
        for value in (
            cu_seqlens.tolist() if cu_seqlens_host is None else cu_seqlens_host
        )
    )
    if len(boundaries) < 2 or boundaries[0] != 0:
        raise ValueError(f"{name} must start with 0 and contain at least one sequence")
    if boundaries[-1] != total_tokens:
        raise ValueError(
            f"{name} must end at the packed token count {total_tokens}, "
            f"got {boundaries[-1]}"
        )
    if any(stop < start for start, stop in zip(boundaries[:-1], boundaries[1:])):
        raise ValueError(f"{name} must be non-decreasing")
    return boundaries


def ascend_fused_infer_attention_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    cu_seqlens_q_host: Sequence[int] | None = None,
    cu_seqlens_k_host: Sequence[int] | None = None,
    softmax_scale: float | None = None,
    return_softmax_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    tensors = {"q": q, "k": k, "v": v}
    invalid_layouts = [name for name, tensor in tensors.items() if tensor.ndim != 3]
    if invalid_layouts:
        raise ValueError(
            "NPU packed attention requires q, k, and v in [T, N, D] layout; "
            f"invalid tensors: {', '.join(invalid_layouts)}"
        )
    invalid_devices = [
        name
        for name, tensor in tensors.items()
        if tensor.device.type != "npu" or tensor.device != q.device
    ]
    if invalid_devices:
        raise ValueError(
            "NPU packed attention requires q, k, and v on the same NPU; "
            f"invalid tensors: {', '.join(invalid_devices)}"
        )
    if not (q.dtype == k.dtype == v.dtype):
        raise ValueError(
            "NPU packed attention requires q, k, and v with the same dtype"
        )
    if k.shape[:2] != v.shape[:2]:
        raise ValueError(
            "NPU packed attention requires matching K/V token and head counts"
        )
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("NPU packed attention requires matching Q/K head dimensions")

    q_boundaries = _packed_boundaries(
        cu_seqlens_q, cu_seqlens_q_host, q.shape[0], "cu_seqlens_q"
    )
    k_boundaries = _packed_boundaries(
        cu_seqlens_k, cu_seqlens_k_host, k.shape[0], "cu_seqlens_k"
    )
    if len(q_boundaries) != len(k_boundaries):
        raise ValueError("cu_seqlens_q and cu_seqlens_k must describe the same batch")

    q_nonempty = [
        stop > start for start, stop in zip(q_boundaries[:-1], q_boundaries[1:])
    ]
    k_nonempty = [
        stop > start for start, stop in zip(k_boundaries[:-1], k_boundaries[1:])
    ]
    if q_nonempty != k_nonempty:
        raise NotImplementedError(
            "NPU packed attention does not support a sequence that is empty only "
            "on the query or key/value side"
        )
    actual_seq_lengths = [
        stop
        for stop, nonempty in zip(q_boundaries[1:], q_nonempty)
        if nonempty
    ]
    actual_seq_lengths_kv = [
        stop
        for stop, nonempty in zip(k_boundaries[1:], k_nonempty)
        if nonempty
    ]
    if not actual_seq_lengths:
        output = torch.empty_like(q)
        if return_softmax_lse:
            lse = torch.empty(
                (q.shape[1], q.shape[0]), dtype=torch.float32, device=q.device
            )
            return output, lse
        return output

    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        if q.shape == k.shape == v.shape:
            q, k, v = torch.stack((q, k, v), dim=0).unbind(0)
        else:
            q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

    output, lse = torch.ops.npu.npu_fused_infer_attention_score(
        q,
        k,
        v,
        num_heads=q.shape[1],
        num_key_value_heads=k.shape[1],
        scale=q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale,
        input_layout="TND",
        actual_seq_lengths=actual_seq_lengths,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        softmax_lse_flag=return_softmax_lse,
    )
    if not return_softmax_lse:
        return output
    if lse.shape != (q.shape[0], q.shape[1], 1):
        raise RuntimeError(
            "Unexpected Ascend TND softmax LSE shape: "
            f"expected {(q.shape[0], q.shape[1], 1)}, got {tuple(lse.shape)}"
        )
    return output, lse.squeeze(-1).transpose(0, 1).contiguous()
