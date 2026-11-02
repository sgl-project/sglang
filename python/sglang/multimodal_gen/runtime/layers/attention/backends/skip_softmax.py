# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import torch

from sglang.multimodal_gen.configs.sample.sampling_params import SkipSoftmaxParams
from sglang.multimodal_gen.runtime.managers.forward_context import (
    get_forward_context_or_none,
)

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.pipelines_core import Req

_WORKSPACE_BYTES = 128 * 1024 * 1024
_REQUEST_KEY = "_skip_softmax_params"
_workspaces: dict[tuple[int, int], torch.Tensor] = {}


@dataclass(frozen=True)
class SkipSoftmaxSequenceMetadata:
    seq_lens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_kv: torch.Tensor
    q_seq_lens_cpu: torch.Tensor
    kv_seq_lens_cpu: torch.Tensor


def set_request_skip_softmax_params(
    batch: "Req", params: SkipSoftmaxParams | None
) -> None:
    batch.extra[_REQUEST_KEY] = params


def get_request_skip_softmax_params() -> SkipSoftmaxParams | None:
    context = get_forward_context_or_none()
    if context is None:
        return None
    batch = context.forward_batch
    if batch is None:
        return None
    params = batch.extra.get(_REQUEST_KEY)
    assert params is None or isinstance(params, SkipSoftmaxParams)
    return params


def get_fixed_sequence_metadata(
    device: torch.device,
    batch_size: int,
    query_length: int,
    kv_length: int,
) -> SkipSoftmaxSequenceMetadata:
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return _cached_fixed_sequence_metadata(
        device_index, batch_size, query_length, kv_length
    )


@lru_cache(maxsize=32)
def _cached_fixed_sequence_metadata(
    device_index: int,
    batch_size: int,
    query_length: int,
    kv_length: int,
) -> SkipSoftmaxSequenceMetadata:
    device = torch.device("cuda", device_index)
    seq_lens = torch.full((batch_size,), kv_length, dtype=torch.int32, device=device)
    cu_seqlens_q = torch.arange(
        0,
        (batch_size + 1) * query_length,
        query_length,
        dtype=torch.int32,
        device=device,
    )
    cu_seqlens_kv = torch.arange(
        0,
        (batch_size + 1) * kv_length,
        kv_length,
        dtype=torch.int32,
        device=device,
    )
    return SkipSoftmaxSequenceMetadata(
        seq_lens=seq_lens,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        q_seq_lens_cpu=torch.full((batch_size,), query_length, dtype=torch.int32),
        kv_seq_lens_cpu=torch.full((batch_size,), kv_length, dtype=torch.int32),
    )


@lru_cache(maxsize=32)
def get_host_sequence_lengths(cu_seqlens: tuple[int, ...]) -> torch.Tensor:
    return torch.tensor(
        [stop - start for start, stop in zip(cu_seqlens[:-1], cu_seqlens[1:])],
        dtype=torch.int32,
    )


def run_skip_softmax(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    seq_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    softmax_scale: float,
    causal: bool,
    threshold_scale_factor: float | None,
    q_seq_lens_cpu: torch.Tensor | None = None,
    kv_seq_lens_cpu: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run FlashInfer TRTLLM attention, optionally with BLASST sparsity."""
    _validate_inputs(query, key, value)
    capability = torch.cuda.get_device_capability(query.device)
    workspace = _get_workspace(query.device)
    query, key, value = (tensor.contiguous() for tensor in (query, key, value))

    common_kwargs = dict(
        seq_lens=seq_lens,
        max_q_len=max_seqlen_q,
        max_kv_len=max_seqlen_kv,
        bmm1_scale=softmax_scale,
        bmm2_scale=1.0,
        batch_size=cu_seqlens_q.numel() - 1,
        cum_seq_lens_q=cu_seqlens_q,
        cum_seq_lens_kv=cu_seqlens_kv,
        skip_softmax_threshold_scale_factor=threshold_scale_factor,
    )
    if capability == (9, 0):
        from flashinfer.prefill import trtllm_fmha_v2_prefill

        return trtllm_fmha_v2_prefill(
            (query, torch.stack((key, value), dim=1)),
            "CONTIGUOUS_Q_KV",
            workspace_buffer=workspace,
            mask_mode="causal" if causal else "padding",
            **common_kwargs,
        )

    if capability in ((10, 0), (10, 3), (10, 7)):
        from flashinfer.prefill import trtllm_ragged_attention_deepseek

        output = torch.empty(
            (query.shape[0], query.shape[1], value.shape[2]),
            dtype=query.dtype,
            device=query.device,
        )
        return trtllm_ragged_attention_deepseek(
            query,
            key,
            value,
            workspace,
            o_sf_scale=-1.0,
            window_left=-1,
            enable_pdl=None,
            is_causal=causal,
            return_lse=False,
            out=output,
            q_seq_lens_cpu=q_seq_lens_cpu,
            kv_seq_lens_cpu=kv_seq_lens_cpu,
            **common_kwargs,
        )

    raise ValueError(
        "Skip Softmax requires Hopper SM90 or Blackwell SM100/SM103/SM107; "
        f"found SM{capability[0]}{capability[1]}."
    )


def _validate_inputs(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> None:
    if query.device.type != "cuda":
        raise ValueError("Skip Softmax requires a CUDA device.")
    if key.device != query.device or value.device != query.device:
        raise ValueError("Skip Softmax requires Q/K/V on the same CUDA device.")
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("Skip Softmax expects packed [tokens, heads, head_dim] Q/K/V.")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f"Skip Softmax requires FP16 or BF16 Q/K/V, got {query.dtype}."
        )
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError("Skip Softmax requires Q/K/V to have the same dtype.")
    if query.shape[-1] != key.shape[-1] or key.shape[-1] != value.shape[-1]:
        raise ValueError("Skip Softmax requires matching Q/K/V head dimensions.")
    if key.shape[:2] != value.shape[:2]:
        raise ValueError("Skip Softmax requires matching K/V token and head counts.")
    if query.shape[1] % key.shape[1] != 0:
        raise ValueError(
            "Skip Softmax requires the query head count to divide by KV heads."
        )
    if query.shape[-1] not in (128, 256):
        raise ValueError(
            f"Skip Softmax supports head dimensions 128 and 256; got {query.shape[-1]}."
        )


def _get_workspace(device: torch.device) -> torch.Tensor:
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    key = (device_index, torch.cuda.current_stream(device).cuda_stream)
    workspace = _workspaces.get(key)
    if workspace is None:
        workspace = torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
        _workspaces[key] = workspace
    return workspace
