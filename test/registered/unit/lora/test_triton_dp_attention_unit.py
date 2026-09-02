"""CPU-only metadata tests for explicit DP-attention LoRA routing."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.srt.lora.backend.triton_backend import (
    TritonLoRABackend,
    gather_dp_attention_lora_batch_info,
)
from sglang.srt.lora.utils import LoRABatchInfo
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.runtime_context import LoRABatchLayout, get_forward
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _batch_info(weight_indices: list[int], seg_lens: list[int]) -> LoRABatchInfo:
    seg_lens_tensor = torch.tensor(seg_lens, dtype=torch.int32)
    seg_indptr = torch.zeros(len(seg_lens) + 1, dtype=torch.int32)
    seg_indptr[1:] = torch.cumsum(seg_lens_tensor, dim=0)
    return LoRABatchInfo(
        bs=len(weight_indices),
        use_cuda_graph=False,
        num_segments=len(weight_indices),
        seg_lens=seg_lens_tensor,
        seg_indptr=seg_indptr,
        max_len=max(seg_lens, default=0),
        weight_indices=torch.tensor(weight_indices, dtype=torch.int32),
        lora_ranks=torch.tensor([0, 8, 4], dtype=torch.int32),
        scalings=torch.tensor([0.0, 2.0, 1.0]),
        permutation=None,
        expected_tokens=sum(seg_lens),
    )


def _routes(batch_info: LoRABatchInfo) -> list[int]:
    routes = torch.repeat_interleave(
        batch_info.weight_indices[: batch_info.num_segments],
        batch_info.seg_lens[: batch_info.num_segments],
    )
    if batch_info.permutation is None:
        return routes.tolist()
    unsorted_routes = torch.empty_like(routes)
    unsorted_routes[batch_info.permutation.long()] = routes
    return unsorted_routes.tolist()


def test_layout_selects_routing_without_shape_inference():
    backend = TritonLoRABackend(3, torch.device("cpu"))
    backend.batch_info = _batch_info([1, 0], [1, 1])
    backend.global_batch_info = _batch_info([2, 1], [1, 1])
    backend.sgemm_batch_info = backend._build_sgemm_routing(backend.batch_info)
    backend.global_sgemm_batch_info = backend._build_sgemm_routing(
        backend.global_batch_info
    )

    assert _routes(backend._sgemm_info()) == [1, 0]
    with get_forward().scoped(lora_batch_layout=LoRABatchLayout.TP_GLOBAL):
        assert _routes(backend._sgemm_info()) == [2, 1]
    assert _routes(backend._sgemm_info()) == [1, 0]


def test_dp_cuda_graph_global_routing_does_not_require_logprob_metadata(
    monkeypatch: pytest.MonkeyPatch,
):
    local_batch_info = _batch_info([1, 2], [1, 1])
    local_batch_info.use_cuda_graph = True
    graph_batch_info = _batch_info([0, 0, 0, 0], [1, 1, 1, 1])
    graph_batch_info.use_cuda_graph = True
    monkeypatch.setattr(
        "sglang.srt.lora.backend.triton_backend.get_attention_dp_rank", lambda: 0
    )

    def gather(output, local, forward_batch):
        assert local.tolist() == [1, 2]
        output.copy_(torch.tensor([1, 2, 0, 0], dtype=torch.int32))

    monkeypatch.setattr(
        "sglang.srt.lora.backend.triton_backend.dp_gather_replicate", gather
    )
    _, global_batch_info, lm_head_batch_infos = gather_dp_attention_lora_batch_info(
        SimpleNamespace(global_num_tokens_cpu=[2, 2]),
        local_batch_info,
        graph_batch_info,
        True,
        0,
        None,
    )

    assert global_batch_info is graph_batch_info
    assert _routes(global_batch_info) == [1, 2, 0, 0]
    assert lm_head_batch_infos is None


def test_prepare_global_routing_gathers_pruned_lm_head_routes(
    monkeypatch: pytest.MonkeyPatch,
):
    backend = TritonLoRABackend(3, torch.device("cpu"))
    backend.batch_info = _batch_info([1, 2], [1, 1])
    backend.lm_head_batch_info = _batch_info([1, 2], [1, 1])
    backend.has_global_active_lora = True
    monkeypatch.setattr(
        "sglang.srt.lora.backend.triton_backend.get_attention_dp_rank", lambda: 0
    )

    gather_count = 0

    def gather(
        output: torch.Tensor,
        local: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> None:
        nonlocal gather_count
        if gather_count == 0:
            assert local.tolist() == [1, 2, 0]
            output.copy_(torch.tensor([1, 2, 0, 0], dtype=torch.int32))
        else:
            assert local.tolist() == [1, 2]
            assert forward_batch.dp_padding_mode is DpPaddingMode.SUM_LEN
            output.copy_(torch.tensor([1, 2, 0, 2, 2], dtype=torch.int32))
        gather_count += 1

    monkeypatch.setattr(
        "sglang.srt.lora.backend.triton_backend.dp_gather_replicate", gather
    )
    forward_batch = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=2,
        input_ids=torch.zeros(3, dtype=torch.int64),
        req_pool_indices=torch.zeros(2, dtype=torch.int64),
        seq_lens=torch.zeros(2, dtype=torch.int64),
        out_cache_loc=torch.zeros(3, dtype=torch.int64),
        seq_lens_sum=0,
        global_num_tokens_cpu=[3, 1],
        global_num_tokens_gpu=torch.tensor([3, 1]),
        global_num_tokens_for_logprob_cpu=[2, 3],
        global_num_tokens_for_logprob_gpu=torch.tensor([2, 3]),
        is_extend_in_batch=True,
    )

    backend.prepare_global_lora_batch(forward_batch)

    assert gather_count == 2
    assert backend.batch_info.num_segments == 3
    assert _routes(backend.batch_info) == [1, 2, 0]
    assert backend.lm_head_batch_info is not None
    assert _routes(backend.lm_head_batch_info) == [1, 2, 0, 2, 2]


def test_prepare_global_routing_gathers_unpadded_decode_lm_head_routes(
    monkeypatch: pytest.MonkeyPatch,
):
    backend = TritonLoRABackend(3, torch.device("cpu"))
    backend.batch_info = _batch_info([1, 2, 0], [1, 1, 1])
    backend.has_global_active_lora = True
    monkeypatch.setattr(
        "sglang.srt.lora.backend.triton_backend.get_attention_dp_rank", lambda: 0
    )

    gather_count = 0

    def gather(
        output: torch.Tensor,
        local: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> None:
        nonlocal gather_count
        if gather_count == 0:
            assert local.tolist() == [1, 2, 0]
            output.copy_(torch.tensor([1, 2, 0, 2, 0, 0], dtype=torch.int32))
        else:
            assert local.tolist() == [1, 2]
            assert forward_batch.dp_padding_mode is DpPaddingMode.SUM_LEN
            output.copy_(torch.tensor([1, 2, 2], dtype=torch.int32))
        gather_count += 1

    monkeypatch.setattr(
        "sglang.srt.lora.backend.triton_backend.dp_gather_replicate", gather
    )
    forward_batch = ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        batch_size=3,
        input_ids=torch.zeros(3, dtype=torch.int64),
        req_pool_indices=torch.zeros(3, dtype=torch.int64),
        seq_lens=torch.zeros(3, dtype=torch.int64),
        out_cache_loc=torch.zeros(3, dtype=torch.int64),
        seq_lens_sum=0,
        global_num_tokens_cpu=[3, 3],
        global_num_tokens_gpu=torch.tensor([3, 3]),
        global_num_tokens_for_logprob_cpu=[2, 1],
        global_num_tokens_for_logprob_gpu=torch.tensor([2, 1]),
        is_extend_in_batch=False,
    )

    backend.prepare_global_lora_batch(forward_batch)

    assert gather_count == 2
    assert backend.global_batch_info is not None
    assert _routes(backend.global_batch_info) == [1, 2, 0, 2, 0, 0]
    assert backend.lm_head_batch_info is not None
    assert _routes(backend.lm_head_batch_info) == [1, 2, 2]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
