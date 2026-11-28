"""Unit tests for long-context DFlash memory fixes:
1. FlashInferAttnBackend sizes ``cuda_graph_kv_indices`` per request (not per
   draft token) for linear DFlash verification and lazily allocates
   ``cuda_graph_custom_mask`` only when a wrapper enables ``use_custom_mask``.
2. FusedKVMaterializeHelper invalidates cached aux CUDA graphs before releasing
   previous workspace tensors, rounds capacity up to 256-row multiples instead
   of doubling or jumping straight to 8192, and leaves no partial state on OOM.
3. build_speculative_verify_target_probs reuses ``scaled_logits`` storage in
   the sparse top-k + top-p branch without mutating ``next_token_logits`` and
   handles non-contiguous inputs safely.
"""

import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.kernels.ops.speculative.fused_kv_materialize import FusedKVMaterializeHelper
from sglang.srt.layers.attention import flashinfer_backend as fi_mod
from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.srt.speculative.dflash_utils import build_speculative_verify_target_probs
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _make_minimal_flashinfer_backend(spec_algorithm, *, max_context_len=262144):
    backend = FlashInferAttnBackend.__new__(FlashInferAttnBackend)
    backend.req_to_token_pool = SimpleNamespace(row_width=max_context_len)
    # Mirrors FlashInferAttnBackend.__init__, which derives max_kv_len from the pool.
    backend.max_kv_len = max_context_len
    backend.spec_algorithm = spec_algorithm
    backend.num_wrappers = 1
    backend.dispatch_reason = None
    backend.use_sliding_window_kv_pool = False
    backend.skip_prefill = False
    backend.workspace_buffer = torch.empty(1, dtype=torch.uint8)
    backend.prefill_backend = "auto"
    backend.kv_indptr = [torch.zeros(16, dtype=torch.int32)]
    backend.kv_last_page_len = torch.zeros(16, dtype=torch.int32)
    return backend


def _make_minimal_fused_kv_helper(on_workspace_reset=None):
    helper = FusedKVMaterializeHelper.__new__(FusedKVMaterializeHelper)
    helper.n_layers = 2
    helper.layer_out_dim = 16
    helper.num_kv_heads = 2
    helper.head_dim = 4
    helper.device = torch.device("cpu")
    helper._on_workspace_reset = on_workspace_reset
    helper._workspace_capacity = 0
    helper._workspace_dtype = None
    helper._proj_workspace = None
    helper._k_workspace = None
    helper._v_workspace = None
    return helper


def test_flashinfer_cuda_graph_kv_indices_and_lazy_custom_mask():
    orig_zeros = torch.zeros

    def _cpu_zeros(*args, **kwargs):
        kwargs["device"] = "cpu"
        return orig_zeros(*args, **kwargs)

    max_bs = 4
    block_size = 16
    max_num_tokens = max_bs * block_size
    max_context_len = 1024

    with mock.patch.object(fi_mod.torch, "zeros", side_effect=_cpu_zeros):
        dflash_be = _make_minimal_flashinfer_backend(
            SpeculativeAlgorithm.DFLASH, max_context_len=max_context_len
        )
        dflash_be.init_cuda_graph_state(max_bs=max_bs, max_num_tokens=max_num_tokens)

        expected_dflash_len = max_bs * max_context_len + 2 * max_num_tokens
        assert dflash_be.cuda_graph_kv_indices[0].shape == (expected_dflash_len,)
        assert dflash_be.cuda_graph_custom_mask is None

        with mock.patch.object(
            fi_mod,
            "BatchPrefillWithPagedKVCacheWrapper",
            return_value=object(),
            create=True,
        ) as wrapper_cls:
            dflash_be._create_prefill_wrappers(bs=max_bs, use_custom_mask=False)
            assert dflash_be.cuda_graph_custom_mask is None
            assert "custom_mask_buf" not in wrapper_cls.call_args.kwargs

            dflash_be._create_prefill_wrappers(bs=max_bs, use_custom_mask=True)
            assert dflash_be.cuda_graph_custom_mask is not None
            assert dflash_be.cuda_graph_custom_mask.shape == (
                max_num_tokens * max_context_len,
            )
            assert (
                wrapper_cls.call_args.kwargs["custom_mask_buf"]
                is dflash_be.cuda_graph_custom_mask
            )

        eagle_be = _make_minimal_flashinfer_backend(
            SpeculativeAlgorithm.EAGLE, max_context_len=max_context_len
        )
        eagle_be.init_cuda_graph_state(max_bs=max_bs, max_num_tokens=max_num_tokens)
        assert eagle_be.cuda_graph_kv_indices[0].shape == (
            max_num_tokens * max_context_len,
        )


def test_fused_kv_materialize_workspace_rounds_256_invalidates_before_release_and_avoids_doubling():
    events = []

    def _on_reset():
        events.append(
            (
                "reset",
                helper._workspace_capacity,
                helper._proj_workspace is not None,
                helper._k_workspace is not None,
                helper._v_workspace is not None,
            )
        )

    helper = _make_minimal_fused_kv_helper(on_workspace_reset=_on_reset)

    fresh_helper = _make_minimal_fused_kv_helper(on_workspace_reset=_on_reset)
    fresh_helper._ensure_workspace(512, torch.float16)
    assert fresh_helper._workspace_capacity == 512
    assert events == []

    helper._ensure_workspace(64, torch.float16)
    assert helper._workspace_capacity == 256
    assert events == []

    orig_empty = torch.empty

    def _spy_empty(*args, **kwargs):
        if len(events) == 1:
            events.append(
                (
                    "alloc",
                    helper._workspace_capacity,
                    helper._proj_workspace is None,
                    helper._k_workspace is None,
                    helper._v_workspace is None,
                )
            )
        return orig_empty(*args, **kwargs)

    with mock.patch(
        "sglang.kernels.ops.speculative.fused_kv_materialize.torch.empty",
        side_effect=_spy_empty,
    ):
        helper._ensure_workspace(272, torch.float16)

    assert events == [
        ("reset", 256, True, True, True),
        ("alloc", 0, True, True, True),
    ]
    assert helper._workspace_capacity == 512

    helper._ensure_workspace(528, torch.float16)
    assert helper._workspace_capacity == 1024
    assert len(events) == 3

    proj_1024 = helper._proj_workspace
    helper._ensure_workspace(784, torch.float16)
    assert helper._workspace_capacity == 1024
    assert helper._proj_workspace is proj_1024
    assert len(events) == 3

    helper._ensure_workspace(8015, torch.float16)
    assert helper._workspace_capacity == 8192
    assert len(events) == 4

    old_proj = helper._proj_workspace
    helper._ensure_workspace(8192, torch.float16)
    assert helper._workspace_capacity == 8192
    assert helper._proj_workspace is old_proj
    assert len(events) == 4

    helper._ensure_workspace(8193, torch.float16)
    assert helper._workspace_capacity == 8448
    assert len(events) == 5


def test_fused_kv_materialize_workspace_oom_leaves_clean_state():
    resets = []
    helper = _make_minimal_fused_kv_helper(on_workspace_reset=lambda: resets.append(1))
    helper._ensure_workspace(64, torch.float16)
    assert helper._workspace_capacity == 256

    orig_empty = torch.empty
    call_count = 0

    def _oom_on_second_empty(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("simulated OOM")
        return orig_empty(*args, **kwargs)

    with (
        mock.patch(
            "sglang.kernels.ops.speculative.fused_kv_materialize.torch.empty",
            side_effect=_oom_on_second_empty,
        ),
        pytest.raises(RuntimeError, match="simulated OOM"),
    ):
        helper._ensure_workspace(512, torch.float16)

    assert resets == [1]
    assert helper._workspace_capacity == 0
    assert helper._workspace_dtype is None
    assert helper._proj_workspace is None
    assert helper._k_workspace is None
    assert helper._v_workspace is None


def test_dflash_worker_invalidate_aux_graphs_syncs_and_clears():
    worker = DFlashWorkerV2.__new__(DFlashWorkerV2)
    worker.device = torch.device("cpu")
    worker._aux_graphs = {((16, 64), (1, 16)): ("static", "graph")}
    worker._aux_pending = True
    worker._aux_done = object()

    order = []

    class _FakeStream:
        def synchronize(self):
            order.append("aux_sync")

    worker._aux_stream = _FakeStream()

    def _fake_join():
        order.append("join")
        worker._aux_pending = False

    worker._join_aux = _fake_join

    with mock.patch("torch.cuda.is_available", return_value=False):
        worker._invalidate_aux_graphs()
        assert order == ["join", "aux_sync"]
        assert worker._aux_graphs == {}

        # No-op once already empty.
        worker._invalidate_aux_graphs()
        assert order == ["join", "aux_sync"]


def test_build_speculative_verify_target_probs_reuses_scaled_logits_without_mutating_input():
    bs = 2
    draft_token_num = 4
    vocab_size = 32
    torch.manual_seed(0)
    logits = torch.randn((bs * draft_token_num, vocab_size), dtype=torch.float32)
    logits_ref = logits.clone()

    sampling_info = SimpleNamespace(
        temperatures=torch.tensor([[0.7], [1.2]], dtype=torch.float32),
        top_ks=torch.tensor([8, 8], dtype=torch.int32),
        top_ps=torch.tensor([0.9, 0.85], dtype=torch.float32),
        need_top_k_sampling=True,
        need_top_p_sampling=True,
        need_min_p_sampling=False,
        sampling_seed=123,
    )

    with mock.patch(
        "sglang.srt.speculative.dflash_utils._dflash_top_p_renorm_prob",
        side_effect=lambda probs, top_ps: probs,
    ):
        probs = build_speculative_verify_target_probs(
            next_token_logits=logits,
            sampling_info=sampling_info,
            bs=bs,
            draft_token_num=draft_token_num,
            max_top_k=8,
            uniform_top_k_value=8,
            use_sparse_topk=True,
        )

        # Non-contiguous logits must not fail on `.view(...)` when falling back.
        noncontig_logits = logits.t().contiguous().t()
        assert not noncontig_logits.is_contiguous()
        probs_noncontig = build_speculative_verify_target_probs(
            next_token_logits=noncontig_logits,
            sampling_info=sampling_info,
            bs=bs,
            draft_token_num=draft_token_num,
            max_top_k=8,
            uniform_top_k_value=8,
            use_sparse_topk=True,
        )

    assert probs.shape == (bs, draft_token_num, vocab_size)
    torch.testing.assert_close(logits, logits_ref)
    torch.testing.assert_close(
        probs.sum(dim=-1), torch.ones((bs, draft_token_num), dtype=torch.float32)
    )
    torch.testing.assert_close(probs_noncontig, probs)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
