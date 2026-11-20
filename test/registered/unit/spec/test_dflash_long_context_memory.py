"""Unit tests for long-context DFlash memory fixes:
1. FlashInferAttnBackend sizes ``cuda_graph_kv_indices`` per request (not per
   draft token) for linear DFlash verification and lazily allocates
   ``cuda_graph_custom_mask`` only when a wrapper enables ``use_custom_mask``.
2. FusedKVMaterializeHelper releases previous workspace tensors before growing
   and avoids doubling past the 8192 chunked-prefill ceiling.
3. build_speculative_verify_target_probs reuses ``scaled_logits`` storage in
   the sparse top-k + top-p branch without mutating ``next_token_logits``.
"""

from types import SimpleNamespace
from unittest import mock

import torch

from sglang.kernels.ops.speculative.fused_kv_materialize import FusedKVMaterializeHelper
from sglang.srt.layers.attention import flashinfer_backend as fi_mod
from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.srt.speculative.dflash_utils import build_speculative_verify_target_probs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _make_minimal_flashinfer_backend(spec_algorithm, *, max_context_len=262144):
    backend = FlashInferAttnBackend.__new__(FlashInferAttnBackend)
    backend.max_context_len = max_context_len
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


def test_fused_kv_materialize_workspace_releases_before_grow_and_avoids_doubling():
    helper = FusedKVMaterializeHelper.__new__(FusedKVMaterializeHelper)
    helper.n_layers = 2
    helper.layer_out_dim = 16
    helper.num_kv_heads = 2
    helper.head_dim = 4
    helper.device = torch.device("cpu")
    helper._workspace_capacity = 0
    helper._workspace_dtype = None
    helper._proj_workspace = None
    helper._k_workspace = None
    helper._v_workspace = None

    helper._ensure_workspace(64, torch.float16)
    assert helper._workspace_capacity == 256

    orig_empty = torch.empty
    seen_released = []

    def _spy_empty(*args, **kwargs):
        if len(seen_released) == 0:
            seen_released.append(
                (
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
        helper._ensure_workspace(8015, torch.float16)

    assert seen_released == [(True, True, True)]
    assert helper._workspace_capacity == 8192

    # Growing from 8015 to 8192 stays at 8192 instead of doubling to 16030.
    old_proj = helper._proj_workspace
    helper._ensure_workspace(8192, torch.float16)
    assert helper._workspace_capacity == 8192
    assert helper._proj_workspace is old_proj

    # Above 8192 rounds up to the next 256-token multiple rather than doubling.
    helper._ensure_workspace(8193, torch.float16)
    assert helper._workspace_capacity == 8448


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

    assert probs.shape == (bs, draft_token_num, vocab_size)
    torch.testing.assert_close(logits, logits_ref)
    torch.testing.assert_close(
        probs.sum(dim=-1), torch.ones((bs, draft_token_num), dtype=torch.float32)
    )
