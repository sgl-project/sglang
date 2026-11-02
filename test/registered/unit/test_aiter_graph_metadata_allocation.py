"""Graph metadata storage must follow its actual paged/legacy consumers."""

import pytest
import torch

from sglang.srt.layers.attention import aiter_backend as module
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=15, suite="stage-b-test-1-gpu-small-amd")


def backend(*, unified=True, verify=True, swa=True, skip_prefill=False):
    obj = module.AiterAttnBackend.__new__(module.AiterAttnBackend)
    obj._use_flydsl_pa_decode = False
    obj._flydsl_dflash_swa_enabled = False
    obj._flydsl_target_verify_swa_enabled = False
    obj.kv_cache_is_vectorized_5d = False
    obj.use_triton_unified_attention = unified
    obj._use_unified_verify = verify
    obj.use_sliding_window_kv_pool = swa
    obj.use_mla = False
    obj.skip_prefill = skip_prefill
    obj.device = "cpu"
    obj.page_size = 64
    obj.max_context_len = 8192
    obj.num_draft_tokens = 8
    # Start smaller to also exercise graph padding's indptr growth.
    obj.kv_indptr = torch.zeros(2, dtype=torch.int32)
    obj.qo_indptr = torch.zeros(2, dtype=torch.int32)
    obj.mask_indptr = torch.zeros(2, dtype=torch.int64)
    return obj


@pytest.mark.parametrize("swa", [False, True])
def test_unified_graphs_keep_page_tables_without_full_history_scratch(swa):
    obj = backend(swa=swa)
    obj.init_cuda_graph_state(4, 32)
    assert obj.cuda_graph_kv_indices is None
    assert obj.cuda_graph_custom_mask is None
    assert obj.cuda_graph_page_table.shape == (4, 128)
    assert obj.kv_indptr.shape == (5,)
    if swa:
        assert obj.cuda_graph_swa_kv_indices is None
        assert obj.cuda_graph_swa_page_table.shape == (4, 128)
        assert obj.cuda_graph_swa_out_cache_loc.shape == (32,)


@pytest.mark.parametrize(
    "unified,verify", [(False, False), (True, False), (False, True)]
)
def test_any_legacy_consumer_retains_token_indices_and_draft_slack(unified, verify):
    obj = backend(unified=unified, verify=verify)
    obj.init_cuda_graph_state(4, 32)
    assert obj.cuda_graph_kv_indices.numel() == 4 * (8192 + 8)
    assert obj.cuda_graph_swa_kv_indices.shape == obj.cuda_graph_kv_indices.shape
    if verify:
        assert obj.cuda_graph_custom_mask is None
    else:
        assert obj.cuda_graph_custom_mask.numel() == 32 * 8192


def test_caller_owned_index_buffer_is_preserved():
    obj = backend()
    scratch = torch.ones(32, dtype=torch.int32)
    obj.init_cuda_graph_state(4, 32, kv_indices_buf=scratch)
    assert obj.cuda_graph_kv_indices is scratch
    assert torch.all(scratch == 1)


def test_decode_only_legacy_backend_does_not_allocate_verify_mask():
    obj = backend(unified=False, verify=False, skip_prefill=True)
    obj.init_cuda_graph_state(4, 32)
    assert obj.cuda_graph_custom_mask is None
    assert obj.cuda_graph_kv_indices.numel() == 4 * (8192 + 8)
