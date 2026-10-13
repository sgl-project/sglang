"""Check FlashInfer's speculative NVFP4 routing without launching GPU kernels.

Kernel replay correctness is covered by test_nvfp4_kv_cache.py. These tests run
the real metadata and forward methods with CPU tensors so that call-contract
changes (including the required KV location plan) cannot escape that coverage.
"""

import sys
from types import SimpleNamespace
from unittest.mock import Mock, create_autospec, patch

import pytest
import torch

from sglang.srt.layers.attention.flashinfer_backend import (
    FlashInferAttnBackend,
    FlashInferIndicesUpdaterPrefill,
)
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=25, suite="base-a-test-cpu")


def _make_backend():
    backend = object.__new__(FlashInferAttnBackend)
    updater = object.__new__(FlashInferIndicesUpdaterPrefill)
    # Autospec preserves update_single_wrapper's required keyword-only plan.
    updater.update = create_autospec(updater.update_single_wrapper)
    backend.indices_updater_prefill = updater
    backend.use_sliding_window_kv_pool = False
    backend.prefill_split_tile_size = None
    backend.num_wrappers = 1
    backend.prefill_uses_dequant_workspace = True
    backend.kv_cache_quant_method = SimpleNamespace(needs_global_scale=lambda: True)
    backend.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.arange(64, dtype=torch.int32).view(2, 32)
    )
    backend.prefill_wrappers_paged = [Mock()]
    backend.prefill_wrappers_verify = [Mock()]
    backend.prefill_cuda_graph_metadata = {2: [Mock()]}
    backend.draft_extend_cuda_graph_metadata = {2: [Mock()]}
    # A previous prompt/full-prefill graph can leave these populated. Neither
    # speculative mode may use that contiguous prompt workspace mapping.
    backend.dq_page_table = torch.tensor([53, 54], dtype=torch.int32)
    backend.cpu_req_pool_indices = None
    pool_methods = object.__new__(MHATokenToKVPool)
    backend.token_to_kv_pool = SimpleNamespace(
        get_flashinfer_speculative_dequant_workspace_kv_buffer=create_autospec(
            pool_methods.get_flashinfer_speculative_dequant_workspace_kv_buffer
        ),
        get_flashinfer_dequant_workspace_kv_buffer=Mock(
            side_effect=AssertionError("speculative forward used prompt workspace")
        ),
        set_kv_buffer=create_autospec(pool_methods.set_kv_buffer),
    )
    return backend


def _make_batch(mode, explicit_prefix):
    width = 3
    prefix_lens = torch.tensor([4, 7], dtype=torch.int32)
    seq_lens = prefix_lens + width if mode.is_draft_extend_v2() else prefix_lens.clone()
    return SimpleNamespace(
        forward_mode=mode,
        batch_size=2,
        req_pool_indices=torch.tensor([1, 0], dtype=torch.int32),
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens.clone(),
        seq_lens_sum=int(seq_lens.sum()),
        extend_prefix_lens=prefix_lens if explicit_prefix else None,
        extend_prefix_lens_cpu=None,
        extend_seq_lens_cpu=None,
        encoder_lens=None,
        out_cache_loc=torch.tensor([12, 13, 14, 21, 22, 23], dtype=torch.int32),
        out_cache_loc_is_physical=True,
        kv_loc_plan=object(),
        spec_info=SimpleNamespace(num_tokens_per_req=width, ragged_verify_layout=None),
    )


@pytest.mark.parametrize(
    "mode,explicit_prefix",
    [
        pytest.param(ForwardMode.TARGET_VERIFY, False, id="target_verify"),
        pytest.param(ForwardMode.DRAFT_EXTEND_V2, False, id="draft_derived_prefix"),
        pytest.param(ForwardMode.DRAFT_EXTEND_V2, True, id="draft_explicit_prefix"),
    ],
)
def test_speculative_metadata_and_forward_use_physical_workspace(mode, explicit_prefix):
    backend = _make_backend()
    batch = _make_batch(mode, explicit_prefix)
    stale_page_table = backend.dq_page_table
    backend.init_forward_metadata(batch)

    update_kwargs = backend.indices_updater_prefill.update.call_args.kwargs
    assert update_kwargs["plan"] is batch.kv_loc_plan
    assert update_kwargs["spec_info"] is batch.spec_info
    assert update_kwargs["use_ragged"] is False
    expected_wrappers = (
        backend.prefill_wrappers_verify
        if mode.is_target_verify()
        else backend.prefill_wrappers_paged
    )
    assert update_kwargs["prefill_wrappers"] is expected_wrappers
    assert backend.forward_metadata.prefill_wrappers is expected_wrappers

    layer = SimpleNamespace(
        layer_id=0,
        logit_cap=0.0,
        is_cross_attention=False,
        attn_type=AttentionType.DECODER,
        tp_q_head_num=2,
        tp_k_head_num=1,
        tp_v_head_num=1,
        head_dim=8,
        scaling=8**-0.5,
        sliding_window_size=-1,
        k_scale_float=1.0,
        v_scale_float=1.0,
    )
    q = torch.randn(6, 2, 8)
    k = torch.randn(6, 1, 8)
    v = torch.randn(6, 1, 8)
    workspace = (torch.empty(64, 1, 8), torch.empty(64, 1, 8))
    pool = backend.token_to_kv_pool
    pool.get_flashinfer_speculative_dequant_workspace_kv_buffer.return_value = workspace
    expected_wrappers[0].forward.return_value = q

    result = backend.forward_extend(q, k, v, layer, batch)

    prepare = pool.get_flashinfer_speculative_dequant_workspace_kv_buffer
    prepare.assert_called_once()
    args = prepare.call_args.args
    assert args[0] is layer
    assert args[1] is backend.req_to_token_pool.req_to_token
    assert args[2] is batch.req_pool_indices
    assert args[3] is batch.seq_lens
    assert args[4] is k
    assert args[5] is v
    assert args[6] is batch.out_cache_loc
    assert args[7] == batch.spec_info.num_tokens_per_req
    assert args[8] == (3 if mode.is_draft_extend_v2() else 0)
    pool.get_flashinfer_dequant_workspace_kv_buffer.assert_not_called()
    write_loc = pool.set_kv_buffer.call_args.args[1]
    assert write_loc.loc is batch.out_cache_loc
    assert write_loc.physical
    assert expected_wrappers[0].forward.call_args.args[1] is workspace
    assert backend.dq_page_table is stale_page_table
    torch.testing.assert_close(result, q.view(6, 16))


@pytest.mark.parametrize(
    "mode",
    [ForwardMode.TARGET_VERIFY, ForwardMode.DRAFT_EXTEND_V2],
)
def test_speculative_graph_metadata_passes_location_plan(mode):
    backend = _make_backend()
    batch = _make_batch(mode, explicit_prefix=False)
    backend.init_forward_metadata_out_graph(batch)

    update_kwargs = backend.indices_updater_prefill.update.call_args.kwargs
    assert update_kwargs["plan"] is batch.kv_loc_plan
    assert update_kwargs["use_ragged"] is False
    expected_wrappers = (
        backend.prefill_cuda_graph_metadata[batch.batch_size]
        if mode.is_target_verify()
        else backend.draft_extend_cuda_graph_metadata[batch.batch_size]
    )
    assert update_kwargs["prefill_wrappers"] is expected_wrappers


def test_non_quantized_draft_extend_preserves_ragged_metadata():
    backend = _make_backend()
    backend.prefill_uses_dequant_workspace = False
    backend.is_multimodal = False
    backend.enable_mis = False
    backend.enable_deterministic = False
    backend.use_paged = False
    batch = _make_batch(ForwardMode.DRAFT_EXTEND_V2, explicit_prefix=True)
    batch.extend_prefix_lens_cpu = [4, 7]
    batch.cross_attention_custom_mask = None
    with patch(
        "sglang.srt.layers.attention.flashinfer_backend.is_in_tc_piecewise_cuda_graph",
        return_value=False,
    ):
        backend.init_forward_metadata(batch)

    update_args = backend.indices_updater_prefill.update.call_args
    assert update_args.args[4] is batch.extend_prefix_lens
    assert update_args.kwargs["spec_info"] is None
    assert update_args.kwargs["use_ragged"] is True
    assert update_args.kwargs["plan"] is batch.kv_loc_plan
    assert backend.forward_metadata.use_ragged is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
