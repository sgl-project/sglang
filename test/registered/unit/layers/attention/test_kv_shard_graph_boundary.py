"""CPU contract tests for live KV-shard metadata across BCG boundaries."""

import importlib
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.attention.dsa.dsa_prefill_cuda_graph import (
    bcg_kv_shard_attention,
    bcg_kv_shard_indexer,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_BCG = importlib.import_module(
    "sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph"
)
_DSA = "sglang.srt.layers.attention.dsa.dsa_prefill_cuda_graph"


def test_indexer_capture_skips_collectives_and_replay_reads_live_batch():
    output = torch.empty((2, 4), dtype=torch.int32)
    pool = SimpleNamespace(forward_batch=SimpleNamespace(value=3))
    indexer = SimpleNamespace(
        _forward_cuda_impl=Mock(
            side_effect=lambda x, q, pos, batch, *a, **kw: torch.full_like(
                output, batch.value
            )
        )
    )
    capture = SimpleNamespace(
        _end_current_segment=Mock(),
        _begin_new_segment=Mock(),
        cuda_graph=SimpleNamespace(_break_fns=[]),
    )
    token = _BCG._current_capture_var.set(capture)
    try:
        assert (
            bcg_kv_shard_indexer(
                indexer,
                pool,
                torch.empty(2, 8),
                torch.empty(2, 8),
                torch.arange(2),
                0,
                True,
                output,
            )
            is None
        )
    finally:
        _BCG._current_capture_var.reset(token)
    indexer._forward_cuda_impl.assert_not_called()
    assert torch.all(output == -1)
    for value in (7, 11):
        pool.forward_batch = SimpleNamespace(value=value)
        assert capture.cuda_graph._break_fns[0]() is None
        assert torch.all(output == value)


def test_indexer_without_output_still_runs_cache_store():
    indexer = SimpleNamespace(_forward_cuda_impl=Mock(return_value=None))
    pool = SimpleNamespace(forward_batch=object())
    assert (
        bcg_kv_shard_indexer(
            indexer,
            pool,
            torch.empty(2, 8),
            torch.empty(2, 8),
            torch.arange(2),
            0,
            False,
            None,
        )
        is None
    )
    indexer._forward_cuda_impl.assert_called_once()


def test_attention_writes_flat_caller_owned_buffer_with_live_metadata():
    layer = SimpleNamespace(tp_q_head_num=2, v_head_dim=4)
    pool = SimpleNamespace(forward_batch=object())
    output = torch.empty((3, 8))
    backend = SimpleNamespace(
        forward=Mock(return_value=torch.arange(24).view(3, 1, 2, 4))
    )
    with patch(f"{_DSA}.get_attn_backend", return_value=backend):
        assert (
            bcg_kv_shard_attention(
                layer,
                pool,
                torch.empty(3, 8),
                None,
                None,
                False,
                output,
            )
            is None
        )
    assert torch.equal(output, torch.arange(24).view(3, 8))
    assert backend.forward.call_args.args[4] is pool.forward_batch


def test_fused_indexer_attention_mutates_both_outputs_at_each_replay():
    from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla import (
        DeepseekMLAForwardMixin,
    )

    output = torch.empty(2, 8)
    topk = torch.empty(2, 4, dtype=torch.int32)
    pool = SimpleNamespace(forward_batch=object())
    indexer = Mock(return_value=torch.full_like(topk, 6))
    attention = Mock(side_effect=lambda *a, **kw: output.fill_(9))
    model = SimpleNamespace(
        should_run_indexer=lambda previous: previous is None,
        indexer=indexer,
        layer_id=3,
        rotary_emb=None,
        w_kc=torch.empty(1),
        _fuse_rope_for_trtllm_mla=lambda batch: False,
        _eager_bmm_attention=SimpleNamespace(run_eager=attention),
    )
    with patch(
        "sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla.prepare_kv_for_attention",
        side_effect=lambda model, batch, k, rope, **kw: (k, rope),
    ) as prepare:
        assert (
            DeepseekMLAForwardMixin._eager_dsa_indexer_attention(
                model,
                pool,
                torch.empty(2, 8),
                torch.empty(2, 8),
                torch.arange(2),
                torch.empty(2, 8),
                torch.empty(2, 8),
                torch.empty(2, 8),
                torch.empty(2, 8),
                torch.empty(2, 8),
                torch.empty(2, 8),
                output,
                topk,
            )
            is None
        )
    assert torch.all(topk == 6)
    assert torch.all(output == 9)
    assert indexer.call_args.kwargs["forward_batch"] is pool.forward_batch
    prepare.assert_called_once()
