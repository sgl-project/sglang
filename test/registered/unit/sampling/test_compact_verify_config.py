from fractions import Fraction
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers import logits_processor as lp
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.model_executor.graph_shared_output import GraphSharedOutput
from sglang.srt.speculative.compact_verify import config
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")
register_cpu_ci(est_time=15, suite="base-c-test-cpu")


def test_conservative_rounding_enclosure():
    u32, u64 = Fraction(1, 2**24), Fraction(1, 2**53)
    g32 = 2048 * u32 / (1 - 2048 * u32)
    g64 = 154884 * u64 / (1 - 154884 * u64)
    lo = (1 - g64) * (1 - u32) / ((1 + g32) * (1 + u64))
    hi = (1 + g64) * (1 + u32) / ((1 - g32) * (1 - u64))
    radius = Fraction(1, 1024)
    assert (1 - radius) * (1 + u64) < lo
    assert (1 + radius) * (1 - u64) > hi


def inputs():
    sampling = SimpleNamespace(
        is_any_greedy=False,
        need_top_k_sampling=False,
        need_top_p_sampling=False,
        need_min_p_sampling=False,
        has_custom_logit_processor=False,
        acc_additive_penalties=None,
        acc_scaling_penalties=None,
        logit_bias=None,
        return_sampling_masks=None,
    )
    batch = SimpleNamespace(
        sampling_info=sampling,
        return_logprob=False,
        reqs=[SimpleNamespace(sampling_params=SimpleNamespace(temperature=1.0))],
    )
    verify = SimpleNamespace(tree_topk=1, max_tree_depth=4, draft_token_num=4)
    return verify, batch


@pytest.mark.parametrize(
    "field",
    [
        "is_any_greedy",
        "need_top_k_sampling",
        "need_top_p_sampling",
        "need_min_p_sampling",
        "has_custom_logit_processor",
    ],
)
def test_sampling_flags_fall_back(field):
    verify, batch = inputs()
    assert config.sampling_supported(verify, batch, None)
    setattr(batch.sampling_info, field, True)
    assert not config.sampling_supported(verify, batch, None)


@pytest.mark.parametrize(
    "field", ["acc_additive_penalties", "acc_scaling_penalties", "logit_bias"]
)
def test_modifiers_fall_back(field):
    verify, batch = inputs()
    setattr(batch.sampling_info, field, torch.zeros(1))
    assert not config.sampling_supported(verify, batch, None)


def test_temperature_logprob_grammar_and_tree_fall_back():
    verify, batch = inputs()
    batch.reqs[0].sampling_params.temperature = 0.7
    assert not config.sampling_supported(verify, batch, None)
    batch.reqs[0].sampling_params.temperature = 1.0
    batch.return_logprob = True
    assert not config.sampling_supported(verify, batch, None)
    batch.return_logprob = False
    assert not config.sampling_supported(verify, batch, object())
    verify.tree_topk = 2
    assert not config.sampling_supported(verify, batch, None)


def test_compact_graph_buffer_does_not_allocate_full_vocabulary():
    buffers = GraphSharedOutput(device=torch.device("cpu"), max_rows=4)
    local = buffers.get_compact_logits_buffer(rows=2)
    assert local.shape == (2, 38720) and local.dtype == torch.bfloat16
    assert buffers._logits_buffers == {}
    assert buffers.get_compact_logits_buffer(rows=1).data_ptr() == local.data_ptr()


def test_local_projection_bypasses_gather(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_COMPACT_SPEC_VERIFY", "1")
    monkeypatch.setattr(config, "configured", lambda: True)
    monkeypatch.setattr(lp, "get_parallel", lambda: SimpleNamespace(tp_rank=0))
    processor = object.__new__(lp.LogitsProcessor)
    torch.nn.Module.__init__(processor)
    processor.do_tensor_parallel_all_gather_dp_attn = False
    processor.do_tensor_parallel_all_gather = True
    processor.use_attn_tp_group = False
    processor.vocab_size = 154880
    processor.logit_scale = None
    processor.final_logit_softcapping = None
    local = torch.ones((2, 38720), dtype=torch.bfloat16)
    processor._compute_lm_head = lambda *_: local
    processor._logits_gatherer = lambda *_: pytest.fail("full gather called")
    head = object.__new__(VocabParallelEmbedding)
    torch.nn.Module.__init__(head)
    head.enable_tp = True
    head.org_vocab_size = head.num_embeddings = 154880
    head.shard_indices = SimpleNamespace(
        org_vocab_start_index=0, org_vocab_end_index=38720
    )
    meta = lp.LogitsMetadata(
        forward_mode=ForwardMode.TARGET_VERIFY,
        capture_hidden_mode=CaptureHiddenMode.NULL,
        next_token_logits_buffer=torch.empty_like(local),
    )
    result = processor._get_logits(torch.zeros(2, 1), head, meta)
    assert (
        meta.compact_verify_sharded
        and result.data_ptr() == meta.next_token_logits_buffer.data_ptr()
    )
    torch.testing.assert_close(result, local)
