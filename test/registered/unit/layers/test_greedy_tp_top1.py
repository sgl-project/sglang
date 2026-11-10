from types import SimpleNamespace

import pytest
import torch
from sglang.srt.layers import logits_processor
from sglang.srt.layers.logits_processor import (
    LogitsProcessor,
    _is_strict_greedy_forward_batch,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


def _strict_batch(**overrides):
    sampling = SimpleNamespace(
        is_all_greedy=True,
        grammars=None,
        has_custom_logit_processor=False,
        logit_bias=None,
        vocab_mask=None,
        apply_mask_func=None,
        acc_additive_penalties=None,
        acc_scaling_penalties=None,
        penalizer_orchestrator=SimpleNamespace(is_required=False),
    )
    values = dict(
        is_prefill_only=False,
        forward_mode=ForwardMode.DECODE,
        spec_algorithm=SpeculativeAlgorithm.NONE,
        sampling_info=sampling,
        return_logprob=False,
        top_logprobs_nums=None,
        token_ids_logprobs=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_strict_greedy_shortcut_rejects_full_logits_consumers(monkeypatch):
    monkeypatch.setenv("SGLANG_V100_GREEDY_TP_TOP1", "1")
    assert _is_strict_greedy_forward_batch(_strict_batch())
    assert not _is_strict_greedy_forward_batch(_strict_batch(return_logprob=True))
    assert not _is_strict_greedy_forward_batch(_strict_batch(top_logprobs_nums=[1]))
    assert not _is_strict_greedy_forward_batch(
        _strict_batch(spec_algorithm=SpeculativeAlgorithm.DFLASH)
    )
    penalized = _strict_batch()
    penalized.sampling_info.penalizer_orchestrator.is_required = True
    assert not _is_strict_greedy_forward_batch(penalized)
    sampled = _strict_batch()
    sampled.sampling_info.is_all_greedy = False
    assert not _is_strict_greedy_forward_batch(sampled)


def test_strict_greedy_shortcut_defaults_off_and_validates_env(monkeypatch):
    monkeypatch.delenv("SGLANG_V100_GREEDY_TP_TOP1", raising=False)
    assert not _is_strict_greedy_forward_batch(_strict_batch())
    monkeypatch.setenv("SGLANG_V100_GREEDY_TP_TOP1", "invalid")
    with pytest.raises(ValueError, match="must be a boolean value"):
        _is_strict_greedy_forward_batch(_strict_batch())


def test_tp_candidate_exchange_matches_full_vocab_argmax(monkeypatch):
    torch.manual_seed(41)
    batch, hidden_size, shard_size, world_size = 4, 8, 5, 4
    hidden = torch.randn(batch, hidden_size, dtype=torch.float16)
    weights = [
        torch.randn(shard_size, hidden_size, dtype=torch.float16)
        for _ in range(world_size)
    ]
    rank = 2

    class FakeGroup:
        def all_gather(self, candidates, dim=-1):
            gathered = []
            for current_rank, weight in enumerate(weights):
                values, indices = torch.max(hidden @ weight.T, dim=-1)
                gathered.append(
                    torch.stack(
                        (
                            values.float(),
                            (indices + current_rank * shard_size).float(),
                        ),
                        dim=-1,
                    )
                )
            torch.testing.assert_close(candidates, gathered[rank])
            return torch.cat(gathered, dim=dim)

    monkeypatch.setattr(
        logits_processor,
        "get_tensor_model_parallel_world_size",
        lambda: world_size,
    )
    monkeypatch.setattr(logits_processor, "get_tp_group", FakeGroup)
    head = SimpleNamespace(
        weight=weights[rank],
        shard_indices=SimpleNamespace(
            num_org_elements=shard_size,
            org_vocab_start_index=rank * shard_size,
        ),
    )
    processor = object.__new__(LogitsProcessor)

    actual = processor._get_greedy_tp_top1(hidden, head)
    expected = torch.argmax(hidden @ torch.cat(weights).T, dim=-1)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
