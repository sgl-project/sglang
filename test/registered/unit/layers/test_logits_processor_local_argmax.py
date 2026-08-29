from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.logits_processor import _vocab_parallel_argmax
from sglang.srt.models.deepseek_nextn import _can_use_draft_local_argmax
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _FakeGroup:
    def __init__(self, gathered_values=None, gathered_ids=None):
        self.world_size = 1 if gathered_values is None else gathered_values.shape[-1]
        self.calls = []
        self._output = (
            None
            if gathered_values is None
            else torch.stack(
                (gathered_values.float(), gathered_ids.float()), dim=-1
            ).flatten(-2)
        )

    def all_gather(self, value, dim=-1):
        self.calls.append((value.clone(), dim))
        assert self._output.device == value.device
        assert self._output.dtype == value.dtype
        return self._output


def _gate_inputs(**overrides):
    values = {
        "requested": True,
        "cuda_backend": True,
        "tp_size": 2,
        "algorithm": "EAGLE",
        "topk": 1,
        "rejection": False,
        "token_map": None,
        "dp_attention": False,
        "dp_lm_head": False,
        "tp_lm_head_all_to_all": False,
        "added_vocab": 0,
        "lora_enabled": False,
        "supported_lm_head": True,
        "vocab_size": 151936,
    }
    values.update(overrides)
    spec = SimpleNamespace(
        speculative_algorithm=values["algorithm"],
        speculative_eagle_topk=values["topk"],
        speculative_use_rejection_sampling=values["rejection"],
        speculative_token_map=values["token_map"],
    )
    parallel = SimpleNamespace(
        enable_dp_attention=values["dp_attention"],
        enable_dp_lm_head=values["dp_lm_head"],
        enable_tp_lm_head_all_to_all=values["tp_lm_head_all_to_all"],
    )
    return values, spec, parallel


def _can_use(**overrides):
    values, spec, parallel = _gate_inputs(**overrides)
    return _can_use_draft_local_argmax(
        values["requested"],
        values["cuda_backend"],
        values["tp_size"],
        spec,
        parallel,
        values["added_vocab"],
        values["lora_enabled"],
        values["supported_lm_head"],
        values["vocab_size"],
    )


def _tp2_candidates(shard0, shard1):
    values = torch.stack((shard0.max(-1).values, shard1.max(-1).values), dim=-1)
    ids = torch.stack(
        (shard0.argmax(-1), shard1.argmax(-1).add(shard0.shape[-1])), dim=-1
    )
    return values, ids


def test_vocab_parallel_argmax_excludes_padding_and_applies_global_offset():
    logits = torch.tensor([[1.0, 3.0, 100.0]])
    actual = _vocab_parallel_argmax(logits, 2, 8, _FakeGroup())
    torch.testing.assert_close(actual, torch.tensor([[9]]))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_vocab_parallel_argmax_matches_full_vocab_and_first_max(dtype):
    shard0 = torch.tensor([[5.0, 1.0], [2.0, 8.0]], dtype=dtype)
    shard1 = torch.tensor([[5.0, 0.0], [9.0, 1.0]], dtype=dtype)
    values, ids = _tp2_candidates(shard0, shard1)

    actual = _vocab_parallel_argmax(shard1, 2, 2, _FakeGroup(values, ids))
    expected = torch.cat((shard0, shard1), dim=-1).argmax(-1, keepdim=True)
    torch.testing.assert_close(actual, expected)


def test_vocab_parallel_argmax_matches_random_tp2_full_vocab():
    generator = torch.Generator().manual_seed(17)
    shard0 = torch.randn(32, 8, generator=generator)
    shard1 = torch.randn(32, 8, generator=generator)
    values, ids = _tp2_candidates(shard0, shard1)
    group = _FakeGroup(values, ids)

    actual = _vocab_parallel_argmax(shard1, 8, 8, group)
    expected = torch.cat((shard0, shard1), dim=-1).argmax(-1, keepdim=True)

    torch.testing.assert_close(actual, expected)
    assert len(group.calls) == 1
    assert group.calls[0][0].shape == (32, 2)
    assert group.calls[0][1] == -1


def test_vocab_parallel_argmax_empty_shard_cannot_win():
    logits = torch.empty((2, 0))
    gathered_values = torch.tensor([[3.0, float("-inf")], [4.0, float("-inf")]])
    gathered_ids = torch.tensor([[1, 2**24 - 1], [2, 2**24 - 1]])
    actual = _vocab_parallel_argmax(
        logits, 0, 8, _FakeGroup(gathered_values, gathered_ids)
    )
    torch.testing.assert_close(actual, torch.tensor([[1], [2]]))


@pytest.mark.parametrize(
    "override",
    [
        {"requested": False},
        {"cuda_backend": False},
        {"tp_size": 1},
        {"algorithm": None},
        {"algorithm": "NGRAM"},
        {"topk": 2},
        {"rejection": True},
        {"token_map": "token-map.json"},
        {"dp_attention": True},
        {"dp_lm_head": True},
        {"tp_lm_head_all_to_all": True},
        {"added_vocab": 1},
        {"lora_enabled": True},
        {"supported_lm_head": False},
        {"vocab_size": 2**24},
    ],
)
def test_local_argmax_gate_fails_closed(override):
    assert not _can_use(**override)


@pytest.mark.parametrize("algorithm", ["EAGLE", "eagle", "NEXTN", "nextn"])
def test_local_argmax_gate_accepts_supported_greedy_tp_path(algorithm):
    assert _can_use(algorithm=algorithm)


def test_local_argmax_proposal_preserves_chain_bookkeeping():
    expected_ids = torch.tensor([[3], [7]])
    worker = object.__new__(EagleDraftWorker)
    worker._draft_vocab_parallel_argmax = lambda _: expected_ids
    positions = torch.tensor([11, 19])
    token_buffer = torch.zeros((2, 3), dtype=torch.long)

    topk_p, topk_index = worker._local_argmax_proposal(
        torch.empty((2, 4)), positions, token_buffer, 2
    )

    torch.testing.assert_close(topk_index, expected_ids)
    torch.testing.assert_close(topk_p, torch.ones((2, 1)))
    torch.testing.assert_close(positions, torch.tensor([12, 20]))
    torch.testing.assert_close(token_buffer[:, 2], expected_ids.flatten())


def test_local_argmax_is_explicitly_disabled_by_default():
    assert ServerArgs.speculative_draft_local_argmax is False
