"""Vocabulary-shard equivalence and stock fallback coverage."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.speculative import eagle_utils, sharded_verify
from sglang.srt.speculative.sharded_verify import (
    VocabShard,
    gather_full_logits,
    key_splits,
    keys_to_tokens,
    shard_keys,
    sharded_target_tokens,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

VOCAB = 248320


def _verify_tree():
    return SimpleNamespace(
        draft_token=torch.tensor([4, 0, 1, 4, 2, 3]),
        draft_probs=None,
        retrieve_index=torch.arange(6).view(2, 3),
        retrieve_next_token=torch.tensor([[1, 2, -1], [1, 2, -1]]),
        retrieve_next_sibling=torch.full((2, 3), -1),
        custom_mask=None,
        positions=None,
        draft_token_num=3,
        max_tree_depth=3,
        tree_topk=1,
    )


def _sampling_info(*, greedy: bool):
    return SimpleNamespace(
        temperatures=torch.ones((2, 1)),
        top_ks=torch.ones(2, dtype=torch.int32),
        top_ps=torch.ones(2),
        min_ps=torch.zeros(2),
        acc_additive_penalties=None,
        acc_scaling_penalties=None,
        logit_bias=None,
        sampling_seed=None,
        is_all_greedy=greedy,
        need_top_k_sampling=False,
        need_top_p_sampling=False,
        need_min_p_sampling=False,
        watermark_keys=None,
        watermark_context_windows=None,
        watermark_enabled=None,
        has_watermark_candidates=False,
        max_top_k=None,
    )


def _verify_batch(sampling_info):
    return SimpleNamespace(
        device=torch.device("cpu"),
        forward_mode=SimpleNamespace(is_idle=lambda: False),
        seq_lens=torch.tensor([7, 11]),
        sampling_info=sampling_info,
        req_pool_indices=torch.tensor([3, 7]),
        return_logprob=False,
    )


def _verify_logits():
    return torch.tensor(
        [
            [9.0, 1.0, 0.0, -1.0, -2.0],
            [0.0, 8.0, 1.0, -1.0, -2.0],
            [0.0, 1.0, 7.0, -1.0, -2.0],
            [0.0, 1.0, 6.0, -1.0, -2.0],
            [0.0, 1.0, 2.0, 5.0, -2.0],
            [0.0, 1.0, 2.0, 3.0, 4.0],
        ]
    )


def _runtime_spec():
    return SimpleNamespace(
        speculative_use_rejection_sampling=False,
        speculative_use_block_verification=False,
        speculative_accept_threshold_single=1.0,
        speculative_accept_threshold_acc=1.0,
    )


class _FakeGroup:
    """all_gather_into_tensor over shards held on one device, rank order."""

    def __init__(self, shards):
        self.shards, self.world_size = shards, len(shards)

    def all_gather_into_tensor(self, output, input):
        output.copy_(torch.cat([s.reshape(-1) for s in self.shards]))


class TestShardedPolicy(CustomTestCase):
    def test_temperature_decline_gathers_then_enters_stock_without_early_rng(self):
        sampling = _sampling_info(greedy=False)
        shard = VocabShard(start=0, valid=5, vocab_size=5)
        logits = SimpleNamespace(
            next_token_logits=_verify_logits().clone(),
            next_token_logits_vocab_shard=shard,
        )
        batch = _verify_batch(sampling)
        group = SimpleNamespace(world_size=1)
        group.all_gather_into_tensor = lambda output, value: output.copy_(value)
        parallel = SimpleNamespace(tp_group=group, attn_tp_group=group)
        events = []
        state_before = torch.random.get_rng_state().clone()

        def local_keys(_view):
            events.append("local")
            self.assertTrue(torch.equal(torch.random.get_rng_state(), state_before))
            return None

        original_gather = eagle_utils.gather_full_logits

        def gather(*args, **kwargs):
            events.append("gather")
            return original_gather(*args, **kwargs)

        def stock_sample(**kwargs):
            events.append("stock")
            self.assertFalse(torch.equal(torch.random.get_rng_state(), state_before))
            kwargs["predicts"].zero_()
            kwargs["accept_index"].fill_(-1)
            kwargs["accept_token_num"].zero_()

        with (
            patch.object(eagle_utils, "_is_cuda", True),
            patch.object(eagle_utils, "_is_cpu", False),
            patch.object(eagle_utils, "_is_hip", False),
            patch.object(eagle_utils, "_is_xpu", False),
            patch.object(eagle_utils, "get_spec", return_value=_runtime_spec()),
            patch.object(eagle_utils, "gather_full_logits", side_effect=gather),
            patch.object(
                eagle_utils,
                "tree_speculative_sampling_target_only",
                side_effect=stock_sample,
            ),
            patch("sglang.srt.runtime_context.get_parallel", return_value=parallel),
            patch(
                "sglang.srt.layers.dp_attention.is_dp_attention_enabled",
                return_value=False,
            ),
        ):
            eagle_utils.eagle_sample(
                _verify_tree(),
                batch,
                logits,
                local_keys=local_keys,
            )

        self.assertEqual(events, ["local", "gather", "stock"])
        self.assertIsNone(logits.next_token_logits_vocab_shard)
        self.assertEqual(tuple(logits.next_token_logits.shape), (6, 5))


def _sharded_tokens(full, tp, split=False):
    width = -(-full.shape[1] // tp)
    splits = key_splits(width) if split else 1
    keys = []
    for rank in range(tp):
        start = rank * width
        part = full[:, start : start + width].contiguous()
        shard = VocabShard(start=start, valid=part.shape[1], vocab_size=full.shape[1])
        keys.append(shard_keys(part, shard, splits).view(-1))
    return keys_to_tokens(torch.cat(keys), tp, splits)


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestShardedGreedy(CustomTestCase):
    def test_matches_argmax(self):
        generator = torch.Generator(device="cuda").manual_seed(0)
        full = (
            torch.randn(7, VOCAB, device="cuda", generator=generator) * 4
        ).bfloat16()
        peak = full.float().max() + 1
        full[0, 5] = full[0, VOCAB - 3] = peak
        full[1].fill_(1.5)
        full[2].zero_()
        want = torch.argmax(full.float(), dim=-1)
        for split in (False, True):
            self.assertTrue(torch.equal(_sharded_tokens(full, 8, split=split), want))

    def test_exchange_falls_back_to_collective(self):
        full = torch.randn(7, 4096, device="cuda").bfloat16()
        shards = list(full.split(2048, dim=1))
        parts = key_splits(shards[0].shape[1])
        keys = [
            shard_keys(
                part,
                VocabShard(rank * 2048, part.shape[1], full.shape[1]),
                splits=parts,
            )
            for rank, part in enumerate(shards)
        ]
        with (
            patch.object(sharded_verify, "_ll_keys", return_value=None) as ll,
            patch.object(sharded_verify, "_mc_gather_keys", return_value=None) as mc,
        ):
            got = sharded_target_tokens(
                shards[0],
                VocabShard(0, shards[0].shape[1], full.shape[1]),
                _FakeGroup(keys),
            )
        ll.assert_called_once()
        mc.assert_called_once()
        self.assertTrue(torch.equal(got, torch.argmax(full.float(), dim=-1)))


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestGatherFullLogits(CustomTestCase):
    def test_matches_concat_slice_cast(self):
        # the fallback must rebuild the stock gathered logits exactly: rank-major
        # vocabulary columns, cut to the vocabulary, FP32
        full = torch.randn(7, 1000, device="cuda").bfloat16()
        for tp in (2, 8):
            with self.subTest(tp=tp):
                width = -(-1000 // tp)
                padded = torch.nn.functional.pad(full, (0, width * tp - 1000))
                shards = list(padded.split(width, dim=1))
                shard = VocabShard(start=0, valid=width, vocab_size=1000)
                got = gather_full_logits(shards[0], shard, _FakeGroup(shards))
                self.assertTrue(torch.equal(got, full.float()))


_TEMPERATURE_ONLY = dict(is_all_greedy=False)
_AGENTX_SAMPLING = dict(
    is_all_greedy=False, need_top_k_sampling=True, need_top_p_sampling=True
)


if __name__ == "__main__":
    unittest.main()
