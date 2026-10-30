"""Target sampling on vocabulary shards.

The per-shard keys reduced across shards pick exactly torch.argmax over the gathered
FP32 logits, ties to the smaller token id included, at TP 2 / 4 / 8. The fallback
gather rebuilds the stock gathered FP32 logits exactly. The LL key exchange returns
every row's max over all ranks' keys, launch after launch, whether the other ranks'
words are already there or land while it polls."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import triton
import triton.language as tl

from sglang.srt.speculative import eagle_utils, sharded_verify
from sglang.srt.speculative.sharded_verify import (
    VocabShard,
    _reduce_keys,
    gather_full_logits,
    key_splits,
    keys_to_tokens,
    reduce_sharded_token_keys,
    shard_keys,
    sharded_target_tokens,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

VOCAB = 248320


def _optional_binding_runtime(test):
    from importlib import import_module

    try:
        speculative_runtime = import_module(
            "sglang.srt.kernel_plugins.nvidia.speculative_runtime"
        )
    except ModuleNotFoundError as exc:
        if exc.name != "sglang.srt.kernel_plugins.nvidia.speculative_runtime":
            raise
        test.skipTest("the standalone binding contract is not installed")
    return speculative_runtime


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
    @staticmethod
    def _applies(
        *, greedy: bool, top_k: bool = False, has_local_keys: bool = False
    ) -> bool:
        info = SimpleNamespace(
            is_all_greedy=greedy,
            need_top_k_sampling=top_k,
            need_top_p_sampling=False,
            need_min_p_sampling=False,
            acc_additive_penalties=None,
            acc_scaling_penalties=None,
            logit_bias=None,
            sampling_seed=None,
        )
        batch = SimpleNamespace(sampling_info=info, return_logprob=False)
        spec = SimpleNamespace(
            speculative_use_rejection_sampling=False,
            speculative_accept_threshold_single=1.0,
            speculative_accept_threshold_acc=1.0,
        )
        with (
            patch.object(eagle_utils, "_is_cuda", True),
            patch.object(eagle_utils, "get_spec", return_value=spec),
        ):
            return eagle_utils._sharded_sampling_applies(
                verify_input=SimpleNamespace(tree_topk=1),
                batch=batch,
                grammar_mask=None,
                watermark=False,
                has_local_keys=has_local_keys,
            )

    def test_only_greedy_rows_stay_sharded(self):
        self.assertTrue(self._applies(greedy=True))
        self.assertFalse(self._applies(greedy=False))
        self.assertTrue(self._applies(greedy=False, has_local_keys=True))
        self.assertFalse(self._applies(greedy=False, top_k=True))

    def test_local_policy_decline_happens_before_runtime_rng(self):
        logits = torch.ones((2, 4))
        shard = VocabShard(start=0, valid=4, vocab_size=4)
        temperatures = torch.ones(2, dtype=torch.float32)
        seen = []

        def local_keys(view):
            seen.append(view)
            self.assertIsNone(view.seeds)
            self.assertEqual(tuple(view.values.shape), (2, 4))
            self.assertEqual(view.values.stride(-1), 1)
            self.assertEqual(tuple(view.temperatures.shape), (2,))
            self.assertEqual(view.temperatures.dtype, torch.float32)
            self.assertTrue(view.temperatures.is_contiguous())
            self.assertGreater(view.splits, 0)
            return None

        rng_before = torch.random.get_rng_state().clone()
        with (
            patch(
                "sglang.srt.runtime_context.get_parallel",
                return_value=SimpleNamespace(tp_group=object()),
            ),
            patch.object(
                torch,
                "randint",
                side_effect=AssertionError("runtime consumed RNG before fallback"),
            ),
        ):
            got = eagle_utils._sharded_tokens(
                logits,
                shard,
                local_keys=local_keys,
                temperatures=temperatures,
            )
        self.assertIsNone(got)
        self.assertEqual(len(seen), 1)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_before))

    def test_existing_sampling_seeds_are_passed_unchanged(self):
        logits = torch.ones((6, 4))
        shard = VocabShard(start=0, valid=4, vocab_size=4)
        seeds = torch.tensor([17, 29], dtype=torch.int64)
        seen = []

        def local_keys(view):
            seen.append(view.seeds)
            return None

        with (
            patch(
                "sglang.srt.runtime_context.get_parallel",
                return_value=SimpleNamespace(tp_group=object()),
            ),
            patch.object(
                torch,
                "randint",
                side_effect=AssertionError("provider fallback must not draw RNG"),
            ),
        ):
            got = eagle_utils._sharded_tokens(
                logits,
                shard,
                local_keys=local_keys,
                temperatures=torch.ones(6),
                seeds=seeds,
            )
        self.assertIsNone(got)
        self.assertIs(seen[0], seeds)
        self.assertEqual(tuple(seen[0].shape), (2,))
        self.assertNotEqual(seen[0].shape[0], logits.shape[0])

    def test_provider_splits_use_rank_uniform_padded_width(self):
        logits = torch.empty((2, 4104))
        shard = VocabShard(start=7 * 4104, valid=4041, vocab_size=32769)
        group = SimpleNamespace(world_size=8)
        tokens = torch.empty(2, dtype=torch.int64)
        seen = []

        def local_keys(view):
            seen.append(view.splits)
            return torch.empty((2, view.splits), dtype=torch.int64)

        with (
            patch(
                "sglang.srt.runtime_context.get_parallel",
                return_value=SimpleNamespace(tp_group=group),
            ),
            patch.object(
                eagle_utils,
                "reduce_sharded_token_keys",
                return_value=tokens,
            ) as reduce,
        ):
            got = eagle_utils._sharded_tokens(
                logits,
                shard,
                local_keys=local_keys,
            )

        self.assertIs(got, tokens)
        self.assertEqual(seen, [4])
        reduce.assert_called_once()
        keys = reduce.call_args.args[0]
        self.assertEqual(tuple(keys.shape), (2, 4))
        self.assertEqual(reduce.call_args.kwargs, {"splits": 4})

    def test_local_key_view_capability_is_cached_before_the_hot_path(self):
        logits = torch.ones((2, 4))
        shard = VocabShard(start=0, valid=4, vocab_size=4)
        with patch.object(
            eagle_utils,
            "import_module",
            side_effect=AssertionError("hot path repeated provider discovery"),
        ):
            view = eagle_utils._local_key_view(
                logits=logits,
                vocab_shard=shard,
                temperatures=None,
                seeds=None,
                splits=1,
            )
        self.assertIs(view.values, logits)
        self.assertEqual(view.shard.start, 0)
        self.assertEqual(view.shard.valid, 4)
        self.assertEqual(view.splits, 1)

    def test_default_local_keys_serves_greedy_and_declines_all_temperature(self):
        runtime = _optional_binding_runtime(self)
        values = torch.ones((2, 4))
        shard = VocabShard(start=0, valid=4, vocab_size=4)
        sentinel = torch.ones(2, dtype=torch.int64)
        greedy = eagle_utils._local_key_view(
            logits=values,
            vocab_shard=shard,
            temperatures=None,
            seeds=None,
            splits=1,
        )
        with patch.object(
            sharded_verify, "shard_keys", return_value=sentinel
        ) as shard_keys:
            self.assertIs(runtime.default_local_keys(greedy), sentinel)
            shard_keys.assert_called_once_with(
                values,
                greedy.shard,
                temperatures=None,
                seeds=None,
                splits=1,
            )
            shard_keys.reset_mock()
            temperature = eagle_utils._local_key_view(
                logits=values,
                vocab_shard=shard,
                temperatures=torch.ones(2),
                seeds=None,
                splits=1,
            )
            self.assertIsNone(runtime.default_local_keys(temperature))
            seeded_temperature = eagle_utils._local_key_view(
                logits=values,
                vocab_shard=shard,
                temperatures=torch.ones(2),
                seeds=torch.arange(2, dtype=torch.int64),
                splits=1,
            )
            self.assertIsNone(runtime.default_local_keys(seeded_temperature))
            shard_keys.assert_not_called()

    def test_default_greedy_with_sampling_seed_matches_stock_bitwise(self):
        runtime = _optional_binding_runtime(self)
        logits = torch.tensor([[1.0, 8.0, 3.0], [7.0, 2.0, 4.0]])
        seeds = torch.tensor([123], dtype=torch.int64)
        stock = torch.argmax(logits, dim=-1)
        encoded = torch.zeros((2, 2), dtype=torch.int64)
        group = SimpleNamespace(world_size=1)
        shard = VocabShard(start=0, valid=3, vocab_size=3)

        with (
            patch(
                "sglang.srt.runtime_context.get_parallel",
                return_value=SimpleNamespace(tp_group=group),
            ),
            patch.object(
                sharded_verify, "shard_keys", return_value=encoded
            ) as shard_keys,
            patch.object(
                eagle_utils,
                "reduce_sharded_token_keys",
                return_value=stock.clone(),
            ) as reduce,
        ):
            got = eagle_utils._sharded_tokens(
                logits,
                shard,
                local_keys=runtime.default_local_keys,
                temperatures=None,
                seeds=seeds,
            )

        self.assertTrue(torch.equal(got, stock))
        provider_shard = eagle_utils._local_key_view(
            logits=logits,
            vocab_shard=shard,
            temperatures=None,
            seeds=seeds,
            splits=2,
        ).shard
        shard_keys.assert_called_once_with(
            logits,
            provider_shard,
            temperatures=None,
            seeds=seeds,
            splits=2,
        )
        reduce.assert_called_once_with(encoded, group, splits=2)

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

    def test_sampled_local_keys_recompute_greedy_real_draft_bonus(self):
        from sglang.srt.speculative import spec_utils

        sampling = _sampling_info(greedy=False)
        logits = SimpleNamespace(
            next_token_logits=_verify_logits().clone(),
            next_token_logits_vocab_shard=VocabShard(
                start=0, valid=5, vocab_size=5
            ),
        )
        sampled = torch.full((6,), 4, dtype=torch.int64)
        greedy_bonus = torch.arange(6, dtype=torch.int64)
        simulated_targets = []

        def greedy_check(**kwargs):
            self.assertTrue(
                torch.equal(kwargs["target_predict"], sampled.reshape(2, 3))
            )
            return (
                kwargs["predicts"],
                kwargs["accept_index"],
                kwargs["accept_token_num"],
            )

        def simulate(**kwargs):
            simulated_targets.append(kwargs["target_predict"].clone())
            return kwargs["accept_index"]

        with (
            patch.object(eagle_utils, "_is_cuda", True),
            patch.object(eagle_utils, "_is_cpu", False),
            patch.object(eagle_utils, "_is_hip", False),
            patch.object(eagle_utils, "_is_xpu", False),
            patch.object(eagle_utils, "get_spec", return_value=_runtime_spec()),
            patch.object(
                eagle_utils,
                "_sharded_tokens",
                side_effect=(sampled, greedy_bonus),
            ) as sharded_tokens,
            patch.object(
                eagle_utils, "verify_tree_greedy_func", side_effect=greedy_check
            ),
            patch.object(
                spec_utils,
                "generate_simulated_accept_index_device",
                side_effect=simulate,
            ),
            patch.object(spec_utils, "SIMULATE_ACC_LEN", 1.0),
            patch.object(spec_utils, "SIMULATE_ACC_TOKEN_MODE", "real-draft-token"),
        ):
            eagle_utils.eagle_sample(
                _verify_tree(),
                _verify_batch(sampling),
                logits,
                simulated_acc_len=torch.tensor([1]),
                local_keys=lambda _view: None,
            )

        self.assertEqual(sharded_tokens.call_count, 2)
        self.assertIsNotNone(sharded_tokens.call_args_list[0].kwargs["temperatures"])
        self.assertIsNone(sharded_tokens.call_args_list[1].kwargs.get("temperatures"))
        self.assertEqual(len(simulated_targets), 1)
        self.assertTrue(torch.equal(simulated_targets[0], greedy_bonus.reshape(2, 3)))


class TestShardedReducerContract(CustomTestCase):
    def test_rejects_noncanonical_provider_keys_before_exchange(self):
        group = SimpleNamespace(world_size=2)
        with self.assertRaisesRegex(TypeError, "torch.int64"):
            reduce_sharded_token_keys(torch.zeros(2), group, splits=1)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            reduce_sharded_token_keys(
                torch.arange(8, dtype=torch.int64).view(2, 4)[:, ::2],
                group,
                splits=2,
            )
        with self.assertRaisesRegex(ValueError, "one split"):
            reduce_sharded_token_keys(
                torch.zeros((2, 1), dtype=torch.int64), group, splits=1
            )
        with self.assertRaisesRegex(ValueError, "incompatible"):
            reduce_sharded_token_keys(
                torch.zeros(2, dtype=torch.int64), group, splits=2
            )

    def test_target_uses_rank_uniform_padded_width_for_splits(self):
        logits = torch.zeros((3, 4104))
        shard = VocabShard(start=7 * 4104, valid=4041, vocab_size=32769)
        keys = torch.zeros((3, key_splits(logits.shape[-1])), dtype=torch.int64)
        group = SimpleNamespace(world_size=8)
        with (
            patch.object(sharded_verify, "shard_keys", return_value=keys) as make,
            patch.object(
                sharded_verify,
                "reduce_sharded_token_keys",
                return_value=torch.zeros(3, dtype=torch.int64),
            ) as reduce,
        ):
            sharded_target_tokens(logits, shard, group)
        splits = key_splits(logits.shape[-1])
        make.assert_called_once_with(logits, shard, splits)
        reduce.assert_called_once_with(keys, group, splits=splits)

    def test_local_key_rows_must_match_logits(self):
        logits = torch.ones((2, 4104))
        shard = VocabShard(start=0, valid=4104, vocab_size=4104)
        with (
            patch(
                "sglang.srt.runtime_context.get_parallel",
                return_value=SimpleNamespace(tp_group=object()),
            ),
            self.assertRaisesRegex(ValueError, "expected"),
        ):
            eagle_utils._sharded_tokens(
                logits,
                shard,
                local_keys=lambda view: torch.zeros(
                    (3, view.splits), dtype=torch.int64
                ),
            )


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
        gen = torch.Generator(device="cuda").manual_seed(0)
        for rows in (1, 7, 112, 896):
            full = (
                torch.randn(rows, VOCAB, device="cuda", generator=gen) * 4
            ).bfloat16()
            # ties across shard boundaries, an all-equal row, a row of zeros of both signs
            peak = full.float().max() + 1
            full[0, 5] = full[0, VOCAB - 3] = peak
            if rows > 2:
                full[1].fill_(1.5)
                full[2].zero_()
                full[2, 1::2] = -0.0
            want = torch.argmax(full.float(), dim=-1)
            for tp in (2, 4, 8, 16):
                for split in (False, True):
                    with self.subTest(rows=rows, tp=tp, split=split):
                        got = _sharded_tokens(full, tp, split=split)
                        self.assertTrue(torch.equal(got, want))

    def test_reduce_strided_rank_rows(self):
        # the multimem exchange's layout: a buffer row per logits row, rank r's parts at
        # column parts * r, the buffer wider than world * parts
        gen = torch.Generator(device="cuda").manual_seed(3)
        rows, world, parts, pad = 7, 8, 16, 32
        keys = torch.randint(
            -(2**62), 2**62, (world, rows, parts), device="cuda", generator=gen
        )
        buf = torch.zeros(rows, world * parts + pad, dtype=torch.int64, device="cuda")
        buf[:, : world * parts] = keys.permute(1, 0, 2).reshape(rows, -1)
        want = keys_to_tokens(keys.reshape(-1), world, parts)
        got = _reduce_keys(buf, rows, world, parts, parts, buf.stride(0))
        self.assertTrue(torch.equal(got, want))
        best = keys.amax(dim=(0, 2))
        self.assertTrue(torch.equal(got, 0xFFFFFFFF - (best & 0xFFFFFFFF)))

    def test_padding_columns_never_win(self):
        full = torch.zeros(3, 1024, device="cuda", dtype=torch.bfloat16)
        full[:, 1000:] = 100.0
        shard = VocabShard(start=0, valid=1024, vocab_size=1000)
        got = keys_to_tokens(shard_keys(full, shard), 1)
        self.assertTrue(
            torch.equal(got, torch.zeros(3, dtype=torch.int64, device="cuda"))
        )

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

    def test_provider_keys_reuse_the_generic_exchange(self):
        def encoded(score, token):
            return (score << 32) | (0xFFFFFFFF - token)

        keys = torch.tensor(
            [
                [encoded(10, 5), encoded(8, 1)],
                [encoded(3, 7), encoded(11, 9)],
                [encoded(6, 2), encoded(6, 4)],
            ],
            dtype=torch.int64,
            device="cuda",
        )
        other = torch.tensor(
            [
                [encoded(12, 30), encoded(7, 0)],
                [encoded(10, 3), encoded(9, 2)],
                [encoded(6, 1), encoded(4, 0)],
            ],
            dtype=torch.int64,
            device="cuda",
        )
        gathered = torch.cat((keys.reshape(-1), other.reshape(-1)))
        group = SimpleNamespace(world_size=2)
        group.all_gather_into_tensor = lambda output, _input: output.copy_(gathered)
        with (
            patch.object(sharded_verify, "_ll_keys", return_value=None),
            patch.object(sharded_verify, "_mc_gather_keys", return_value=None),
        ):
            got = reduce_sharded_token_keys(keys, group, splits=2)
        want = torch.tensor([30, 9, 1], dtype=torch.int64, device="cuda")
        self.assertTrue(torch.equal(got, want))


@triton.jit
def _plain_st_b64(addr, val, mask):
    # _ll_st_b64 on one GPU: a plain store into this rank's own buffer
    return tl.inline_asm_elementwise(
        """
        {
            .reg .pred %p0;
            setp.eq.s32 %p0, $3, 1;
            @!%p0 bra end;
            st.relaxed.sys.global.b64 [$1], $2;
            end:
        }
        """,
        "=r,l,l,r",
        args=[addr, val, mask.to(tl.int32)],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestLLKeys(CustomTestCase):
    """_ll_keys_kernel as rank 0 of WORLD ranks on one GPU (its multimem store made a plain
    one), the other ranks' words written by the test. Each launch is followed at once by
    another kernel and nothing syncs the host until the end. Every row must get the max over
    all ranks' keys. A warp once left the poll on a stale block-wide count and took words a
    slot held from an earlier launch: wrong tokens for a block of rows (168 of 600 launches
    at 56 rows on one rank)."""

    LAUNCHES = 600

    def _check(self, world, rows, late):
        maxr = sharded_verify._LL_MAX_ROWS
        gen = torch.Generator(device="cuda").manual_seed(world * 100 + rows)
        buf = torch.zeros(
            sharded_verify._LL_SLOTS * world * maxr * 2,
            dtype=torch.int64,
            device="cuda",
        )
        epoch = torch.zeros(2, dtype=torch.int32, device="cuda")
        writer = torch.cuda.Stream()
        # load the writer's kernels now: lazy loading them would wait for a spinning poll
        torch.cuda._sleep(1)
        torch.empty_like(buf)[torch.arange(2, device="cuda")] = 0
        launches = []
        for i in range(self.LAUNCHES):
            keys = torch.randint(
                -(2**62), 2**62, (world, rows, 2), device="cuda", generator=gen
            )
            want = 0xFFFFFFFF - (keys.amax(dim=(0, 2)) & 0xFFFFFFFF)
            tag, slot = i + 1, (i + 1) % 3
            at = torch.cat(
                [
                    ((slot * world + q) * maxr + torch.arange(rows, device="cuda")) * 2
                    for q in range(1, world)
                ]
                or [torch.empty(0, dtype=torch.int64, device="cuda")]
            )
            others = keys[1:].amax(dim=2).reshape(-1)
            lo = (others & 0xFFFFFFFF) | (tag << 32)
            hi = ((others >> 32) & 0xFFFFFFFF) | (tag << 32)
            if late:
                # the other ranks' words land ~10 us after this launch begins to poll
                writer.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(writer):
                    torch.cuda._sleep(20_000)
                    buf[at] = lo
                    buf[at + 1] = hi
            else:
                buf[at] = lo
                buf[at + 1] = hi
            tokens = torch.full((rows,), -5, dtype=torch.int64, device="cuda")
            sharded_verify._ll_keys_kernel[(1,)](
                keys[0].contiguous(),
                2,
                rows,
                buf.data_ptr(),
                buf.data_ptr(),
                epoch,
                tokens,
                RANK=0,
                WORLD=world,
                MAXR=maxr,
                BLOCK_R=max(16, triton.next_power_of_2(rows)),
                BLOCK_P=2,
                KEY_FLOOR=sharded_verify._KEY_FLOOR,
                SPIN_LIMIT=1 << 22,
                num_warps=4,
            )
            torch.empty(rows, dtype=torch.int32, device="cuda").fill_(3)
            launches.append((tokens, want))
        torch.cuda.synchronize()
        wrong = sum(not torch.equal(t, w) for t, w in launches)
        self.assertEqual(wrong, 0, f"{wrong} of {self.LAUNCHES} launches")
        self.assertEqual(int(epoch[0]), self.LAUNCHES)
        self.assertEqual(int(epoch[1]), self.LAUNCHES % sharded_verify._LL_SLOTS)

    def _check_epoch_wrap(self, last_tag):
        rows, parts = 7, 2
        maxr = sharded_verify._LL_MAX_ROWS
        buf = torch.zeros(
            sharded_verify._LL_SLOTS * maxr * 2,
            dtype=torch.int64,
            device="cuda",
        )
        state = torch.tensor(
            [
                last_tag if last_tag < 2**31 else last_tag - 2**32,
                last_tag % sharded_verify._LL_SLOTS,
            ],
            dtype=torch.int32,
            device="cuda",
        )
        launches = []
        gen = torch.Generator(device="cuda").manual_seed(last_tag)
        for _ in range(4):
            logits = torch.randn(
                (rows, 257), device="cuda", dtype=torch.bfloat16, generator=gen
            )
            want = torch.argmax(logits.float(), dim=-1)
            keys = shard_keys(
                logits,
                VocabShard(start=0, valid=257, vocab_size=257),
                splits=parts,
            )
            tokens = torch.full((rows,), -5, dtype=torch.int64, device="cuda")
            sharded_verify._ll_keys_kernel[(1,)](
                keys,
                parts,
                rows,
                buf.data_ptr(),
                buf.data_ptr(),
                state,
                tokens,
                RANK=0,
                WORLD=1,
                MAXR=maxr,
                BLOCK_R=16,
                BLOCK_P=2,
                KEY_FLOOR=sharded_verify._KEY_FLOOR,
                SPIN_LIMIT=1 << 10,
                num_warps=4,
            )
            launches.append((tokens, want))

        # Do not synchronize between launches: the epoch is device-owned and this
        # path must remain capturable without a GPU-to-CPU round trip.
        torch.cuda.synchronize()
        for got, want in launches:
            self.assertTrue(torch.equal(got, want))
        expected_tag = last_tag
        for _ in range(4):
            expected_tag = (expected_tag + 1) & 0xFFFFFFFF
            if expected_tag == 0:
                expected_tag = 1
        self.assertEqual(int(state[0]) & 0xFFFFFFFF, expected_tag)
        self.assertEqual(
            int(state[1]),
            (last_tag % sharded_verify._LL_SLOTS + 4) % sharded_verify._LL_SLOTS,
        )

    def test_every_row_gets_every_ranks_max(self):
        with patch.object(sharded_verify, "_ll_st_b64", _plain_st_b64):
            for world in (1, 2, 8):
                for rows in (7, 28, 56, 63):
                    for late in (False, True) if world > 1 else (False,):
                        with self.subTest(world=world, rows=rows, late=late):
                            self._check(world, rows, late)

    def test_epoch_wrap_keeps_tags_nonzero_and_slots_independent(self):
        with patch.object(sharded_verify, "_ll_st_b64", _plain_st_b64):
            for last_tag in (2**31 - 2, 0xFFFFFFFE):
                with self.subTest(last_tag=hex(last_tag)):
                    self._check_epoch_wrap(last_tag)


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


if __name__ == "__main__":
    unittest.main()
