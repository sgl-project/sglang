"""Vocabulary-shard equivalence, provider verify seams and stock fallback coverage."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.kernels.plugin_kernels.nvidia import spec
from sglang.srt.environ import envs
from sglang.srt.kernel_plugins.nvidia import binding
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.runtime_context import reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative import eagle_utils, eagle_worker_common, sharded_verify
from sglang.srt.speculative.sharded_verify import (
    VocabShard,
    gather_full_logits,
    key_splits,
    keys_to_tokens,
    shard_keys,
    sharded_target_tokens,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.parallel_groups import publish
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
        step_penalties=None,
        logit_bias=None,
        sampling_seed=None,
        is_all_greedy=greedy,
        need_top_k_sampling=False,
        need_top_p_sampling=False,
        need_min_p_sampling=False,
        watermark=None,
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


class _BroadcastGroup:
    def __init__(self, world_size):
        self.world_size, self.sources = world_size, []

    def broadcast(self, tensor, src):
        self.sources.append(src)


class TestGatheredPolicy(CustomTestCase):
    """A provider target sampler on gathered verify logits."""

    def _sample(
        self,
        sampling,
        *,
        gathered_tokens,
        world_size=1,
        fused=False,
    ):
        """eagle_sample on CPU gathered logits; returns its outputs, the target the
        greedy check consumed, the stock sampler's calls and the TP group."""
        logits = SimpleNamespace(
            next_token_logits=_verify_logits().clone(),
            next_token_logits_vocab_shard=None,
        )
        group = _BroadcastGroup(world_size)
        targets, stock = [], []

        def greedy_check(**kwargs):
            targets.append(kwargs["target_predict"].clone())
            kwargs["accept_token_num"].zero_()
            return (
                kwargs["predicts"],
                kwargs["accept_index"],
                kwargs["accept_token_num"],
            )

        def chain_accept(**kwargs):
            if kwargs["finish"]:
                return kwargs["num_correct_drafts"] + 1, kwargs["accept_index"]
            targets.append(kwargs["target_predict"].clone())
            kwargs["predict"].zero_()
            kwargs["accept_index"].fill_(-1)
            kwargs["num_correct_drafts"].zero_()

        def stock_sample(**kwargs):
            # outputs that depend on the coins, so RNG use shows in them
            stock.append(kwargs)
            draws = (kwargs["uniform_samples"].reshape(-1) * 1000).int()
            kwargs["predicts"].copy_(draws)
            kwargs["accept_index"].fill_(-1)
            kwargs["accept_token_num"].zero_()

        with (
            patch.object(eagle_utils, "_is_cuda", True),
            patch.object(eagle_utils, "_is_cpu", False),
            patch.object(eagle_utils, "_is_hip", False),
            patch.object(eagle_utils, "_is_xpu", False),
            patch.object(eagle_utils, "get_spec", return_value=_runtime_spec()),
            patch.object(eagle_utils, "verify_tree_greedy_func", greedy_check),
            patch.object(eagle_utils, "_chain_accept_applies", return_value=fused),
            patch.object(eagle_utils, "_chain_accept", side_effect=chain_accept),
            patch.object(
                eagle_utils,
                "tree_speculative_sampling_target_only",
                side_effect=stock_sample,
            ),
            patch(
                "sglang.srt.runtime_context.get_parallel",
                return_value=SimpleNamespace(tp_group=group, attn_tp_group=group),
            ),
            patch(
                "sglang.srt.layers.dp_attention.is_dp_attention_enabled",
                return_value=True,
            ),
        ):
            outputs = eagle_utils.eagle_sample(
                _verify_tree(),
                _verify_batch(sampling),
                logits,
                gathered_tokens=gathered_tokens,
            )
        return outputs, targets, stock, group

    def test_provider_tokens_are_the_target_of_the_chain_check(self):
        """Temperature-only rows: the view holds the gathered logits and each row's
        temperature, and the provider's tokens are what the drafts are checked
        against; ranks of an attention-TP group still take rank 0's result."""
        sampling = _sampling_info(greedy=False)
        sampling.temperatures = torch.tensor([[0.5], [2.0]])
        tokens = torch.tensor([4, 0, 1, 2, 3, 3])
        views = []

        def policy(view):
            views.append(view)
            return tokens

        for fused, world_size in ((False, 1), (True, 1), (True, 2), (False, 2)):
            with self.subTest(fused=fused, world_size=world_size):
                views.clear()
                _, targets, stock, group = self._sample(
                    sampling, gathered_tokens=policy, world_size=world_size, fused=fused
                )
                self.assertEqual(len(views), 1)
                self.assertTrue(torch.equal(views[0].values, _verify_logits()))
                self.assertTrue(
                    torch.equal(
                        views[0].temperatures,
                        torch.tensor([0.5, 0.5, 0.5, 2.0, 2.0, 2.0]),
                    )
                )
                self.assertEqual(len(targets), 1)
                self.assertTrue(torch.equal(targets[0], tokens.view(2, 3)))
                self.assertEqual(stock, [])
                self.assertEqual(set(group.sources), set() if world_size == 1 else {0})

    def test_decline_keeps_the_stock_sampler_and_its_rng(self):
        """A provider that declines sees the RNG untouched, and the stock sampler then
        draws exactly as it does with no provider."""
        sampling = _sampling_info(greedy=False)
        events = []

        def decline(_view):
            events.append(torch.random.get_rng_state().clone())
            return None

        torch.manual_seed(0)
        seeded = torch.random.get_rng_state().clone()
        runs = []
        for policy in (decline, None):
            torch.random.set_rng_state(seeded)
            outputs, targets, stock, _ = self._sample(sampling, gathered_tokens=policy)
            runs.append((outputs, torch.random.get_rng_state().clone()))
            self.assertEqual(targets, [])
            self.assertEqual(len(stock), 1)
        self.assertEqual(len(events), 1)
        self.assertTrue(torch.equal(events[0], seeded))
        (declined, declined_rng), (plain, plain_rng) = runs
        self.assertTrue(torch.equal(declined_rng, plain_rng))
        for got, want in zip(declined, plain):
            self.assertTrue(torch.equal(got, want))

    def test_only_temperature_rows_of_a_top1_chain_reach_the_provider(self):
        """Every round the stock rule or a logits-editing step needs stays stock;
        logprobs read the logits after verify, so they do not exclude a round."""
        # (case, admitted, sampling-info fields, server-spec fields, call arguments)
        cases = [
            ("temperature-only", True, {}, {}, {}),
            ("logprobs", True, {}, {}, dict(return_logprob=True)),
            ("greedy", False, {}, {}, dict(greedy=True)),
            ("compact top-k", False, {}, {}, dict(sparse=True)),
            ("vocab shard", False, {}, {}, dict(vocab_shard=VocabShard(0, 5, 5))),
            ("top-k", False, dict(need_top_k_sampling=True), {}, {}),
            ("logit bias", False, dict(logit_bias=torch.zeros(2, 5)), {}, {}),
            ("request seed", False, dict(sampling_seed=torch.ones(2)), {}, {}),
            ("grammar", False, {}, {}, dict(grammar_mask=object())),
            ("watermark", False, {}, {}, dict(watermark=True)),
            ("custom logit processor", False, {}, {}, dict(custom_processor=True)),
            ("rejection", False, {}, dict(speculative_use_rejection_sampling=True), {}),
            ("tree", False, {}, {}, dict(topk=2)),
            ("not CUDA", False, {}, {}, dict(cuda=False)),
        ]
        for name, admitted, info, spec_fields, call in cases:
            with self.subTest(case=name):
                sampling = _sampling_info(greedy=False)
                sampling.__dict__.update(info)
                server_spec = _runtime_spec()
                server_spec.__dict__.update(spec_fields)
                batch = _verify_batch(sampling)
                batch.return_logprob = call.get("return_logprob", False)
                verify_input = _verify_tree()
                verify_input.tree_topk = call.get("topk", 1)
                with (
                    patch.object(eagle_utils, "_is_cuda", call.get("cuda", True)),
                    patch.object(eagle_utils, "get_spec", return_value=server_spec),
                ):
                    applies = eagle_utils._gathered_sampling_applies(
                        greedy=call.get("greedy", False),
                        sparse=call.get("sparse", False),
                        vocab_shard=call.get("vocab_shard"),
                        verify_input=verify_input,
                        batch=batch,
                        grammar_mask=call.get("grammar_mask"),
                        watermark=call.get("watermark", False),
                        has_custom_logit_processor=call.get("custom_processor", False),
                    )
                self.assertEqual(applies, admitted)

    def test_malformed_tokens_are_refused(self):
        sampling = _sampling_info(greedy=False)
        for tokens, error in (
            ([1, 2, 3, 4, 5, 6], TypeError),
            (torch.zeros(6, dtype=torch.int32), ValueError),
            (torch.zeros(2, 3, dtype=torch.int64), ValueError),
        ):
            with self.subTest(tokens=tokens), self.assertRaises(error):
                self._sample(sampling, gathered_tokens=lambda _view: tokens)


class _LoraHead(nn.Linear):
    def set_lora(self, *args):
        pass

    def apply_lora(self, *args):
        pass


class _QuantMethod:
    def apply(self, layer, hidden_states, bias=None):
        return torch.zeros(hidden_states.shape[0], layer.weight.shape[0])


class _OwnHeadProcessor(LogitsProcessor):
    # the override signature models with their own LM head use
    def _compute_lm_head(
        self, hidden_states, lm_head, embedding_bias=None, *, out=None
    ):
        return super()._compute_lm_head(hidden_states, lm_head, embedding_bias, out=out)


class TestVerifyShardHead(CustomTestCase):
    """A provider head over the sharded verify's LM-head shard."""

    def setUp(self):
        super().setUp()
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy", device="cpu"), role="test")
        torch.manual_seed(0)
        self.hidden = torch.randn(3, 4).bfloat16()
        self.head = nn.Linear(4, 8, bias=False, dtype=torch.bfloat16)
        self.head.requires_grad_(False)
        self.want = torch.matmul(self.hidden, self.head.weight.T)

    def _logits(self, *, shard_head, head=None, backing=None, configure=None):
        processor = LogitsProcessor(SimpleNamespace(vocab_size=8), skip_all_gather=True)
        processor.enable_sharded_verify(shard_head=shard_head)
        if configure is not None:
            configure(processor)
        metadata = SimpleNamespace(next_token_logits_buffer=backing)
        return processor._get_shard_logits(
            self.hidden, self.head if head is None else head, metadata
        )

    def test_head_serves_the_plain_gemm_into_the_borrowed_buffer(self):
        """The head sees the GEMM's operands and the graph's borrowed buffer, and its
        logits replace the GEMM's; a head with a logit scale keeps the GEMM."""
        views = []

        def head(view):
            views.append(view)
            return view.out.fill_(1.5)

        backing = torch.empty(3 * 8, dtype=torch.bfloat16)
        logits = self._logits(shard_head=head, backing=backing)
        (view,) = views
        self.assertIs(view.weight, self.head.weight)
        self.assertTrue(torch.equal(view.hidden_states, self.hidden))
        self.assertEqual(view.out.data_ptr(), backing.data_ptr())
        self.assertEqual(logits.data_ptr(), backing.data_ptr())
        self.assertTrue(torch.all(logits == 1.5))

        views.clear()
        logits = self._logits(
            shard_head=head,
            backing=backing,
            configure=lambda p: setattr(p, "logit_scale", 2.0),
        )
        self.assertEqual(views, [])
        self.assertTrue(torch.equal(logits, self.want * 2.0))

    def test_declined_or_absent_head_keeps_the_stock_gemm(self):
        """Value-identical to the GEMM with no buffer, a borrowed buffer of the
        weight dtype, or one viewing an fp32 backing storage."""
        decline = lambda view: None
        for backing in (
            None,
            torch.empty(3 * 8, dtype=torch.bfloat16),
            torch.empty(3 * 8, dtype=torch.float32),
        ):
            for shard_head in (None, decline):
                with self.subTest(
                    backing=None if backing is None else backing.dtype,
                    head=shard_head is not None,
                ):
                    logits = self._logits(shard_head=shard_head, backing=backing)
                    self.assertEqual(logits.dtype, torch.bfloat16)
                    self.assertTrue(torch.equal(logits, self.want))

    def test_head_is_offered_only_the_plain_gemm(self):
        """FP32 heads, quantized heads, LoRA heads and a subclass's own LM head keep
        their own path."""

        def fp32(processor):
            processor.use_fp32_lm_head = True

        quantized = nn.Linear(4, 8, bias=False, dtype=torch.bfloat16)
        quantized.quant_method = _QuantMethod()
        lora = _LoraHead(4, 8, bias=False, dtype=torch.bfloat16)
        for name, kwargs in (
            ("fp32", dict(configure=fp32)),
            ("quantized", dict(head=quantized)),
            ("lora", dict(head=lora.requires_grad_(False))),
        ):
            with self.subTest(head=name):
                self._logits(
                    shard_head=lambda view: self.fail("head offered a non-plain GEMM"),
                    **kwargs,
                )
        for shard_head in (None, lambda view: self.fail("head offered an own head")):
            with self.subTest(head="subclass", bound=shard_head is not None):
                processor = _OwnHeadProcessor(
                    SimpleNamespace(vocab_size=8), skip_all_gather=True
                )
                processor.enable_sharded_verify(shard_head=shard_head)
                logits = processor._get_shard_logits(
                    self.hidden,
                    self.head,
                    SimpleNamespace(next_token_logits_buffer=None),
                )
                self.assertTrue(torch.equal(logits, self.want))

    def test_binding_hands_a_provider_head_to_the_target_processor(self):
        head = lambda view: None

        def factory(stock):
            return spec.SpeculativeRuntimeHooks(
                post_verify=stock.post_verify, verify_shard_head=head
            )

        # an outer target class the sharded verify admits
        model = type("Qwen3_5MoeForCausalLM", (nn.Module,), {})()
        model.logits_processor = LogitsProcessor(
            SimpleNamespace(vocab_size=8), skip_all_gather=True
        )
        model.__dict__["_megakernel"] = binding._InstalledMegakernel(
            built=SimpleNamespace(speculative_runtime_hooks=factory),
            binding=None,
            prepare=None,
        )
        worker = SimpleNamespace(model_runner=SimpleNamespace(model=model))
        with (
            envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(True),
            patch.object(binding, "_gather_tp", side_effect=lambda v, group=None: [v]),
        ):
            eagle_worker_common.bind_eagle_runtime_hooks(
                worker, supports_sharded_verify=True
            )
        self.assertIs(model.logits_processor._verify_shard_head, head)

    def test_malformed_head_logits_are_refused(self):
        for logits in (torch.zeros(3, 7).bfloat16(), torch.zeros(3, 8)):
            with self.subTest(shape=tuple(logits.shape), dtype=logits.dtype):
                with self.assertRaisesRegex(ValueError, "verify-shard head"):
                    self._logits(shard_head=lambda view: logits)
        # the borrowed buffer is what verify graphs read; a head must write into it
        with self.assertRaisesRegex(ValueError, "written into out"):
            self._logits(
                shard_head=lambda view: torch.zeros(3, 8).bfloat16(),
                backing=torch.empty(3 * 8, dtype=torch.bfloat16),
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
