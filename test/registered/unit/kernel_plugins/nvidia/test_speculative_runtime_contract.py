"""CPU contracts for provider-neutral speculative runtime strategies."""

import typing

import msgspec
import torch

from sglang.kernels.plugin_kernels.nvidia import spec
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, stage="base-a", runner_config="cpu")


class TestSpeculativeRuntimeContract(CustomTestCase):
    def test_frozen_field_lists(self):
        expected = {
            spec.VocabShardView: ("start", "valid", "vocab_size"),
            spec.ShardedLogitsView: (
                "values",
                "shard",
                "temperatures",
                "seeds",
                "splits",
            ),
            spec.VerifyTreeView: (
                "draft_tokens",
                "draft_probabilities",
                "retrieve_index",
                "next_child",
                "next_sibling",
                "custom_mask",
                "positions",
                "tokens_per_request",
                "max_depth",
                "topk",
            ),
            spec.VerifySamplingView: (
                "sequence_lengths",
                "temperatures",
                "top_ks",
                "top_ps",
                "min_ps",
                "additive_penalties",
                "scaling_penalties",
                "logit_bias",
                "sampling_seed",
                "is_all_greedy",
                "need_top_k_sampling",
                "need_top_p_sampling",
                "need_min_p_sampling",
                "watermark_keys",
                "watermark_context_windows",
                "watermark_enabled",
                "has_watermark_candidates",
                "max_top_k",
                "device",
            ),
            spec.PostVerifyIO: (
                "logits",
                "verify_tree",
                "sampling",
                "grammar_mask",
                "watermark",
                "watermark_full_mask",
                "request_pool_indices",
                "return_logprob",
                "is_idle",
                "simulated_accept_length",
                "simulated_accept_length_rng",
                "sparse_target_max_top_k",
                "sparse_bonus_argmax",
            ),
            spec.PostVerifyOutput: (
                "token_ids",
                "accepted_lengths",
                "accepted_indices",
            ),
            spec.PostVerifyStrategy: ("run", "local_keys"),
            spec.SpeculativeRuntimeHooks: ("post_verify",),
        }
        for contract, fields in expected.items():
            with self.subTest(contract=contract.__name__):
                self.assertTrue(issubclass(contract, msgspec.Struct))
                self.assertEqual(
                    tuple(field.name for field in msgspec.structs.fields(contract)),
                    fields,
                )
                self.assertTrue(contract.__struct_config__.frozen)

    def test_frozen_contract_field_types_resolve_exactly_without_any(self):
        tensor = torch.Tensor
        optional = typing.Optional
        expected = {
            spec.VocabShardView: {"start": int, "valid": int, "vocab_size": int},
            spec.ShardedLogitsView: {
                "values": tensor,
                "shard": spec.VocabShardView,
                "temperatures": optional[tensor],
                "seeds": optional[tensor],
                "splits": int,
            },
            spec.VerifyTreeView: {
                "draft_tokens": tensor,
                "draft_probabilities": optional[tensor],
                "retrieve_index": tensor,
                "next_child": tensor,
                "next_sibling": tensor,
                "custom_mask": optional[tensor],
                "positions": optional[tensor],
                "tokens_per_request": int,
                "max_depth": int,
                "topk": int,
            },
            spec.VerifySamplingView: {
                "sequence_lengths": tensor,
                "temperatures": tensor,
                "top_ks": tensor,
                "top_ps": tensor,
                "min_ps": tensor,
                "additive_penalties": optional[tensor],
                "scaling_penalties": optional[tensor],
                "logit_bias": optional[tensor],
                "sampling_seed": optional[tensor],
                "is_all_greedy": bool,
                "need_top_k_sampling": bool,
                "need_top_p_sampling": bool,
                "need_min_p_sampling": bool,
                "watermark_keys": optional[tensor],
                "watermark_context_windows": optional[tensor],
                "watermark_enabled": optional[tensor],
                "has_watermark_candidates": bool,
                "max_top_k": optional[int],
                "device": torch.device,
            },
            spec.PostVerifyIO: {
                "logits": spec.VerifyLogitsView,
                "verify_tree": spec.VerifyTreeView,
                "sampling": spec.VerifySamplingView,
                "grammar_mask": optional[spec.GrammarMaskView],
                "watermark": optional[spec.WatermarkStateView],
                "watermark_full_mask": bool,
                "request_pool_indices": tensor,
                "return_logprob": bool,
                "is_idle": bool,
                "simulated_accept_length": optional[tensor],
                "simulated_accept_length_rng": optional[torch.Generator],
                "sparse_target_max_top_k": optional[int],
                "sparse_bonus_argmax": bool,
            },
            spec.PostVerifyOutput: {
                "token_ids": tensor,
                "accepted_lengths": tensor,
                "accepted_indices": tensor,
            },
            spec.PostVerifyStrategy: {
                "run": spec.VerifyRunner,
                "local_keys": spec.LocalKeysFn,
            },
            spec.SpeculativeRuntimeHooks: {
                "post_verify": spec.PostVerifyStrategy,
            },
        }
        for contract, types in expected.items():
            with self.subTest(contract=contract.__name__):
                resolved = typing.get_type_hints(contract)
                self.assertEqual(resolved, types)
                self.assertNotIn("typing.Any", repr(resolved))

        self.assertEqual(
            typing.get_type_hints(spec.VerifyLogitsView),
            {
                "next_token_logits": tensor,
                "next_token_logits_vocab_shard": optional[spec.VocabShardView],
            },
        )

        self.assertEqual(
            spec.LocalKeysFn,
            typing.Callable[[spec.ShardedLogitsView], typing.Optional[torch.Tensor]],
        )
        self.assertEqual(
            spec.HooksFactory,
            typing.Callable[
                [spec.SpeculativeRuntimeHooks], spec.SpeculativeRuntimeHooks
            ],
        )

    def test_local_key_output_contract_and_logits_writeback_are_documented(self):
        # The runtime PR owns ``sharded_verify``.  When both PRs are present its
        # public alias must still resolve to this one canonical ABI type; the
        # standalone binding PR can validate the provider contract by itself.
        try:
            from sglang.srt.speculative.sharded_verify import VocabShard
        except ModuleNotFoundError as exc:
            if exc.name != "sglang.srt.speculative.sharded_verify":
                raise
        else:
            self.assertIs(VocabShard, spec.VocabShardView)

        key_contract = " ".join(spec.ShardedLogitsView.__doc__.split())
        for phrase in (
            "floating ``[rows, local_cols]``",
            "unit last stride",
            "contiguous float32 ``[rows]``",
            "contiguous int64 ``[batch]``",
            "positive runtime-selected",
            "contiguous int64",
            "high 32 bits",
            "0xffffffff - global_token_id",
            "one split returns ``[rows]``",
            "``[rows, splits]``",
            "``splits`` must be used unchanged",
            "identical on every TP rank",
            "rank-uniform predicates",
            "``shard.start``/``shard.valid`` alone",
            "collectively establish the complete descriptor",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, key_contract)
        logits_contract = spec.VerifyLogitsView.__doc__
        self.assertIn("writes the full tensor back", logits_contract)
        self.assertIn("clears ``next_token_logits_vocab_shard``", logits_contract)

    def test_strategy_passes_the_local_policy_to_runtime(self):
        output = spec.PostVerifyOutput(
            token_ids=torch.tensor([1]),
            accepted_lengths=torch.tensor([1]),
            accepted_indices=torch.tensor([0]),
        )
        local_keys = lambda view: view.values.argmax(-1)
        calls = []

        def run(io, policy):
            calls.append((io, policy))
            return output

        io = object()
        strategy = spec.PostVerifyStrategy(run=run, local_keys=local_keys)
        self.assertIs(strategy(io), output)
        self.assertEqual(calls, [(io, local_keys)])

    def test_built_provider_exposes_optional_factory(self):
        hints = typing.get_type_hints(spec.BuiltMegakernel)
        self.assertEqual(
            hints["speculative_runtime_hooks"], typing.Optional[spec.HooksFactory]
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
