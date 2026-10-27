# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The DeepSpec Authors
# See python/sglang/srt/speculative/ubtree_components/LICENSE.

"""UBTree configuration, selector, and tree topology contracts."""

import inspect
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_registry import (
    CustomSpecAlgo,
    _assert_custom_spec_algo_conforms,
)
from sglang.srt.speculative.ubtree_components.config import (
    parse_ubtree_config,
    resolve_verify_width,
    validate_runtime_config,
)
from sglang.srt.speculative.ubtree_components.selector import UBTreeSelector
from sglang.srt.speculative.ubtree_components.tree import (
    build_sglang_full_mask,
    build_ubtree_batch,
    compact_tree_values_to_front,
)
from sglang.srt.speculative.ubtree_components.validation import validate_ubtree_request
from sglang.srt.speculative.ubtree_components.vocab import vocab_parallel_topk
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestUBTreeKernelCompatibility(CustomTestCase):
    def test_shared_kernels_preserve_legacy_positional_arguments(self):
        from sglang.kernels.ops.attention.linear.fused_sigmoid_gating_recurrent import (
            fused_sigmoid_gating_delta_rule_update,
        )
        from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update

        # These are the pre-UBTree positional call contracts. Supplying the tail
        # positionally must still bind to padding / ReplaySSM, not tree options.
        contracts = (
            (
                causal_conv1d_update,
                "x conv_state weight bias activation cache_seqlens conv_state_indices "
                "num_accept_tokens intermediate_conv_window intermediate_state_indices "
                "retrieve_next_token retrieve_next_sibling retrieve_parent_token "
                "pad_slot_id metadata validate_data",
                {"direct_parent": True, "save_tree_raw_only": True},
            ),
            (
                fused_sigmoid_gating_delta_rule_update,
                "A_log a dt_bias softplus_beta softplus_threshold q k v b "
                "initial_state_source initial_state_indices scale use_qk_l2norm_in_kernel "
                "cu_seqlens is_kda lower_bound disable_state_update "
                "intermediate_states_buffer intermediate_state_indices cache_steps "
                "retrieve_parent_token cache_ring replayssm_rawv replayssm_rawk "
                "replayssm_g replayssm_beta match_cutedsl_decode",
                {"tree_depths": object(), "force_factorized_tree": True},
            ),
        )
        for function, names, tree_args in contracts:
            with self.subTest(function=function.__name__):
                names = names.split()
                values = [object() for _ in names]
                signature = inspect.signature(function)
                bound = signature.bind(*values, **tree_args)
                for name, value in zip(names, values):
                    self.assertIs(bound.arguments[name], value)
                for name, value in tree_args.items():
                    self.assertIs(bound.arguments[name], value)
                positional = [
                    name
                    for name, parameter in signature.parameters.items()
                    if parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
                ]
                self.assertEqual(positional, names)
                with self.assertRaises(TypeError):
                    signature.bind(*values, object())


def _config():
    return SimpleNamespace(
        vocab_size=17,
        hidden_size=8,
        block_size=5,
        ubtree_config={
            "rank": 4,
            "candidate_k": 3,
            "tree_budget": 5,
            "tree_supertree_width": 3,
            "factorizer_hidden_width": 6,
            "target_embedding_dim": 8,
            "depth_condition_count": 4,
            "objective": "kl",
            "adds_unary_logits": True,
            "token_factorizer": "target_embedding_mlp",
            "hidden_projection_type": "shared",
            "tree_depth_bonus": 0.0,
        },
    )


def _lookup_config():
    config = _config()
    config.ubtree_config = {
        **config.ubtree_config,
        "token_factorizer": "lookup",
        "factorizer_hidden_width": 0,
        "target_embedding_dim": 0,
    }
    return config


class TestUBTreeComponents(CustomTestCase):
    def test_config_separates_draft_width_from_verify_width(self):
        config = parse_ubtree_config(_config())
        assert config.block_size == 5
        assert config.verify_node_count == 6

    def test_lookup_selector_has_checkpoint_compatible_five_tensor_layout(self):
        selector = UBTreeSelector(_lookup_config())
        assert selector.tied_token_factorizer is None
        assert set(selector.state_dict()) == {
            "hidden_projection.weight",
            "predecessor_codebook.weight",
            "successor_codebook.weight",
            "depth_conditioner.unary_log_scale",
            "depth_conditioner.correction_log_scale",
        }

        hidden = torch.randn(2, 4, 8)
        candidate_ids = torch.randint(0, 17, (2, 4, 3))
        unary = torch.randn(2, 4, 3)
        edges = selector.build_edge_log_probs(
            hidden,
            torch.tensor([1, 2]),
            candidate_ids,
            unary_candidate_logits=unary,
        )
        assert edges.shape == (2, 4, 3, 3)
        torch.testing.assert_close(edges.exp().sum(-1), torch.ones(2, 4, 3))

    def test_selector_cache_can_be_materialized_through_parallel_embedding_api(self):
        class FakeParallelEmbedding(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = torch.nn.Parameter(weight)

            def forward(self, tokens):
                return torch.nn.functional.embedding(tokens, self.weight)

        torch.manual_seed(11)
        direct = UBTreeSelector(_config())
        parallel = UBTreeSelector(_config())
        parallel.load_state_dict(direct.state_dict())
        embedding = torch.randn(17, 8)
        direct.materialize_cache(embedding)
        parallel.materialize_cache_from_embedding(
            FakeParallelEmbedding(embedding.clone()),
            device=torch.device("cpu"),
            chunk_size=5,
        )
        tokens = torch.arange(17)
        torch.testing.assert_close(
            direct.tied_token_factorizer.factors(tokens, role="predecessor"),
            parallel.tied_token_factorizer.factors(tokens, role="predecessor"),
        )
        torch.testing.assert_close(
            direct.tied_token_factorizer.factors(tokens, role="successor"),
            parallel.tied_token_factorizer.factors(tokens, role="successor"),
        )

    def test_selector_cache_gathers_factorized_vocab_shards(self):
        torch.manual_seed(19)
        direct = UBTreeSelector(_config())
        parallel = UBTreeSelector(_config())
        parallel.load_state_dict(direct.state_dict())
        embedding = torch.randn(18, 8)
        direct.materialize_cache(embedding[:17])

        pred_mlp = parallel.tied_token_factorizer.predecessor_mlp
        succ_mlp = parallel.tied_token_factorizer.successor_mlp
        remote_pred = pred_mlp(embedding[9:]).detach()
        remote_succ = succ_mlp(embedding[9:]).detach()

        class FakeGroup:
            world_size = 2
            calls = 0

            def all_gather_into_tensor(self, output, local):
                remote = remote_pred if self.calls == 0 else remote_succ
                self.calls += 1
                output.copy_(torch.cat([local, remote]))

        sharded_embedding = SimpleNamespace(
            weight=torch.nn.Parameter(embedding[:9].clone()),
            shard_indices=SimpleNamespace(num_org_elements_padded=9),
        )
        parallel.materialize_cache_from_vocab_parallel_embedding(
            sharded_embedding,
            tp_group=FakeGroup(),
            chunk_size=4,
        )
        tokens = torch.arange(17)
        torch.testing.assert_close(
            direct.tied_token_factorizer.factors(tokens, role="predecessor"),
            parallel.tied_token_factorizer.factors(tokens, role="predecessor"),
        )
        torch.testing.assert_close(
            direct.tied_token_factorizer.factors(tokens, role="successor"),
            parallel.tied_token_factorizer.factors(tokens, role="successor"),
        )

    def test_vocab_parallel_topk_merges_tensor_parallel_shards(self):
        hidden = torch.tensor([[1.0, 0.5], [-0.25, 2.0]])
        full_weight = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.5, 0.5],
                [-1.0, 0.0],
                [0.0, -1.0],
                [2.0, 1.0],
                [1.0, 2.0],
                [-2.0, 1.0],
            ]
        )
        expected_values, expected_ids = torch.nn.functional.linear(
            hidden, full_weight
        ).topk(3, dim=-1)

        local_weight = full_weight[:4]
        remote_values, remote_ids = torch.nn.functional.linear(
            hidden, full_weight[4:]
        ).topk(3, dim=-1)
        remote_ids += 4

        class FakeGroup:
            world_size = 2

            def all_gather_into_tensor(self, output, local):
                remote = remote_values if output.dtype.is_floating_point else remote_ids
                output.copy_(torch.cat([local, remote.reshape(-1)]))

        head = SimpleNamespace(
            weight=local_weight,
            shard_indices=SimpleNamespace(
                num_org_elements=4,
                org_vocab_start_index=0,
            ),
        )
        values, ids = vocab_parallel_topk(
            hidden,
            head,
            k=3,
            vocab_size=8,
            tp_group=FakeGroup(),
        )
        torch.testing.assert_close(values, expected_values)
        assert torch.equal(ids, expected_ids)

    def test_batched_tree_is_prefix_closed_and_has_sglang_links(self):
        torch.manual_seed(7)
        candidates = torch.randint(0, 50, (2, 4, 5))
        edges = torch.log_softmax(torch.randn(2, 4, 5, 5), dim=-1)
        tree = build_ubtree_batch(
            anchor_tokens=torch.tensor([8, 9]),
            candidate_ids=candidates,
            edge_log_probs=edges,
            budget=7,
            supertree_width=4,
        )
        assert tree.tokens.shape == (2, 8)
        assert torch.all(tree.parent_ids[:, 1:] < torch.arange(1, 8))
        assert torch.all(tree.visibility.diagonal(dim1=1, dim2=2))
        assert tree.retrieve_index.tolist() == [list(range(8)), list(range(8, 16))]

        for batch_index in range(2):
            for node in range(1, tree.node_count):
                parent = int(tree.parent_ids[batch_index, node])
                child = int(tree.retrieve_next_token[batch_index, parent])
                siblings = []
                while child != -1:
                    siblings.append(child)
                    child = int(tree.retrieve_next_sibling[batch_index, child])
                assert node in siblings

    def test_sglang_mask_contains_prefix_and_only_ancestors(self):
        candidates = torch.tensor([[[1, 2], [3, 4]]])
        edges = torch.log_softmax(torch.randn(1, 2, 2, 2), dim=-1)
        tree = build_ubtree_batch(
            anchor_tokens=torch.tensor([0]),
            candidate_ids=candidates,
            edge_log_probs=edges,
            budget=3,
            supertree_width=2,
        )
        packed = build_sglang_full_mask(tree.visibility, [3])
        mask = packed.view(tree.node_count, 3 + tree.node_count)
        assert torch.all(mask[:, :3])
        assert torch.equal(mask[:, 3:], tree.visibility[0])
        device_only = build_sglang_full_mask(
            tree.visibility, torch.tensor([3]), prefix_lens_sum=3
        )
        assert torch.equal(device_only, packed)

    def test_compact_to_front_reuses_contiguous_storage_without_changing_values(self):
        values = torch.arange(2 * 5 * 3).view(10, 3)
        original = values.clone()
        accept_index = torch.tensor([[2, 4, -1], [6, 9, 8]], dtype=torch.int32)
        expected = original.clone()
        expected.view(2, 5, 3)[:, :3].copy_(
            original[accept_index.to(torch.int64).clamp_min(0).reshape(-1)].view(
                2, 3, 3
            )
        )

        output = compact_tree_values_to_front(
            values,
            accept_index,
            batch_size=2,
            node_count=5,
        )

        assert output.data_ptr() == values.data_ptr()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_compact_to_front_keeps_noncontiguous_fallback(self):
        values = torch.arange(2 * 5 * 4).view(10, 4)[:, ::2]
        original = values.clone()
        accept_index = torch.tensor([[2, 4, -1], [6, 9, 8]], dtype=torch.int32)
        expected = original.clone()
        expected.view(2, 5, 2)[:, :3].copy_(
            original[accept_index.to(torch.int64).clamp_min(0).reshape(-1)].view(
                2, 3, 2
            )
        )

        output = compact_tree_values_to_front(
            values,
            accept_index,
            batch_size=2,
            node_count=5,
        )

        assert output.is_contiguous()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


class TestUBTreeRuntimeConfig(CustomTestCase):
    def test_verify_width_is_fixed_and_within_checkpoint(self):
        config = replace(parse_ubtree_config(_config()), tree_budget=64)
        self.assertEqual(resolve_verify_width(config, None), 49)
        for width in (2, 17, 31, 32, 33, 48, 49, 64, 65):
            with self.subTest(width=width):
                self.assertEqual(resolve_verify_width(config, width), width)
        for budget in (1, 16, 31, 39):
            with self.subTest(budget=budget):
                self.assertEqual(
                    resolve_verify_width(replace(config, tree_budget=budget), None),
                    budget + 1,
                )
        for width in (-1, 0, 1, 66):
            with self.subTest(width=width), self.assertRaises(ValueError):
                resolve_verify_width(config, width)
        with self.assertRaises(ValueError):
            resolve_verify_width(replace(config, tree_budget=31), 49)

    def test_builtin_identity_and_capabilities_preserve_plugin_protocol(self):
        algo = SpeculativeAlgorithm.from_string("ubtree")
        self.assertIs(algo, SpeculativeAlgorithm.UBTREE)
        self.assertFalse(algo.is_dflash())
        self.assertTrue(algo.is_dflash_family())
        self.assertFalse(algo.supports_mixed_chunk())
        self.assertFalse(algo.supports_grammar_overlap())
        self.assertFalse(algo.supports_ragged_verify())
        self.assertFalse(algo.supports_prefill_shared_read_done())
        self.assertFalse(algo.can_run_cuda_graph(None, is_draft_worker=True))
        self.assertTrue(algo.can_run_cuda_graph(None, is_draft_worker=False))
        _assert_custom_spec_algo_conforms(CustomSpecAlgo)

    def test_invalid_runtime_shapes_are_rejected(self):
        cfg = SimpleNamespace(
            device="cuda",
            pp_size=1,
            dp_size=1,
            max_running_requests=1,
            disaggregation_mode="null",
            enable_pd_role_switch=False,
            linear_attn_verify_backend=None,
            enable_strict_thinking=False,
            speculative_draft_window_size=None,
            speculative_draft_sink_size=None,
            speculative_use_rejection_sampling=False,
            speculative_use_block_verification=False,
            speculative_adaptive=False,
            page_size=1,
            speculative_draft_attention_backend="fa3",
            disable_cuda_graph=False,
            disable_cuda_graph_padding=True,
            cuda_graph_config=SimpleNamespace(
                decode=SimpleNamespace(backend="full", bs=[1]),
                prefill=SimpleNamespace(backend="disabled"),
            ),
        )
        validate_runtime_config(cfg, ("fa3",))
        for backend in ("flashinfer", "triton"):
            with self.subTest(backend=backend), self.assertRaises(ValueError):
                validate_runtime_config(cfg, (backend,))
        for field, value in [
            ("device", "cpu"),
            ("speculative_draft_attention_backend", "flashinfer"),
            ("pp_size", 2),
            ("dp_size", 2),
            ("max_running_requests", 2),
            ("disaggregation_mode", "decode"),
            ("disaggregation_mode", "prefill"),
            ("enable_pd_role_switch", True),
            ("linear_attn_verify_backend", "helion"),
            ("linear_attn_verify_backend", "flashinfer"),
            ("enable_strict_thinking", True),
            ("page_size", 16),
            ("disable_cuda_graph_padding", False),
            ("speculative_adaptive", True),
            ("speculative_use_rejection_sampling", True),
            ("speculative_draft_window_size", 32),
        ]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_runtime_config(
                    SimpleNamespace(**(vars(cfg) | {field: value})), ("fa3",)
                )
        for backend in ("triton", "nv_cutedsl"):
            validate_runtime_config(
                SimpleNamespace(
                    **(vars(cfg) | {"linear_attn_verify_backend": backend})
                ),
                ("fa3",),
            )


class TestUBTreeRequestValidation(CustomTestCase):
    @staticmethod
    def _request(**overrides):
        fields = dict(
            sampling_params=SamplingParams(temperature=0),
            custom_logit_processor=None,
            return_hidden_states=False,
            return_sampling_mask=False,
            return_logprob=False,
            grammar=None,
        )
        return SimpleNamespace(**(fields | overrides))

    def test_requests_without_custom_processor_are_allowed(self):
        for processor in (None, ""):
            with self.subTest(processor=processor):
                req = self._request(custom_logit_processor=processor)
                self.assertIsNone(validate_ubtree_request(req))

    def test_custom_processors_are_rejected_without_deserialization(self):
        for processor in ('{"callable": "invalid-hex"}', "invalid-processor"):
            with self.subTest(processor=processor):
                req = self._request(custom_logit_processor=processor)
                self.assertEqual(
                    validate_ubtree_request(req),
                    "UBTREE does not support custom_logit_processor.",
                )

    def test_unsupported_outputs_are_still_rejected(self):
        for field in ("return_hidden_states", "return_sampling_mask"):
            with self.subTest(field=field):
                req = self._request(
                    return_hidden_states=field == "return_hidden_states",
                    return_sampling_mask=field == "return_sampling_mask",
                )
                self.assertIn(field, validate_ubtree_request(req))

    def test_greedy_normalization_and_unsupported_sampling_options(self):
        for params in ({"temperature": 0}, {"temperature": 0.8, "top_k": 1}):
            req = self._request(sampling_params=SamplingParams(**params))
            self.assertIsNone(validate_ubtree_request(req))
        for params, message in [
            ({"temperature": 0.8, "top_k": 10}, "greedy"),
            ({"json_schema": '{"type":"object"}'}, "grammar"),
            ({"regex": "[a-z]+"}, "grammar"),
            ({"ebnf": 'root ::= "hello"'}, "grammar"),
            ({"frequency_penalty": 0.1}, "penalties"),
            ({"presence_penalty": 0.1}, "penalties"),
            ({"repetition_penalty": 1.1}, "penalties"),
            ({"min_new_tokens": 1}, "min_new_tokens"),
            ({"logit_bias": {1: 1.0}}, "logit_bias"),
        ]:
            with self.subTest(params=params):
                req = self._request(
                    sampling_params=SamplingParams(**({"temperature": 0} | params))
                )
                self.assertIn(message, validate_ubtree_request(req))
        self.assertIn(
            "return_logprob",
            validate_ubtree_request(self._request(return_logprob=True)),
        )
        self.assertIn(
            "grammar", validate_ubtree_request(self._request(grammar=object()))
        )


class TestUBTreeLoading(CustomTestCase):
    @staticmethod
    def _model(config):
        from sglang.srt.models.ubtree import Qwen3UBTreeModel

        # Exercise the actual inherited loader with a small selector, without
        # constructing distributed attention or downloading a checkpoint.
        model = Qwen3UBTreeModel.__new__(Qwen3UBTreeModel)
        torch.nn.Module.__init__(model)
        model.projector_type = "linear"
        model.ubtree_selector = UBTreeSelector(config)
        model._ubtree_selector_loaded = False
        return model

    def test_initial_load_requires_every_selector_parameter(self):
        for config in (_config(), _lookup_config()):
            source = self._model(config)
            weights = [
                (name, torch.ones_like(p)) for name, p in source.named_parameters()
            ]
            for missing, _ in weights:
                with self.subTest(
                    layout=config.ubtree_config["token_factorizer"], missing=missing
                ):
                    model = self._model(config)
                    with self.assertRaisesRegex(ValueError, missing):
                        model.load_weights(
                            (name, value) for name, value in weights if name != missing
                        )
                    self.assertFalse(model._ubtree_selector_loaded)
            for prefix in ("", "model."):
                model = self._model(config)
                model.load_weights((prefix + name, value) for name, value in weights)
                for name, p in model.named_parameters():
                    torch.testing.assert_close(p, torch.ones_like(p))
                # A successful initial load must not turn subsequent partial
                # updates into a false missing-checkpoint error.
                name, value = weights[0]
                model.load_weights([(name, value * 2)])
                torch.testing.assert_close(
                    dict(model.named_parameters())[name], value * 2
                )

    def test_quantized_head_is_rejected_before_matmul(self):
        for dtype in (torch.float8_e4m3fn, torch.int8):
            with (
                self.subTest(dtype=dtype),
                patch("torch.nn.functional.linear") as linear,
            ):
                with self.assertRaisesRegex(ValueError, "dense FP16/BF16/FP32"):
                    vocab_parallel_topk(
                        torch.ones(1, 8),
                        SimpleNamespace(weight=torch.ones((17, 8)).to(dtype)),
                        k=3,
                        vocab_size=17,
                    )
                linear.assert_not_called()

    def test_bound_selector_rejects_target_and_draft_weight_updates(self):
        from sglang.srt.model_executor.model_runner_components.weight_updater import (
            WeightUpdater,
        )
        from sglang.srt.runtime_context import get_context

        selector = UBTreeSelector(_config())
        embedding = torch.nn.Embedding(17, 8)
        selector.bind_target_embedding(embedding, device="cpu")
        for model in (selector, embedding):
            original = {
                name: p.detach().clone() for name, p in model.named_parameters()
            }
            updater = WeightUpdater(
                tp_rank=0,
                device="cpu",
                gpu_id=0,
                model_config=None,
                custom_weight_loaders={},
                get_model=lambda: model,
                update_model_fields=lambda **kwargs: None,
                recapture_cuda_graph=lambda: None,
                get_model_runner=lambda: None,
            )
            updates = [
                (name, torch.zeros_like(p)) for name, p in model.named_parameters()
            ]
            with get_context().override_server_args(weight_cache_mode="off"):
                results = [
                    updater.update_weights_from_disk("unused", "auto"),
                    updater.update_weights_from_tensor(updates, load_format="direct"),
                    updater.update_weights_from_tensor(
                        {}, load_format="flattened_bucket"
                    ),
                    updater.load_weights_from_distributed(updates),
                    updater.update_weights_from_ipc(None),
                ]
            for success, message in results:
                self.assertFalse(success)
                self.assertIn("UBTREE", message)
            for name, p in model.named_parameters():
                torch.testing.assert_close(p, original[name])

    def test_lookup_selector_does_not_restrict_target_updates(self):
        selector = UBTreeSelector(_lookup_config())
        embedding = torch.nn.Embedding(17, 8)
        selector.bind_target_embedding(embedding, device="cpu")
        self.assertFalse(hasattr(selector, "_derived_weight_cache_error"))
        self.assertFalse(hasattr(embedding, "_derived_weight_cache_error"))


if __name__ == "__main__":
    unittest.main()
