"""Runtime-side execution tests for optional speculative provider hooks."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.speculative import eagle_worker_common
from sglang.srt.speculative.eagle_post_verify_graphs import _sample
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, stage="base-a", runner_config="cpu")


def _sampling():
    empty = torch.empty(0)
    return SimpleNamespace(
        temperatures=torch.ones((2, 1)),
        top_ks=torch.ones(2, dtype=torch.int32),
        top_ps=torch.ones(2),
        min_ps=torch.zeros(2),
        acc_additive_penalties=None,
        acc_scaling_penalties=None,
        logit_bias=None,
        sampling_seed=None,
        is_all_greedy=True,
        need_top_k_sampling=False,
        need_top_p_sampling=False,
        need_min_p_sampling=False,
        watermark_keys=None,
        watermark_context_windows=None,
        watermark_enabled=None,
        has_watermark_candidates=False,
        max_top_k=None,
        sequence_lengths=torch.tensor([7, 11]),
        unused=empty,
    )


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


def _worker_with_model_class(model_class):
    processor = LogitsProcessor.__new__(LogitsProcessor)
    torch.nn.Module.__init__(processor)
    processor._sharded_verify_eligible = True
    processor.sharded_verify = False
    model = type(model_class, (torch.nn.Module,), {})()
    model.logits_processor = processor
    worker = SimpleNamespace(model_runner=SimpleNamespace(model=model))
    return worker, model, processor


class _PostVerify:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def __call__(self, io):
        self.calls.append(io)
        return self.output


class TestRuntimeHookExecution(CustomTestCase):
    def _assert_worker_sharded_verify_support(
        self,
        *,
        module_name,
        worker_name,
        topk,
        expected,
        tree_mode=None,
    ):
        from importlib import import_module

        module = import_module(module_name)
        worker = getattr(module, worker_name).__new__(getattr(module, worker_name))
        worker.topk = topk
        worker._target_worker = SimpleNamespace()
        worker._draft_worker = None
        if tree_mode is not None:
            worker.tree_mode = tree_mode

        hooks = object()
        with (
            patch(
                "sglang.srt.speculative.base_spec_worker.BaseSpecWorker.init_attention_backends"
            ),
            patch.object(
                module, "bind_eagle_runtime_hooks", return_value=hooks
            ) as bind,
        ):
            worker.init_attention_backends()

        bind.assert_called_once_with(
            worker.target_worker,
            supports_sharded_verify=expected,
        )
        self.assertIs(worker._speculative_runtime_hooks, hooks)

    def test_eagle_sharded_verify_opt_in_requires_topk_one(self):
        for topk, expected in ((1, True), (2, False)):
            with self.subTest(topk=topk):
                self._assert_worker_sharded_verify_support(
                    module_name="sglang.srt.speculative.eagle_worker_v2",
                    worker_name="EAGLEWorkerV2",
                    topk=topk,
                    expected=expected,
                )

    def test_multi_layer_eagle_sharded_verify_opt_in_requires_topk_one(self):
        for topk, expected in ((1, True), (2, False)):
            with self.subTest(topk=topk):
                self._assert_worker_sharded_verify_support(
                    module_name="sglang.srt.speculative.multi_layer_eagle_worker_v2",
                    worker_name="MultiLayerEagleWorkerV2",
                    topk=topk,
                    expected=expected,
                )

    def test_ngram_and_uno_keep_full_vocab(self):
        for module_name, worker_name, tree_mode in (
            ("sglang.srt.speculative.ngram_worker", "NGRAMWorker", None),
            ("sglang.srt.speculative.uno_worker_v2", "UnoWorkerV2", False),
        ):
            with self.subTest(worker=worker_name):
                self._assert_worker_sharded_verify_support(
                    module_name=module_name,
                    worker_name=worker_name,
                    topk=1,
                    expected=False,
                    tree_mode=tree_mode,
                )

    def test_sharded_verify_opts_in_only_for_validated_qwen35_models(self):
        validated_model_classes = (
            "Qwen3_5ForCausalLM",
            "Qwen3_5MoeForCausalLM",
            "Qwen3_5ForConditionalGeneration",
            "Qwen3_5MoeForConditionalGeneration",
        )

        for model_class in validated_model_classes:
            with self.subTest(model_class=model_class):
                worker, _, processor = _worker_with_model_class(model_class)

                selected = eagle_worker_common.bind_eagle_runtime_hooks(
                    worker, supports_sharded_verify=True
                )

                self.assertTrue(processor.sharded_verify)
                self.assertIs(
                    selected, eagle_worker_common.STOCK_SPECULATIVE_RUNTIME_HOOKS
                )

    def test_unvalidated_models_keep_full_vocab_and_still_bind_provider(self):
        unvalidated_model_classes = (
            "Gemma4UnifiedForConditionalGeneration",
            "UnknownForCausalLM",
            "Qwen3_5ForCausalLMMTP",
        )

        for model_class in unvalidated_model_classes:
            with self.subTest(model_class=model_class):
                worker, model, processor = _worker_with_model_class(model_class)
                provider_hooks = object()
                bind_provider = Mock(return_value=provider_hooks)

                with patch.object(
                    eagle_worker_common,
                    "import_module",
                    return_value=SimpleNamespace(
                        bind_speculative_runtime_hooks=bind_provider
                    ),
                ):
                    selected = eagle_worker_common.bind_eagle_runtime_hooks(
                        worker, supports_sharded_verify=True
                    )

                self.assertFalse(processor.sharded_verify)
                bind_provider.assert_called_once_with(
                    model, eagle_worker_common.STOCK_SPECULATIVE_RUNTIME_HOOKS
                )
                self.assertIs(selected, provider_hooks)

    def test_runtime_alone_binds_stock_for_supported_model_with_master_off(self):
        worker, _, processor = _worker_with_model_class("Qwen3_5ForCausalLM")

        with envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(False):
            selected = eagle_worker_common.bind_eagle_runtime_hooks(
                worker, supports_sharded_verify=True
            )

        self.assertTrue(processor.sharded_verify)
        self.assertIs(
            selected, eagle_worker_common.STOCK_SPECULATIVE_RUNTIME_HOOKS
        )

    def test_unsupported_consumer_keeps_full_vocab(self):
        worker, _, processor = _worker_with_model_class("Qwen3_5ForCausalLM")

        eagle_worker_common.bind_eagle_runtime_hooks(
            worker, supports_sharded_verify=False
        )
        self.assertFalse(processor.sharded_verify)

    def test_graph_sample_calls_the_selected_strategy_unconditionally(self):
        output = SimpleNamespace(
            token_ids=torch.tensor([5]),
            accepted_lengths=torch.tensor([1]),
            accepted_indices=torch.tensor([0]),
        )
        strategy = _PostVerify(output)
        hooks = SimpleNamespace(post_verify=strategy)
        sampling = _sampling()
        tree = SimpleNamespace(
            draft_tokens=torch.tensor([1, 2, 3, 4, 5, 6]),
            retrieve_index=torch.arange(6),
            next_child=torch.full((6,), -1),
            next_sibling=torch.full((6,), -1),
        )
        slot = SimpleNamespace(
            staged={
                "seq_lens": sampling.sequence_lengths,
                "req_pool_indices": torch.tensor([3, 7]),
                "draft_tokens": tree.draft_tokens,
                "retrieve_index": tree.retrieve_index,
                "retrieve_next_token": tree.next_child,
                "retrieve_next_sibling": tree.next_sibling,
            },
            sampling_info=sampling,
            is_all_greedy=True,
            need_top_k=False,
            need_top_p=False,
            unused_sampling_tensor=torch.empty(0),
            tokens_per_request=3,
            max_tree_depth=3,
            device="cpu",
            logits=torch.ones((6, 5)),
            vocab_shard=None,
            return_logprob=False,
            max_top_k=None,
        )

        got = _sample(slot, hooks)

        self.assertEqual(len(strategy.calls), 1)
        self.assertIs(strategy.calls[0].logits.next_token_logits, slot.logits)
        self.assertIs(got[0], output.token_ids)
        self.assertIs(got[1], output.accepted_lengths)
        self.assertIs(got[2], output.accepted_indices)

    def test_stock_fallback_invokes_eagle_sample_without_provider_keys(self):
        sampling = _sampling()
        tree = SimpleNamespace(
            draft_token=torch.tensor([1]),
            draft_probs=None,
            retrieve_index=torch.tensor([0]),
            retrieve_next_token=torch.tensor([-1]),
            retrieve_next_sibling=torch.tensor([-1]),
            custom_mask=None,
            positions=None,
            draft_token_num=1,
            max_tree_depth=1,
            tree_topk=1,
        )
        io = eagle_worker_common.make_post_verify_io(
            verify_input=tree,
            sampling_info=sampling,
            sequence_lengths=torch.tensor([7]),
            request_pool_indices=torch.tensor([3]),
            device="cpu",
            logits_output=SimpleNamespace(
                next_token_logits=torch.ones((1, 5)),
                next_token_logits_vocab_shard=None,
            ),
            grammar_mask=None,
            watermark=None,
            watermark_full_mask=True,
            return_logprob=False,
            is_idle=False,
        )
        expected = (torch.tensor([1]), torch.tensor([1]), torch.tensor([0]))
        with patch(
            "sglang.srt.speculative.eagle_utils.eagle_sample", return_value=expected
        ) as sample:
            got = eagle_worker_common.STOCK_SPECULATIVE_RUNTIME_HOOKS.post_verify(io)

        sample.assert_called_once()
        self.assertIs(got.token_ids, expected[0])
        self.assertIs(got.accepted_lengths, expected[1])
        self.assertIs(got.accepted_indices, expected[2])

    def test_ngram_uses_the_same_unconditional_post_verify_strategy(self):
        source = (
            Path(eagle_worker_common.__file__).with_name("ngram_worker.py").read_text()
        )
        self.assertNotIn("eagle_sample(", source)
        self.assertIn("self._speculative_runtime_hooks.post_verify(", source)

        # NGRAM has no proposal probabilities. The shared constructor must form
        # the same model-independent view without requiring an EAGLE-only field.
        io = eagle_worker_common.make_post_verify_io(
            verify_input=_verify_tree(),
            sampling_info=_sampling(),
            sequence_lengths=torch.tensor([7, 11]),
            request_pool_indices=torch.tensor([3, 7]),
            device="cpu",
            logits_output=SimpleNamespace(
                next_token_logits=_verify_logits(),
                next_token_logits_vocab_shard=None,
            ),
            grammar_mask=None,
            watermark=None,
            watermark_full_mask=True,
            return_logprob=False,
            is_idle=False,
        )
        probabilities = getattr(
            io.verify_tree,
            "draft_probabilities",
            getattr(io.verify_tree, "draft_probs", None),
        )
        self.assertIsNone(probabilities)

    def test_default_post_verify_matches_direct_stock_greedy_bitwise(self):
        runtime = _optional_binding_runtime(self)
        from sglang.srt.speculative import eagle_utils

        sampling = _sampling()
        tree = _verify_tree()
        runtime_spec = SimpleNamespace(
            speculative_use_rejection_sampling=False,
            speculative_accept_threshold_single=1.0,
            speculative_accept_threshold_acc=1.0,
        )

        def cpu_greedy(
            *,
            predicts,
            accept_index,
            accept_token_num,
            candidates,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            target_predict,
            topk,
        ):
            del topk
            target_flat = target_predict.reshape(-1)
            for row in range(candidates.shape[0]):
                last = int(retrieve_index[row, 0])
                accept_index[row, 0] = last
                accepted = 0
                child = 0
                for _ in range(1, accept_index.shape[1]):
                    child = int(retrieve_next_token[row, child])
                    while child != -1:
                        draft_index = int(retrieve_index[row, child])
                        if int(candidates[row, child]) == int(target_flat[last]):
                            predicts[last] = target_flat[last]
                            accepted += 1
                            accept_index[row, accepted] = draft_index
                            last = draft_index
                            break
                        child = int(retrieve_next_sibling[row, child])
                    if child == -1:
                        break
                accept_token_num[row] = accepted
                predicts[last] = target_flat[last]
            return predicts, accept_index, accept_token_num

        def logits_view():
            return SimpleNamespace(
                next_token_logits=_verify_logits().clone(),
                next_token_logits_vocab_shard=None,
            )

        stock_batch = SimpleNamespace(
            device=torch.device("cpu"),
            forward_mode=SimpleNamespace(is_idle=lambda: False),
            seq_lens=sampling.sequence_lengths,
            sampling_info=sampling,
            req_pool_indices=torch.tensor([3, 7]),
            return_logprob=False,
        )
        io = eagle_worker_common.make_post_verify_io(
            verify_input=tree,
            sampling_info=sampling,
            sequence_lengths=stock_batch.seq_lens,
            request_pool_indices=stock_batch.req_pool_indices,
            device="cpu",
            logits_output=logits_view(),
            grammar_mask=None,
            watermark=None,
            watermark_full_mask=True,
            return_logprob=False,
            is_idle=False,
        )

        with (
            envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(False),
            patch.object(eagle_utils, "get_spec", return_value=runtime_spec),
            patch.object(
                eagle_utils, "verify_tree_greedy_func", side_effect=cpu_greedy
            ),
        ):
            direct = eagle_utils.eagle_sample(
                tree,
                stock_batch,
                logits_view(),
            )
            actual = runtime.default_speculative_runtime_hooks().post_verify(io)

        selected = (
            actual.token_ids,
            actual.accepted_lengths,
            actual.accepted_indices,
        )
        for stock, hooked in zip(direct, selected):
            self.assertTrue(torch.equal(stock, hooked))


if __name__ == "__main__":
    import unittest

    unittest.main()
