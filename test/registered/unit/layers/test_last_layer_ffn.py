import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.last_layer_ffn import (
    configure_last_layer_ffn,
    eligible_batch,
    prepare_last_layer_ffn,
    select_attention_rows,
)
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layer_boundary.residual.add_norm import PLAIN_ADD
from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_mode import CaptureHiddenMode, ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, enter_scope

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def batch():
    return SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        return_logprob=False,
        is_prefill_only=False,
        spec_info=None,
        capture_hidden_mode=CaptureHiddenMode.NULL,
        return_hidden_states_before_norm=False,
        token_indices_to_pool=None,
        multi_item_delimiter_indices=None,
        lora_ids=[None, None],
        mm_inputs=None,
        extend_seq_lens_cpu=[3, 5],
        extend_prefix_lens_cpu=[0, 0],
        extend_is_final_cpu=[True, True],
        batch_size=2,
        input_ids=torch.arange(8),
        extend_seq_lens=torch.tensor([3, 5]),
        residual_stream=None,
    )


class LastLayerFFNTests(CustomTestCase):
    def configure_model(self, **changes):
        model = SimpleNamespace(
            pp_group=SimpleNamespace(world_size=changes.get("pp_size", 1))
        )
        parallel = SimpleNamespace(
            tp_size=changes.get("tp_size", 1),
            attn_dp_size=changes.get("attn_dp_size", 1),
            attn_cp_size=changes.get("attn_cp_size", 1),
        )
        execution = SimpleNamespace(
            graph=SimpleNamespace(
                cuda_graph_config=SimpleNamespace(
                    prefill=SimpleNamespace(
                        backend=changes.get("backend", Backend.BREAKABLE)
                    )
                )
            )
        )
        with (
            patch("sglang.srt.runtime_context.get_parallel", return_value=parallel),
            patch("sglang.srt.runtime_context.get_exec", return_value=execution),
            patch("sglang.srt.utils.is_cuda", return_value=changes.get("cuda", True)),
        ):
            configure_last_layer_ffn(
                model,
                supported_architecture=changes.get("supported_architecture", True),
                quant_config=changes.get("quant_config"),
                model_type=changes.get("model_type", "qwen3"),
            )
        return model

    def test_supported_architectures_enable_pruning_by_default(self):
        for model_type, minimum in [("qwen2", 512), ("qwen3", 512), ("llama", 2048)]:
            for backend in (Backend.DISABLED, Backend.BREAKABLE):
                with self.subTest(model_type=model_type, backend=backend):
                    model = self.configure_model(model_type=model_type, backend=backend)
                    self.assertEqual(model.last_layer_ffn_min_extend_tokens, minimum)

    def test_unsupported_runtime_disables_pruning(self):
        for changes in [
            {"supported_architecture": False},
            {"model_type": "unsupported"},
            {"quant_config": object()},
            {"cuda": False},
            {"tp_size": 2},
            {"pp_size": 2},
            {"attn_dp_size": 2},
            {"attn_cp_size": 2},
            {"backend": Backend.FULL},
            {"backend": Backend.TC_PIECEWISE},
        ]:
            with self.subTest(changes=changes):
                model = self.configure_model(**changes)
                self.assertIsNone(model.last_layer_ffn_min_extend_tokens)

    def test_eligibility(self):
        self.assertTrue(eligible_batch(batch()))
        cases = {
            "forward_mode": ForwardMode.DECODE,
            "tbo_split_seq_index": 0,
            "tbo_parent_token_range": (0, 8),
            "tbo_children": [],
            "return_logprob": True,
            "is_prefill_only": True,
            "spec_info": object(),
            "capture_hidden_mode": CaptureHiddenMode.FULL,
            "return_hidden_states_before_norm": True,
            "token_indices_to_pool": [1],
            "multi_item_delimiter_indices": [1],
            "lora_ids": ["adapter", None],
            "mm_inputs": [object()],
            "extend_is_final_cpu": [True, False],
            "extend_prefix_lens_cpu": [-1, 0],
            "extend_seq_lens_cpu": [0, 8],
            "batch_size": 3,
            "input_ids": torch.arange(9),
            "extend_seq_lens": None,
        }
        for name, value in cases.items():
            with self.subTest(name=name):
                b = batch()
                setattr(b, name, value)
                self.assertFalse(eligible_batch(b))

    def test_cached_prefix_and_final_chunk_use_suffix_local_indices(self):
        for prefixes in ([128, 0], [128, 256]):
            with self.subTest(prefixes=prefixes):
                b = batch()
                b.extend_prefix_lens_cpu = prefixes
                self.assertTrue(eligible_batch(b))
                indices = torch.cumsum(b.extend_seq_lens, dim=0) - 1
                self.assertEqual(indices.tolist(), [2, 7])
                b.extend_is_final_cpu = [True, False]
                self.assertFalse(eligible_batch(b))

    def test_empty_and_single_token_requests_keep_original_path(self):
        for lens in ([], [1, 1]):
            b = batch()
            b.batch_size = len(lens)
            b.extend_seq_lens_cpu = lens
            b.extend_prefix_lens_cpu = [0] * len(lens)
            b.extend_is_final_cpu = [True] * len(lens)
            b.input_ids = torch.zeros(sum(lens), dtype=torch.long)
            self.assertFalse(eligible_batch(b))

    def test_tokenwise_ffn_selection_is_mathematically_equivalent(self):
        # Exact arithmetic with small integer-valued doubles isolates the
        # dependency property from reduced-precision GEMM shape effects.
        x = torch.arange(32, dtype=torch.float64).reshape(8, 4)
        residual = x + 1
        up = torch.arange(24, dtype=torch.float64).reshape(4, 6)
        down = torch.arange(24, dtype=torch.float64).reshape(6, 4)
        indices = torch.tensor([2, 7])
        full = torch.relu((x + residual) @ up) @ down
        b = batch()
        hidden, b.residual_stream = ResidualStream.from_handoff(x, residual, PLAIN_ADD)
        selected = select_attention_rows(hidden, b, indices)
        h, res = residual_batch.stream_of(b).export(selected)
        compact = torch.relu((h + res) @ up) @ down
        torch.testing.assert_close(compact, full[indices], rtol=0, atol=0)

    def test_operands_selected_without_adding(self):
        b = batch()
        x = torch.arange(32.0).reshape(8, 4)
        r = x + 100
        hidden, b.residual_stream = ResidualStream.from_handoff(x, r, PLAIN_ADD)
        idx = torch.tensor([2, 7])
        selected = select_attention_rows(hidden, b, idx)
        h, res = residual_batch.stream_of(b).export(selected)
        torch.testing.assert_close(h, x[idx])
        torch.testing.assert_close(res, r[idx])
        self.assertIsNot(h, x)
        self.assertIsNot(res, r)
        torch.testing.assert_close(r, x + 100)

    def test_no_double_indexing(self):
        x = torch.randn(2, 4)
        meta = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            extend_return_logprob=False,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            extend_seq_lens_cpu=[3, 5],
        )
        out = LogitsProcessor._get_pruned_states(
            None, x, None, None, meta, already_pruned=True
        )
        self.assertIs(out[0], x)
        self.assertEqual(out[1:], (None, None, None, None, []))
        meta.extend_return_logprob = True
        with self.assertRaises(AssertionError):
            LogitsProcessor._get_pruned_states(
                None, x, None, None, meta, already_pruned=True
            )


class PrepareLastLayerFFNTests(CustomTestCase):
    def setUp(self):
        self.model = SimpleNamespace(
            last_layer_ffn_min_extend_tokens=512,
            capture_aux_hidden_states=False,
            model=SimpleNamespace(layers_to_capture=[]),
        )
        self.batch = batch()
        self.batch.extend_seq_lens_cpu = [300, 212]
        self.batch.extend_seq_lens = torch.tensor([300, 212])
        self.batch.input_ids = torch.arange(512)
        self.batch.last_layer_ffn_indices = None
        self.batch.last_layer_ffn_pruned = False
        self.batch.last_layer_ffn_graph_rows = 0
        self.scope = enter_scope(
            self,
            patch(
                "sglang.srt.model_executor.runner_utils.prefill_graph.is_in_prefill_graph_scope",
                return_value=False,
            ),
        )
        self.compiling = enter_scope(
            self, patch("torch.compiler.is_compiling", return_value=False)
        )
        self.capturing = enter_scope(
            self, patch("torch.cuda.is_current_stream_capturing", return_value=False)
        )

    def assert_unpruned(self):
        self.assertIsNone(self.batch.last_layer_ffn_indices)
        self.assertFalse(self.batch.last_layer_ffn_pruned)

    def test_eager_replans_and_clears_stale_selection_on_fallback(self):
        self.assertEqual(prepare_last_layer_ffn(self.model, self.batch), 0)
        self.assertEqual(self.batch.last_layer_ffn_indices.tolist(), [299, 511])
        # Eager selection is planned here; the terminal boundary marks execution.
        self.assertFalse(self.batch.last_layer_ffn_pruned)
        self.batch.last_layer_ffn_pruned = True
        self.batch.return_logprob = True
        prepare_last_layer_ffn(self.model, self.batch)
        self.assert_unpruned()
        self.batch.return_logprob = False
        self.batch.extend_seq_lens_cpu = [256, 256]
        self.batch.extend_seq_lens = torch.tensor([256, 256])
        prepare_last_layer_ffn(self.model, self.batch)
        self.assertEqual(self.batch.last_layer_ffn_indices.tolist(), [255, 511])

    def test_graph_selection_then_ordinary_graph_clears_stale_state(self):
        self.scope.return_value = True
        indices = torch.tensor([299, 511, 0, 0])
        self.batch.last_layer_ffn_indices = indices
        self.batch.last_layer_ffn_graph_rows = 4
        self.assertEqual(prepare_last_layer_ffn(self.model, self.batch), 4)
        self.assertIs(self.batch.last_layer_ffn_indices, indices)
        self.assertTrue(self.batch.last_layer_ffn_pruned)
        self.capturing.assert_not_called()
        self.batch.last_layer_ffn_graph_rows = 0
        self.assertEqual(prepare_last_layer_ffn(self.model, self.batch), 0)
        self.assert_unpruned()
        self.scope.return_value = False
        prepare_last_layer_ffn(self.model, self.batch)
        self.assertEqual(self.batch.last_layer_ffn_indices.tolist(), [299, 511])

    def test_unsupported_model_clears_state_without_cuda_queries(self):
        self.model.last_layer_ffn_min_extend_tokens = None
        self.batch.last_layer_ffn_indices = torch.tensor([299, 511])
        self.batch.last_layer_ffn_pruned = True
        prepare_last_layer_ffn(self.model, self.batch)
        self.assert_unpruned()
        self.capturing.assert_not_called()
        self.compiling.assert_not_called()

    def test_eager_model_and_execution_guards(self):
        for case in [
            "embedding",
            "input_embeds",
            "aux",
            "layers",
            "compile",
            "capture",
        ]:
            with self.subTest(case=case):
                self.model.capture_aux_hidden_states = case == "aux"
                self.model.model.layers_to_capture = [0] if case == "layers" else []
                self.compiling.return_value = case == "compile"
                self.capturing.return_value = case == "capture"
                self.batch.last_layer_ffn_indices = torch.tensor([299, 511])
                self.batch.last_layer_ffn_pruned = True
                prepare_last_layer_ffn(
                    self.model,
                    self.batch,
                    get_embedding=case == "embedding",
                    input_embeds=torch.ones(1) if case == "input_embeds" else None,
                )
                self.assert_unpruned()

    def test_eager_selects_more_than_sixteen_requests(self):
        self.batch.batch_size = 17
        self.batch.extend_seq_lens_cpu = [32] * 17
        self.batch.extend_seq_lens = torch.tensor([32] * 17)
        self.batch.extend_prefix_lens_cpu = [0] * 17
        self.batch.extend_is_final_cpu = [True] * 17
        self.batch.input_ids = torch.arange(544)
        prepare_last_layer_ffn(self.model, self.batch)
        self.assertEqual(
            self.batch.last_layer_ffn_indices.tolist(), list(range(31, 544, 32))
        )
        self.model.last_layer_ffn_min_extend_tokens = 2048
        prepare_last_layer_ffn(self.model, self.batch)
        self.assert_unpruned()


if __name__ == "__main__":
    unittest.main()
