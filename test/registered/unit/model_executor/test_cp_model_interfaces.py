"""CPU regressions for explicit CP batch fields and model capabilities."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

import sglang.srt.model_executor.runner.prefill_cuda_graph_runner as prefill
from sglang.srt.layers.cp.bcg.bcg import execute_prefill_cp_bcg
from sglang.srt.layers.cp.bcg.dsv4_bcg import DSV4CPBCGInput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_interfaces import (
    CPBoundedTailLogits,
    LanguageModelInputPreparation,
    prepare_language_model_inputs,
)
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

MODEL = "sglang.srt.models.deepseek_v4"
BCG = "sglang.srt.layers.cp.bcg.bcg"


class _PreparedModel(NS, LanguageModelInputPreparation):
    pass


class _TailModel(NS, CPBoundedTailLogits):
    prepare_cp_logits_inputs = DeepseekV4ForCausalLM.prepare_cp_logits_inputs


def make_batch():
    return ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=1,
        input_ids=torch.arange(6),
        req_pool_indices=torch.tensor([0]),
        seq_lens=torch.tensor([6]),
        out_cache_loc=torch.arange(6),
        seq_lens_sum=6,
        positions=torch.arange(6),
        extend_seq_lens=torch.tensor([6]),
        extend_seq_lens_cpu=[6],
        extend_prefix_lens_cpu=[0],
        dp_padding_mode=NS(is_max_len=lambda: True),
    )


class TestCPModelInterfaces(CustomTestCase):
    def test_text_model_preparation_is_identity(self):
        batch = make_batch()
        embeds = torch.ones(6, 2)
        for model in (object(), LanguageModelInputPreparation()):
            ids, prepared = prepare_language_model_inputs(
                model, batch.input_ids, batch, embeds
            )
            self.assertIs(ids, batch.input_ids)
            self.assertIs(prepared, embeds)

    def test_method_presence_does_not_declare_preparation(self):
        batch = make_batch()
        model = NS(prepare_language_model_inputs=Mock(side_effect=AssertionError))
        ids, embeds = prepare_language_model_inputs(model, batch.input_ids, batch)
        self.assertIs(ids, batch.input_ids)
        self.assertIsNone(embeds)
        model.prepare_language_model_inputs.assert_not_called()

    def test_declared_preparation_runs_and_propagates_failures(self):
        batch = make_batch()
        normalized = batch.input_ids + 100
        embeds = torch.ones(6, 2)
        model = _PreparedModel(
            prepare_language_model_inputs=Mock(return_value=(normalized, embeds))
        )
        ids, prepared = prepare_language_model_inputs(model, batch.input_ids, batch)
        self.assertIs(ids, normalized)
        self.assertIs(prepared, embeds)
        model.prepare_language_model_inputs.assert_called_once_with(
            batch.input_ids, batch, None
        )
        model.prepare_language_model_inputs.side_effect = ValueError("vision failed")
        with self.assertRaisesRegex(ValueError, "vision failed"):
            prepare_language_model_inputs(model, batch.input_ids, batch)

    def _make_capture_runner(self, cp):
        runner = prefill.PrefillCudaGraphRunner.__new__(prefill.PrefillCudaGraphRunner)
        runner.enable_cp_bcg_capture = cp
        runner.prefill_cp_bcg_input = DSV4CPBCGInput(
            input_embeds=torch.empty(0), positions=torch.empty(0)
        )
        runner._prefill_forward_context = lambda *a, **kw: nullcontext()
        runner._capture_pp_proxy_tensors = lambda n: None
        runner._uses_eager_prefill_tail = lambda: True
        runner._pack_qwen_bcg_hc_output = lambda output: output
        runner.layer_model = NS(forward=Mock(return_value=torch.ones(3, 2)))
        return runner

    def _capture(self, runner, batch):
        with (
            patch.object(prefill, "set_dp_buffer_len"),
            patch.object(prefill, "set_is_extend_in_batch"),
            patch.object(prefill, "_build_layer_model_forward_kwargs", return_value={}),
        ):
            return runner._run_forward(batch, 6)

    def test_non_cp_capture_keeps_global_inputs(self):
        batch = make_batch()
        self.assertIsNone(batch.cp_positions)
        self.assertIsNone(batch.cp_input_ids)
        runner = self._make_capture_runner(cp=False)
        self._capture(runner, batch)
        args = runner.layer_model.forward.call_args.args
        self.assertIs(args[0], batch.input_ids)
        self.assertIs(args[1], batch.positions)

    def test_cp_capture_requires_prepared_positions_and_ids(self):
        batch = make_batch()
        runner = self._make_capture_runner(cp=True)
        with self.assertRaisesRegex(RuntimeError, "positions must be prepared"):
            self._capture(runner, batch)
        batch.cp_positions = batch.positions[::2]
        with self.assertRaisesRegex(RuntimeError, "token IDs must be prepared"):
            self._capture(runner, batch)
        runner.layer_model.forward.assert_not_called()

    def test_cp_capture_uses_local_fields_without_mutating_global_inputs(self):
        batch = make_batch()
        global_ids = batch.input_ids.clone()
        global_positions = batch.positions.clone()
        batch.cp_positions = batch.positions[::2].clone()
        batch.cp_input_ids = batch.input_ids[::2].clone()
        runner = self._make_capture_runner(cp=True)
        self._capture(runner, batch)
        args = runner.layer_model.forward.call_args.args
        self.assertIs(args[0], batch.cp_input_ids)
        self.assertIs(args[1], batch.cp_positions)
        torch.testing.assert_close(batch.input_ids, global_ids)
        torch.testing.assert_close(batch.positions, global_positions)

    def _tail_model(self, **kwargs):
        return _TailModel(
            capture_aux_hidden_states=True,
            model=NS(late_layer_start=1),
            pp_group=NS(is_last_rank=True),
            lm_head=object(),
            logits_processor=Mock(return_value=NS()),
            **kwargs,
        )

    def _replay(self, model, batch, local_output):
        runner = NS(
            prefill_cp_bcg_input=NS(live_local_tokens=6),
            model_runner=NS(model=model),
            backend=NS(replay=Mock(return_value=local_output)),
            _prefill_forward_context=lambda *a, **kw: nullcontext(),
        )
        with (
            patch(BCG + ".cp_gather_after_forward", side_effect=lambda x, *a: x),
            patch(BCG + ".torch.cuda.current_stream", return_value=None),
        ):
            return execute_prefill_cp_bcg(runner, batch, batch, 6, 6)

    def test_bounded_tail_metadata_and_output_mapping_survive_cp_replay(self):
        batch = make_batch()
        indices = torch.tensor([1, 4])
        tail = NS(
            rows=lambda x: x[indices],
            extend_seq_lens=torch.tensor([2]),
            extend_seq_lens_cpu=[2],
            token_indices=indices,
        )
        backend = NS(tail_forward_metadata=NS(late_layer_tail=tail))
        model = self._tail_model()
        hidden = torch.ones(2, 3)
        aux = [torch.full((2, 3), 2.0)]
        with (
            patch(MODEL + ".get_attn_backend", return_value=backend),
            patch(MODEL + ".LogitsMetadata.from_forward_batch", return_value=NS()),
        ):
            output = self._replay(model, batch, (hidden, aux))
        args = model.logits_processor.call_args.args
        torch.testing.assert_close(args[0], batch.input_ids[indices])
        torch.testing.assert_close(args[1], hidden)
        self.assertEqual(args[1].data_ptr(), hidden.data_ptr())
        self.assertIs(args[3].extend_seq_lens, tail.extend_seq_lens)
        self.assertEqual(args[3].extend_seq_lens_cpu, [2])
        self.assertEqual(args[3].extend_logprob_start_lens_cpu, [2])
        torch.testing.assert_close(args[4][0], aux[0])
        self.assertIs(output.hidden_states_token_indices, indices)
        self.assertEqual(batch.extend_seq_lens_cpu, [6])

    def test_bounded_tail_requires_metadata_when_enabled(self):
        batch = make_batch()
        model = self._tail_model()
        backend = NS(tail_forward_metadata=NS(late_layer_tail=None))
        with patch(MODEL + ".get_attn_backend", return_value=backend):
            with self.assertRaisesRegex(RuntimeError, "prepared tail metadata"):
                model.prepare_cp_logits_inputs(batch.input_ids, batch)

    def test_bounded_tail_is_identity_outside_aux_extend(self):
        batch = make_batch()
        model = self._tail_model()
        cases = (
            (False, 1, ForwardMode.EXTEND),
            (True, None, ForwardMode.EXTEND),
            (True, 1, ForwardMode.DECODE),
            (True, 1, ForwardMode.TARGET_VERIFY),
        )
        for capture, start, mode in cases:
            with self.subTest(capture=capture, start=start, mode=mode):
                model.capture_aux_hidden_states = capture
                model.model.late_layer_start = start
                batch.forward_mode = mode
                with patch(MODEL + ".get_attn_backend") as get_backend:
                    inputs = model.prepare_cp_logits_inputs(batch.input_ids, batch)
                self.assertIs(inputs.input_ids, batch.input_ids)
                self.assertIs(inputs.logits_metadata, batch)
                self.assertIsNone(inputs.hidden_states_token_indices)
                get_backend.assert_not_called()

    def test_cp_logits_without_tail_capability_keep_global_metadata(self):
        batch = make_batch()
        model = NS(
            capture_aux_hidden_states=False,
            pp_group=NS(is_last_rank=True),
            lm_head=object(),
            logits_processor=Mock(return_value=NS()),
        )
        self._replay(model, batch, torch.ones(6, 3))
        args = model.logits_processor.call_args.args
        self.assertIs(args[0], batch.input_ids)
        self.assertIs(args[3], batch)


if __name__ == "__main__":
    unittest.main()
