"""Residual ownership across layer execution and batch reuse."""

import unittest
import weakref
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.batch_overlap.two_batch_overlap import TboForwardBatchPreparer
from sglang.srt.layers.layer_boundary import PLAIN_ADD
from sglang.srt.layers.layer_boundary.output import UnreducedOutput
from sglang.srt.layers.layer_boundary.residual import batch
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=9, suite="base-a-test-cpu")


class TestBatchOwnedResidual(CustomTestCase):
    def test_start_releases_previous_call_and_resets_state(self):
        fb = SimpleNamespace(residual_stream=None)
        with self.assertRaisesRegex(RuntimeError, "start"):
            batch.stream_of(fb)
        batch.start(fb)
        old = batch.stream_of(fb)
        self.assertIs(old, batch.stream_of(fb))
        value = torch.ones(2, 4)
        ref = weakref.ref(value)
        old.write(value)
        del value, old
        batch.start(fb)
        fresh = batch.stream_of(fb)
        self.assertIs(fresh, batch.stream_of(fb))
        self.assertIsNone(fresh.pending)
        self.assertIsNone(fresh.residual)
        self.assertIsNone(ref())

    def test_terminal_completion_releases_batch_before_final_norm(self):
        fb = SimpleNamespace(residual_stream=None)
        batch.start(fb)
        stream = batch.stream_of(fb)
        residual = torch.full((2, 4), 3.0)
        partial = torch.ones(2, 4)
        group = SimpleNamespace(all_reduce=Mock(side_effect=lambda x: x * 2))
        stream.write(residual)
        hidden = stream.record(UnreducedOutput(partial, group=group), PLAIN_ADD)
        result = batch.final_norm(
            hidden, fb, lambda value, prior: (value + prior, prior)
        )
        self.assertIsNone(fb.residual_stream)
        group.all_reduce.assert_called_once()
        torch.testing.assert_close(result, torch.full((2, 4), 5.0))
        with self.assertRaisesRegex(RuntimeError, "start"):
            batch.stream_of(fb)

    def test_pp_export_releases_the_batch_owner(self):
        fb = SimpleNamespace(residual_stream=None)
        batch.start(fb)
        hidden = torch.full((2, 4), 2.0)
        residual = torch.ones(2, 4)
        stream = batch.stream_of(fb)
        stream.write(residual)
        output = stream.record(hidden, PLAIN_ADD)
        proxy = batch.to_pp(output, fb)
        self.assertIs(proxy["hidden_states"], hidden)
        self.assertIs(proxy["residual"], residual)
        self.assertIsNone(fb.residual_stream)

    def test_tbo_metadata_and_reused_batch_remain_stream_free_between_calls(self):
        size = 4
        parent = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=size,
            input_ids=torch.zeros(size, dtype=torch.long),
            positions=torch.zeros(size, dtype=torch.long),
            out_cache_loc=torch.zeros(size, dtype=torch.long),
            req_pool_indices=torch.zeros(size, dtype=torch.long),
            seq_lens=torch.ones(size, dtype=torch.int32),
            seq_lens_cpu=torch.ones(size, dtype=torch.int32),
            seq_lens_sum=size,
            spec_info=None,
        )
        for _ in range(2):
            self.assertIsNone(parent.residual_stream)
            with (
                get_context().override_server_args(
                    attention_backend="fa3", moe_dense_tp_size=None
                ),
                get_parallel().override(attn_tp_size=1),
            ):
                children = [
                    TboForwardBatchPreparer.filter_batch(
                        parent,
                        start_token_index=start,
                        end_token_index=start + 2,
                        start_seq_index=start,
                        end_seq_index=start + 2,
                        out_num_token_non_padded=torch.tensor(2),
                        out_num_token_non_padded_cpu=2,
                    )
                    for start in (0, 2)
                ]
            self.assertTrue(all(child.residual_stream is None for child in children))
            batch.start(parent)
            owner = batch.stream_of(parent)
            self.assertTrue(all(child.residual_stream is None for child in children))
            hidden = owner.write(torch.ones(size, 4))
            batch.final_norm(hidden, parent, lambda value: value)
            cloned = replace(parent)
            self.assertIsNone(cloned.residual_stream)
            self.assertIsNone(parent.residual_stream)

    def test_split_prefill_retains_then_releases_batch_owner(self):
        from sglang.srt.models.llama4 import Llama4ForCausalLM
        from sglang.srt.models.qwen2_moe import Qwen2MoeForCausalLM
        from sglang.srt.models.qwen3 import Qwen3ForCausalLM
        from sglang.srt.models.qwen3_moe import Qwen3MoeForCausalLM
        from sglang.srt.models.sarvam_moe import (
            SarvamMLAForCausalLM,
            SarvamMoEForCausalLM,
        )

        for cls in (
            Llama4ForCausalLM,
            Qwen3ForCausalLM,
            Qwen3MoeForCausalLM,
            Qwen2MoeForCausalLM,
            SarvamMLAForCausalLM,
            SarvamMoEForCausalLM,
        ):
            with self.subTest(model=cls.__name__):
                context = (
                    nullcontext()
                    if cls in (Llama4ForCausalLM, Qwen3ForCausalLM)
                    else patch(
                        cls.__module__ + ".get_global_expert_distribution_recorder",
                        return_value=SimpleNamespace(
                            with_current_layer=lambda i: nullcontext()
                        ),
                    )
                )
                with context:
                    self._check_split_prefill(cls)

    def _check_split_prefill(self, cls):
        owner_ids = []
        group = SimpleNamespace(all_reduce=Mock(side_effect=lambda x: x * 2))

        class Layer:
            def __call__(self, positions, hidden, forward_batch, **kwargs):
                stream = batch.stream_of(forward_batch)
                owner_ids.append(id(stream))
                hidden, old = stream.export(hidden)
                stream.write(hidden if old is None else hidden + old)
                return stream.record(
                    UnreducedOutput(torch.ones_like(hidden), group=group), PLAIN_ADD
                )

        self_outer = self
        fb = SimpleNamespace(
            residual_stream=None, residual=None, forward_mode=ForwardMode.DECODE
        )

        def norm(hidden, residual):
            self.assertIsNone(fb.residual_stream)
            return hidden + residual, residual

        model = SimpleNamespace(
            config=SimpleNamespace(num_hidden_layers=4),
            layers=[Layer() for _ in range(4)],
            norm=norm,
        )
        lm = SimpleNamespace(
            model=model,
            lm_head=None,
            logits_processor=lambda ids, hidden, head, fb: hidden,
        )
        value = torch.ones(2, 4)
        result = cls.forward_split_prefill(lm, None, None, fb, (0, 2), value)
        self.assertIsNone(result)
        self.assertIsNotNone(fb.residual_stream)
        result = cls.forward_split_prefill(lm, None, None, fb, (2, 4))
        self.assertEqual(len(set(owner_ids)), 1)
        self.assertIsNone(fb.residual_stream)
        self.assertIsNone(fb.residual)
        self.assertEqual(group.all_reduce.call_count, 4)
        torch.testing.assert_close(result, torch.full((2, 4), 9.0))

    def test_mtp_returns_written_normalized_output_and_releases_batch(self):
        from sglang.srt.models.nemotron_h_mtp import NemotronHMultiTokenPredictor

        def terminal_layer(*, inputs_embeds, hidden_states, forward_batch):
            stream = batch.stream_of(forward_batch)
            stream.write(hidden_states)
            hidden_states = stream.record(torch.ones_like(hidden_states), PLAIN_ADD)
            hidden_states = batch.fold(hidden_states, forward_batch)
            normalized = hidden_states * 3
            return batch.set_written(normalized, forward_batch)

        predictor = SimpleNamespace(pattern_len=1, layers={"0": terminal_layer})
        fb = SimpleNamespace(
            residual_stream=None,
            spec_info=SimpleNamespace(hidden_states=torch.ones(2, 4)),
        )
        result = NemotronHMultiTokenPredictor.forward(
            predictor, None, None, fb, inputs_embeds=torch.zeros(2, 4)
        )
        torch.testing.assert_close(result, torch.full((2, 4), 6.0))
        self.assertIsNone(fb.residual_stream)

    def test_take_output_rejects_pending_or_mismatched_outputs(self):
        fb = SimpleNamespace(residual_stream=None)
        batch.start(fb)
        stream = batch.stream_of(fb)
        value = torch.ones(2, 4)
        stream.write(value)
        pending = stream.record(value * 2, PLAIN_ADD)
        with self.assertRaises(RuntimeError):
            batch.take_output(pending, fb)
        self.assertIs(batch.stream_of(fb), stream)
        stream.write(value)
        with self.assertRaises(RuntimeError):
            batch.take_output(value.clone(), fb)
        self.assertIs(batch.take_output(value, fb), value)
        self.assertIsNone(fb.residual_stream)


if __name__ == "__main__":
    unittest.main()
