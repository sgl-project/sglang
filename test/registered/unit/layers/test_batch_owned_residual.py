"""Residual ownership across layer execution and batch reuse."""

import unittest
import weakref
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.batch_overlap.two_batch_overlap import TboForwardBatchPreparer
from sglang.srt.layers.communicator import (
    ADD,
    BoundarySteps,
    EdgeDecl,
    LayerCommunicator,
    Layout,
    StageEntry,
    StageInput,
    StageOutput,
    make_boundary,
)
from sglang.srt.layers.communicator.legacy_stage import StageCommunicator
from sglang.srt.layers.communicator.output import UnreducedOutput
from sglang.srt.layers.communicator.residual import batch
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestBatchOwnedResidual(CustomTestCase):
    def test_start_releases_previous_call_and_resets_state(self):
        fb = SimpleNamespace(residual_stream=None)
        with self.assertRaisesRegex(RuntimeError, "start"):
            batch.current(fb)
        old = batch.start(fb)
        self.assertIs(old, batch.current(fb))
        value = torch.ones(2, 4)
        ref = weakref.ref(value)
        old.write(value)
        del value, old
        fresh = batch.start(fb)
        self.assertIs(fresh, batch.current(fb))
        self.assertIsNone(fresh.pending)
        self.assertIsNone(fresh.residual)
        self.assertIsNone(ref())

    def test_terminal_completion_releases_batch_before_final_norm(self):
        fb = SimpleNamespace(residual_stream=None)
        stream = batch.start(fb)
        residual = torch.full((2, 4), 3.0)
        partial = torch.ones(2, 4)
        group = SimpleNamespace(all_reduce=Mock(side_effect=lambda x: x * 2))
        stream.write(residual)
        hidden = stream.leave(UnreducedOutput(partial, group=group), ADD)
        complete, residual_value = LayerCommunicator.finish_layer_stack(
            hidden, stream, fb
        )
        self.assertIsNone(fb.residual_stream)
        self.assertIs(residual_value, residual)
        group.all_reduce.assert_called_once()
        torch.testing.assert_close(complete + residual_value, torch.full((2, 4), 5.0))
        with self.assertRaisesRegex(RuntimeError, "start"):
            batch.current(fb)

    def test_pp_receive_attaches_and_export_releases_the_same_stream(self):
        communicator = LayerCommunicator.__new__(LayerCommunicator)
        communicator._batch_steps = lambda fb: SimpleNamespace(
            attention=SimpleNamespace(input_sum=None), ffn=None
        )
        fb = SimpleNamespace(residual_stream=None)
        hidden = torch.full((2, 4), 2.0)
        residual = torch.ones(2, 4)
        received, stream = communicator.from_pp(
            PPProxyTensors({"hidden_states": hidden, "residual": residual}), fb
        )
        self.assertIs(stream, batch.current(fb))
        self.assertIs(stream.pending.value, received)
        self.assertIs(stream.pending.update, ADD)
        exported, exported_residual = communicator.finish_layer_stack(
            received, stream, fb
        )
        self.assertIs(exported, hidden)
        self.assertIs(exported_residual, residual)
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
            owner = batch.start(parent)
            self.assertTrue(all(child.residual_stream is None for child in children))
            hidden = owner.write(torch.ones(size, 4))
            LayerCommunicator.finish_layer_stack(hidden, owner, parent)
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
            layer_communicator = LayerCommunicator.__new__(LayerCommunicator)

            def __call__(self, positions, hidden, forward_batch, *legacy_residual):
                stream = batch.current(forward_batch)
                if legacy_residual:
                    self_outer.assertIs(legacy_residual[0], stream)
                owner_ids.append(id(stream))
                hidden, old = stream.finish(hidden)
                stream.write(hidden if old is None else hidden + old)
                result = stream.leave(
                    UnreducedOutput(torch.ones_like(hidden), group=group), ADD
                )
                return (result, stream) if legacy_residual else result

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
            stream = batch.current(forward_batch)
            stream.write(hidden_states)
            hidden_states = stream.leave(torch.ones_like(hidden_states), ADD)
            hidden_states = batch.fold(hidden_states, forward_batch)
            normalized = hidden_states * 3
            return batch.written(normalized, forward_batch)

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
        stream = batch.start(fb)
        value = torch.ones(2, 4)
        stream.write(value)
        pending = stream.leave(value * 2, ADD)
        with self.assertRaises(RuntimeError):
            batch.take_output(pending, fb)
        self.assertIs(batch.current(fb), stream)
        stream.write(value)
        with self.assertRaises(RuntimeError):
            batch.take_output(value.clone(), fb)
        self.assertIs(batch.take_output(value, fb), value)
        self.assertIsNone(fb.residual_stream)

    def test_interleaved_stages_share_only_their_own_batch_stream(self):
        class Read:
            norms_plainly = False

            def enter(self, value):
                return value

            def read(self, value, norm, quant_format="", **kwargs):
                return value * 2, value

            def update_and_read(self, update, value, residual, norm, **kwargs):
                updated = update.update(value, residual)
                return updated * 2, updated

        rows = Layout(frozenset())
        boundary = make_boundary(
            EdgeDecl(StageOutput(rows), StageInput(rows, read=Read()), rows, rows)
        )
        steps = BoundarySteps(
            StageEntry(boundary.prepare, rows), None, StageOutput(rows), None, False
        )
        stage = StageCommunicator(
            SimpleNamespace(input_layernorm=None),
            "attention",
            "input_layernorm",
        )
        a, b = [SimpleNamespace(residual_stream=None) for _ in range(2)]
        sa, sb = batch.start(a), batch.start(b)
        first, first_alias = stage._prepare(torch.ones(2, 4), sa, a, steps)
        second, second_alias = stage._prepare(torch.full((2, 4), 10.0), sb, b, steps)
        self.assertIs(first_alias, batch.current(a))
        self.assertIs(second_alias, batch.current(b))
        first = sa.leave(first * 3, ADD)
        second = sb.leave(second * 5, ADD)
        torch.testing.assert_close(
            stage._prepare(first, sa, a, steps)[0], torch.full((2, 4), 14.0)
        )
        torch.testing.assert_close(
            stage._prepare(second, sb, b, steps)[0], torch.full((2, 4), 220.0)
        )
        self.assertIsNot(batch.current(a), batch.current(b))
        batch.start(a)
        self.assertIsNone(batch.current(a).residual)
        self.assertIsNotNone(batch.current(b).residual)


if __name__ == "__main__":
    unittest.main()
