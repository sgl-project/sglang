"""GLM-5.3-Flash layerwise prefill must match an uninterrupted forward."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from sglang.srt.layers.layer_boundary import PLAIN_ADD
from sglang.srt.layers.layer_boundary.output import UnreducedOutput
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.models.glm5_next import (
    Glm5NextDecoderLayer,
    Glm5NextForConditionalGeneration,
    Glm5NextModel,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class FakeLayer(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.seen_topk = []
        self.group = SimpleNamespace(all_reduce=lambda value: value * 2)

    def forward(
        self,
        positions,
        hidden_states,
        forward_batch,
        zero_allocator,
        gemm_output_zero_allocator,
        prev_topk_indices=None,
        capture_output=None,
    ):
        self.seen_topk.append(prev_topk_indices)
        prev = 0 if prev_topk_indices is None else prev_topk_indices.item()
        stream = residual_batch.stream_of(forward_batch)
        hidden_states, residual = stream.export(hidden_states)
        written = hidden_states if residual is None else hidden_states + residual
        stream.write(written)
        if capture_output is not None:
            capture_output(written)
        output = torch.full_like(written, self.layer_id + prev + 1)
        hidden_states = stream.record(
            UnreducedOutput(output, group=self.group), PLAIN_ADD
        )
        return hidden_states, torch.tensor([self.layer_id + 1])


class FakeNorm(nn.Module):
    def forward(self, hidden_states, residual=None):
        if residual is None:
            return hidden_states + 10
        return hidden_states + residual + 10, None


def make_model():
    model = Glm5NextModel.__new__(Glm5NextModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(mhc=False)
    model.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    model.embed_tokens = nn.Embedding(8, 2)
    model.layers = nn.ModuleList(FakeLayer(i) for i in range(3))
    model.start_layer = 0
    model.end_layer = 3
    model.first_k_dense_replace = 0
    model.gemm_output_zero_allocator_size = 0
    model.layers_to_capture = [2]
    model.dflash_capture = False
    model.enable_a2a_moe = False
    model.norm = FakeNorm()
    return model


def make_batch():
    return SimpleNamespace(
        forward_mode=ForwardMode.SPLIT_PREFILL,
        can_run_tbo=False,
        hidden_states=None,
        residual_stream=None,
        model_specific_states=None,
    )


class TestGlm5NextPDMux(unittest.TestCase):
    def test_decode_helper_is_reset_on_the_next_prefill_slice(self):
        helper = object()
        layer = SimpleNamespace(
            pdmux_alt_stream=helper,
            is_linear_attn=False,
            is_layer_sparse=True,
            self_attn=SimpleNamespace(
                alt_stream=None,
                use_dsa=True,
                indexer=SimpleNamespace(alt_stream=None),
            ),
            mlp=SimpleNamespace(alt_stream=None),
        )
        with patch(
            "sglang.srt.models.glm5_next.get_pdmux_decode_alt_stream",
            return_value=helper,
        ):
            for mode, expected in (
                (ForwardMode.DECODE, helper),
                (ForwardMode.SPLIT_PREFILL, None),
                (ForwardMode.IDLE, helper),
                (ForwardMode.EXTEND, None),
            ):
                Glm5NextDecoderLayer._set_pdmux_alt_stream(
                    layer, SimpleNamespace(forward_mode=mode)
                )
                self.assertIs(layer.self_attn.alt_stream, expected)
                self.assertIs(layer.self_attn.indexer.alt_stream, expected)
                self.assertIs(layer.mlp.alt_stream, expected)

            # DSA layers that reuse the previous top-k do not own an indexer.
            layer.self_attn.indexer = None
            Glm5NextDecoderLayer._set_pdmux_alt_stream(
                layer, SimpleNamespace(forward_mode=ForwardMode.DECODE)
            )
            self.assertIs(layer.self_attn.alt_stream, helper)

    def test_idle_dp_rank_skips_split_prefill_attention_plan(self):
        runner = SimpleNamespace(
            attn_backend=Mock(),
            model=SimpleNamespace(forward_split_prefill=Mock(return_value=None)),
            model_config=SimpleNamespace(num_hidden_layers=2),
            device_timer=None,
        )
        batch = SimpleNamespace(
            split_index=0,
            forward_mode=ForwardMode.IDLE,
            input_ids=torch.empty(0, dtype=torch.long),
            positions=torch.empty(0, dtype=torch.long),
        )

        with patch(
            "sglang.srt.model_executor.model_runner.device_timer_ctx",
            return_value=nullcontext(),
        ):
            result = ModelRunner.forward_split_prefill(runner, batch, forward_count=1)

        self.assertIsNone(result)
        self.assertEqual(batch.split_index, 1)
        runner.attn_backend.init_forward_metadata.assert_not_called()
        runner.model.forward_split_prefill.assert_called_once()

    def test_intermediate_segment_restores_mlp_sync_padding(self):
        batch = SimpleNamespace(
            _original_forward_mode=None,
            _original_batch_size=2,
            _original_num_tokens=3,
            batch_size=4,
            spec_info=None,
            positions=torch.arange(5),
            seq_lens=torch.arange(4),
            req_pool_indices=torch.arange(4),
            req_pool_indices_cpu=list(range(4)),
            seq_lens_cpu=list(range(4)),
        )

        ForwardBatch.post_forward_mlp_sync_batch(batch, None)

        self.assertEqual(batch.batch_size, 2)
        self.assertEqual(len(batch.positions), 3)
        self.assertEqual(len(batch.seq_lens), 2)
        self.assertEqual(len(batch.req_pool_indices), 2)
        self.assertEqual(batch.req_pool_indices_cpu, [0, 1])
        self.assertEqual(batch.seq_lens_cpu, [0, 1])

    def test_split_prefill_preserves_residual_topk_and_aux_capture(self):
        model = make_model()
        batch = make_batch()
        input_ids = torch.tensor([1, 2])
        positions = torch.tensor([0, 1])
        embeddings = model.embed_tokens(input_ids)
        recorder = SimpleNamespace(with_current_layer=lambda i: nullcontext())

        with (
            patch(
                "sglang.srt.models.glm5_next.get_global_expert_distribution_recorder",
                return_value=recorder,
            ),
            patch(
                "sglang.srt.models.glm5_next.check_cuda_graph_backend",
                return_value=False,
            ),
        ):
            expected, expected_aux = model.forward(
                input_ids, positions, make_batch(), input_embeds=embeddings
            )
            self.assertIsNone(
                model.forward(
                    None,
                    positions,
                    batch,
                    input_embeds=embeddings,
                    split_interval=(0, 1),
                )
            )
            stream = batch.residual_stream
            self.assertIsNotNone(stream.pending)
            self.assertIsNone(
                model.forward_split_prefill(input_ids, positions, batch, (1, 2))
            )
            self.assertIs(batch.residual_stream, stream)
            actual, actual_aux = model.forward_split_prefill(
                input_ids, positions, batch, (2, 3)
            )

        self.assertIsNone(batch.residual_stream)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_aux[0], expected_aux[0])
        self.assertEqual(len(actual_aux), 1)
        self.assertEqual(model.layers[1].seen_topk[-1].item(), 1)
        self.assertEqual(model.layers[2].seen_topk[-1].item(), 2)

    def test_multimodal_embedding_runs_only_for_first_segment(self):
        wrapper = Glm5NextForConditionalGeneration.__new__(
            Glm5NextForConditionalGeneration
        )
        nn.Module.__init__(wrapper)
        language_model = Mock(start_layer=0)
        language_model.forward_split_prefill.return_value = torch.ones(2, 2)
        wrapper.model = language_model
        wrapper.is_mrope_enabled = True
        wrapper.capture_aux_hidden_states = False
        wrapper.logits_processor = Mock(return_value="logits")
        wrapper.lm_head = None
        batch = SimpleNamespace(mrope_positions=torch.tensor([10, 11]))
        input_ids = torch.tensor([1, 2])
        attn_context = SimpleNamespace(
            maybe_input_scattered=lambda forward_batch: nullcontext()
        )

        with (
            patch(
                "sglang.srt.models.glm5_next.get_attn_tp_context",
                return_value=attn_context,
            ),
            patch(
                "sglang.srt.models.glm5_next.general_mm_embed_routine",
                return_value=None,
            ) as embed,
        ):
            self.assertIsNone(
                wrapper.forward_split_prefill(
                    input_ids, torch.tensor([0, 1]), batch, (0, 1)
                )
            )
            self.assertEqual(
                wrapper.forward_split_prefill(
                    input_ids, torch.tensor([0, 1]), batch, (1, 2)
                ),
                "logits",
            )

        embed.assert_called_once()
        self.assertIs(embed.call_args.kwargs["language_model"], language_model)
        torch.testing.assert_close(
            embed.call_args.kwargs["positions"], batch.mrope_positions
        )
        language_model.forward_split_prefill.assert_called_once()
        torch.testing.assert_close(
            language_model.forward_split_prefill.call_args.args[1],
            batch.mrope_positions,
        )


if __name__ == "__main__":
    unittest.main()
