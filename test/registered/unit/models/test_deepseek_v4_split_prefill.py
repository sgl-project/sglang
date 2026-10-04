import unittest
from contextlib import ExitStack, nullcontext
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.logits_processor import (
    LogitsMetadata,
    LogitsProcessor,
    LogitsProcessorOutput,
)
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM, DeepseekV4Model
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Layer:
    def __init__(self, index):
        self.index = index
        self.hc_post_calls = 0
        self.calls = []

    def __call__(self, *, hidden_states, prev_residual, prev_post, prev_comb, **kw):
        self.calls.append((prev_residual, prev_post, prev_comb))
        value = self.index + 1
        return hidden_states + value, value, value + 10, value + 20

    def hc_post(self, hidden_states, residual, post, comb):
        self.hc_post_calls += 1
        return hidden_states + residual + post + comb


class _Engram:
    layer_hash_index = 0

    def __init__(self):
        self.calls = []

    def __call__(self, hidden, hashes, batch, **kwargs):
        self.calls.append(hashes.clone())
        return hidden + hashes[:, None, None]


class _PrevLayer:
    def __init__(self, index):
        self.index = index
        self.engram = None
        self.calls = []
        self.input_layernorm = object()

    def forward_hc_pre_from_prev(
        self,
        *,
        hidden_states,
        prev_pre,
        combined_attn,
        normalized_attn,
        next_combined,
        input_ids,
        positions,
        **kwargs,
    ):
        self.calls.append(
            (prev_pre, combined_attn, normalized_attn, input_ids, positions)
        )
        value = self.index + 1
        hidden = hidden_states + value
        if combined_attn is not None:
            hidden = hidden + combined_attn[:, None, :] * 0.01
        if normalized_attn is not None:
            hidden = hidden + normalized_attn[:, None, :] * 0.02
        if next_combined is not None:
            next_combined.append((hidden.mean(dim=1), hidden.mean(dim=1) + 10))
        pre = hidden.new_full((hidden.shape[0], 1), value)
        return hidden, pre


class _Tail:
    def __init__(self):
        self.token_indices = torch.tensor([1, 3])
        self.positions = torch.tensor([1, 3])
        self.extend_seq_lens = torch.tensor([2], dtype=torch.int32)
        self.extend_seq_lens_cpu = [2]

    def rows(self, tensor):
        return None if tensor is None else tensor[self.token_indices]

    def scatter(self, rows, num_tokens):
        out = rows.new_zeros((num_tokens, *rows.shape[1:]))
        out[self.token_indices] = rows
        return out


class TestDeepseekV4SplitPrefill(unittest.TestCase):
    def setUp(self):
        self.backend = SimpleNamespace()
        self.stack = ExitStack()
        for name, value in (
            ("get_parallel", SimpleNamespace(attn_dp_size=1)),
            ("is_cp_active", False),
            ("check_cuda_graph_backend", True),
            ("get_attn_backend", self.backend),
            ("get_platform", SimpleNamespace(is_blackwell=False)),
        ):
            self.stack.enter_context(
                patch("sglang.srt.models.deepseek_v4." + name, return_value=value)
            )
        self.stack.enter_context(
            patch(
                "sglang.kernels.ops.layernorm.mhc.hc_combine",
                side_effect=lambda x, pre, *_: (
                    x.reshape(x.shape[0], 2, x.shape[1] // 2).sum(dim=1) + pre
                ),
            )
        )
        self.addCleanup(self.stack.close)

    def model(self, *, predecessor=False, count=4):
        layers = [(_PrevLayer if predecessor else _Layer)(i) for i in range(count)]
        model = SimpleNamespace(
            pp_group=SimpleNamespace(
                world_size=1, is_first_rank=True, is_last_rank=True
            ),
            embed_tokens=lambda ids: ids.float()[:, None],
            hc_mult=2,
            layers=layers,
            start_layer=0,
            end_layer=count,
            hc_pre_from_prev_sublayer=predecessor,
            use_fused_mhc_post_pre=True,
            hc_head=Mock(side_effect=lambda hidden, *_: hidden.sum(dim=1)),
            hc_head_fn=None,
            hc_head_scale=None,
            hc_head_base=None,
            norm=Mock(side_effect=lambda hidden: hidden * 2),
            dspark_layers_to_capture=None,
            engram_hasher=None,
            late_layer_start=None,
            config=SimpleNamespace(model_type="deepseek_v41", vision_n_layers=0),
            _can_run_tbo=lambda batch: False,
            _check_late_layer_tail_readers=lambda batch: None,
        )
        for method in ("_forward_layers_hc_pre_from_prev", "_finalize_hidden_states"):
            setattr(model, method, MethodType(getattr(DeepseekV4Model, method), model))
        return model

    def batch(self, count=4, offset=0):
        return SimpleNamespace(
            input_ids=torch.arange(count) + 1 + offset,
            positions=torch.arange(count),
            forward_mode=ForwardMode.EXTEND,
            attn_cp_metadata=None,
            hidden_states=None,
            model_specific_states=None,
        )

    def normal(self, model, batch):
        return DeepseekV4Model.forward(
            model, batch.input_ids, batch.positions, batch, None
        )

    def split(self, model, batch, interval):
        return DeepseekV4Model.forward_split_prefill(
            model, batch.input_ids, batch.positions, batch, interval
        )

    def assert_results_equal(self, actual, expected):
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e)

    def test_dp_gather_cannot_modify_input_ids_reused_by_later_slices(self):
        model = self.model()
        batch = self.batch()
        original_ids = batch.input_ids.clone()

        def gather(output, local_ids, forward_batch):
            output.copy_(local_ids)
            # MAX_LEN's replicate gather may zero local padding in place.
            local_ids.zero_()

        with (
            patch(
                "sglang.srt.models.deepseek_v4.get_parallel",
                return_value=SimpleNamespace(attn_dp_size=2),
            ),
            patch(
                "sglang.srt.models.deepseek_v4.get_moe_a2a_backend",
                return_value=SimpleNamespace(is_none=lambda: True),
            ),
            patch(
                "sglang.srt.models.deepseek_v4.get_global_dp_buffer_len",
                return_value=4,
            ),
            patch(
                "sglang.srt.models.deepseek_v4.dp_gather_replicate", side_effect=gather
            ) as gather_call,
        ):
            self.assertIsNone(self.split(model, batch, (0, 1)))
            torch.testing.assert_close(batch.input_ids, original_ids)
            torch.testing.assert_close(
                batch.model_specific_states["input_ids_global"], original_ids
            )
            self.split(model, batch, (1, 4))
            torch.testing.assert_close(batch.input_ids, original_ids)
            gather_call.assert_called_once()

    def test_fused_mhc_split_matches_ordinary_forward_and_finalizes_once(self):
        model = self.model()
        batch = self.batch()
        batch.freqs_cis_c4 = batch.freqs_cis_c128 = object()
        self.assertIsNone(self.split(model, batch, (0, 1)))
        self.assertFalse(hasattr(batch, "freqs_cis_c4"))
        self.assertFalse(hasattr(batch, "freqs_cis_c128"))
        self.assertIsNone(self.split(model, batch, (1, 3)))
        model.norm.assert_not_called()
        model.hc_head.assert_not_called()
        self.assertEqual(model.layers[1].calls[0], (1, 11, 21))
        result = self.split(model, batch, (3, 4))
        self.assertEqual([layer.hc_post_calls for layer in model.layers], [0, 0, 0, 1])
        model.norm.assert_called_once()
        model.hc_head.assert_called_once()
        self.assert_results_equal(result, self.normal(self.model(), self.batch()))

    def test_predecessor_combined_and_normalized_inputs_cross_slice(self):
        model = self.model(predecessor=True)
        batch = self.batch(128)
        self.assertIsNone(self.split(model, batch, (0, 1)))
        result = self.split(model, batch, (1, 4))
        self.assertIsNotNone(model.layers[1].calls[0][0])
        self.assertIsNotNone(model.layers[1].calls[0][1])
        self.assertIsNotNone(model.layers[1].calls[0][2])
        self.assert_results_equal(
            result, self.normal(self.model(predecessor=True), self.batch(128))
        )

    def test_engram_hashes_are_batch_owned_and_reset_collapsed_carry(self):
        def configure():
            model = self.model(predecessor=True)
            model.engram_hasher = Mock(side_effect=lambda ids, _: ids[:, None])
            model.layers[2].engram = _Engram()
            return model

        model = configure()
        batch = self.batch(128)
        self.assertIsNone(self.split(model, batch, (0, 2)))
        result = self.split(model, batch, (2, 4))
        model.engram_hasher.assert_called_once()
        self.assertEqual(model.layers[2].calls[0][1:3], (None, None))
        self.assert_results_equal(result, self.normal(configure(), self.batch(128)))

    def test_tail_crosses_slice_and_restores_full_rows(self):
        tail = _Tail()
        self.backend.tail_forward_metadata = SimpleNamespace(late_layer_tail=tail)

        def enter(batch):
            saved = batch.input_ids
            batch.input_ids = tail.rows(batch.input_ids)
            return saved

        def leave(saved, batch):
            batch.input_ids = saved

        self.backend.enter_late_layer_tail = Mock(side_effect=enter)
        self.backend.exit_late_layer_tail = Mock(side_effect=leave)
        self.stack.enter_context(
            patch(
                "sglang.srt.models.deepseek_v4._scatter_tail_rows",
                side_effect=lambda *, tail, rows, num_tokens: tail.scatter(
                    rows, num_tokens
                ),
            )
        )
        model = self.model(predecessor=True)
        model.late_layer_start = 2
        batch = self.batch()
        full_ids = batch.input_ids.clone()
        self.assertIsNone(self.split(model, batch, (0, 1)))
        self.assertIsNone(self.split(model, batch, (1, 3)))
        self.backend.exit_late_layer_tail.assert_not_called()
        result = self.split(model, batch, (3, 4))
        torch.testing.assert_close(batch.input_ids, full_ids)
        self.assertEqual(result[0].shape[0], 4)
        self.assertEqual(result[1].shape[0], 4)
        self.backend.exit_late_layer_tail.assert_called_once()
        ordinary = self.model(predecessor=True)
        ordinary.late_layer_start = 2
        self.assert_results_equal(result, self.normal(ordinary, self.batch()))

    def test_interleaved_batches_do_not_share_mhc_or_engram_state(self):
        model = self.model(predecessor=True)
        model.engram_hasher = Mock(side_effect=lambda ids, _: ids[:, None])
        a, b = self.batch(), self.batch(offset=20)
        self.assertIsNone(self.split(model, a, (0, 1)))
        self.assertIsNone(self.split(model, b, (0, 2)))
        self.normal(model, self.batch(offset=40))
        result_a, result_b = self.split(model, a, (1, 4)), self.split(model, b, (2, 4))
        self.assertIsNot(
            a.model_specific_states["hc_pre"], b.model_specific_states["hc_pre"]
        )
        self.assert_results_equal(
            result_a, self.normal(self.model(predecessor=True), self.batch())
        )
        self.assert_results_equal(
            result_b, self.normal(self.model(predecessor=True), self.batch(offset=20))
        )

    def test_wrapper_prepares_multimodal_embedding_once_and_emits_final_logits(self):
        ids = torch.tensor([1, 2**30])
        embeds = torch.tensor([[1.0], [99.0]])

        def split_forward(input_ids, positions, batch, interval, input_embeds):
            if interval[0] == 0:
                batch.model_specific_states = {"logits_input_ids": input_ids}
                return None
            return torch.ones(2, 1), torch.ones(2, 2)

        model = SimpleNamespace(
            vision=object(),
            config=SimpleNamespace(image_token_id=5),
            model=SimpleNamespace(
                start_layer=0, forward_split_prefill=Mock(side_effect=split_forward)
            ),
            _prepare_mm_embeddings=Mock(return_value=embeds),
            logits_processor=Mock(return_value="logits"),
            lm_head=object(),
            capture_aux_hidden_states=False,
        )
        batch = self.batch(2)
        batch.mm_inputs = [object()]
        with patch(
            "sglang.srt.models.deepseek_v4.get_attn_tp_context",
            return_value=SimpleNamespace(maybe_input_scattered=lambda _: nullcontext()),
        ):
            self.assertIsNone(
                DeepseekV4ForCausalLM.forward_split_prefill(
                    model, ids, batch.positions, batch, (0, 1)
                )
            )
            self.assertEqual(
                DeepseekV4ForCausalLM.forward_split_prefill(
                    model, ids, batch.positions, batch, (1, 2)
                ),
                "logits",
            )
        model._prepare_mm_embeddings.assert_called_once()
        model.logits_processor.assert_called_once()
        self.assertIs(
            model.model.forward_split_prefill.call_args_list[0].args[4], embeds
        )
        torch.testing.assert_close(
            model.logits_processor.call_args.args[0], torch.tensor([1, 5])
        )
        self.assertEqual(
            model.logits_processor.call_args.kwargs["hidden_states_before_norm"].shape,
            (2, 2),
        )
        torch.testing.assert_close(ids, torch.tensor([1, 2**30]))

    def wrapper(self, body, *, capture_aux=False):
        body.forward_split_prefill = MethodType(
            DeepseekV4Model.forward_split_prefill, body
        )

        def logits(ids, hidden, head, metadata, aux, *, hidden_states_before_norm):
            # Use the real FULL hidden-selection behavior consumed by NextN
            # and DSpark, while keeping LM-head kernels out of the CPU test.
            captured = LogitsProcessor._get_hidden_states_to_store(
                None,
                hidden,
                hidden_states_before_norm,
                aux,
                hidden,
                hidden_states_before_norm,
                aux,
                None,
                metadata,
            )
            return LogitsProcessorOutput(
                next_token_logits=hidden, hidden_states=captured
            )

        return SimpleNamespace(
            vision=None,
            model=body,
            capture_aux_hidden_states=capture_aux,
            logits_processor=Mock(side_effect=logits),
            lm_head=object(),
        )

    def wrapper_split(self, wrapper, batch, interval):
        with patch(
            "sglang.srt.models.deepseek_v4.get_attn_tp_context",
            return_value=SimpleNamespace(maybe_input_scattered=lambda _: nullcontext()),
        ):
            return DeepseekV4ForCausalLM.forward_split_prefill(
                wrapper, batch.input_ids, batch.positions, batch, interval
            )

    def test_mtp_full_capture_preserves_flattened_pre_mhc_for_nextn(self):
        model = self.model(predecessor=True)
        model.embed_tokens = lambda ids: torch.stack(
            (ids.float(), ids.float() + 10, ids.float() + 20), dim=1
        )
        wrapper = self.wrapper(model)
        batch = self.batch()
        batch.capture_hidden_mode = CaptureHiddenMode.FULL
        self.assertIsNone(self.wrapper_split(wrapper, batch, (0, 2)))
        wrapper.logits_processor.assert_not_called()
        output = self.wrapper_split(wrapper, batch, (2, 4))
        self.assertEqual(output.hidden_states.shape, (4, 6))
        ordinary = self.model(predecessor=True)
        ordinary.embed_tokens = model.embed_tokens
        expected = self.normal(ordinary, self.batch())[1]
        torch.testing.assert_close(output.hidden_states, expected)
        torch.testing.assert_close(
            output.hidden_states.view(4 * model.hc_mult, 3),
            expected.view(4 * model.hc_mult, 3),
        )
        wrapper.logits_processor.assert_called_once()

    def test_mtp_idle_final_slice_publishes_empty_flattened_hidden(self):
        model = self.model(predecessor=True)
        wrapper = self.wrapper(model)
        batch = self.batch(0)
        batch.forward_mode = ForwardMode.IDLE
        batch.capture_hidden_mode = CaptureHiddenMode.FULL
        self.assertIsNone(self.wrapper_split(wrapper, batch, (0, 1)))
        output = self.wrapper_split(wrapper, batch, (1, 4))
        self.assertEqual(output.hidden_states.shape, (0, model.hc_mult))
        wrapper.logits_processor.assert_called_once()

    def test_dspark_capture_across_slices_matches_ordinary_forward(self):
        for predecessor in (False, True):
            with self.subTest(predecessor=predecessor):
                model = self.model(predecessor=predecessor)
                model.dspark_layers_to_capture = [0, 2]
                batch = self.batch()
                self.assertIsNone(self.split(model, batch, (0, 1)))
                aux = batch.model_specific_states["dspark_aux_hidden_states"]
                self.assertEqual(len(aux), 1)
                first_capture = aux[0].clone()
                result, captures = self.split(model, batch, (1, 4))
                self.assertIs(captures, aux)
                self.assertEqual(len(captures), 2)
                torch.testing.assert_close(captures[0], first_capture)
                ordinary = self.model(predecessor=predecessor)
                ordinary.dspark_layers_to_capture = model.dspark_layers_to_capture
                expected, expected_captures = self.normal(ordinary, self.batch())
                self.assert_results_equal(result, expected)
                for actual, expected in zip(captures, expected_captures):
                    torch.testing.assert_close(actual, expected)

    def test_dspark_aux_storage_is_independent_for_interleaved_prefill(self):
        model = self.model(predecessor=True)
        model.dspark_layers_to_capture = [0, 2]
        a, b = self.batch(), self.batch(offset=20)
        self.assertIsNone(self.split(model, a, (0, 1)))
        a_aux = a.model_specific_states["dspark_aux_hidden_states"]
        self.assertIsNone(self.split(model, b, (0, 3)))
        b_aux = b.model_specific_states["dspark_aux_hidden_states"]
        self.assertIsNot(a_aux, b_aux)
        self.assertEqual(len(a_aux), 1)
        self.assertEqual(len(b_aux), 2)
        _, final_a = self.split(model, a, (1, 4))
        _, final_b = self.split(model, b, (3, 4))
        ordinary = self.model(predecessor=True)
        ordinary.dspark_layers_to_capture = model.dspark_layers_to_capture
        for result, batch in (
            (final_a, self.batch()),
            (final_b, self.batch(offset=20)),
        ):
            _, expected = self.normal(ordinary, batch)
            for actual, expected in zip(result, expected):
                torch.testing.assert_close(actual, expected)

    def test_dspark_wrapper_captures_aux_instead_of_pre_mhc(self):
        body = self.model(predecessor=True)
        body.dspark_layers_to_capture = [0, 2]
        wrapper = self.wrapper(body, capture_aux=True)
        batch = self.batch()
        batch.capture_hidden_mode = CaptureHiddenMode.FULL
        self.assertIsNone(self.wrapper_split(wrapper, batch, (0, 1)))
        output = self.wrapper_split(wrapper, batch, (1, 4))
        aux = batch.model_specific_states["dspark_aux_hidden_states"]
        torch.testing.assert_close(output.hidden_states, torch.cat(aux, dim=-1))
        self.assertIsNone(
            wrapper.logits_processor.call_args.kwargs["hidden_states_before_norm"]
        )

    def test_dspark_tail_final_logits_use_batch_owned_rows_and_indices(self):
        tail = _Tail()
        self.backend.tail_forward_metadata = SimpleNamespace(late_layer_tail=tail)
        self.backend.enter_late_layer_tail = Mock(return_value=object())
        self.backend.exit_late_layer_tail = Mock()
        body = self.model(predecessor=True)
        body.late_layer_start = 2
        body.dspark_layers_to_capture = [0, 3]
        wrapper = self.wrapper(body, capture_aux=True)
        batch = self.batch()
        batch.capture_hidden_mode = CaptureHiddenMode.FULL
        batch.extend_seq_lens = torch.tensor([4], dtype=torch.int32)
        batch.extend_seq_lens_cpu = [4]
        batch.extend_logprob_start_lens_cpu = [4]
        full_ids = batch.input_ids.clone()
        self.assertIsNone(self.wrapper_split(wrapper, batch, (0, 1)))
        self.assertIsNone(self.wrapper_split(wrapper, batch, (1, 3)))
        wrapper.logits_processor.assert_not_called()
        # A later forward may refresh backend scratch. Finalization must use
        # the tail associated with these persisted target captures.
        self.backend.tail_forward_metadata = SimpleNamespace(late_layer_tail=object())
        metadata = LogitsMetadata(
            forward_mode=ForwardMode.EXTEND,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            extend_seq_lens=batch.extend_seq_lens,
            extend_seq_lens_cpu=batch.extend_seq_lens_cpu,
            extend_logprob_start_lens_cpu=batch.extend_logprob_start_lens_cpu,
        )
        with patch(
            "sglang.srt.models.deepseek_v4.LogitsMetadata.from_forward_batch",
            return_value=metadata,
        ):
            output = self.wrapper_split(wrapper, batch, (3, 4))
        call = wrapper.logits_processor.call_args
        torch.testing.assert_close(call.args[0], tail.rows(full_ids))
        self.assertIs(call.args[3], metadata)
        self.assertIs(metadata.extend_seq_lens, tail.extend_seq_lens)
        self.assertEqual(metadata.extend_seq_lens_cpu, [2])
        self.assertEqual(metadata.extend_logprob_start_lens_cpu, [2])
        self.assertEqual(batch.extend_seq_lens_cpu, [4])
        self.assertIs(output.hidden_states_token_indices, tail.token_indices)
        self.assertEqual(output.hidden_states.shape, (2, 2))
        torch.testing.assert_close(
            output.hidden_states, torch.cat(call.args[4], dim=-1)
        )
        self.backend.exit_late_layer_tail.assert_called_once()
        self.assertTrue(DeepseekV4ForCausalLM.supports_pdmux_dspark_prefill)


if __name__ == "__main__":
    unittest.main()
