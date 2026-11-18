"""Regression tests for multimodal IDs and Engram masking during graph replay."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import sglang.srt.models.deepseek_v4 as dsv4
from sglang.srt.layers.engram import engram_gate
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestDeepseekV41BCGMultimodal(CustomTestCase):
    def test_cp_engram_graph_writes_global_hash_buffer_before_sharding(self):
        class ReachedEngram(Exception):
            pass

        for cp, rank, local_rows in ((2, 0, 4), (2, 1, 4), (8, 0, 1), (8, 7, 1)):
            with self.subTest(cp=cp, rank=rank):
                ids = torch.arange(10, 16)
                output = torch.empty((6, 1), dtype=torch.long)
                hasher = Mock(side_effect=lambda ids, batch: ids[:, None])
                hasher.new_hash_ids.return_value = output
                expected = torch.zeros((local_rows, 1), dtype=torch.long)
                live = ids[rank::cp]
                expected[: live.numel(), 0] = live

                def read_engram(residual, hashes, batch, **kwargs):
                    torch.testing.assert_close(hashes, expected[:, 0])
                    self.assertTrue(kwargs["cp_all_tokens"])
                    raise ReachedEngram

                engram = Mock(side_effect=read_engram)
                engram.layer_hash_index = 0
                model = SimpleNamespace(
                    pp_group=SimpleNamespace(world_size=1),
                    engram_hasher=hasher,
                    late_layer_start=None,
                    decoder_replay_tail=None,
                    start_layer=0,
                    end_layer=1,
                    layers=[SimpleNamespace(engram=engram)],
                )
                batch = SimpleNamespace(
                    forward_mode=ForwardMode.EXTEND,
                    input_ids=ids,
                    attn_cp_metadata=SimpleNamespace(total_seq_lens=6),
                )
                with (
                    patch.object(dsv4, "is_cp_active", return_value=True),
                    patch.object(dsv4, "is_in_breakable_cuda_graph", return_value=True),
                    patch.object(dsv4, "_is_cuda", False),
                    patch.object(dsv4, "_is_hip", False),
                    patch.object(
                        dsv4,
                        "get_parallel",
                        return_value=SimpleNamespace(
                            attn_cp_size=cp, attn_cp_rank=rank, attn_dp_size=1
                        ),
                    ),
                    patch.object(
                        dsv4,
                        "get_forward",
                        return_value=SimpleNamespace(sp_active=False),
                    ),
                    patch.object(
                        dsv4,
                        "get_tc_piecewise_forward_context",
                        return_value=SimpleNamespace(forward_batch=batch),
                    ),
                    patch.object(
                        dsv4,
                        "bcg_deepseek_v4_engram_hash_ids",
                        side_effect=dsv4.deepseek_v4_engram_hash_ids,
                    ) as graph_hash,
                    self.assertRaises(ReachedEngram),
                ):
                    dsv4.DeepseekV4Model._forward_layers_hc_pre_from_prev(
                        model,
                        torch.arange(local_rows),
                        torch.zeros(local_rows, 1, 2),
                        batch,
                        live,
                        ids,
                        False,
                        None,
                    )
                self.assertEqual(graph_hash.call_count, 1)
                self.assertIs(graph_hash.call_args.args[2], output)
                torch.testing.assert_close(output[:, 0], ids)
                hasher.new_hash_ids.assert_called_once()
                hasher.assert_called_once()

    def test_cp_low_ratio_sources_keep_an_eager_break_with_graph_backend(self):
        batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND)
        backend = SimpleNamespace(low_ratio_prefill_graph=True)
        for graph, cp, tail, expected in (
            (True, True, False, True),
            (True, False, False, False),
            (True, False, True, True),
            (False, True, False, False),
        ):
            with (
                self.subTest(graph=graph, cp=cp, tail=tail),
                patch.object(dsv4, "is_in_breakable_cuda_graph", return_value=graph),
                patch.object(dsv4, "is_cp_active", return_value=cp),
                patch.object(dsv4, "in_replay_tail_graph", return_value=tail),
            ):
                self.assertEqual(
                    dsv4._low_ratio_sources_break(backend, batch), expected
                )

    def test_replay_uses_prepared_image_ids_without_mutating_scheduler_ids(self):
        for keyword in (False, True):
            with self.subTest(keyword=keyword):
                scheduler_ids = torch.tensor([7, dsv4.MM_PAD_SHIFT_VALUE + 1, 9, 0])
                static_ids = scheduler_ids.clone()
                image_id = 42
                layer = SimpleNamespace(forward=None)
                runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
                runner._is_full_backend = False
                runner.is_dsv41_model = True
                runner._qwen_bcg_hc_sidechannel = False
                runner._use_draft_input_embeds = False
                runner.layer_model = layer
                runner.buffer_registry = SimpleNamespace(has_slot=lambda name: False)
                runner._prefill_forward_context = lambda *a, **kw: nullcontext()
                runner.backend = SimpleNamespace(
                    replay=lambda *a, **kw: static_ids.clone()
                )

                def outer_forward(ids, positions, batch):
                    model = SimpleNamespace(
                        vision=object(), config=SimpleNamespace(image_token_id=image_id)
                    )
                    mapped, _ = (
                        dsv4.DeepseekV4ForCausalLM.prepare_language_model_inputs(
                            model, ids, batch
                        )
                    )
                    if keyword:
                        return layer.forward(
                            input_ids=mapped, positions=positions, forward_batch=batch
                        )
                    return layer.forward(mapped, positions, batch)

                runner.model_runner = SimpleNamespace(
                    pp_group=SimpleNamespace(is_first_rank=True),
                    model=SimpleNamespace(forward=outer_forward),
                )
                live = SimpleNamespace(input_ids=scheduler_ids, mm_input_embeds=None)
                static = SimpleNamespace(
                    input_ids=static_ids,
                    positions=torch.arange(4),
                    mm_input_embeds=None,
                    mm_inputs=None,
                    forward_mode=ForwardMode.EXTEND,
                )
                actual = runner._execute_body_capture(live, static, 4, 3, None)
                torch.testing.assert_close(actual, torch.tensor([7, image_id, 9, 0]))
                self.assertEqual(scheduler_ids[1].item(), dsv4.MM_PAD_SHIFT_VALUE + 1)

    def test_image_engram_mask_does_not_depend_on_capture_batch_mm_metadata(self):
        image_id = 42
        ids = torch.tensor([7, image_id, 9])
        hidden = torch.arange(6, dtype=torch.float32).reshape(3, 1, 2)
        kv = torch.ones(3, 4)
        weight = torch.ones(1, 2)

        class Engram:
            layer_hash_index = 0

            def __call__(self, residual, *args, input_ids, **kwargs):
                return engram_gate(
                    residual, kv, weight, weight, 1e-6, 1e-6, input_ids, image_id
                )

        class Layer(SimpleNamespace):
            def __call__(self, *, hidden_states, **kwargs):
                return hidden_states

        layer = Layer(
            engram=Engram(),
            hc_cfg=None,
            boundary_mhc_fusions=False,
            forward_hc_pre_from_prev=lambda **kw: kw["state"],
        )
        model = SimpleNamespace(
            pp_group=SimpleNamespace(world_size=1),
            engram_hasher=lambda *a: torch.zeros((3, 1), dtype=torch.long),
            late_layer_start=None,
            decoder_replay_tail=None,
            start_layer=0,
            end_layer=1,
            layers=[layer],
            config=SimpleNamespace(
                model_type="deepseek_v41", vision_n_layers=1, image_token_id=image_id
            ),
        )
        expected = engram_gate(hidden, kv, weight, weight, 1e-6, 1e-6)
        self.assertFalse(torch.equal(expected[1], hidden[1]))
        expected[1] = hidden[1]
        for contains_mm in (False, True):
            batch = SimpleNamespace(
                forward_mode=ForwardMode.EXTEND, contains_mm_inputs=lambda: contains_mm
            )
            with (
                patch.object(dsv4, "is_cp_active", return_value=False),
                patch.object(dsv4, "is_in_breakable_cuda_graph", return_value=False),
                patch.object(
                    dsv4, "get_forward", return_value=SimpleNamespace(sp_active=False)
                ),
                patch.object(dsv4, "check_cuda_graph_backend", return_value=True),
                patch.object(dsv4, "_is_hip", False),
            ):
                actual, _, _ = dsv4.DeepseekV4Model._forward_layers_hc_pre_from_prev(
                    model, torch.arange(3), hidden, batch, ids, ids, False, []
                )
            torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
