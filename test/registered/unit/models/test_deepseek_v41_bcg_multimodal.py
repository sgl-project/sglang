"""Regression tests for multimodal IDs and Engram masking during graph replay."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.models.deepseek_v4 as dsv4
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestDeepseekV41BCGMultimodal(CustomTestCase):
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

        class Engram:
            layer_hash_index = 0

            def __call__(self, residual, *args, **kwargs):
                return residual + 10

        layer = SimpleNamespace(
            engram=Engram(),
            hc_cfg=None,
            forward_hc_pre_from_prev=lambda **kw: kw["state"],
        )
        model = SimpleNamespace(
            pp_group=SimpleNamespace(world_size=1),
            engram_hasher=lambda *a: torch.zeros((3, 1), dtype=torch.long),
            late_layer_start=None,
            start_layer=0,
            end_layer=1,
            layers=[layer],
            config=SimpleNamespace(
                model_type="deepseek_v41", vision_n_layers=1, image_token_id=image_id
            ),
        )
        expected = hidden + 10
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
