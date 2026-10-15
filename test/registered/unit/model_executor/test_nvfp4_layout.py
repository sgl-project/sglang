"""Host-side eligibility and cycle ownership for native Nemotron models."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.model_runner_components import nvfp4_layout
from sglang.srt.models.nano_nemotron_vl import NemotronH_Omni_Reasoning_V3
from sglang.srt.models.nemotron_h import NemotronHForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def bare_model(cls):
    model = object.__new__(cls)
    nn.Module.__init__(model)
    return model


def language_model(count=40):
    model = bare_model(NemotronHForCausalLM)
    model.experts = nn.ModuleList()
    for _ in range(count):
        expert = nn.Module()
        expert.w2_weight = torch.empty(
            (512, 1024, 1344), dtype=torch.uint8, device="meta"
        )
        expert.w13_weight = torch.empty(
            (512, 2688, 512), dtype=torch.uint8, device="meta"
        )
        expert.quant_method = SimpleNamespace(enable_flashinfer_cutlass_moe=True)
        model.experts.append(expert)
    return model


class TestNvfp4LayoutEligibility(CustomTestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(envs.SGLANG_SM120_NVFP4_PHASE_LAYOUT.override(True))
        self.parallel = SimpleNamespace(tp_size=1, ep_size=1, pp_size=1)
        self.stack.enter_context(
            patch.object(nvfp4_layout, "get_parallel", return_value=self.parallel)
        )
        self.stack.enter_context(
            patch.object(
                nvfp4_layout,
                "get_schedule",
                return_value=SimpleNamespace(max_running_requests=1),
            )
        )
        self.stack.enter_context(
            patch("torch.cuda.get_device_capability", return_value=(12, 0))
        )
        self.stack.enter_context(
            patch(
                "sglang.srt.layers.moe.moe_runner.flashinfer_cutlass._sm120_mtp_mma_supported",
                return_value=True,
            )
        )
        self.cycle = SimpleNamespace(workspace_bytes=0)
        self.stack.enter_context(
            patch(
                "sglang.kernels.ops.gemm.sm120_nvfp4_layout.Nvfp4LayoutCycle",
                return_value=self.cycle,
            )
        )

    def test_cycle_is_owned_by_forward_model(self):
        for wrapped in (False, True):
            with self.subTest(wrapped=wrapped):
                llm = language_model()
                model = bare_model(NemotronH_Omni_Reasoning_V3) if wrapped else llm
                if wrapped:
                    model.language_model = llm
                    # Matching-shaped vision modules must never enter the LLM cycle.
                    model.vision_model = language_model(1)
                result = nvfp4_layout.initialize_nvfp4_layout(
                    model, is_draft_worker=False
                )
                self.assertIs(result, self.cycle)
                self.assertIs(model._sglang_nvfp4_layout_cycle, self.cycle)
                self.assertEqual(
                    [m._sglang_nvfp4_layout_slot[1] for m in llm.experts],
                    list(range(40)),
                )
                if wrapped:
                    self.assertIs(llm._sglang_nvfp4_layout_cycle, self.cycle)
                    self.assertFalse(
                        hasattr(
                            model.vision_model.experts[0], "_sglang_nvfp4_layout_slot"
                        )
                    )

    def test_inner_capture_warmup_and_outer_replay_share_state(self):
        model = bare_model(NemotronH_Omni_Reasoning_V3)
        model.language_model = language_model()
        transitions = []
        self.cycle.is_pair = False

        def ensure(pair):
            if self.cycle.is_pair != pair:
                transitions.append(pair)
                self.cycle.is_pair = pair

        self.cycle.ensure = ensure
        nvfp4_layout.initialize_nvfp4_layout(model, is_draft_worker=False)
        verify = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY, input_ids=torch.zeros(4)
        )
        prefill = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND, input_ids=torch.zeros(1024)
        )
        with patch(
            "sglang.srt.layers.moe.moe_runner.flashinfer_cutlass._flashinfer_autotune_active",
            return_value=False,
        ):
            # Direct language-model warmup must select the pair kernels before capture.
            nvfp4_layout.prepare_nvfp4_layout(model.language_model, verify)
            self.assertTrue(self.cycle.is_pair)
            nvfp4_layout.prepare_nvfp4_layout(model, verify)
            self.assertEqual(transitions, [True])
            # Eager prefill restores original storage, and the next replay switches back.
            nvfp4_layout.prepare_nvfp4_layout(model, prefill)
            self.assertFalse(self.cycle.is_pair)
            nvfp4_layout.prepare_nvfp4_layout(model, verify)
            self.assertEqual(transitions, [True, False, True])

    def test_unrecognized_wrapper_is_rejected(self):
        model = nn.Module()
        model.language_model = language_model()
        with self.assertRaisesRegex(ValueError, "requires serial Nemotron"):
            nvfp4_layout.initialize_nvfp4_layout(model, is_draft_worker=False)

    def test_vl_inner_architecture_is_checked(self):
        model = bare_model(NemotronH_Omni_Reasoning_V3)
        model.language_model = nn.Module()
        with self.assertRaisesRegex(ValueError, "requires serial Nemotron"):
            nvfp4_layout.initialize_nvfp4_layout(model, is_draft_worker=False)

    def test_vl_keeps_layer_count_and_parallel_guards(self):
        model = bare_model(NemotronH_Omni_Reasoning_V3)
        model.language_model = language_model(39)
        with self.assertRaisesRegex(ValueError, "all 40"):
            nvfp4_layout.initialize_nvfp4_layout(model, is_draft_worker=False)
        model.language_model = language_model()
        self.parallel.tp_size = 2
        with self.assertRaisesRegex(ValueError, "requires serial Nemotron"):
            nvfp4_layout.initialize_nvfp4_layout(model, is_draft_worker=False)

    def test_draft_never_gets_layout_cycle(self):
        model = language_model()
        self.assertIsNone(
            nvfp4_layout.initialize_nvfp4_layout(model, is_draft_worker=True)
        )
        self.assertFalse(hasattr(model, "_sglang_nvfp4_layout_cycle"))


if __name__ == "__main__":
    unittest.main()
