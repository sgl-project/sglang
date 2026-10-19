"""CPU-only AST tests for the opt-in SM80 candidate graph factory.

No CUDA/torch installation is needed: test the real factory/selector source with
only hardware/runtime imports mocked. GPU metadata fill/replay is a separate test.
"""
import ast
import dataclasses
import logging
import os
import runpy
import sys
import types
import unittest
from pathlib import Path
from typing import Optional
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "python/sglang/srt/layers/attention/graph_variants.py"
register_cpu_ci = runpy.run_path(
    str(ROOT / "python/sglang/test/ci/ci_register.py")
)["register_cpu_ci"]
register_cpu_ci(est_time=2, suite="base-a-test-cpu")
FLAGS = (
    "SGLANG_DSV41_SM80_EXPERIMENTAL",
    "SGLANG_DSV41_SM80_FULL_GRAPH",
    "SGLANG_DSV41_SM80_GRAPH_VARIANTS",
)


class Lengths:
    def __init__(self, values, device="cpu"):
        self.values = values
        self.device = types.SimpleNamespace(type=device)

    def numel(self):
        return len(self.values)

    def max(self):
        return max(self.values)


class TestSM80GraphVariants(unittest.TestCase):
    def setUp(self):
        self.capability = (8, 0)
        self.hip = False
        self.gfx95 = False
        self.cap_devices = []
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(get_device_capability=self._capability)
        fake_modes = types.ModuleType("sglang.srt.model_executor.forward_batch_info")
        fake_modes.ForwardMode = types.SimpleNamespace(DECODE="decode", TARGET_VERIFY="verify")
        fake_utils = types.ModuleType("sglang.srt.utils")
        fake_utils.is_hip = lambda: self.hip
        fake_utils.is_gfx95_supported = lambda: self.gfx95
        self.context = patch.dict(sys.modules, {
            "torch": fake_torch,
            "sglang.srt.model_executor.forward_batch_info": fake_modes,
            "sglang.srt.utils": fake_utils,
        })
        self.context.start()
        self.addCleanup(self.context.stop)
        self.env = patch.dict(os.environ, {flag: "1" for flag in FLAGS})
        self.env.start()
        self.addCleanup(self.env.stop)
        tree = ast.parse(SOURCE.read_text())
        keep = {"Dsv41CandidateGraphVariants", "create_dsv41_candidate_graph_variants"}
        body = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in keep]
        module = types.ModuleType("_sm80_graph_variants_ast_test")
        self.module_patch = patch.dict(sys.modules, {module.__name__: module})
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        envs = runpy.run_path(str(SOURCE.parents[2] / "environ.py"))["envs"]
        module.__dict__.update(envs=envs, dataclass=dataclasses.dataclass, Optional=Optional,
                               DSV41_CANDIDATE_FILTERED="candidate_filtered",
                               logger=logging.getLogger(__name__), ForwardBatch=object)
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        isolated = ast.fix_missing_locations(ast.Module(body=[future] + body, type_ignores=[]))
        exec(compile(isolated, str(SOURCE), "exec"), module.__dict__)
        self.factory = module.create_dsv41_candidate_graph_variants
        config = types.SimpleNamespace(model_type="deepseek_v41", candidate_source_layer_id=2,
                                       candidate_topk_blocks=16, candidate_block_size=256,
                                       compress_ratios=[0, 1, 2], index_topk=512)
        self.runner = types.SimpleNamespace(device="cuda", gpu_id=7,
                                           model_config=types.SimpleNamespace(hf_text_config=config),
                                           spec_algorithm=types.SimpleNamespace(is_dspark=lambda: True),
                                           is_draft_worker=False)

    def _capability(self, device):
        self.cap_devices.append(device)
        return self.capability

    def make(self, mode="decode", width=0):
        return self.factory(self.runner, mode, width)

    def test_exact_gate_all_three_flags(self):
        self.assertIsNotNone(self.make())
        self.assertTrue(self.cap_devices)
        self.assertEqual(set(self.cap_devices), {7})
        for flag in FLAGS:
            for value in (None, "0", "true", "yes", "01"):
                with self.subTest(flag=flag, value=value):
                    with patch.dict(os.environ):
                        if value is None:
                            os.environ.pop(flag, None)
                        else:
                            os.environ[flag] = value
                        self.assertIsNone(self.make())

    def test_arch_device_model_and_mode_rejections(self):
        for capability in ((8, 6), (8, 9), (9, 0), (7, 5)):
            self.capability = capability
            self.assertIsNone(self.make())
        self.capability = (8, 0)
        self.runner.device = "cpu"
        self.assertIsNone(self.make())
        self.runner.device = "cuda"
        self.hip = True
        self.assertIsNone(self.make())
        self.hip = False
        self.assertIsNone(self.make("verify", 1))
        self.assertIsNone(self.make("prefill"))
        config = self.runner.model_config.hf_text_config
        config.model_type = "deepseek_v4"
        self.assertIsNone(self.make())
        config.model_type = "deepseek_v41"
        config.candidate_source_layer_id = -1
        self.assertIsNone(self.make())
        config.candidate_source_layer_id = 2
        config.candidate_topk_blocks = 0
        self.assertIsNone(self.make())

    def test_cpu_length_boundaries_and_missing_mirror(self):
        variants = self.make()
        for length, expected in ((511, "candidate_all"), (512, "candidate_all"),
                                 (513, "candidate_c2_all"), (1023, "candidate_c2_all"),
                                 (1024, "candidate_c2_all"), (1025, "candidate_unfiltered"),
                                 (4096, "candidate_unfiltered"), (4097, "candidate_filtered")):
            with self.subTest(length=length):
                batch = types.SimpleNamespace(seq_lens_cpu=Lengths([1, length]))
                self.assertEqual(variants.select(batch), expected)
        for mirror in (None, Lengths([]), Lengths([14], device="cuda")):
            self.assertEqual(variants.select(types.SimpleNamespace(seq_lens_cpu=mirror)),
                             "candidate_filtered")
        self.assertEqual(variants.select(types.SimpleNamespace()), "candidate_filtered")

    def test_capture_order_and_limits(self):
        variants = self.make()
        self.assertEqual(variants.graph_limits, (("candidate_all", 512),
                                                ("candidate_c2_all", 1024),
                                                ("candidate_unfiltered", 4096)))
        self.assertEqual(variants.capture_labels, ("candidate_all", "candidate_c2_all",
                                                  "candidate_unfiltered", "candidate_filtered"))
        self.assertEqual(variants.verify_extra_tokens, 0)

    def test_existing_sm100_and_gfx95_behavior_unchanged(self):
        with patch.dict(os.environ, {flag: "0" for flag in FLAGS}):
            self.capability = (10, 0)
            self.assertIsNotNone(self.make())
            self.assertEqual(self.make("verify", 3).verify_extra_tokens, 3)
            self.hip, self.gfx95 = True, True
            self.assertIsNotNone(self.make())
            self.assertIsNotNone(self.make("verify", 3))
            self.gfx95 = False
            self.assertIsNone(self.make())

    def test_selector_never_reads_gpu_lengths(self):
        class ForbiddenGPU:
            def __getattr__(self, name):
                raise AssertionError("GPU length access: " + name)
        batch = types.SimpleNamespace(seq_lens_cpu=None, seq_lens=ForbiddenGPU())
        self.assertEqual(self.make().select(batch), "candidate_filtered")


if __name__ == "__main__":
    unittest.main()
