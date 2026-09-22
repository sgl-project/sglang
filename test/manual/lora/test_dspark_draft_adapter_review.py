"""Regression tests from the independent review of the draft adapter patch."""

import ast
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
from test_dspark_draft_adapter_routing import config, routing

ROOT = Path(__file__).resolve().parents[3]
HPC_MODULE = "sglang.kernels.ops.attention.dsv4.gemm"


def load_weight_updater_functions(spec):
    source = (
        ROOT
        / "python/sglang/srt/model_executor/model_runner_components/weight_updater.py"
    )
    tree = ast.parse(source.read_text())
    guard = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "_unsupported_derived_weight_cache_error"
    )
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "WeightUpdater"
    )
    methods = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name
        in {
            "update_weights_from_disk",
            "update_weights_from_distributed",
            "update_weights_from_tensor",
            "update_weights_from_ipc",
        }
    ]
    namespace = {
        "get_spec": lambda: spec,
        "get_context": lambda: NS(is_config_namespace_published=lambda name: True),
        "torch": torch,
    }
    for function in [guard, *methods]:
        exec(
            compile(
                "from __future__ import annotations\n" + ast.unparse(function),
                str(source),
                "exec",
            ),
            namespace,
        )
    return namespace


class TestDraftAdapterReviewRegressions(unittest.TestCase):
    def test_new_flags_do_not_change_frozen_positional_signature(self):
        path = ROOT / "python/sglang/srt/arg_groups/field_order.py"
        spec = importlib.util.spec_from_file_location("dspark_field_order_review", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # collect_input_fields appends fields omitted from this compatibility
        # table. Inserting either here moves every later existing argument.
        for name in ("speculative_dspark_lora_path", "speculative_dspark_lora_paths"):
            self.assertNotIn(name, module.POSITIONAL_FIELD_ORDER)
        index = module.POSITIONAL_FIELD_ORDER.index("uno_lora_path")
        self.assertEqual(
            module.POSITIONAL_FIELD_ORDER[index + 1], "speculative_draft_model_path"
        )

    def test_lowercase_algorithm_uses_existing_case_insensitive_contract(self):
        for algorithm in ("dspark", "DsPaRk", "DSPARK"):
            cfg = config()
            cfg.speculative_algorithm = algorithm
            with self.subTest(algorithm=algorithm):
                routing.validate_draft_adapter_server_config(cfg)

    def test_online_weight_updates_rejected_before_target_or_draft_mutation(self):
        functions = load_weight_updater_functions(
            NS(speculative_dspark_lora_paths='{"rust":"/a"}')
        )
        # The guard must apply even when the first runner being updated is the
        # target, which does not own a draft_adapter_bank attribute.
        hpc = types.ModuleType(HPC_MODULE)
        hpc.hpc_bf16xfp32_gemm_enabled = lambda: False
        worker = NS(
            get_model=lambda: None, _assert_weight_cache_inactive=lambda op: None
        )
        calls = {
            "update_weights_from_disk": ("new-checkpoint", "auto"),
            "update_weights_from_distributed": (
                ["fc.weight"],
                ["bfloat16"],
                [[4, 4]],
                "test",
            ),
            "update_weights_from_tensor": ([],),
            "update_weights_from_ipc": (NS(zmq_handles={}),),
        }
        with patch.dict(sys.modules, {HPC_MODULE: hpc}):
            error = functions["_unsupported_derived_weight_cache_error"]()
            self.assertIsNotNone(error)
            self.assertIn("draft adapter", error.lower())
            for method, args in calls.items():
                with self.subTest(method=method):
                    success, message = functions[method](worker, *args)
                    self.assertFalse(success)
                    self.assertEqual(message, error)

    def test_ordinary_weight_updates_keep_existing_guard_behavior(self):
        functions = load_weight_updater_functions(
            NS(speculative_dspark_lora_paths=None)
        )
        hpc = types.ModuleType(HPC_MODULE)
        hpc.hpc_bf16xfp32_gemm_enabled = lambda: False
        with patch.dict(sys.modules, {HPC_MODULE: hpc}):
            self.assertIsNone(functions["_unsupported_derived_weight_cache_error"]())
            hpc.hpc_bf16xfp32_gemm_enabled = lambda: True
            self.assertIn(
                "HPC-Ops", functions["_unsupported_derived_weight_cache_error"]()
            )

    def test_guard_still_works_before_runtime_configuration_is_published(self):
        functions = load_weight_updater_functions(None)
        functions["get_context"] = lambda: NS(
            is_config_namespace_published=lambda name: False
        )
        hpc = types.ModuleType(HPC_MODULE)
        hpc.hpc_bf16xfp32_gemm_enabled = lambda: False
        with patch.dict(sys.modules, {HPC_MODULE: hpc}):
            self.assertIsNone(functions["_unsupported_derived_weight_cache_error"]())
            model = torch.nn.Sequential(torch.nn.Linear(1, 1))
            model[0]._hc_attn_tf32_parts = (torch.ones(1), torch.zeros(1))
            self.assertIn(
                "compensated mHC",
                functions["_unsupported_derived_weight_cache_error"](model),
            )

    def test_deferred_weight_initialization_is_rejected_at_startup(self):
        cfg = config()
        cfg.checkpoint_engine_wait_weights_before_ready = True
        with self.assertRaisesRegex(
            ValueError, "checkpoint_engine_wait_weights_before_ready"
        ):
            routing.validate_draft_adapter_server_config(cfg)


if __name__ == "__main__":
    unittest.main()
