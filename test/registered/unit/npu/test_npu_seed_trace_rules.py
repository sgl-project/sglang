"""CPU regression tests for the transfer_to_npu/Dynamo seed-rule conflict.

Load the production helper without importing the NPU runtime. Only NPU rule
registration and function aliases are mocked; Dynamo builds its real rule map.
"""

import ast
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import torch
from torch._dynamo import trace_rules
from torch._dynamo.variables import SkipFunctionVariable, TorchInGraphFunctionVariable

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")

UTILS = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/hardware_backend/npu/utils.py"
)


def _load_helper():
    node = next(
        node
        for node in ast.parse(UTILS.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_patch_npu_seed_trace_rules"
    )
    namespace = {"torch": torch}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(UTILS), "exec"), namespace
    )
    return namespace[node.name]


class TestNpuSeedTraceRules(unittest.TestCase):
    def setUp(self):
        self.patch_rules = _load_helper()
        self.npu = ModuleType("torch.npu")
        self.npu.random = ModuleType("torch.npu.random")
        self.npu_rules = {
            f"torch.npu.random.{name}": TorchInGraphFunctionVariable
            for name in ("manual_seed", "manual_seed_all")
        }
        for name in ("manual_seed", "manual_seed_all"):
            setattr(self.npu.random, name, getattr(torch.cuda, name))
        rules_module = ModuleType("torch_npu.dynamo.trace_rule")
        rules_module.torch_non_c_binding_in_graph_functions_npu = self.npu_rules

        # Explicitly model the PyTorch 2.13 rules even on an older CPU test host.
        self.cuda_rules = dict(trace_rules.manual_torch_name_rule_map)
        for prefix in ("torch.cuda", "torch.cuda.random"):
            for name in ("manual_seed", "manual_seed_all"):
                self.cuda_rules[f"{prefix}.{name}"] = SkipFunctionVariable
        self.maps = [self.cuda_rules] + [
            {
                name: rule
                for name, rule in mapping.items()
                if name not in self.cuda_rules
            }
            for mapping in trace_rules.torch_name_rule_map
            if mapping is not trace_rules.manual_torch_name_rule_map
        ]
        stack = self.enterContext(ExitStack())
        stack.enter_context(patch.object(torch, "npu", self.npu, create=True))
        stack.enter_context(
            patch.dict(
                "sys.modules",
                {
                    "torch.npu": self.npu,
                    "torch.npu.random": self.npu.random,
                    "torch_npu.dynamo.trace_rule": rules_module,
                },
            )
        )
        stack.enter_context(
            patch.object(trace_rules, "manual_torch_name_rule_map", self.cuda_rules)
        )
        stack.enter_context(patch.object(trace_rules, "torch_name_rule_map", self.maps))
        trace_rules.clear_lru_cache()
        self.addCleanup(trace_rules.clear_lru_cache)

    def test_registered_alias_conflict_is_fixed_without_disabling_compile(self):
        for name in ("manual_seed", "manual_seed_all"):
            self.enterContext(patch.object(torch.cuda.random, name, lambda seed: None))
            self.assertIsNot(
                getattr(torch.cuda.random, name), getattr(self.npu.random, name)
            )
        self.maps.append(self.npu_rules)
        with self.assertRaisesRegex(AssertionError, "Duplicate torch object"):
            trace_rules.get_torch_obj_rule_map()
        self.patch_rules()
        rules = trace_rules.get_torch_obj_rule_map()
        for name in ("manual_seed", "manual_seed_all"):
            self.assertIs(rules[getattr(torch.cuda, name)], SkipFunctionVariable)
            self.assertIs(getattr(torch.cuda, name), getattr(self.npu.random, name))
        compiled = torch.compile(lambda x: x + 1, backend="eager", fullgraph=True)
        torch.testing.assert_close(compiled(torch.ones(2)), torch.full((2,), 2.0))

    def test_random_submodule_alias_conflict_is_fixed(self):
        # transfer_to_npu can alias cuda.random while the top-level CUDA
        # functions remain distinct from the NPU random functions.
        for name in ("manual_seed", "manual_seed_all"):

            def function(seed):
                pass

            setattr(self.npu.random, name, function)
            self.enterContext(patch.object(torch.cuda.random, name, function))
            self.assertIsNot(getattr(torch.cuda, name), function)
        self.maps.append(self.npu_rules)
        with self.assertRaisesRegex(AssertionError, "Duplicate torch object"):
            trace_rules.get_torch_obj_rule_map()

        self.patch_rules()

        rules = trace_rules.get_torch_obj_rule_map()
        for name in ("manual_seed", "manual_seed_all"):
            self.assertIs(rules[getattr(self.npu.random, name)], SkipFunctionVariable)
            self.assertIs(
                getattr(torch.cuda.random, name), getattr(self.npu.random, name)
            )
            self.assertIsNot(getattr(torch.cuda, name), getattr(self.npu.random, name))

    def test_skip_rule_must_belong_to_the_aliased_entry(self):
        for name in ("manual_seed", "manual_seed_all"):

            def function(seed):
                pass

            setattr(self.npu.random, name, function)
            self.enterContext(patch.object(torch.cuda.random, name, function))
            self.cuda_rules[f"torch.cuda.random.{name}"] = TorchInGraphFunctionVariable
        before = self.npu_rules.copy()
        with patch.object(trace_rules, "clear_lru_cache") as clear:
            self.patch_rules()
        self.assertEqual(self.npu_rules, before)
        clear.assert_not_called()

    def test_lazy_registration_uses_patched_source_rules(self):
        self.patch_rules()
        self.maps.append(self.npu_rules)
        trace_rules.clear_lru_cache()
        rules = trace_rules.get_torch_obj_rule_map()
        self.assertIs(rules[torch.cuda.manual_seed_all], SkipFunctionVariable)

    def test_cached_rules_are_invalidated_after_aliasing(self):
        original = (lambda seed: None, lambda seed: None)
        self.npu.random.manual_seed, self.npu.random.manual_seed_all = original
        self.maps.append(self.npu_rules)
        cached = trace_rules.get_torch_obj_rule_map()
        self.assertIs(cached[original[0]], TorchInGraphFunctionVariable)
        for name in ("manual_seed", "manual_seed_all"):
            setattr(self.npu.random, name, getattr(torch.cuda, name))
        self.patch_rules()
        refreshed = trace_rules.get_torch_obj_rule_map()
        self.assertIsNot(refreshed, cached)
        for name, old_function in zip(("manual_seed", "manual_seed_all"), original):
            self.assertNotIn(old_function, refreshed)
            self.assertIs(refreshed[getattr(torch.cuda, name)], SkipFunctionVariable)

    def test_older_torch_rules_are_unchanged(self):
        for old_rule in (TorchInGraphFunctionVariable, None):
            with self.subTest(old_rule=old_rule):
                for prefix in ("torch.cuda", "torch.cuda.random"):
                    for name in ("manual_seed", "manual_seed_all"):
                        key = f"{prefix}.{name}"
                        if old_rule is None:
                            self.cuda_rules.pop(key)
                        else:
                            self.cuda_rules[key] = old_rule
                before = self.npu_rules.copy()
                with patch.object(trace_rules, "clear_lru_cache") as clear:
                    self.patch_rules()
                self.assertEqual(self.npu_rules, before)
                clear.assert_not_called()

    def test_non_aliased_functions_are_unchanged(self):
        self.npu.random.manual_seed = lambda seed: None
        self.npu.random.manual_seed_all = lambda seed: None
        before = self.npu_rules.copy()
        with patch.object(trace_rules, "clear_lru_cache") as clear:
            self.patch_rules()
        self.assertEqual(self.npu_rules, before)
        clear.assert_not_called()

    def test_repeated_patch_is_noop_and_unrelated_rules_are_preserved(self):
        self.npu_rules["torch.npu.random.seed"] = TorchInGraphFunctionVariable
        self.patch_rules()
        self.assertIs(
            self.npu_rules["torch.npu.random.seed"], TorchInGraphFunctionVariable
        )
        with patch.object(trace_rules, "clear_lru_cache") as clear:
            self.patch_rules()
        clear.assert_not_called()

    def test_already_fixed_or_missing_npu_rules_are_unchanged(self):
        self.npu_rules["torch.npu.random.manual_seed"] = SkipFunctionVariable
        del self.npu_rules["torch.npu.random.manual_seed_all"]
        before = self.npu_rules.copy()
        with patch.object(trace_rules, "clear_lru_cache") as clear:
            self.patch_rules()
        self.assertEqual(self.npu_rules, before)
        clear.assert_not_called()


if __name__ == "__main__":
    unittest.main()
