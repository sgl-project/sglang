"""Dependency-light contract test for Bailing V3 VL expert topology."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2.0, suite="base-a-test-cpu")


class TestBailingMoeV3VLExpertLocation(unittest.TestCase):
    @staticmethod
    def _wrapper_method(name):
        model_path = (
            Path(__file__).parents[4] / "python/sglang/srt/models/bailing_mm_v3.py"
        )
        tree = ast.parse(model_path.read_text())
        wrapper = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "BailingMoeV3VLForConditionalGeneration"
        )
        return model_path, next(
            node
            for node in wrapper.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )

    def test_delegates_expert_location_config_to_language_backbone(self):
        model_path, method = self._wrapper_method(
            "get_model_config_for_expert_location"
        )

        delegated = object()
        text_config = SimpleNamespace()
        test_case = self

        class LanguageBackbone:
            @classmethod
            def get_model_config_for_expert_location(cls, config):
                test_case.assertIs(config, text_config)
                return delegated

        synthetic_wrapper = ast.ClassDef(
            name="Wrapper",
            bases=[],
            keywords=[],
            body=[method],
            decorator_list=[],
        )
        module = ast.fix_missing_locations(
            ast.Module(body=[synthetic_wrapper], type_ignores=[])
        )
        namespace = {"BailingMoeV3ForCausalLM": LanguageBackbone}
        exec(compile(module, str(model_path), "exec"), namespace)

        result = namespace["Wrapper"].get_model_config_for_expert_location(
            SimpleNamespace(text_config=text_config)
        )
        self.assertIs(result, delegated)

    def test_delegates_shared_expert_fusion_gate_to_language_backbone(self):
        model_path, method = self._wrapper_method(
            "shared_experts_fusion_disable_reason"
        )
        delegated = object()
        text_config = SimpleNamespace()
        quant_config = object()
        test_case = self

        class LanguageBackbone:
            @classmethod
            def shared_experts_fusion_disable_reason(cls, config, quant):
                test_case.assertIs(config, text_config)
                test_case.assertIs(quant, quant_config)
                return delegated

        synthetic_wrapper = ast.ClassDef(
            name="Wrapper",
            bases=[],
            keywords=[],
            body=[method],
            decorator_list=[],
        )
        module = ast.fix_missing_locations(
            ast.Module(body=[synthetic_wrapper], type_ignores=[])
        )
        namespace = {"BailingMoeV3ForCausalLM": LanguageBackbone}
        exec(compile(module, str(model_path), "exec"), namespace)

        result = namespace["Wrapper"].shared_experts_fusion_disable_reason(
            SimpleNamespace(text_config=text_config), quant_config
        )
        self.assertIs(result, delegated)


if __name__ == "__main__":
    unittest.main()
