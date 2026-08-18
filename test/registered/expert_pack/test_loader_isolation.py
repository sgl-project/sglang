from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[3]


class TestLoaderIsolation(unittest.TestCase):
    def _run_isolated(self, source: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        paths = [str(ROOT), str(ROOT / "python")]
        if env.get("PYTHONPATH"):
            paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(paths)
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(source)],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            self.fail(
                f"isolated Python failed with code {result.returncode}:\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        return result

    def test_load_formats_are_exact_and_expert_pack_is_lazy(self) -> None:
        result = self._run_isolated("""
            import sys
            from sglang.srt.configs.load_config import LoadConfig, LoadFormat
            from sglang.srt.model_loader.loader import (
                DefaultModelLoader,
                GGUFModelLoader,
                get_model_loader,
            )

            prefix = "sglang.srt.layers.moe.expert_pack"
            assert not any(name.startswith(prefix) for name in sys.modules)
            assert LoadConfig().load_format is LoadFormat.AUTO
            assert isinstance(get_model_loader(LoadConfig(load_format="auto")), DefaultModelLoader)
            assert isinstance(
                get_model_loader(LoadConfig(load_format="safetensors")),
                DefaultModelLoader,
            )
            assert isinstance(get_model_loader(LoadConfig(load_format="gguf")), GGUFModelLoader)
            assert not any(name.startswith(prefix) for name in sys.modules)

            try:
                get_model_loader(LoadConfig(load_format="expert_pack"))
            except ValueError as exc:
                assert "requires pack_path" in str(exc)
            else:
                raise AssertionError("expert_pack without pack_path must fail")
            assert "sglang.srt.model_loader.expert_pack_loader" in sys.modules
            assert "sglang.srt.layers.moe.expert_pack" in sys.modules
            print("loader isolation: OK")
            """)
        self.assertIn("loader isolation: OK", result.stdout)

    def test_native_model_registry_and_configs_do_not_import_expert_pack(self) -> None:
        result = self._run_isolated("""
            import sys
            from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config
            from sglang.srt.configs.kimi_k3 import KimiK3Config
            from sglang.srt.configs.kimi_linear import KimiLinearConfig
            from sglang.srt.models.registry import ModelRegistry

            expected = {
                "DeepseekV4ForCausalLM",
                "KimiK3ForConditionalGeneration",
                "KimiK3LinearForCausalLM",
            }
            for architecture in expected:
                model_cls, resolved = ModelRegistry.resolve_model_cls([architecture])
                assert resolved == architecture
                assert model_cls.__name__ == architecture

            deepseek = DeepSeekV4Config(
                architectures=["DeepseekV4ForCausalLM"],
                quantization_config={},
                rope_scaling={},
                compress_ratios=[],
            )
            kimi_text = KimiLinearConfig(
                architectures=["KimiK3LinearForCausalLM"],
                model_type="kimi_linear",
            )
            kimi_vl = KimiK3Config(
                architectures=["KimiK3ForConditionalGeneration"],
                text_config=kimi_text,
            )
            assert deepseek.architectures == ["DeepseekV4ForCausalLM"]
            assert kimi_text.architectures == ["KimiK3LinearForCausalLM"]
            assert kimi_vl.architectures == ["KimiK3ForConditionalGeneration"]
            assert not any(
                name.startswith("sglang.srt.layers.moe.expert_pack")
                or name == "sglang.srt.model_loader.expert_pack_loader"
                for name in sys.modules
            )
            print("native registry isolation: OK")
            """)
        self.assertIn("native registry isolation: OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
