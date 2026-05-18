import json
import os
import subprocess
import textwrap
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.true_on_policy import (
    QWEN3_DENSE_TRUE_ON_POLICY_V1,
    get_on_policy_rms_norm_kwargs,
    should_force_bfloat16_dense_tensor_math,
    should_force_bfloat16_lm_head,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="stage-a-test-cpu")


def _run_dense_math_script(script_body: str) -> dict[str, object]:
    stubbed_imports = textwrap.dedent("""
        import importlib.machinery
        import json
        import sys
        import types
        from pydantic import BaseModel

        def install_openai_stubs():
            openai_mod = types.ModuleType("openai")
            openai_types_mod = types.ModuleType("openai.types")
            openai_responses_mod = types.ModuleType("openai.types.responses")
            openai_response_mod = types.ModuleType("openai.types.responses.response")
            openai_tool_mod = types.ModuleType("openai.types.responses.tool")

            openai_mod.__spec__ = importlib.machinery.ModuleSpec("openai", loader=None)
            openai_types_mod.__spec__ = importlib.machinery.ModuleSpec("openai.types", loader=None)
            openai_responses_mod.__spec__ = importlib.machinery.ModuleSpec(
                "openai.types.responses", loader=None
            )
            openai_response_mod.__spec__ = importlib.machinery.ModuleSpec(
                "openai.types.responses.response", loader=None
            )
            openai_tool_mod.__spec__ = importlib.machinery.ModuleSpec(
                "openai.types.responses.tool", loader=None
            )

            for name in [
                "ResponseFunctionToolCall",
                "ResponseInputItemParam",
                "ResponseOutputItem",
                "ResponseOutputMessage",
                "ResponseOutputText",
                "ResponseReasoningItem",
            ]:
                setattr(openai_responses_mod, name, type(name, (BaseModel,), {}))

            openai_response_mod.ToolChoice = type("ToolChoice", (BaseModel,), {})
            openai_tool_mod.Tool = type("Tool", (BaseModel,), {})

            sys.modules.setdefault("openai", openai_mod)
            sys.modules.setdefault("openai.types", openai_types_mod)
            sys.modules.setdefault("openai.types.responses", openai_responses_mod)
            sys.modules.setdefault("openai.types.responses.response", openai_response_mod)
            sys.modules.setdefault("openai.types.responses.tool", openai_tool_mod)

        install_openai_stubs()

        hf_utils_mod = types.ModuleType("sglang.srt.utils.hf_transformers_utils")
        hf_utils_mod.__spec__ = importlib.machinery.ModuleSpec(
            "sglang.srt.utils.hf_transformers_utils", loader=None
        )
        hf_utils_mod.check_gguf_file = lambda *args, **kwargs: False
        hf_utils_mod.get_rope_config = lambda config: (
            getattr(config, "rope_theta", 1000000),
            getattr(config, "rope_scaling", None),
        )
        sys.modules.setdefault("sglang.srt.utils.hf_transformers_utils", hf_utils_mod)

        gguf_mod = types.ModuleType("gguf")
        gguf_mod.__spec__ = importlib.machinery.ModuleSpec("gguf", loader=None)
        gguf_mod.GGMLQuantizationType = type(
            "GGMLQuantizationType",
            (),
            {
                "F32": 0,
                "F16": 1,
                "BF16": 2,
                "Q4_0": 3,
                "Q4_1": 4,
                "Q5_0": 5,
                "Q5_1": 6,
                "Q8_0": 7,
                "Q8_1": 8,
                "Q2_K": 9,
                "Q3_K": 10,
                "Q4_K": 11,
                "Q5_K": 12,
                "Q6_K": 13,
                "IQ1_S": 14,
                "IQ1_M": 15,
                "IQ2_XXS": 16,
                "IQ2_XS": 17,
                "IQ2_S": 18,
                "IQ3_XXS": 19,
                "IQ3_S": 20,
                "IQ4_NL": 21,
                "IQ4_XS": 22,
            },
        )
        sys.modules.setdefault("gguf", gguf_mod)
        """)

    env = dict(os.environ)
    pythonpath = env.get("PYTHONPATH")
    repo_python = "python"
    env["PYTHONPATH"] = (
        f"{repo_python}{os.pathsep}{pythonpath}" if pythonpath else repo_python
    )
    script = f"{stubbed_imports}\n{script_body}"
    completed = subprocess.run(
        ["python", "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return json.loads(completed.stdout)


class TestDenseOnPolicyHelpers(unittest.TestCase):
    def test_default_dense_math_helpers_are_inactive(self):
        server_args = SimpleNamespace(
            true_on_policy_contract=None,
            tp_size=1,
        )

        self.assertFalse(should_force_bfloat16_dense_tensor_math(server_args))
        self.assertFalse(
            should_force_bfloat16_lm_head(
                server_args=server_args,
                use_fp32_lm_head=False,
            )
        )
        self.assertEqual(get_on_policy_rms_norm_kwargs(server_args), {})

    def test_on_policy_dense_math_helpers_enable_bfloat16_and_rms_norm_kwargs(self):
        server_args = SimpleNamespace(
            true_on_policy_contract=QWEN3_DENSE_TRUE_ON_POLICY_V1,
            tp_size=1,
        )

        kwargs = get_on_policy_rms_norm_kwargs(
            server_args,
            weight_dtype=torch.float32,
            override_orig_dtype=torch.float32,
            fp32_residual=True,
        )

        self.assertTrue(should_force_bfloat16_dense_tensor_math(server_args))
        self.assertTrue(
            should_force_bfloat16_lm_head(
                server_args=server_args,
                use_fp32_lm_head=False,
            )
        )
        self.assertFalse(
            should_force_bfloat16_lm_head(
                server_args=server_args,
                use_fp32_lm_head=True,
            )
        )
        self.assertEqual(kwargs["weight_dtype"], torch.float32)
        self.assertEqual(kwargs["override_orig_dtype"], torch.float32)
        self.assertTrue(kwargs["cast_x_before_out_mul"])
        self.assertTrue(kwargs["fp32_residual"])


class TestDenseOnPolicyContracts(unittest.TestCase):
    def test_qwen3_style_rms_norm_keeps_fp32_weight_output_and_residual(self):
        result = _run_dense_math_script(textwrap.dedent("""
                import json
                from types import SimpleNamespace

                import torch

                from sglang.srt.layers.layernorm import RMSNorm
                from sglang.srt.true_on_policy import get_on_policy_rms_norm_kwargs

                from sglang.srt.true_on_policy import QWEN3_DENSE_TRUE_ON_POLICY_V1

                server_args = SimpleNamespace(
                    true_on_policy_contract=QWEN3_DENSE_TRUE_ON_POLICY_V1,
                    tp_size=1,
                )
                norm = RMSNorm(
                    4,
                    eps=1e-6,
                    **get_on_policy_rms_norm_kwargs(
                        server_args,
                        weight_dtype=torch.float32,
                        override_orig_dtype=torch.float32,
                        fp32_residual=True,
                    ),
                )
                x = torch.randn(2, 4, dtype=torch.bfloat16)
                residual = torch.randn(2, 4, dtype=torch.bfloat16)
                out, residual_out = norm.forward_native(x, residual)
                print(
                    json.dumps(
                        {
                            "weight_dtype": str(norm.weight.dtype),
                            "out_dtype": str(out.dtype),
                            "residual_dtype": str(residual_out.dtype),
                        }
                    )
                )
                """))

        self.assertEqual(result["weight_dtype"], "torch.float32")
        self.assertEqual(result["out_dtype"], "torch.float32")
        self.assertEqual(result["residual_dtype"], "torch.float32")

    def test_rms_norm_can_self_configure_from_true_on_policy_role_hints(self):
        result = _run_dense_math_script(textwrap.dedent("""
                import json

                import torch

                from sglang.srt.layers.layernorm import RMSNorm
                from sglang.srt.server_args import (
                    ServerArgs,
                    get_global_server_args,
                    set_global_server_args_for_scheduler,
                )
                from sglang.srt.true_on_policy import QWEN3_DENSE_TRUE_ON_POLICY_V1

                set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
                server_args = get_global_server_args()
                server_args.true_on_policy_contract = QWEN3_DENSE_TRUE_ON_POLICY_V1
                server_args.tp_size = 1
                norm = RMSNorm(
                    4,
                    eps=1e-6,
                    true_on_policy_weight_dtype=torch.float32,
                    true_on_policy_override_orig_dtype=torch.float32,
                    true_on_policy_fp32_residual=True,
                )
                print(
                    json.dumps(
                        {
                            "weight_dtype": str(norm.weight.dtype),
                            "cast_x_before_out_mul": norm.cast_x_before_out_mul,
                            "fp32_residual": norm.fp32_residual,
                            "override_orig_dtype": str(norm.override_orig_dtype),
                        }
                    )
                )
                """))

        self.assertEqual(result["weight_dtype"], "torch.float32")
        self.assertTrue(result["cast_x_before_out_mul"])
        self.assertTrue(result["fp32_residual"])
        self.assertEqual(result["override_orig_dtype"], "torch.float32")

    def test_on_policy_lm_head_forces_bfloat16_matmul_inputs(self):
        result = _run_dense_math_script(textwrap.dedent("""
                import json
                from types import SimpleNamespace
                from unittest.mock import patch

                import torch
                import torch.nn as nn

                from sglang.srt.layers.logits_processor import LogitsProcessor
                from sglang.srt.server_args import (
                    ServerArgs,
                    get_global_server_args,
                    set_global_server_args_for_scheduler,
                )
                from sglang.srt.true_on_policy import QWEN3_DENSE_TRUE_ON_POLICY_V1

                class DummyMeta:
                    gathered_buffer = None
                    next_token_logits_buffer = None

                    def compute_dp_attention_metadata(self):
                        return None

                class LMHeadStub(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.weight = nn.Parameter(torch.randn(8, 4, dtype=torch.float32))

                set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
                get_global_server_args().enable_dp_lm_head = False
                get_global_server_args().enable_fp32_lm_head = False
                get_global_server_args().true_on_policy_contract = QWEN3_DENSE_TRUE_ON_POLICY_V1
                get_global_server_args().tp_size = 1

                processor = LogitsProcessor(
                    SimpleNamespace(vocab_size=8, final_logit_softcapping=None),
                    skip_all_gather=True,
                    logit_scale=None,
                )
                hidden_states = torch.randn(2, 4, dtype=torch.float32)
                head = LMHeadStub()
                captured = {}

                original_matmul = torch.matmul

                def probe_matmul(a, b, *args, **kwargs):
                    if not captured:
                        captured["a_dtype"] = str(a.dtype)
                        captured["b_dtype"] = str(b.dtype)
                    return original_matmul(a, b, *args, **kwargs)

                with patch("torch.matmul", new=probe_matmul):
                    logits = processor._get_logits(hidden_states, head, DummyMeta())

                print(
                    json.dumps(
                        {
                            "a_dtype": captured["a_dtype"],
                            "b_dtype": captured["b_dtype"],
                            "logits_dtype": str(logits.dtype),
                        }
                    )
                )
                """))

        self.assertEqual(result["a_dtype"], "torch.bfloat16")
        self.assertEqual(result["b_dtype"], "torch.bfloat16")
        self.assertEqual(result["logits_dtype"], "torch.bfloat16")


if __name__ == "__main__":
    unittest.main(verbosity=2)
