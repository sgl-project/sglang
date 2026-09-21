import json
import os
import subprocess
import sys
import textwrap
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.true_on_policy import (
    QWEN3_DENSE_TRUE_ON_POLICY_V1,
    get_on_policy_rms_norm_kwargs,
    should_force_bfloat16_dense_tensor_math,
    should_force_bfloat16_lm_head,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def _run_dense_math_script(script_body: str) -> dict[str, object]:
    env = dict(os.environ)
    pythonpath = env.get("PYTHONPATH")
    repo_python = "python"
    env["PYTHONPATH"] = (
        f"{repo_python}{os.pathsep}{pythonpath}" if pythonpath else repo_python
    )
    completed = subprocess.run(
        [sys.executable, "-c", script_body],
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

        with patch(
            "sglang.srt.runtime_context.get_server_args", return_value=server_args
        ):
            self.assertFalse(should_force_bfloat16_dense_tensor_math())
            self.assertFalse(should_force_bfloat16_lm_head(use_fp32_lm_head=False))
            self.assertEqual(get_on_policy_rms_norm_kwargs(), {})

    def test_on_policy_dense_math_helpers_enable_bfloat16_and_rms_norm_kwargs(self):
        server_args = SimpleNamespace(
            true_on_policy_contract=QWEN3_DENSE_TRUE_ON_POLICY_V1,
            tp_size=1,
        )

        with patch(
            "sglang.srt.runtime_context.get_server_args", return_value=server_args
        ):
            kwargs = get_on_policy_rms_norm_kwargs(
                weight_dtype=torch.float32,
                override_orig_dtype=torch.float32,
                fp32_residual=True,
            )

            self.assertTrue(should_force_bfloat16_dense_tensor_math())
            self.assertTrue(should_force_bfloat16_lm_head(use_fp32_lm_head=False))
            self.assertFalse(should_force_bfloat16_lm_head(use_fp32_lm_head=True))
        self.assertEqual(kwargs["weight_dtype"], torch.float32)
        self.assertEqual(kwargs["override_orig_dtype"], torch.float32)
        self.assertTrue(kwargs["cast_x_before_out_mul"])
        self.assertTrue(kwargs["fp32_residual"])


class TestDenseOnPolicyContracts(unittest.TestCase):
    def test_qwen3_style_rms_norm_keeps_fp32_weight_output_and_residual(self):
        result = _run_dense_math_script(
            textwrap.dedent("""
                import json

                import torch

                from sglang.srt.layers.layernorm import RMSNorm
                from sglang.srt.runtime_context import publish
                from sglang.srt.server_args import ServerArgs
                from sglang.srt.true_on_policy import QWEN3_DENSE_TRUE_ON_POLICY_V1

                publish(
                    ServerArgs(
                        model_path="dummy",
                        true_on_policy_contract=QWEN3_DENSE_TRUE_ON_POLICY_V1,
                        tp_size=1,
                    ),
                    role="test",
                )
                norm = RMSNorm(
                    4,
                    eps=1e-6,
                    true_on_policy_weight_dtype=torch.float32,
                    true_on_policy_override_orig_dtype=torch.float32,
                    true_on_policy_fp32_residual=True,
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
                """)
        )

        self.assertEqual(result["weight_dtype"], "torch.float32")
        self.assertEqual(result["out_dtype"], "torch.float32")
        self.assertEqual(result["residual_dtype"], "torch.float32")

    def test_rms_norm_can_self_configure_from_true_on_policy_role_hints(self):
        result = _run_dense_math_script(
            textwrap.dedent("""
                import json

                import torch

                from sglang.srt.layers.layernorm import RMSNorm
                from sglang.srt.runtime_context import publish
                from sglang.srt.server_args import ServerArgs
                from sglang.srt.true_on_policy import QWEN3_DENSE_TRUE_ON_POLICY_V1

                publish(
                    ServerArgs(
                        model_path="dummy",
                        true_on_policy_contract=QWEN3_DENSE_TRUE_ON_POLICY_V1,
                        tp_size=1,
                    ),
                    role="test",
                )
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
                """)
        )

        self.assertEqual(result["weight_dtype"], "torch.float32")
        self.assertTrue(result["cast_x_before_out_mul"])
        self.assertTrue(result["fp32_residual"])
        self.assertEqual(result["override_orig_dtype"], "torch.float32")

    def test_on_policy_lm_head_forces_bfloat16_matmul_inputs(self):
        result = _run_dense_math_script(
            textwrap.dedent("""
                import json
                from types import SimpleNamespace
                from unittest.mock import patch

                import torch
                import torch.nn as nn

                from sglang.srt.layers.logits_processor import LogitsProcessor
                from sglang.srt.runtime_context import publish
                from sglang.srt.server_args import ServerArgs
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

                publish(
                    ServerArgs(
                        model_path="dummy",
                        enable_dp_lm_head=False,
                        enable_fp32_lm_head=False,
                        true_on_policy_contract=QWEN3_DENSE_TRUE_ON_POLICY_V1,
                        tp_size=1,
                    ),
                    role="test",
                )

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
                """)
        )

        self.assertEqual(result["a_dtype"], "torch.bfloat16")
        self.assertEqual(result["b_dtype"], "torch.bfloat16")
        self.assertEqual(result["logits_dtype"], "torch.bfloat16")


if __name__ == "__main__":
    unittest.main(verbosity=2)
