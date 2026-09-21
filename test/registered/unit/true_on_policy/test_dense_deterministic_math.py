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
    def test_qwen_projections_use_activation_dtype_with_quantized_weights(self):
        from sglang.srt.models import qwen3
        from sglang.srt.models.qwen2 import Qwen2MLP

        class ProjectionStub:
            def __init__(self, dtype, weight_dtype):
                self.params_dtype = dtype
                self.inputs = []
                if weight_dtype is not None:
                    self.weight = torch.empty(1, dtype=weight_dtype)

            def __call__(self, x, **kwargs):
                self.inputs.append(x)
                return x, None

        cases = [
            ("quant_bf16", torch.bfloat16, torch.bfloat16, None, False),
            ("quant_fp16", torch.float16, torch.float16, None, False),
            ("packed_int8", torch.bfloat16, torch.bfloat16, torch.int8, False),
            (
                "fp8_weight",
                torch.bfloat16,
                torch.bfloat16,
                torch.float8_e4m3fn,
                False,
            ),
            ("on_policy", torch.float32, torch.bfloat16, torch.bfloat16, True),
            ("cleared_flag", torch.float32, torch.bfloat16, torch.bfloat16, False),
            ("dense_bf16", torch.bfloat16, torch.bfloat16, torch.bfloat16, False),
            ("dense_fp16", torch.float16, torch.float16, torch.float16, False),
            ("dense_fp32", torch.float32, torch.float32, torch.float32, False),
        ]
        for name, input_dtype, dtype, weight_dtype, on_policy in cases:
            for forward in (qwen3.Qwen3Attention.forward, Qwen2MLP.forward):
                with self.subTest(case=name, forward=forward.__qualname__):
                    projection = ProjectionStub(dtype, weight_dtype)
                    output_projection = ProjectionStub(dtype, weight_dtype)
                    x = torch.randn(2, 4, dtype=input_dtype)

                    def prepare(positions, hidden_states):
                        projected, _ = projection(hidden_states)
                        return projected.float(), projected.float(), projected

                    def attend(q, k, v, forward_batch, save_kv_cache):
                        self.assertEqual((q.dtype, k.dtype, v.dtype), (dtype,) * 3)
                        self.assertTrue(save_kv_cache)
                        return v

                    model = SimpleNamespace(
                        qkv_proj=projection,
                        gate_up_proj=projection,
                        o_proj=output_projection,
                        down_proj=output_projection,
                        act_fn=lambda x: x,
                        use_fused_qk_norm_mrope=False,
                        forward_prepare_native=prepare,
                        attn=attend,
                    )
                    server_args = SimpleNamespace(
                        true_on_policy_contract=(
                            QWEN3_DENSE_TRUE_ON_POLICY_V1 if on_policy else None
                        ),
                        tp_size=1,
                    )
                    with (
                        patch(
                            "sglang.srt.runtime_context.get_server_args",
                            return_value=server_args,
                        ),
                        patch.object(qwen3, "_is_npu", False),
                    ):
                        if forward is qwen3.Qwen3Attention.forward:
                            output = forward(model, None, x, None)
                        else:
                            output = forward(model, x)

                    self.assertEqual(len(projection.inputs), 1)
                    self.assertEqual(len(output_projection.inputs), 1)
                    self.assertEqual(projection.inputs[0].dtype, dtype)
                    self.assertEqual(output_projection.inputs[0].dtype, dtype)
                    torch.testing.assert_close(output, x.to(dtype))
                    if input_dtype == dtype:
                        self.assertIs(projection.inputs[0], x)

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
