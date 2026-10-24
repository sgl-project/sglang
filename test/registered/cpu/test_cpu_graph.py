import copy
import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("SGLANG_USE_CPU_ENGINE", "1")

import torch

from sglang.srt.utils import get_cpu_ids_by_node, kill_process_tree
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.sgl_eval_utils import run_sgl_eval
from sglang.test.test_utils import (
    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    intel_amx_benchmark,
    is_in_ci,
    popen_launch_server,
)

register_cpu_ci(est_time=315, suite="stage-a-tp-test-cpu-intel")


class TestCPUGraph(CustomTestCase):
    def test_cpu_cpp_wrapper_config_and_shims_registration(self):
        from unittest.mock import patch

        import torch._inductor.config as ic

        from sglang.srt.model_executor.cpu_graph_runner import set_torch_compile_config

        with patch.dict(os.environ, {"SGLANG_CPU_CPP_WRAPPER": "1"}):
            set_torch_compile_config()
            self.assertTrue(ic.cpp_wrapper)
            shims = ic.aot_inductor.custom_ops_to_c_shims
            self.assertGreaterEqual(len(shims), 6)
            registered_op_names = {op._schema.name for op in shims.keys()}
            self.assertIn(
                "sgl_kernel::multimodal_rotary_embedding_cpu", registered_op_names
            )
            self.assertIn("sgl_kernel::fp8_scaled_mm_cpu", registered_op_names)
            self.assertIn("sgl_kernel::fused_experts_cpu", registered_op_names)
            self.assertIn("sgl_kernel::shared_expert_cpu", registered_op_names)
            self.assertIn("sgl_kernel::qkv_proj_with_rope", registered_op_names)
            self.assertIn(
                "sgl_kernel::qkv_proj_with_rope_fused_weight", registered_op_names
            )

    def test_cpu_cpp_wrapper_custom_ops_accuracy(self):

        from sglang.srt.model_executor.cpu_graph_runner import register_fake_ops
        from sglang.srt.model_executor.runner_utils.cpu_c_shims import (
            enable_cpu_cpp_wrapper,
        )

        register_fake_ops(1)
        enable_cpu_cpp_wrapper()

        class CustomOpModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.randn(64, dtype=torch.bfloat16))

            def forward(self, pos, q, k, cs):
                torch.ops.sgl_kernel.multimodal_rotary_embedding_cpu(
                    pos, q, k, 64, cs, [16, 24, 24], True, True
                )
                normed = torch.ops.sgl_kernel.rmsnorm_cpu(q, self.w, 1e-5)
                act = torch.ops.sgl_kernel.silu_and_mul_cpu(normed)
                return act

        model = CustomOpModel()
        pos = torch.zeros(3, 1, dtype=torch.int64)
        q = torch.randn(1, 64, dtype=torch.bfloat16)
        k = torch.randn(1, 64, dtype=torch.bfloat16)
        cs = torch.randn(1024, 64, dtype=torch.bfloat16)

        with torch.no_grad():
            eager_out = model(pos, q.clone(), k.clone(), cs)
            compiled = torch.compile(model, backend="inductor")
            comp_out = compiled(pos, q.clone(), k.clone(), cs)

        self.assertEqual(comp_out.shape, eager_out.shape)
        self.assertTrue(torch.allclose(comp_out, eager_out, atol=1e-3, rtol=1e-3))

    def test_cpu_cpp_wrapper_transformer_block_execution(self):
        from sglang.srt.model_executor.runner_utils.cpu_c_shims import (
            enable_cpu_cpp_wrapper,
        )

        enable_cpu_cpp_wrapper()

        class MiniBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = torch.nn.Linear(
                    256, 256, bias=False, dtype=torch.bfloat16
                )
                self.k_proj = torch.nn.Linear(
                    256, 256, bias=False, dtype=torch.bfloat16
                )
                self.v_proj = torch.nn.Linear(
                    256, 256, bias=False, dtype=torch.bfloat16
                )
                self.out_proj = torch.nn.Linear(
                    256, 256, bias=False, dtype=torch.bfloat16
                )
                self.w_norm = torch.nn.Parameter(torch.randn(256, dtype=torch.bfloat16))

            def forward(self, x):
                q = self.q_proj(x)
                k = self.k_proj(x)
                v = self.v_proj(x)
                attn = self.out_proj(q * k + v)
                normed = torch.ops.sgl_kernel.rmsnorm_cpu(attn, self.w_norm, 1e-5)
                return torch.ops.sgl_kernel.silu_and_mul_cpu(normed)

        model = MiniBlock()
        x = torch.randn(4, 256, dtype=torch.bfloat16)

        with torch.no_grad():
            compiled = torch.compile(model, backend="inductor")
            out = compiled(x)

        self.assertEqual(out.shape, (4, 128))

    @intel_amx_benchmark(
        extra_args=[
            "--batch-size",
            "1",
            "--mem-fraction-static",
            "0.05",
            "--enable-torch-compile",
            "--torch-compile-max-bs",
            "2",
            "--cuda-graph-bs-decode",
            "2",
        ],
        min_throughput=7,
    )
    def test_latency_torch_compile_cpu(self):
        return DEFAULT_MLA_MODEL_NAME_FOR_TEST

    def test_mmlu_torch_compile_cpu(self):
        model = DEFAULT_MLA_MODEL_NAME_FOR_TEST
        base_url = DEFAULT_URL_FOR_TEST
        cpu_ids_by_node = get_cpu_ids_by_node()
        n_numa_node = len(cpu_ids_by_node)
        env = copy.deepcopy(os.environ)
        env["SGLANG_CPU_OMP_THREADS_BIND"] = "all"
        process = popen_launch_server(
            model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "intel_amx",
                "--mem-fraction-static",
                "0.05",
                "--disable-radix",
                "--trust-remote-code",
                "--disable-overlap-schedule",
                "--enable-torch-compile",
                "--cuda-graph-bs-decode",
                "2",
                "--tp",
                f"{n_numa_node}",
            ],
            env=env,
        )

        try:
            args = SimpleNamespace(
                base_url=base_url,
                model=model,
                eval_name="mmlu",
                num_examples=64,
                num_threads=32,
            )

            metrics = run_sgl_eval(args)
            if is_in_ci():
                self.assertGreater(metrics["score"], 0.45)
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
