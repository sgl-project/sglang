import inspect
import itertools
import unittest
from unittest.mock import patch

import torch

import sglang.kernels.ops.activation.kimi_k3_situ_and_mul as situ_module
import sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe as moe
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")


class TestTritonMoeSitu(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.version.hip is not None:
            raise unittest.SkipTest("requires NVIDIA CUDA")
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def setUp(self):
        self.enterContext(get_parallel().override(tp_group=None))
        self.enterContext(torch.no_grad())
        self.gen = torch.Generator(device="cuda").manual_seed(4598)

    def tensor(self, shape, dtype=torch.bfloat16):
        return torch.randn(shape, generator=self.gen, device="cuda", dtype=dtype)

    def block_fp8(self, weight):
        e, n, k = weight.shape
        blocks = weight.float().view(e, n // 128, 128, k // 128, 128)
        scales = blocks.abs().amax(dim=(2, 4)).clamp_min(1e-12) / 448
        q = blocks / scales[:, :, None, :, None]
        return q.reshape_as(weight).to(torch.float8_e4m3fn), scales

    def layer(
        self,
        m,
        *,
        dtype=torch.bfloat16,
        fp8=False,
        inter=128,
        linear_beta=25.0,
        filter_expert=False,
    ):
        e, h, k = 16, 512, 8
        x = self.tensor((m, h), dtype)
        w1 = self.tensor((e, 2 * inter, h), dtype) / h**0.5
        w2 = self.tensor((e, h, inter), dtype) / inter**0.5
        vals, ids = self.tensor((m, e), torch.float32).topk(k)
        ids = ids.int()
        if filter_expert:
            ids[:, -1] = -1
        routes = StandardTopKOutput(vals.softmax(-1), ids, None)
        cfg = MoeRunnerConfig(
            num_experts=e * 2 if filter_expert else e,
            num_local_experts=e,
            top_k=k,
            activation="situ",
            is_gated=True,
            inplace=False,
            gemm1_clamp_limit=linear_beta,
        )
        kwargs = {}
        if fp8:
            w1, s1 = self.block_fp8(w1)
            w2, s2 = self.block_fp8(w2)
            kwargs = dict(
                use_fp8_w8a8=True, w1_scale=s1, w2_scale=s2, block_shape=[128, 128]
            )

        def run(hidden=x):
            return moe.fused_experts(hidden, w1, w2, routes, cfg, **kwargs)

        return x, run

    def reference(self, run):
        with patch.object(moe, "get_device_capability", return_value=(9, 0)):
            return run()

    def test_fallback_guards(self):
        _, run = self.layer(512)
        self.assert_fallback(run, sm=(9, 0))
        with patch.object(moe, "is_batch_invariant_mode_enabled", return_value=True):
            self.assert_fallback(run)

        for name, options in (
            ("fp16", {"m": 512, "dtype": torch.float16}),
            ("expert_filter", {"m": 512, "filter_expert": True}),
            ("width", {"m": 512, "inter": 127}),
        ):
            with self.subTest(name=name):
                _, run = self.layer(**options)
                self.assert_fallback(run)

    def assert_fallback(self, run, sm=(12, 0)):
        ref = self.reference(run)
        with (
            patch.object(moe, "get_device_capability", return_value=sm),
            patch.object(situ_module, "situ_and_mul") as fused,
        ):
            out = run()
        fused.assert_not_called()
        torch.testing.assert_close(out, ref, rtol=0, atol=0)

    def test_row_stride_falls_back(self):
        _, run = self.layer(512)
        empty = torch.empty
        gate_up_shape = (512 * 8, 256)
        seen = False

        def padded_gate_up(shape, *args, **kwargs):
            nonlocal seen
            if shape == gate_up_shape:
                seen = True
                return empty((shape[0], shape[1] + 1), *args, **kwargs)[:, :-1]
            return empty(shape, *args, **kwargs)

        with patch.object(moe.torch, "empty", side_effect=padded_gate_up):
            self.assert_fallback(run)
        self.assertTrue(seen)

    def test_tma_down_falls_back(self):
        if not moe._moe_support_tma():
            self.skipTest("requires TMA")
        _, run = self.layer(512)
        config = moe.try_get_optimal_moe_config

        def request_tma(*args, **kwargs):
            up, (down, extra) = config(*args, **kwargs)
            down = dict(up if down is None else down, USE_TMA=True)
            return up, (down, extra)

        with (
            patch.object(moe, "try_get_optimal_moe_config", side_effect=request_tma),
            patch.object(
                moe, "_fused_moe_kernel_sequence", wraps=moe._fused_moe_kernel_sequence
            ) as sequence,
        ):
            self.assert_fallback(run)
        bound = inspect.signature(moe._fused_moe_kernel_sequence).bind_partial(
            *sequence.call_args.args, **sequence.call_args.kwargs
        )
        self.assertTrue(bound.arguments["down_moe_use_tma"])

    def assert_layer_close(self, out, ref):
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.isfinite(ref).all())
        torch.testing.assert_close(out.float(), ref.float(), rtol=0.02, atol=0.04)

    def test_fused_experts_eager_and_cuda_graph(self):
        if torch.cuda.get_device_capability() != (12, 0):
            self.skipTest("requires SM120")
        for m, fp8, beta in itertools.product((64, 512), (False, True), (None, 25.0)):
            with self.subTest(m=m, fp8=fp8, linear_beta=beta):
                _, run = self.layer(m, fp8=fp8, linear_beta=beta)
                ref = self.reference(run)
                with patch.object(
                    situ_module, "situ_and_mul", wraps=situ_module.situ_and_mul
                ) as fused:
                    eager = run()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured = run()
                    graph.replay()
                self.assertEqual(fused.call_count, 2 if m >= 512 else 0)
                for out in (eager, captured):
                    self.assert_layer_close(out, ref)
                    if m < 512:
                        torch.testing.assert_close(out, ref, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
