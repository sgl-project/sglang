"""Real wide/narrow MLP prefill without the optional native gate GEMM API."""

import unittest
from types import MethodType
from unittest.mock import patch

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from sgl_kernel import gelu_tanh_and_mul

from sglang.kernels.fused_op import get_fused_op_backend, set_fused_op_backend
from sglang.kernels.spec import KernelBackend
from sglang.srt.models.gemma3_causal import Gemma3MLP
from sglang.srt.models.gemma4_causal import Gemma4MLP, Gemma4TextModel
from sglang.srt.runtime_context import SpawnRanks, create_context, use_context
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import CustomTestCase


@unittest.skipUnless(
    torch.xpu.is_available() and "B70" in torch.xpu.get_device_name(), "requires B70"
)
class TestGemma4PrefillLut(CustomTestCase):
    def test_original_projections_shared_owner_compile_replay_and_fallback(self):
        snapshot = snapshot_download(
            "google/gemma-4-E2B-it",
            revision="3e22461f65e89153144f8adb70e3b8c2cc9845a7",
            local_files_only=True,
        )
        context = create_context(
            ServerArgs(
                model_path=snapshot,
                device="xpu",
                model_impl="sglang",
                language_model_only=True,
                attention_backend="intel_xpu",
                page_size=64,
                disable_overlap_schedule=True,
            ),
            role="test",
            ranks=SpawnRanks(world_rank=0, gpu_id=0),
        )
        previous = get_fused_op_backend()
        set_fused_op_backend(None)
        try:
            with (
                use_context(context),
                context.parallel.override(tp_group=None, attn_tp_group=None),
                torch.no_grad(),
            ):
                for intermediate in (6144, 12288):
                    with self.subTest(intermediate=intermediate):
                        # Exercise the standalone lookup fallback even when a
                        # newer optional wheel provides fast fused prefill.
                        with patch(
                            "sgl_kernel.bf16_dense_gelu_tanh_mul_lut_xe20_prefill_supported",
                            return_value=False,
                            create=True,
                        ):
                            mlp = Gemma4MLP(1536, intermediate, "gelu_pytorch_tanh").to(
                                device="xpu", dtype=torch.bfloat16
                            )
                        mlp.gate_up_proj.weight.fill_(1 / 256)
                        mlp.down_proj.weight.fill_(1 / 256)
                        self.assertIsNotNone(mlp._xpu_gelu_mul_lut)
                        owner = Gemma4TextModel.__new__(Gemma4TextModel)
                        torch.nn.Module.__init__(owner)
                        owner.add_module("mlp", mlp)
                        owner._initialize_xpu_gate_up_lut()
                        owner._assert_xpu_gate_up_lut_initialized()
                        self.assertIs(
                            mlp._xpu_gate_up_gelu_table, owner._xpu_gate_up_lut_owner
                        )
                        x = torch.full(
                            (4096, 1536), 1 / 32, device="xpu", dtype=torch.bfloat16
                        )

                        def reference():
                            return F.linear(
                                gelu_tanh_and_mul(F.linear(x, mlp.gate_up_proj.weight)),
                                mlp.down_proj.weight,
                            )

                        graphs = []

                        def backend(graph, inputs):
                            graphs.append(str(graph.graph))
                            return graph.forward

                        compiled = torch.compile(mlp, backend=backend, fullgraph=True)
                        self.assertIsNone(mlp.act_fn._forward_method)
                        torch.testing.assert_close(
                            compiled(x), reference(), atol=0, rtol=0
                        )
                        self.assertTrue(graphs)
                        self.assertTrue(
                            all("triton_kernel_wrapper" in g for g in graphs)
                        )
                        self.assertTrue(
                            all(
                                "bf16_dense_gelu_tanh_mul_lut_xe20" not in g
                                for g in graphs
                            )
                        )
                        self.assertIsNone(mlp.act_fn._forward_method)
                        graph = torch.xpu.XPUGraph()
                        with torch.xpu.graph(graph):
                            output = mlp(x)
                        for scale in (1.0, -0.5):
                            x.mul_(scale)
                            expected = reference()
                            output.fill_(float("nan"))
                            graph.replay()
                            torch.testing.assert_close(output, expected, atol=0, rtol=0)

                        def forbidden_lookup(*args):
                            raise AssertionError(
                                "hook/backend reached lookup fast path"
                            )

                        mlp._xpu_gelu_mul_lut = forbidden_lookup
                        # Replacements before the first standalone activation
                        # must be selected by normal dispatch, not bypassed.
                        for method_name in ("forward_xpu", "_forward_impl"):
                            original = getattr(mlp.act_fn, method_name)
                            calls = []

                            def replacement(module, value):
                                calls.append(method_name)
                                return torch.zeros_like(value[..., :intermediate])

                            setattr(
                                mlp.act_fn,
                                method_name,
                                MethodType(replacement, mlp.act_fn),
                            )
                            mlp.act_fn._forward_method = None
                            try:
                                actual = mlp(x)
                                torch.testing.assert_close(
                                    actual, torch.zeros_like(actual), atol=0, rtol=0
                                )
                                self.assertEqual(calls, [method_name])
                            finally:
                                setattr(mlp.act_fn, method_name, original)
                                mlp.act_fn._forward_method = None
                        observed = []
                        handle = mlp.act_fn.register_forward_hook(
                            lambda module, inputs, output: observed.append(output)
                        )
                        try:
                            torch.testing.assert_close(
                                mlp(x), reference(), atol=0, rtol=0
                            )
                            self.assertEqual(len(observed), 1)
                        finally:
                            handle.remove()
                        set_fused_op_backend(KernelBackend.TORCH)
                        expected = Gemma3MLP.forward(mlp, x)
                        torch.testing.assert_close(mlp(x), expected, atol=0, rtol=0)
                        set_fused_op_backend(None)
                        torch._dynamo.reset()
        finally:
            set_fused_op_backend(previous)
            torch._dynamo.reset()


if __name__ == "__main__":
    unittest.main()
