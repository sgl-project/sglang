"""Actual decoder tail: keep norms/hooks and all non-prefill fallback routes."""

import unittest
from types import MethodType

import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig

from sglang.kernels.fused_op import get_fused_op_backend, set_fused_op_backend
from sglang.kernels.spec import KernelBackend
from sglang.srt.models.gemma4_causal import Gemma4DecoderLayer, Gemma4TextModel
from sglang.srt.runtime_context import SpawnRanks, create_context, use_context
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import CustomTestCase


@unittest.skipUnless(
    torch.xpu.is_available() and "B70" in torch.xpu.get_device_name(), "requires B70"
)
class TestGemma4PleAddScale(CustomTestCase):
    def test_real_tail_compile_replay_norm_hooks_and_fallbacks(self):
        snapshot = snapshot_download(
            "google/gemma-4-E2B-it",
            revision="3e22461f65e89153144f8adb70e3b8c2cc9845a7",
            local_files_only=True,
        )
        config = AutoConfig.from_pretrained(snapshot, local_files_only=True).text_config
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
                layer = Gemma4DecoderLayer(0, config).to(
                    device="xpu", dtype=torch.bfloat16
                )
                for parameter in layer.parameters():
                    parameter.requires_grad_(False)
                    parameter.fill_(1 / 256)
                layer.layer_scalar.fill_(1.5)
                owner = Gemma4TextModel.__new__(Gemma4TextModel)
                torch.nn.Module.__init__(owner)
                owner.add_module("decoder", layer)
                owner._initialize_xpu_gate_up_lut()
                owner._assert_xpu_gate_up_lut_initialized()
                callback = layer._xpu_ple_add_scale
                self.assertIsNotNone(callback)

                def tail(x, residual, per_layer):
                    return layer._next_stream(x, residual, per_layer, None)

                def reference(x, residual, per_layer):
                    saved = layer._xpu_ple_add_scale
                    layer._xpu_ple_add_scale = None
                    try:
                        return tail(x, residual, per_layer)
                    finally:
                        layer._xpu_ple_add_scale = saved

                x = torch.randn(4096, 1536, device="xpu", dtype=torch.bfloat16) / 16
                residual = torch.randn_like(x) / 8
                per_layer = torch.full(
                    (4096, 256), 1 / 16, device="xpu", dtype=torch.bfloat16
                )
                torch.testing.assert_close(
                    tail(x, residual, per_layer),
                    reference(x, residual, per_layer),
                    atol=0,
                    rtol=0,
                )
                compiled = torch.compile(tail, backend="eager", fullgraph=True)
                torch.testing.assert_close(
                    compiled(x, residual, per_layer),
                    reference(x, residual, per_layer),
                    atol=0,
                    rtol=0,
                )
                tail(x, residual, per_layer)
                torch.xpu.synchronize()
                graph = torch.xpu.XPUGraph()
                with torch.xpu.graph(graph):
                    output = tail(x, residual, per_layer)
                scalar_pointer = layer.layer_scalar.data_ptr()
                for scalar in (-0.5, 1.75):
                    x.mul_(0.5)
                    residual.add_(0.125)
                    per_layer.mul_(0.5)
                    layer.layer_scalar.fill_(scalar)
                    self.assertEqual(layer.layer_scalar.data_ptr(), scalar_pointer)
                    output.fill_(float("nan"))
                    graph.replay()
                    torch.xpu.synchronize()
                    torch.testing.assert_close(
                        output, reference(x, residual, per_layer), atol=0, rtol=0
                    )

                # The optimized pair consumes the norm's actual __call__ result;
                # it must not bypass hooks or a delegated norm implementation.
                observed = []
                hook = layer.post_per_layer_input_norm.register_forward_hook(
                    lambda module, inputs, value: observed.append(True)
                )
                try:
                    torch.testing.assert_close(
                        tail(x, residual, per_layer),
                        reference(x, residual, per_layer),
                        atol=0,
                        rtol=0,
                    )
                    self.assertEqual(observed, [True, True])
                finally:
                    hook.remove()
                norm = layer.post_per_layer_input_norm
                original = norm.forward_xpu
                norm.forward_xpu = MethodType(
                    lambda module, value: torch.zeros_like(value), norm
                )
                norm._forward_method = None
                try:
                    torch.testing.assert_close(
                        tail(x, residual, per_layer),
                        reference(x, residual, per_layer),
                        atol=0,
                        rtol=0,
                    )
                finally:
                    norm.forward_xpu = original
                    norm._forward_method = None

                def forbidden(*args):
                    raise AssertionError("fallback reached PLE pair callback")

                layer._xpu_ple_add_scale = forbidden
                for rows in (128,):
                    torch.testing.assert_close(
                        tail(x[:rows], residual[:rows], per_layer[:rows]),
                        reference(x[:rows], residual[:rows], per_layer[:rows]),
                        atol=0,
                        rtol=0,
                    )
                set_fused_op_backend(KernelBackend.TORCH)
                torch.testing.assert_close(
                    tail(x, residual, per_layer),
                    reference(x, residual, per_layer),
                    atol=0,
                    rtol=0,
                )
                set_fused_op_backend(None)
                with torch.enable_grad():
                    torch.testing.assert_close(
                        tail(x, residual, per_layer),
                        reference(x, residual, per_layer),
                        atol=0,
                        rtol=0,
                    )
                layer.layer_scalar = layer.layer_scalar.float()
                actual = tail(x, residual, per_layer)
                expected = reference(x, residual, per_layer)
                self.assertEqual(actual.dtype, expected.dtype)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                set_fused_op_backend(KernelBackend.TORCH)
                disabled = Gemma4DecoderLayer(0, config)
                self.assertIsNone(disabled._xpu_ple_add_scale)
        finally:
            set_fused_op_backend(previous)
            torch._dynamo.reset()


if __name__ == "__main__":
    unittest.main()
