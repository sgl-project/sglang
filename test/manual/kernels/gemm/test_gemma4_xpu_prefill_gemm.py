"""Actual optional fast-prefill wheel, model guards, and legacy-wheel fallback."""

import unittest
from types import MethodType

import sgl_kernel
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download

from sglang.kernels.fused_op import get_fused_op_backend, set_fused_op_backend
from sglang.kernels.spec import KernelBackend
from sglang.srt.models.gemma3_causal import Gemma3MLP
from sglang.srt.models.gemma4_causal import Gemma4MLP, Gemma4TextModel
from sglang.srt.runtime_context import SpawnRanks, create_context, use_context
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import CustomTestCase

_QUERY = "bf16_dense_gelu_tanh_mul_lut_xe20_prefill_supported"
_HAS_FAST_PREFILL = (
    torch.xpu.is_available()
    and "B70" in torch.xpu.get_device_name()
    and all(
        getattr(sgl_kernel, _QUERY, lambda *args: False)(4096, n, 1536)
        for n in (12288, 24576)
    )
)


def make_context():
    snapshot = snapshot_download(
        "google/gemma-4-E2B-it",
        revision="3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        local_files_only=True,
    )
    return create_context(
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


def make_mlp(intermediate):
    mlp = Gemma4MLP(1536, intermediate, "gelu_pytorch_tanh").to(
        device="xpu", dtype=torch.bfloat16
    )
    mlp.gate_up_proj.weight.fill_(1 / 256)
    mlp.down_proj.weight.fill_(1 / 256)
    return mlp


def make_owner(*modules):
    owner = Gemma4TextModel.__new__(Gemma4TextModel)
    torch.nn.Module.__init__(owner)
    for index, module in enumerate(modules):
        owner.add_module(str(index), module)
    owner._initialize_xpu_gate_up_lut()
    owner._assert_xpu_gate_up_lut_initialized()
    return owner


def reference(mlp, x):
    # Aligned clones avoid Torch's known BF16 offset-one GEMM issue. These
    # binary-fraction operands also have exactly representable FP32 dot products.
    projected = F.linear(x.clone(), mlp.gate_up_proj.weight.clone())
    return F.linear(sgl_kernel.gelu_tanh_and_mul(projected), mlp.down_proj.weight)


def native_count(graph):
    return sum(
        node.op == "call_function"
        and node.target
        in (
            torch.ops.sgl_kernel.bf16_dense_gelu_tanh_mul_lut_xe20,
            torch.ops.sgl_kernel.bf16_dense_gelu_tanh_mul_lut_xe20.default,
        )
        for node in graph.graph.nodes
    )


@unittest.skipUnless(_HAS_FAST_PREFILL, "requires the fast-prefill native wheel on B70")
class TestGemma4PrefillGemm(CustomTestCase):
    def test_shared_owner_native_compile_replay_and_fallback_guards(self):
        context = make_context()
        previous = get_fused_op_backend()
        set_fused_op_backend(None)
        try:
            with (
                use_context(context),
                context.parallel.override(tp_group=None, attn_tp_group=None),
                torch.no_grad(),
            ):
                wide, narrow = make_mlp(12288), make_mlp(6144)
                self.assertIsNotNone(wide._xpu_prefill_gate_up_lut)
                self.assertIsNotNone(narrow._xpu_prefill_gate_up_lut)
                owner = make_owner(wide, narrow)
                for mlp in (wide, narrow):
                    self.assertIs(
                        mlp._xpu_gate_up_gelu_table, owner._xpu_gate_up_lut_owner
                    )
                for mlp in (wide, narrow):
                    torch._dynamo.reset()
                    graphs = []

                    def backend(graph, _):
                        graphs.append(graph)
                        return graph.forward

                    compiled = torch.compile(mlp, backend=backend, fullgraph=True)
                    for rows in mlp._xpu_prefill_gate_up_rows:
                        short = torch.full(
                            (rows, 1536), 1 / 32, device="xpu", dtype=torch.bfloat16
                        )
                        torch.testing.assert_close(
                            compiled(short), reference(mlp, short), atol=0, rtol=0
                        )
                    original_weight = mlp.gate_up_proj.weight.detach().clone()
                    for offset, scale in ((32, 1), (1, 0.5), (0, -1)):
                        x = torch.full(
                            (4096 * 1536 + offset,),
                            1 / 32 * scale,
                            device="xpu",
                            dtype=torch.bfloat16,
                        )[offset:].view(4096, 1536)
                        weight = torch.empty(
                            original_weight.numel() + offset,
                            device="xpu",
                            dtype=torch.bfloat16,
                        )[offset:].view_as(original_weight)
                        weight.copy_(original_weight * scale)
                        mlp.gate_up_proj.weight.data = weight
                        torch.testing.assert_close(
                            compiled(x), reference(mlp, x), atol=0, rtol=0
                        )
                    self.assertTrue(graphs)
                    self.assertTrue(all(native_count(graph) == 1 for graph in graphs))
                    self.assertTrue(
                        all(
                            "triton_kernel_wrapper" not in str(graph.graph)
                            for graph in graphs
                        )
                    )

                    # Both operands are offset-one views so host alignment clones
                    # must themselves be captured and read changed data on replay.
                    x = torch.full(
                        (4096 * 1536 + 1,), 1 / 32, device="xpu", dtype=torch.bfloat16
                    )[1:].view(4096, 1536)
                    weight = torch.empty(
                        original_weight.numel() + 1, device="xpu", dtype=torch.bfloat16
                    )[1:].view_as(original_weight)
                    weight.copy_(original_weight)
                    mlp.gate_up_proj.weight.data = weight
                    mlp(x)
                    torch.xpu.synchronize()
                    graph = torch.xpu.XPUGraph()
                    with torch.xpu.graph(graph):
                        output = mlp(x)
                    for scale_x, scale_w in ((0.5, 2), (-1, 0.25)):
                        x.mul_(scale_x)
                        weight.mul_(scale_w)
                        output.fill_(float("nan"))
                        graph.replay()
                        torch.xpu.synchronize()
                        torch.testing.assert_close(
                            output, reference(mlp, x), atol=0, rtol=0
                        )

                    mlp.gate_up_proj.weight.data = weight.clone()
                    x = x.clone()

                    def forbidden(*args):
                        raise AssertionError(
                            "guarded fallback reached an optimized callback"
                        )

                    mlp._xpu_prefill_gate_up_lut = forbidden
                    # Unadmitted rows retain the original parent MLP path.
                    mlp._xpu_gelu_mul_lut = forbidden
                    for rows in (1, 26, 61, 128, 512):
                        small = x[:rows].contiguous()
                        torch.testing.assert_close(
                            mlp(small), Gemma3MLP.forward(mlp, small), atol=0, rtol=0
                        )
                    mlp._xpu_gelu_mul_lut = forbidden
                    for method in ("forward_xpu", "_forward_impl"):
                        original = getattr(mlp.act_fn, method)
                        calls = []

                        def replacement(module, value):
                            calls.append(method)
                            return torch.zeros_like(value[..., : value.shape[-1] // 2])

                        setattr(mlp.act_fn, method, MethodType(replacement, mlp.act_fn))
                        mlp.act_fn._forward_method = None
                        try:
                            self.assertEqual(torch.count_nonzero(mlp(x)).item(), 0)
                            self.assertEqual(calls, [method])
                        finally:
                            setattr(mlp.act_fn, method, original)
                            mlp.act_fn._forward_method = None
                    observed = []
                    handle = mlp.act_fn.register_forward_hook(
                        lambda *args: observed.append(True)
                    )
                    try:
                        torch.testing.assert_close(
                            mlp(x), reference(mlp, x), atol=0, rtol=0
                        )
                        self.assertEqual(len(observed), 1)
                    finally:
                        handle.remove()
                    set_fused_op_backend(KernelBackend.TORCH)
                    torch.testing.assert_close(
                        mlp(x), Gemma3MLP.forward(mlp, x), atol=0, rtol=0
                    )
                    set_fused_op_backend(None)
        finally:
            set_fused_op_backend(previous)
            torch._dynamo.reset()

    def test_legacy_api_with_missing_or_false_query_keeps_lookup(self):
        context = make_context()
        previous = get_fused_op_backend()
        set_fused_op_backend(None)
        try:
            with (
                use_context(context),
                context.parallel.override(tp_group=None, attn_tp_group=None),
                torch.no_grad(),
            ):
                self.assertTrue(
                    hasattr(torch.ops.sgl_kernel, "bf16_dense_gelu_tanh_mul_lut_xe20")
                )
                for mode in ("missing", "false", "wide_only", "legacy_4096"):
                    with self.subTest(mode=mode):
                        for intermediate in (6144, 12288):
                            original = getattr(sgl_kernel, _QUERY)
                            calls = []

                            def unsupported(*shape):
                                calls.append(shape)
                                return (mode == "wide_only" and shape[1] == 24576) or (
                                    mode == "legacy_4096" and shape[0] == 4096
                                )

                            try:
                                if mode == "missing":
                                    delattr(sgl_kernel, _QUERY)
                                else:
                                    setattr(sgl_kernel, _QUERY, unsupported)
                                mlp = make_mlp(intermediate)
                            finally:
                                setattr(sgl_kernel, _QUERY, original)
                            native_expected = mode == "legacy_4096" or (
                                mode == "wide_only" and intermediate == 12288
                            )
                            self.assertEqual(
                                mlp._xpu_prefill_gate_up_lut is not None,
                                native_expected,
                            )
                            if mode != "missing":
                                self.assertEqual(
                                    calls,
                                    [
                                        (rows, 2 * intermediate, 1536)
                                        for rows in (256, 1024, 2048, 4095, 4096)
                                    ],
                                )
                            self.assertEqual(
                                mlp._xpu_prefill_gate_up_rows,
                                (4096,)
                                if mode == "legacy_4096"
                                else (
                                    (256, 1024, 2048, 4095, 4096)
                                    if native_expected
                                    else ()
                                ),
                            )
                            owner = make_owner(mlp)
                            self.assertIs(
                                mlp._xpu_gate_up_gelu_table,
                                owner._xpu_gate_up_lut_owner,
                            )
                            x = torch.full(
                                (4096, 1536), 1 / 32, device="xpu", dtype=torch.bfloat16
                            )
                            graphs = []

                            def backend(graph, _):
                                graphs.append(graph)
                                return graph.forward

                            compiled = torch.compile(
                                mlp, backend=backend, fullgraph=True
                            )
                            torch.testing.assert_close(
                                compiled(x), reference(mlp, x), atol=0, rtol=0
                            )
                            self.assertTrue(graphs)
                            self.assertTrue(
                                all(
                                    native_count(graph) == int(native_expected)
                                    for graph in graphs
                                )
                            )
                            self.assertTrue(
                                all(
                                    ("triton_kernel_wrapper" in str(graph.graph))
                                    == (not native_expected)
                                    for graph in graphs
                                )
                            )
                            torch._dynamo.reset()
        finally:
            set_fused_op_backend(previous)
            torch._dynamo.reset()


if __name__ == "__main__":
    unittest.main()
