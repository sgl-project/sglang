"""GLM-5.3-Flash KDA projections in FP8 (SGLANG_OPT_GLM53_KDA_PTPC_MODULES) through the
real load-time quantization and padding, the real quantizer and GEMM kernels, and CUDA
graph replay. On H200 and H20 it also checks the tuned tile tables."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

from sglang.kernels.fused_op import get_fused_op_backend, set_fused_op_backend
from sglang.kernels.ops.gemm import fp8_scaled_mm
from sglang.kernels.ops.gemm.fp8_kernel import (
    get_w8a8_channelwise_fp8_config,
    get_w8a8_channelwise_fp8_configs,
    triton_scaled_mm,
)
from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_quant_fp8
from sglang.kernels.spec import KernelBackend
from sglang.srt.environ import envs
from sglang.srt.layers.quantization import fp8_utils
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
from sglang.srt.model_executor.model_runner_components.weight_exporter import (
    WeightExporter,
)
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
    _unsupported_derived_weight_cache_error,
)
from sglang.srt.model_loader.loader import _modules_with_quant_method
from sglang.srt.models import glm5_next
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# H100 runs the default path of a Hopper GPU without tuned tables, in the per-commit stage.
register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")
register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="8-gpu-h200")
register_cuda_ci(est_time=60, stage="base-c", runner_config="8-gpu-h20")

HIDDEN, HEADS, HEAD_DIM = 4096, 64, 128
PREFIX = "model.layers.0.self_attn"
# Per-rank (N, K) of the fused [q|k|v|b|f_a|g_a] projection and of o_proj,
# at attention TP 1, 4 and 8; the load pads TP8's N = 3336 to 3344.
SHAPES = {
    "input_tp1": (24896, 4096),
    "input_tp4": (6416, 4096),
    "input_tp8": (3336, 4096),
    "o_proj_tp1": (4096, 8192),
    "o_proj_tp4": (4096, 2048),
    "o_proj_tp8": (4096, 1024),
}
TUNED = torch.cuda.is_available() and torch.cuda.get_device_name() in (
    "NVIDIA H200",
    "NVIDIA H20",
)


def padded(n):
    return n + -n % 16


def magnitudes(rows, generator):
    # Rows span 2^-6..2^6, so a scale applied to the wrong row or column is visible.
    exponents = torch.randint(-6, 7, (rows, 1), generator=generator, device="cuda")
    return torch.pow(2.0, exponents.float())


def random_weight(n, k, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    weight = torch.randn(n, k, generator=generator, device="cuda") * 0.02
    return (weight * magnitudes(n, generator)).to(torch.bfloat16)


def make_input(m, k, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, generator=generator, device="cuda")
    return (x * magnitudes(m, generator)).to(torch.bfloat16)


def make_layer(n, k, seed=0):
    """One projection with the quant method the model uses, loaded and post-processed."""
    method = glm5_next._KdaFp8Config(is_checkpoint_fp8_serialized=False)
    method = method.get_quant_method(None, "")
    layer = torch.nn.Module()
    layer.quant_method = (
        method  # as LinearBase sets it; the model loader finds layers by it
    )
    with torch.device("cuda"):
        method.create_weights(layer, k, [n], k, n, torch.bfloat16)
    weight = random_weight(n, k, seed)
    layer.weight.data.copy_(weight)
    layer.input_scale = None
    method.process_weights_after_loading(layer)
    return method, layer, weight


def dequantized_reference(layer, x, n, bias=None):
    """The GEMM the FP8 kernels compute, in FP32 from the FP8 weight, the per-token FP8
    activations and both scales, with an elementwise error bound."""
    x_q, x_scale = sglang_per_token_quant_fp8(x)
    a = x_q.float() * x_scale.view(-1, 1)
    b = layer.weight.float() * layer.weight_scale.view(1, -1)
    ref = (a @ b)[:, :n]
    if bias is not None:
        ref = ref + bias.float()
    # BF16 output rounding, plus the FP8 tensor cores' reduced-precision accumulation.
    bound = 2**-8 * ref.abs() + 2**-9 * (a.abs() @ b.abs())[:, :n]
    return ref, bound


def assert_matches(test, y, ref, bound):
    test.assertEqual(y.dtype, torch.bfloat16)
    test.assertTrue(torch.all((y.float() - ref).abs() <= bound))
    test.assertLess(((y.float() - ref).norm() / ref.norm()).item(), 1e-2)


def table_tokens(n, k):
    """One token count per distinct tuned tile of the table, and its first CUTLASS point."""
    table = get_w8a8_channelwise_fp8_configs(padded(n), k) or {}
    tiles, cutlass = {}, []
    for m in sorted(table):
        if table[m]:
            tiles.setdefault(tuple(sorted(table[m].items())), m)
        else:
            cutlass.append(m)
    return sorted(tiles.values()), cutlass[:1]


def relative_error(actual, reference):
    return (
        (actual.float() - reference.float()).norm() / reference.float().norm()
    ).item()


def kernel_names(fn):
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return {e.name for e in prof.events() if e.device_type.name == "CUDA"}


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0),
    "the KDA FP8 projections are tested on Hopper",
)
class TestGlm53KdaFp8Projection(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        print(f"{torch.cuda.get_device_name()}: tuned KDA tables checked: {TUNED}")

    def setUp(self):
        # EnvField.override skips its restore when the body raises;
        # this outer override exits normally and restores the variable.
        tune = envs.SGLANG_ENABLE_FP8_GEMM_CONFIG_TUNE.override(True)
        tune.__enter__()
        self.addCleanup(tune.__exit__, None, None, None)

    def test_load_pads_and_quantizes_per_channel(self):
        for name, (n, k) in SHAPES.items():
            with self.subTest(shape=name):
                _, layer, weight = make_layer(n, k)
                self.assertEqual(layer.weight.dtype, torch.float8_e4m3fn)
                self.assertEqual(tuple(layer.weight.shape), (k, padded(n)))
                self.assertTrue(torch.all(layer.weight[:, n:].float() == 0))
                restored = layer.weight.float() * layer.weight_scale.view(1, -1)
                error = (restored[:, :n].t() - weight.float()).abs()
                # e4m3 keeps 3 mantissa bits: relative error at most 2^-4 per element.
                row_max = weight.float().abs().amax(dim=1, keepdim=True)
                bound = 2**-4 * weight.float().abs() + 2**-9 * row_max
                self.assertTrue(torch.all(error <= bound))
                if TUNED:
                    self.assertIsNotNone(get_w8a8_channelwise_fp8_configs(padded(n), k))

    def test_output_matches_dequantized_gemm(self):
        tiles_seen = 0
        for name, (n, k) in SHAPES.items():
            tiles, cutlass = table_tokens(n, k)
            tiles_seen += len(tiles)
            for tune, tokens in (
                (True, {1, 8191, 8192, *tiles, *cutlass}),
                (False, {1, 64}),
            ):
                with envs.SGLANG_ENABLE_FP8_GEMM_CONFIG_TUNE.override(tune):
                    method, layer, weight = make_layer(n, k)
                for m in sorted(tokens):
                    with self.subTest(shape=name, tune=tune, m=m):
                        x = make_input(m, k, seed=m)
                        y = method.apply(layer, x)
                        self.assertEqual(tuple(y.shape), (m, n))
                        assert_matches(self, y, *dequantized_reference(layer, x, n))
                        bf16 = x.float() @ weight.float().t()
                        self.assertLess(relative_error(y, bf16), 0.08)
        if TUNED:
            self.assertGreater(tiles_seen, 0)

    def test_kernel_choice_matches_apply_fp8_linear(self):
        # The projection calls the kernel apply_fp8_linear would pick;
        # its output equals that kernel's bitwise, and the profiler shows which kernels ran.
        import sgl_kernel

        n, k = SHAPES["input_tp1"]

        def check(method, layer, m, expected, other=None):
            n, k = layer.output_size_per_partition, layer.input_size_per_partition
            x = make_input(m, k, seed=7)
            x_q, x_scale = sglang_per_token_quant_fp8(x)
            gemm = (x_q, layer.weight, x_scale, layer.weight_scale)
            ran = kernel_names(lambda: method.apply(layer, x))
            want = kernel_names(lambda: expected(x, gemm))
            self.assertTrue(
                torch.equal(method.apply(layer, x), expected(x, gemm)[:, :n])
            )
            self.assertTrue(want)
            self.assertLessEqual(want, ran)
            if other is not None:
                self.assertTrue(kernel_names(lambda: other(gemm)) - want)
                self.assertFalse((kernel_names(lambda: other(gemm)) - want) & ran)

        def aot(x, gemm):
            return sgl_kernel.fp8_scaled_mm(*gemm, torch.bfloat16, None)

        def generic(x, gemm):
            return fp8_scaled_mm(*gemm, out_dtype=torch.bfloat16)

        def triton_default(x, gemm):
            return triton_scaled_mm(*gemm, torch.bfloat16, None)

        with envs.SGLANG_ENABLE_FP8_GEMM_CONFIG_TUNE.override(False):
            method, layer, _ = make_layer(n, k)
        for m in (16, 8191):
            with self.subTest(case="aot", m=m):
                check(method, layer, m, aot, other=lambda g: triton_default(None, g))
                check(method, layer, m, generic)
        with self.subTest(case="8192 rows"):
            check(method, layer, 8192, generic)

        previous = get_fused_op_backend()
        set_fused_op_backend(KernelBackend.TORCH)
        try:
            with self.subTest(case="forced backend"):
                check(method, layer, 16, generic, other=lambda g: aot(None, g))
        finally:
            set_fused_op_backend(previous)

        with (
            self.subTest(case="USE_TRITON_W8A8_FP8_KERNEL"),
            patch.object(glm5_next, "use_triton_w8a8_fp8_kernel", True),
            patch.object(fp8_utils, "use_triton_w8a8_fp8_kernel", True),
        ):
            triton_method, triton_layer, _ = make_layer(n, k)
            check(
                triton_method,
                triton_layer,
                16,
                lambda x, g: Fp8LinearMethod.apply(triton_method, triton_layer, x),
                other=lambda g: aot(None, g),
            )

        for name in ("input_tp1", "input_tp8"):
            n, k = SHAPES[name]
            tiles, _ = table_tokens(n, k)
            if not tiles:
                self.assertFalse(TUNED, f"the {name} table selects no tuned tile")
                continue
            tune_method, tune_layer, _ = make_layer(n, k)
            tile = get_w8a8_channelwise_fp8_config(N=padded(n), K=k, M=tiles[0])

            def tiled(x, gemm, tile=tile):
                return triton_scaled_mm(
                    *gemm,
                    torch.bfloat16,
                    None,
                    block_size_m=tile["BLOCK_SIZE_M"],
                    block_size_n=tile["BLOCK_SIZE_N"],
                    block_size_k=tile["BLOCK_SIZE_K"],
                    use_heuristic=False,
                    num_warps=tile["num_warps"],
                    num_stages=tile["num_stages"],
                )

            with self.subTest(case="tuned tile", shape=name, m=tiles[0]):
                check(
                    tune_method,
                    tune_layer,
                    tiles[0],
                    tiled,
                    other=lambda g: aot(None, g),
                )

    @torch.no_grad()
    def test_model_projections_match_bf16_reference(self):
        # The model's modules, weight loaders, padding and fused forward;
        # q, k, v and b are split per rank, f_a and g_a are whole on every rank.
        dim = HEAD_DIM * HEADS
        shapes = {name: (dim, HIDDEN) for name in ("q_proj", "k_proj", "v_proj")}
        shapes.update(
            b_proj=(HEADS, HIDDEN),
            f_a_proj=(HEAD_DIM, HIDDEN),
            g_a_proj=(HEAD_DIM, HIDDEN),
            f_b_proj=(dim, HEAD_DIM),
            g_b_proj=(dim, HEAD_DIM),
            o_proj=(HIDDEN, dim),
        )
        weights = {
            name: random_weight(*shape, seed=i)
            for i, (name, shape) in enumerate(shapes.items())
        }
        server = get_context().override_server_args(
            device="cuda", enable_lora=False, lora_paths=None
        )
        server.install()
        self.addCleanup(server.restore)
        # The model loader builds the modules with the checkpoint dtype as the default.
        self.addCleanup(torch.set_default_dtype, torch.get_default_dtype())
        torch.set_default_dtype(torch.bfloat16)
        for tp, rank in ((1, 0), (8, 0), (8, 7)):
            with (
                self.subTest(attn_tp=tp, rank=rank),
                get_parallel().override(
                    tp_size=tp,
                    tp_rank=rank,
                    attn_tp_size=tp,
                    attn_tp_rank=rank,
                    attn_dp_size=1,
                    attn_dp_rank=0,
                    attn_cp_size=1,
                    attn_cp_rank=0,
                    moe_tp_size=tp,
                ),
                envs.SGLANG_OPT_GLM53_KDA_PTPC_MODULES.override(
                    "qkv_proj,f_a_proj,g_a_proj,o_proj"
                ),
                torch.device("cuda"),
            ):
                attention = glm5_next.Glm5NextLinearAttention(
                    layer_idx=0,
                    hidden_size=HIDDEN,
                    config=SimpleNamespace(
                        linear_attn_config={
                            "head_dim": HEAD_DIM,
                            "num_heads": HEADS,
                            "short_conv_kernel_size": 4,
                        }
                    ),
                    quant_config=None,
                    prefix=PREFIX,
                )
                model = SimpleNamespace(
                    config=SimpleNamespace(n_routed_experts=0),
                    num_fused_shared_experts=0,
                    quant_config=None,
                    named_parameters=lambda: (
                        (f"{PREFIX}.{name}", param)
                        for name, param in attention.named_parameters()
                    ),
                )
                with patch.object(
                    glm5_next.DeepseekV2WeightLoaderMixin, "post_load_weights"
                ):
                    glm5_next.Glm5NextForConditionalGeneration.load_weights(
                        model,
                        [(f"{PREFIX}.{name}.weight", w) for name, w in weights.items()],
                    )
                for module, quant_method in _modules_with_quant_method(attention):
                    quant_method.process_weights_after_loading(module)
                for layer in (attention.fused_qkvbfg_a_proj, attention.o_proj):
                    self.assertIs(
                        type(layer.quant_method), glm5_next._KdaFp8LinearMethod
                    )
                    self.assertEqual(layer.weight.dtype, torch.float8_e4m3fn)

                def shard(name, dim=0):
                    return weights[name].chunk(tp, dim=dim)[rank]

                x = make_input(64, HIDDEN, seed=tp + rank)
                expected = (
                    torch.cat(
                        [F.linear(x, shard(n)) for n in ("q_proj", "k_proj", "v_proj")],
                        dim=-1,
                    ),
                    F.linear(x, shard("b_proj")),
                    F.linear(F.linear(x, weights["f_a_proj"]), shard("f_b_proj")),
                    F.linear(F.linear(x, weights["g_a_proj"]), shard("g_b_proj")),
                )
                outputs = attention.forward_qkvbfg_fused(x, None)
                for part, actual, reference in zip(
                    ("qkv", "beta", "forget_gate", "g_proj"), outputs, expected
                ):
                    self.assertEqual(actual.shape, reference.shape, part)
                    self.assertLess(relative_error(actual, reference), 0.08, part)
                z = make_input(64, dim // tp, seed=100 + rank)
                # The row-parallel forward around it only adds the TP all-reduce.
                out = attention.o_proj.quant_method.apply(attention.o_proj, z)
                reference = F.linear(z, shard("o_proj", dim=1))
                self.assertLess(relative_error(out, reference), 0.08)

    def test_bias_takes_the_generic_fp8_path(self):
        # The KDA projections are built with bias=False;
        # with a bias, an unpadded projection still goes to apply_fp8_linear.
        n, k = SHAPES["o_proj_tp8"]
        method, layer, _ = make_layer(n, k)
        x = make_input(16, k, seed=1)
        generator = torch.Generator(device="cuda").manual_seed(2)
        bias = torch.randn(n, generator=generator, device="cuda").to(torch.bfloat16)
        y = method.apply(layer, x, bias)
        self.assertEqual(tuple(y.shape), (16, n))
        assert_matches(self, y, *dequantized_reference(layer, x, n, bias))

    def test_cuda_graph_replay_matches_eager(self):
        tile_captures = 0
        for name in ("input_tp1", "input_tp8", "o_proj_tp8"):
            n, k = SHAPES[name]
            method, layer, _ = make_layer(n, k)
            tiles, cutlass = table_tokens(n, k)
            tokens = {1, *tiles[:1], *cutlass} if TUNED else {1, 8, 64}
            for m in sorted(tokens):
                with self.subTest(shape=name, m=m):
                    x = make_input(m, k, seed=0)
                    method.apply(layer, x)  # compiles the kernels before capture
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        y = method.apply(layer, x)
                    for seed in (1, 2):
                        x.copy_(make_input(m, k, seed=seed))
                        y.fill_(float("nan"))
                        graph.replay()
                        torch.cuda.synchronize()
                        self.assertTrue(torch.equal(y, method.apply(layer, x)))
                        assert_matches(self, y, *dequantized_reference(layer, x, n))
                    tile_captures += m in tiles
        if TUNED:
            self.assertGreater(tile_captures, 0)

    def test_second_post_load_pass_changes_nothing_and_updates_are_refused(self):
        # end_weight_update reruns post-load processing; the FP8 weights must survive it,
        # and the weight-update and weight-export guards must refuse these projections.
        _, layer, _ = make_layer(*SHAPES["input_tp8"])
        weight, scale = layer.weight.clone(), layer.weight_scale.clone()
        updater = SimpleNamespace(get_model=lambda: layer, device="cuda")
        WeightUpdater.end_weight_update(updater, run_post_load=False)
        self.assertTrue(torch.equal(layer.weight, weight))
        self.assertTrue(torch.equal(layer.weight_scale, scale))
        error = _unsupported_derived_weight_cache_error(layer)
        self.assertIsNotNone(error)
        self.assertIn("SGLANG_OPT_GLM53_KDA_PTPC_MODULES", error)
        exporter = WeightExporter(
            tp_rank=0,
            tp_size=1,
            gpu_id=0,
            get_model_path=lambda: "",
            get_model=lambda: layer,
        )
        with self.assertRaisesRegex(RuntimeError, "SGLANG_OPT_GLM53_KDA_PTPC_MODULES"):
            exporter._check_weight_export_supported()

    def test_bf16_weight_memory_is_released_after_quantization(self):
        # TP8's padding copies the BF16 weight;
        # without the method's own pool, the FP8 weight can keep the freed copy reserved.
        n, k = SHAPES["input_tp8"]
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        before = torch.cuda.memory_reserved()
        method = glm5_next._KdaFp8Config(is_checkpoint_fp8_serialized=False)
        method = method.get_quant_method(None, "")
        layer = torch.nn.Module()
        with torch.device("cuda"):
            method.create_weights(layer, k, [n], k, n, torch.bfloat16)
        layer.weight.data.normal_()
        layer.input_scale = None
        method.process_weights_after_loading(layer)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        # What stays is the FP8 weight (1 byte per element) and its scales.
        # Allocator rounding adds a few MiB; keeping the BF16 copy would add 26 MiB.
        self.assertLess(
            torch.cuda.memory_reserved() - before, padded(n) * k + 6 * 2**20
        )


if __name__ == "__main__":
    unittest.main()
