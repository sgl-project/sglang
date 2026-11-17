"""ModelOptNvFp4FusedMoEMethod with --moe-runner-backend nvfp4_w4a16_sm90 on a synthetic NVFP4 MoE layer: create
weights, load, repack, run through the MoE runner, and reload in place, against an FP32 reference with exact NVFP4
dequantization."""

import math
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptNvFp4FusedMoEMethod,
    _check_nvfp4_sm90_runner_config,
)
from sglang.srt.runtime_context import get_context, get_exec, get_flags
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

E, H, I, TOPK = 16, 1024, 384, 4
E2M1 = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def _dequant(w, sf):
    tab = torch.tensor(E2M1, device=w.device)
    v = torch.stack((tab[(w & 15).long()], tab[(w >> 4).long()]), dim=-1).flatten(-2)
    return v * sf.float().repeat_interleave(16, dim=-1)


def _fill(layer, seed):
    """Random checkpoint values written into the registered parameters, as the weight loader would."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    for name in ("w13", "w2"):
        w = getattr(layer, name + "_weight")
        w.data.copy_(
            torch.randint(
                0, 256, w.shape, dtype=torch.uint8, generator=g, device="cuda"
            )
        )
        sf = getattr(layer, name + "_weight_scale")
        sf.data.copy_(
            torch.exp2(6.5 + torch.rand(sf.shape, generator=g, device="cuda")).to(
                sf.dtype
            )
        )
        k = w.shape[2] * 2
        gs = getattr(layer, name + "_weight_scale_2")
        gs.data.copy_(
            torch.exp(0.5 * torch.randn(gs.shape, generator=g, device="cuda"))
            / math.sqrt(8.6 * 1.3e4 * k)
        )
    return {n: p.data.clone() for n, p in layer.named_parameters()}


def _reference(ck, x, ids, w, scale):
    out = torch.zeros(x.shape, dtype=torch.float32, device="cuda")
    for e in range(E):
        t, k = (ids == e).nonzero(as_tuple=True)
        if t.numel() == 0:
            continue
        h = x.float()[t] @ _dequant(ck["w13_weight"][e], ck["w13_weight_scale"][e]).T
        gs = ck["w13_weight_scale_2"][e]
        a = torch.nn.functional.silu(h[:, :I] * gs[0]) * (h[:, I:] * gs[1])
        y = (
            a
            @ (
                _dequant(ck["w2_weight"][e], ck["w2_weight_scale"][e])
                * ck["w2_weight_scale_2"][e]
            ).T
        )
        out.index_add_(0, t, y * w[t, k].unsqueeze(1))
    return out * scale


class TestRunnerConfigRefusals(CustomTestCase):
    """The op computes plain SiLU(gate) * up with router weights applied after the down projection. Any other expert
    math must be refused, not run without it: GLM-5.3-Flash and DeepSeek-V4.1-Flash set a SwiGLU clamp of 10, and the
    runner used to drop it with no error."""

    def test_plain_gated_silu_passes(self):
        _check_nvfp4_sm90_runner_config(MoeRunnerConfig())

    def test_refuses_each_unsupported_variant(self):
        for field, value in (
            ("swiglu_limit", 10.0),
            ("gemm1_clamp_limit", 10.0),
            ("gemm1_alpha", 1.702),
            ("gemm1_beta", 1.0),
            ("apply_router_weight_on_input", True),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    _check_nvfp4_sm90_runner_config(MoeRunnerConfig(**{field: value}))

    def test_refuses_other_activations(self):
        with self.assertRaisesRegex(ValueError, "gated SiLU"):
            _check_nvfp4_sm90_runner_config(MoeRunnerConfig(activation="gelu"))


@unittest.skipIf(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    "needs an sm_90 GPU",
)
class TestModelOptNvfp4Sm90Runner(CustomTestCase):
    def setUp(self):
        self.server_args = get_context().override_server_args(model_path="dummy")
        self.server_args.install()
        self.backend = get_flags().moe.override(
            runner_backend=MoeRunnerBackend.NVFP4_W4A16_SM90
        )
        self.backend.__enter__()

    def tearDown(self):
        self.backend.__exit__(None, None, None)
        self.server_args.restore()

    def test_load_keeps_one_copy_of_the_experts(self):
        """Loading must not leave the checkpoint-layout copy behind in the allocator: a model's KV pool is sized from
        the memory left after loading, and on DeepSeek-R1 EP8 a repack into separately allocated buffers left 10.8GB
        per GPU unusable (162,615 KV tokens against 493,518 on Marlin). Fails on a repack into new buffers."""
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        start = torch.cuda.memory_reserved()
        layers = [self._layer(32, 2048, 768) for _ in range(4)]
        e, h, i = 32, 2048, 768
        per_layer = (
            e * (2 * i // 256) * (h // 64) * 9216 + e * (h // 256) * (i // 64) * 9216
        )
        for k, (method, layer) in enumerate(layers):
            _fill(layer, seed=k)
            method.process_weights_after_loading(layer)
        torch.cuda.synchronize()
        grown = torch.cuda.memory_reserved() - start
        # The four layers' experts, one layer's transient copy and the allocator's rounding.
        self.assertLess(grown, 4 * per_layer + 1.5 * per_layer + (64 << 20))

    def test_rejects_eplb(self):
        with get_exec().moe.override(enable_eplb=True):
            with self.assertRaisesRegex(ValueError, "enable-eplb"):
                self._layer()

    def _layer(self, e=E, h=H, i=I):
        method = ModelOptNvFp4FusedMoEMethod(
            SimpleNamespace(
                is_checkpoint_nvfp4_serialized=True,
                group_size=16,
                get_name=lambda: "modelopt_fp4",
            )
        )
        config = MoeRunnerConfig(
            num_experts=e,
            num_local_experts=e,
            hidden_size=h,
            intermediate_size_per_partition=i,
            top_k=TOPK,
            activation="silu",
            is_gated=True,
            routed_scaling_factor=2.5,
        )
        layer = torch.nn.Module()
        layer.num_experts = layer.num_local_experts = e
        layer.moe_runner_config = config
        layer._mega_moe_nvfp4 = False  # FusedMoE.__init__ sets it
        with torch.device("cuda"):
            method.create_weights(layer, e, h, i, torch.bfloat16, weight_loader=None)
        method.create_moe_runner(layer, config)
        return method, layer

    def _inputs(self, T, seed):
        g = torch.Generator(device="cuda").manual_seed(seed)
        x = torch.randn((T, H), generator=g, device="cuda").bfloat16()
        ids = torch.argsort(torch.rand((T, E), generator=g, device="cuda"), dim=1)[
            :, :TOPK
        ].int()
        ids[0, 0] = -1  # another rank's route
        w = torch.rand((T, TOPK), generator=g, device="cuda")
        return x, ids, w

    def _apply(self, method, layer, x, ids, w):
        topk = StandardTopKOutput(
            topk_weights=w,
            topk_ids=ids,
            router_logits=torch.zeros((x.shape[0], E), device="cuda"),
        )
        return method.apply(
            layer,
            StandardDispatchOutput(
                hidden_states=x, hidden_states_scale=None, topk_output=topk
            ),
        ).hidden_states

    def _assert_close(self, out, ck, x, ids, w):
        ref = _reference(ck, x, ids, w, 2.5)
        rel = float((out.float() - ref).norm() / ref.norm())
        self.assertLess(rel, 0.01)

    def _check(self, method, layer, ck, T, seed):
        x, ids, w = self._inputs(T, seed)
        self._assert_close(self._apply(method, layer, x, ids, w), ck, x, ids, w)

    def test_load_run_and_reload(self):
        """Fails on: a repack that drops the up global scale or the routed scaling factor, a method that keeps the
        checkpoint copy (memory doubles), a reload that skips the restore (the loader writes into empty tensors) or
        the repack (the graph keeps the old weights), a restore that allocates a second copy of the experts, or a
        reload that repacks into new buffers (the captured graph keeps reading the old ones)."""
        method, layer = self._layer()
        ck = _fill(layer, seed=1)
        ck["w13_weight_scale_2"][:, 1] *= 3.0  # gate and up global scales differ
        layer.w13_weight_scale_2.data.copy_(ck["w13_weight_scale_2"])
        method.process_weights_after_loading(layer)
        self.assertEqual(layer.w13_weight.numel() + layer.w2_weight_scale.numel(), 0)
        experts = layer.nvfp4_sm90_experts
        for T in (1, 37, 1024):
            self._check(method, layer, ck, T, seed=T)

        # A CUDA graph captured on the first weights must serve the reloaded ones.
        x, ids, w = self._inputs(64, seed=4)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self._apply(method, layer, x, ids, w)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_out = self._apply(method, layer, x, ids, w)
        graph.replay()
        self._assert_close(graph_out, ck, x, ids, w)

        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        method.restore_weights_before_loading(layer)
        self.assertEqual(torch.cuda.memory_allocated(), allocated)
        for name in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"):
            param = getattr(layer, name)
            self.assertEqual(param.shape, ck[name].shape)
            self.assertEqual(param.dtype, ck[name].dtype)
            buf = experts.w13 if name.startswith("w13") else experts.w2
            self.assertEqual(
                param.untyped_storage().data_ptr(), buf.untyped_storage().data_ptr()
            )
        ck2 = _fill(layer, seed=2)
        method.process_weights_after_loading(layer)
        self.assertIs(layer.nvfp4_sm90_experts, experts)
        self.assertEqual(layer.w13_weight.numel() + layer.w2_weight_scale.numel(), 0)
        self._check(method, layer, ck2, 300, seed=3)
        graph.replay()
        self._assert_close(graph_out, ck2, x, ids, w)


class TestRoutedScalingPlacement(CustomTestCase):
    def test_runner_scales_the_combined_output(self):
        """The runner applies the routed scaling factor to its output, so top-k must not fold it in as well."""
        from sglang.srt.layers.moe.fused_moe_triton.layer import (
            _fuses_routed_scaling_factor_in_topk,
        )

        method = object.__new__(ModelOptNvFp4FusedMoEMethod)
        self.assertFalse(
            _fuses_routed_scaling_factor_in_topk(
                method, MoeRunnerBackend.NVFP4_W4A16_SM90
            )
        )
        self.assertTrue(
            _fuses_routed_scaling_factor_in_topk(method, MoeRunnerBackend.TRITON)
        )


if __name__ == "__main__":
    unittest.main()
