"""Online FP8 MoE on XPU (``--quantization fp8`` on an unquantized checkpoint) quantizes each
of w13 / w2 while it loads, one layer at a time
(``XPUOnlineFp8MoEMethod`` in ``hardware_backend/xpu/quantization/fp8_online_moe.py``).
The result must be bitwise identical to the path every other platform takes
(``Fp8MoEMethod``): load the experts in the checkpoint dtype and quantize them after
loading."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.hardware_backend.xpu.quantization.fp8_online_moe import (
    XPUOnlineFp8MoEMethod,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
from sglang.srt.utils import is_xpu
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=15, suite="extra-test-1-gpu-xpu")

DEV = "xpu"
HIDDEN = 256
INTERMEDIATE = 384
NUM_EXPERTS = 4
SHARD_DIM = {"w1": 0, "w3": 0, "w2": 1}
FP8_MODULE = "sglang.srt.layers.quantization.fp8"


class _StubFusedMoE(torch.nn.Module):
    """Only the surface create_weights and the loader's narrowing helpers touch. The
    narrowing helpers are the real ones."""

    _load_model_weight_or_group_weight_scale = (
        FusedMoE._load_model_weight_or_group_weight_scale
    )
    _load_w13 = FusedMoE._load_w13
    _load_w2 = FusedMoE._load_w2

    quant_config = None
    use_padded_loading = False
    use_presharded_weights = False

    def __init__(self, tp_size: int, tp_rank: int):
        super().__init__()
        self.moe_tp_size = tp_size
        self.moe_tp_rank = tp_rank
        self.num_local_experts = NUM_EXPERTS
        self.moe_runner_config = SimpleNamespace(is_gated=True)

    def weight_loader(self, param, loaded_weight, weight_name, shard_id, expert_id):
        """FusedMoE.weight_loader for local experts: TP narrowing into the expert."""
        self._load_model_weight_or_group_weight_scale(
            shard_dim=SHARD_DIM[shard_id],
            expert_data=param.data[expert_id],
            shard_id=shard_id,
            loaded_weight=loaded_weight,
        )


def _checkpoint():
    """Per-expert gate / up / down in BF16, with a different magnitude per expert so a
    scale stored into the wrong expert is visible."""
    gen = torch.Generator().manual_seed(0)
    experts = []
    for e in range(NUM_EXPERTS):
        mag = 0.01 * (e + 1)

        def rand(*shape):
            return (torch.randn(*shape, generator=gen) * mag).to(torch.bfloat16)

        experts.append(
            {
                "w1": rand(INTERMEDIATE, HIDDEN),
                "w3": rand(INTERMEDIATE, HIDDEN),
                "w2": rand(HIDDEN, INTERMEDIATE),
            }
        )
    return experts


def _make(tp_size, tp_rank, xpu=True):
    """A layer as create_weights leaves it: ``XPUOnlineFp8MoEMethod``, or with
    ``xpu=False`` the ``Fp8MoEMethod`` every other platform uses."""
    cls = XPUOnlineFp8MoEMethod if xpu else Fp8MoEMethod
    method = cls(Fp8Config(is_checkpoint_fp8_serialized=False))
    layer = _StubFusedMoE(tp_size, tp_rank)
    layer.quant_method = method
    with torch.device(DEV):
        method.create_weights(
            layer,
            num_experts=NUM_EXPERTS,
            hidden_size=HIDDEN,
            intermediate_size_per_partition=INTERMEDIATE // tp_size,
            params_dtype=torch.bfloat16,
            weight_loader=layer.weight_loader,
        )
    return method, layer


def _load(layer, order, ckpt):
    """What a model's load_weights does: look the parameters up once, then call each
    one's weight_loader."""
    params = dict(layer.named_parameters())
    for e, shard_id in order:
        name = "w2_weight" if shard_id == "w2" else "w13_weight"
        param = params[name]
        param.weight_loader(param, ckpt[e][shard_id], name, shard_id, e)


def _process(method, layer):
    method.process_weights_after_loading(layer)


# Qwen3.5 / Qwen3.6 fused checkpoints: every expert's gate, then every expert's up,
# then every down. Per-expert checkpoints: gate, up, down of one expert at a time.
ORDERS = {
    "fused": [(e, s) for s in ("w1", "w3", "w2") for e in range(NUM_EXPERTS)],
    "per_expert": [(e, s) for e in range(NUM_EXPERTS) for s in ("w3", "w2", "w1")],
}


class _BackendMocked(CustomTestCase):
    def setUp(self):
        backend = mock.patch(
            f"{FP8_MODULE}.get_moe_runner_backend",
            return_value=SimpleNamespace(
                is_cutlass=lambda: False,
                is_flashinfer_trtllm=lambda: False,
                is_flashinfer_trtllm_routed=lambda: False,
                is_hpc_ops=lambda: False,
            ),
        )
        backend.start()
        self.addCleanup(backend.stop)


@unittest.skipUnless(is_xpu(), "needs an XPU device")
class TestXpuOnlineFp8MoE(_BackendMocked):
    def assertSameWeights(self, layer, ref):
        for attr in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
            got, want = getattr(layer, attr), getattr(ref, attr)
            self.assertEqual(got.shape, want.shape, attr)
            self.assertEqual(got.dtype, want.dtype, attr)
            if got.dtype == torch.float32:
                self.assertTrue(torch.equal(got, want), attr)
            else:
                self.assertTrue(
                    torch.equal(got.view(torch.uint8), want.view(torch.uint8)), attr
                )

    def test_bitwise_identical_to_quantizing_after_loading(self):
        ckpt = _checkpoint()
        for tp_size, tp_rank in ((1, 0), (2, 0), (2, 1)):
            for order_name, order in ORDERS.items():
                with self.subTest(tp_size=tp_size, tp_rank=tp_rank, order=order_name):
                    ref_method, ref = _make(tp_size, tp_rank, xpu=False)
                    _load(ref, order, ckpt)
                    self.assertEqual(ref.w13_weight.dtype, torch.bfloat16)
                    _process(ref_method, ref)

                    method, layer = _make(tp_size, tp_rank)
                    self.assertTrue(layer.w13_weight.is_meta)
                    _load(layer, order, ckpt)
                    # Quantized by the loader, before process_weights_after_loading.
                    self.assertEqual(layer.w13_weight.dtype, ref.w13_weight.dtype)
                    self.assertEqual(layer.w2_weight.dtype, ref.w2_weight.dtype)
                    _process(method, layer)

                    self.assertSameWeights(layer, ref)
                    # Distinct per-expert scales, so a swapped expert would fail above.
                    self.assertEqual(
                        layer.w2_weight_scale.unique().numel(), NUM_EXPERTS
                    )

    def test_reload_quantizes_again(self):
        ckpt = _checkpoint()
        method, layer = _make(1, 0)
        _load(layer, ORDERS["per_expert"], ckpt)
        _process(method, layer)
        first = layer.w13_weight.clone()
        # A weight update looks the (now FP8) parameters up again and reloads them.
        _load(layer, ORDERS["fused"], ckpt)
        _process(method, layer)
        self.assertTrue(
            torch.equal(layer.w13_weight.view(torch.uint8), first.view(torch.uint8))
        )

    def test_incomplete_weight_is_quantized_after_loading(self):
        ckpt = _checkpoint()
        order = [o for o in ORDERS["per_expert"] if o != (2, "w3")]
        method, layer = _make(1, 0)
        _load(layer, order, ckpt)
        self.assertEqual(layer.w13_weight.dtype, torch.bfloat16)
        _process(method, layer)
        self.assertNotEqual(layer.w13_weight.dtype, torch.bfloat16)
        self.assertEqual(tuple(layer.w13_weight_scale.shape), (NUM_EXPERTS,))

    def test_never_loaded_weight_is_materialized(self):
        # --load-format dummy never calls the weight loader.
        method, layer = _make(1, 0)
        _process(method, layer)
        for name in ("w13_weight", "w2_weight"):
            self.assertFalse(getattr(layer, name).is_meta, name)
        self.assertTrue(torch.isfinite(layer.w13_weight_scale).all())


def _quant_method(config, xpu):
    layer = FusedMoE.__new__(FusedMoE)
    torch.nn.Module.__init__(layer)
    with mock.patch(f"{FP8_MODULE}._is_xpu", xpu):
        return config.get_quant_method(layer, prefix="model.layers.0.mlp.experts")


@unittest.skipUnless(is_xpu(), "needs an XPU device")
class TestQuantMethodSelection(_BackendMocked):
    def test_xpu_online_checkpoint(self):
        method = _quant_method(Fp8Config(is_checkpoint_fp8_serialized=False), True)
        self.assertIs(type(method), XPUOnlineFp8MoEMethod)

    def test_fp8_checkpoint_and_other_platforms_unchanged(self):
        for serialized, xpu in ((True, True), (False, False), (True, False)):
            with self.subTest(serialized=serialized, xpu=xpu):
                config = Fp8Config(is_checkpoint_fp8_serialized=serialized)
                self.assertIs(type(_quant_method(config, xpu)), Fp8MoEMethod)

    def test_fp8_moe_method_loads_in_checkpoint_dtype(self):
        _, layer = _make(1, 0, xpu=False)
        self.assertFalse(getattr(layer, "xpu_online_fp8_moe", False))
        self.assertFalse(layer.w13_weight.is_meta)
        self.assertEqual(layer.w13_weight.dtype, torch.bfloat16)
        self.assertEqual(layer.w13_weight.weight_loader, layer.weight_loader)


if __name__ == "__main__":
    unittest.main()
