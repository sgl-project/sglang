"""Unit tests for the load-time shared-expert MXFP4 conversion — CPU-only, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.test.test_utils import CustomTestCase

HIDDEN = 64
INTERMEDIATE = 32
NUM_LOCAL_ROUTED = 2
SHARED_SLOT = NUM_LOCAL_ROUTED


def _fake_packed(rows: int, cols: int, offset: int) -> torch.Tensor:
    """Position-encoding bytes, never zero, so a misplaced copy is visible."""
    flat = torch.arange(rows * cols, dtype=torch.int64) + offset
    return (flat % 251 + 1).to(torch.uint8).reshape(rows, cols)


class _StubScheme:
    """Stands in for QuarkW4A4MXFp4MoE without aiter or a GPU.

    Returns bytes that encode their own position, and uses a different offset
    for weights and scales, so swapping the two buffers cannot go unnoticed.
    """

    def __init__(self, online: bool = True):
        self.quantize_shared_expert_online = online
        self.seen = []

    def quantize_shared_expert(self, loaded_weight):
        self.seen.append(tuple(loaded_weight.shape))
        rows, cols = loaded_weight.shape
        return (
            _fake_packed(rows, cols // 2, offset=0),
            _fake_packed(rows, cols // 32, offset=1000),
        )


class _StubFusedMoE:
    """Only the surface the loader hook and its narrowing helpers touch.

    Borrowing the real methods rather than reimplementing them is the point:
    the test exercises the same placement logic the routed experts use.
    """

    _maybe_load_bf16_shared_expert_as_fp4 = (
        FusedMoE._maybe_load_bf16_shared_expert_as_fp4
    )
    _load_model_weight_or_group_weight_scale = (
        FusedMoE._load_model_weight_or_group_weight_scale
    )
    _load_w13 = FusedMoE._load_w13
    _load_w2 = FusedMoE._load_w2

    use_padded_loading = False
    use_presharded_weights = False
    use_triton_kernels = False
    quant_config = None
    quant_method = None
    moe_tp_size = 1

    def __init__(self, online: bool = True, has_fused_shared: bool = True):
        self._has_fused_shared = has_fused_shared
        self._num_local_routed = NUM_LOCAL_ROUTED
        self.moe_runner_config = SimpleNamespace(is_gated=True)
        self.scheme = _StubScheme(online)

        slots = NUM_LOCAL_ROUTED + 1
        self.w13_weight = torch.nn.Parameter(
            torch.zeros(slots, 2 * INTERMEDIATE, HIDDEN // 2, dtype=torch.uint8),
            requires_grad=False,
        )
        self.w13_weight_scale = torch.nn.Parameter(
            torch.zeros(slots, 2 * INTERMEDIATE, HIDDEN // 32, dtype=torch.uint8),
            requires_grad=False,
        )
        self.w2_weight = torch.nn.Parameter(
            torch.zeros(slots, HIDDEN, INTERMEDIATE // 2, dtype=torch.uint8),
            requires_grad=False,
        )
        self.w2_weight_scale = torch.nn.Parameter(
            torch.zeros(slots, HIDDEN, INTERMEDIATE // 32, dtype=torch.uint8),
            requires_grad=False,
        )

    def load(self, shard_id, expert_id=SHARED_SLOT, dtype=torch.bfloat16, param=None):
        if shard_id == "w2":
            loaded = torch.ones(HIDDEN, INTERMEDIATE, dtype=dtype)
            default_param, shard_dim = self.w2_weight, 1
        else:
            loaded = torch.ones(INTERMEDIATE, HIDDEN, dtype=dtype)
            default_param, shard_dim = self.w13_weight, 0
        return self._maybe_load_bf16_shared_expert_as_fp4(
            param=default_param if param is None else param,
            loaded_weight=loaded,
            shard_id=shard_id,
            expert_id=expert_id,
            shard_dim=shard_dim,
            tp_rank=0,
        )


class TestSharedExpertOnlineMxfp4Loader(CustomTestCase):
    """Pins where the converted shared expert lands.

    The gate tests cover whether the conversion happens; this covers whether it
    is written to the right place. Getting that wrong is the silent failure the
    feature exists to prevent: a wrong scale buffer or shard dim serves garbage
    rather than raising.
    """

    # ---- the conversion lands in the fused slot ----------------------------

    def test_gate_and_up_land_in_their_own_halves(self):
        layer = _StubFusedMoE()
        self.assertTrue(layer.load("w1"))
        self.assertTrue(layer.load("w3"))

        expected = _fake_packed(INTERMEDIATE, HIDDEN // 2, offset=0)
        weight = layer.w13_weight.data[SHARED_SLOT]
        torch.testing.assert_close(weight[:INTERMEDIATE], expected)
        torch.testing.assert_close(weight[INTERMEDIATE:], expected)

        expected_scale = _fake_packed(INTERMEDIATE, HIDDEN // 32, offset=1000)
        scale = layer.w13_weight_scale.data[SHARED_SLOT]
        torch.testing.assert_close(scale[:INTERMEDIATE], expected_scale)
        torch.testing.assert_close(scale[INTERMEDIATE:], expected_scale)

    def test_gate_alone_leaves_the_up_half_untouched(self):
        # w1 must not spill into w3's half of the fused buffer.
        layer = _StubFusedMoE()
        self.assertTrue(layer.load("w1"))
        self.assertTrue((layer.w13_weight.data[SHARED_SLOT, INTERMEDIATE:] == 0).all())
        self.assertTrue(
            (layer.w13_weight_scale.data[SHARED_SLOT, INTERMEDIATE:] == 0).all()
        )

    def test_down_projection_lands_in_w2(self):
        layer = _StubFusedMoE()
        self.assertTrue(layer.load("w2"))
        torch.testing.assert_close(
            layer.w2_weight.data[SHARED_SLOT],
            _fake_packed(HIDDEN, INTERMEDIATE // 2, offset=0),
        )
        torch.testing.assert_close(
            layer.w2_weight_scale.data[SHARED_SLOT],
            _fake_packed(HIDDEN, INTERMEDIATE // 32, offset=1000),
        )

    def test_weights_and_scales_do_not_swap_buffers(self):
        # The two buffers carry different byte patterns, so a scale written
        # into the weight buffer (or vice versa) shows up here.
        layer = _StubFusedMoE()
        layer.load("w1")
        layer.load("w2")
        for buf, offset in (
            (layer.w13_weight.data[SHARED_SLOT, :INTERMEDIATE], 0),
            (layer.w13_weight_scale.data[SHARED_SLOT, :INTERMEDIATE], 1000),
            (layer.w2_weight.data[SHARED_SLOT], 0),
            (layer.w2_weight_scale.data[SHARED_SLOT], 1000),
        ):
            rows, cols = buf.shape
            torch.testing.assert_close(buf, _fake_packed(rows, cols, offset))

    def test_routed_slots_are_left_alone(self):
        layer = _StubFusedMoE()
        for shard_id in ("w1", "w3", "w2"):
            layer.load(shard_id)
        for slot in range(NUM_LOCAL_ROUTED):
            self.assertTrue((layer.w13_weight.data[slot] == 0).all())
            self.assertTrue((layer.w13_weight_scale.data[slot] == 0).all())
            self.assertTrue((layer.w2_weight.data[slot] == 0).all())
            self.assertTrue((layer.w2_weight_scale.data[slot] == 0).all())

    # ---- and everything else passes straight through -----------------------

    def test_routed_expert_is_not_intercepted(self):
        layer = _StubFusedMoE()
        self.assertFalse(layer.load("w1", expert_id=NUM_LOCAL_ROUTED - 1))
        self.assertEqual(layer.scheme.seen, [])

    def test_already_quantized_weight_is_not_intercepted(self):
        # A packed uint8 tensor is a routed expert's serialized MXFP4, which
        # must reach the normal loader unconverted.
        layer = _StubFusedMoE()
        self.assertFalse(layer.load("w1", dtype=torch.uint8))
        self.assertEqual(layer.scheme.seen, [])

    def test_opt_out_scheme_is_not_intercepted(self):
        layer = _StubFusedMoE(online=False)
        self.assertFalse(layer.load("w1"))
        self.assertEqual(layer.scheme.seen, [])

    def test_unfused_layer_is_not_intercepted(self):
        layer = _StubFusedMoE(has_fused_shared=False)
        self.assertFalse(layer.load("w1"))
        self.assertEqual(layer.scheme.seen, [])

    def test_a_scale_param_is_not_mistaken_for_the_weight(self):
        # Only the weight parameter triggers the conversion; the scheme
        # produces both buffers from it in one go.
        layer = _StubFusedMoE()
        self.assertFalse(layer.load("w1", param=layer.w13_weight_scale))
        self.assertEqual(layer.scheme.seen, [])

    def test_the_whole_tensor_is_quantized_before_sharding(self):
        layer = _StubFusedMoE()
        layer.load("w1")
        layer.load("w2")
        self.assertEqual(
            layer.scheme.seen, [(INTERMEDIATE, HIDDEN), (HIDDEN, INTERMEDIATE)]
        )


if __name__ == "__main__":
    unittest.main()
