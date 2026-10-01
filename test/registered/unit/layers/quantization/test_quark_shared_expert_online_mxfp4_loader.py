"""Unit tests for the load-time shared-expert MXFP4 conversion — CPU-only, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4_moe import (
    OCP_MX_BLOCK_SIZE as MX_BLOCK_SIZE,
)
from sglang.test.test_utils import CustomTestCase

# Both K dimensions and every per-rank intermediate below are multiples of the
# MX block size, which is what create_weights enforces at startup.

HIDDEN = 64
# Divisible by the widest TP size times the block size, so every rank's
# intermediate stays block-aligned: at TP8 each rank holds exactly one MX block
# of the down projection, which is the tightest shape the feature has to serve.
# Kept different from HIDDEN so a transposed shard cannot pass unnoticed.
INTERMEDIATE = 256
# amd/Qwen3.8-2.4T-A95B-Quark-MXFP4 only runs at TP8, so that is the size that
# matters; TP2 is kept alongside it to catch anything specific to a single
# divisor rather than to the sharding arithmetic.
TP_SIZES = (2, 8)
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

    def quantize_shared_expert(self, loaded_weight, device):
        # Moved as the real scheme does, so a caller that passes something
        # that is not a device fails here rather than silently.
        loaded_weight = loaded_weight.to(device)
        self.seen.append(tuple(loaded_weight.shape))
        rows, cols = loaded_weight.shape
        return (
            _fake_packed(rows, cols // 2, offset=0),
            _fake_packed(rows, cols // MX_BLOCK_SIZE, offset=1000),
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

    use_triton_kernels = False
    quant_config = None
    quant_method = None
    # Recorded by the scheme's create_weights on the real layer.
    _load_device = torch.device("cpu")

    def __init__(
        self,
        online: bool = True,
        has_fused_shared: bool = True,
        moe_tp_size: int = 1,
        intermediate_pad: int = 0,
        use_padded_loading: bool = False,
        use_presharded_weights: bool = False,
    ):
        self.moe_tp_size = moe_tp_size
        self.use_padded_loading = use_padded_loading
        self.use_presharded_weights = use_presharded_weights
        self._has_fused_shared = has_fused_shared
        self._num_local_routed = NUM_LOCAL_ROUTED
        self.moe_runner_config = SimpleNamespace(is_gated=True)
        self.scheme = _StubScheme(online)

        # What a rank's buffers actually hold: its slice of the intermediate
        # size, plus aiter's padding on top.
        self.intermediate_per_partition = INTERMEDIATE // moe_tp_size
        rows = self.intermediate_per_partition + intermediate_pad
        self.buffer_rows = rows

        slots = NUM_LOCAL_ROUTED + 1
        self.w13_weight = torch.nn.Parameter(
            torch.zeros(slots, 2 * rows, HIDDEN // 2, dtype=torch.uint8),
            requires_grad=False,
        )
        self.w13_weight_scale = torch.nn.Parameter(
            torch.zeros(slots, 2 * rows, HIDDEN // MX_BLOCK_SIZE, dtype=torch.uint8),
            requires_grad=False,
        )
        self.w2_weight = torch.nn.Parameter(
            torch.zeros(slots, HIDDEN, rows // 2, dtype=torch.uint8),
            requires_grad=False,
        )
        self.w2_weight_scale = torch.nn.Parameter(
            torch.zeros(slots, HIDDEN, rows // MX_BLOCK_SIZE, dtype=torch.uint8),
            requires_grad=False,
        )

    def load(
        self,
        shard_id,
        expert_id=SHARED_SLOT,
        dtype=torch.bfloat16,
        param=None,
        tp_rank=0,
    ):
        # A presharded checkpoint already holds only this rank's slice; an
        # ordinary one holds the whole intermediate size on every rank.
        loaded_intermediate = (
            self.intermediate_per_partition
            if self.use_presharded_weights
            else INTERMEDIATE
        )
        if shard_id == "w2":
            loaded = torch.ones(HIDDEN, loaded_intermediate, dtype=dtype)
            default_param, shard_dim = self.w2_weight, 1
        else:
            loaded = torch.ones(loaded_intermediate, HIDDEN, dtype=dtype)
            default_param, shard_dim = self.w13_weight, 0
        return self._maybe_load_bf16_shared_expert_as_fp4(
            param=default_param if param is None else param,
            loaded_weight=loaded,
            shard_id=shard_id,
            expert_id=expert_id,
            shard_dim=shard_dim,
            tp_rank=tp_rank,
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

        expected_scale = _fake_packed(
            INTERMEDIATE, HIDDEN // MX_BLOCK_SIZE, offset=1000
        )
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
            _fake_packed(HIDDEN, INTERMEDIATE // MX_BLOCK_SIZE, offset=1000),
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

    # ---- tensor parallelism: w2 is the projection whose K dimension splits --

    def test_down_projection_k_split_gives_each_rank_its_own_slice(self):
        # w2's K is the intermediate size, so TP cuts across the MX blocks.
        # Quantizing whole and sharding after is only equivalent to sharding
        # first if both cuts land on a block boundary, which is what this pins.
        full_weight = _fake_packed(HIDDEN, INTERMEDIATE // 2, offset=0)
        full_scale = _fake_packed(HIDDEN, INTERMEDIATE // MX_BLOCK_SIZE, offset=1000)
        for tp_size in TP_SIZES:
            for tp_rank in range(tp_size):
                layer = _StubFusedMoE(moe_tp_size=tp_size)
                self.assertTrue(layer.load("w2", tp_rank=tp_rank))

                weight_cols = layer.intermediate_per_partition // 2
                scale_cols = layer.intermediate_per_partition // MX_BLOCK_SIZE
                with self.subTest(tp_size=tp_size, tp_rank=tp_rank):
                    torch.testing.assert_close(
                        layer.w2_weight.data[SHARED_SLOT],
                        full_weight[
                            :, weight_cols * tp_rank : weight_cols * (tp_rank + 1)
                        ],
                    )
                    torch.testing.assert_close(
                        layer.w2_weight_scale.data[SHARED_SLOT],
                        full_scale[
                            :, scale_cols * tp_rank : scale_cols * (tp_rank + 1)
                        ],
                    )
                    # Both cuts describe the same boundary: a packed byte holds
                    # 2 FP4 values and a scale covers MX_BLOCK_SIZE of them. If
                    # they disagreed, a shard would carry another rank's scales.
                    self.assertEqual(weight_cols * 2, scale_cols * MX_BLOCK_SIZE)

    def test_the_ranks_together_reconstruct_the_whole_projection(self):
        # Per-rank correctness is not enough: the shards also have to tile the
        # tensor with no gap and no overlap. Concatenating them in rank order
        # must give back exactly what a single rank quantized.
        for tp_size in TP_SIZES:
            with self.subTest(tp_size=tp_size):
                down, down_scale, gate = [], [], []
                for tp_rank in range(tp_size):
                    layer = _StubFusedMoE(moe_tp_size=tp_size)
                    layer.load("w2", tp_rank=tp_rank)
                    layer.load("w1", tp_rank=tp_rank)
                    rows = layer.intermediate_per_partition
                    down.append(layer.w2_weight.data[SHARED_SLOT])
                    down_scale.append(layer.w2_weight_scale.data[SHARED_SLOT])
                    gate.append(layer.w13_weight.data[SHARED_SLOT, :rows])

                torch.testing.assert_close(
                    torch.cat(down, dim=1),
                    _fake_packed(HIDDEN, INTERMEDIATE // 2, offset=0),
                )
                torch.testing.assert_close(
                    torch.cat(down_scale, dim=1),
                    _fake_packed(HIDDEN, INTERMEDIATE // MX_BLOCK_SIZE, offset=1000),
                )
                torch.testing.assert_close(
                    torch.cat(gate, dim=0),
                    _fake_packed(INTERMEDIATE, HIDDEN // 2, offset=0),
                )

    def test_gate_and_up_take_their_own_intermediate_slice(self):
        # w1/w3 split the output dimension instead, so K (hidden) stays whole
        # and each rank takes a row band of the quantized tensor.
        full = _fake_packed(INTERMEDIATE, HIDDEN // 2, offset=0)
        for tp_size in TP_SIZES:
            for tp_rank in range(tp_size):
                layer = _StubFusedMoE(moe_tp_size=tp_size)
                self.assertTrue(layer.load("w1", tp_rank=tp_rank))
                self.assertTrue(layer.load("w3", tp_rank=tp_rank))

                rows = layer.intermediate_per_partition
                expected = full[rows * tp_rank : rows * (tp_rank + 1)]
                weight = layer.w13_weight.data[SHARED_SLOT]
                with self.subTest(tp_size=tp_size, tp_rank=tp_rank):
                    torch.testing.assert_close(weight[:rows], expected)
                    torch.testing.assert_close(weight[rows:], expected)

    def test_presharded_weights_are_not_narrowed_again(self):
        # The checkpoint already holds this rank's slice, so narrowing it a
        # second time would silently keep only part of it.
        for tp_size in TP_SIZES:
            layer = _StubFusedMoE(moe_tp_size=tp_size, use_presharded_weights=True)
            last_rank = tp_size - 1
            self.assertTrue(layer.load("w2", tp_rank=last_rank))
            self.assertTrue(layer.load("w1", tp_rank=last_rank))

            rows = layer.intermediate_per_partition
            with self.subTest(tp_size=tp_size):
                torch.testing.assert_close(
                    layer.w2_weight.data[SHARED_SLOT],
                    _fake_packed(HIDDEN, rows // 2, offset=0),
                )
                torch.testing.assert_close(
                    layer.w13_weight.data[SHARED_SLOT, :rows],
                    _fake_packed(rows, HIDDEN // 2, offset=0),
                )

    # ---- aiter's intermediate padding --------------------------------------

    def test_data_goes_in_the_leading_slice_and_padding_stays_zero(self):
        pad = MX_BLOCK_SIZE
        layer = _StubFusedMoE(intermediate_pad=pad)
        for shard_id in ("w1", "w3", "w2"):
            self.assertTrue(layer.load(shard_id))

        rows = layer.intermediate_per_partition
        weight = layer.w13_weight.data[SHARED_SLOT]
        expected = _fake_packed(INTERMEDIATE, HIDDEN // 2, offset=0)
        # gate occupies the first `rows` of its half, up the first `rows` of
        # the second half, and the pad rows after each stay untouched.
        torch.testing.assert_close(weight[:rows], expected)
        self.assertTrue((weight[rows : rows + pad] == 0).all())
        torch.testing.assert_close(weight[rows + pad : rows + pad + rows], expected)
        self.assertTrue((weight[rows + pad + rows :] == 0).all())

        w2 = layer.w2_weight.data[SHARED_SLOT]
        cols = rows // 2
        torch.testing.assert_close(
            w2[:, :cols], _fake_packed(HIDDEN, INTERMEDIATE // 2, offset=0)
        )
        self.assertTrue((w2[:, cols:] == 0).all())

    def test_padded_loading_agrees_with_the_plain_path(self):
        # use_padded_loading routes through narrow_padded_param_and_loaded_weight
        # instead of the leading-slice copy. On a padded buffer the two have to
        # produce the same bytes, including the zeroed tail.
        pad = MX_BLOCK_SIZE
        plain = _StubFusedMoE(intermediate_pad=pad, use_padded_loading=False)
        padded = _StubFusedMoE(intermediate_pad=pad, use_padded_loading=True)
        for shard_id in ("w1", "w3", "w2"):
            self.assertTrue(plain.load(shard_id))
            self.assertTrue(padded.load(shard_id))

        for name in (
            "w13_weight",
            "w13_weight_scale",
            "w2_weight",
            "w2_weight_scale",
        ):
            torch.testing.assert_close(
                getattr(plain, name).data,
                getattr(padded, name).data,
                msg=f"{name} differs between padded and plain loading",
            )

    def test_padding_and_tensor_parallelism_together(self):
        # Padding and sharding combined: each rank's slice goes in the leading
        # part of its own padded band, and the pad after it stays zero.
        #
        # Not the shape amd/Qwen3.8-2.4T-A95B-Quark-MXFP4 serves in. Its
        # moe_intermediate_size is 2048, so TP8 gives 256 per rank and a
        # w2_down_dim of 128, which is exactly AITER_PADDING_SIZE; aiter reports
        # is_padded=False and the measured configuration carries no padding at
        # all. Padding only appears at a TP size that leaves w2_down_dim short
        # of 128, e.g. TP16.
        #
        # The padded buffer is driven through the plain leading-slice copy here,
        # not through use_padded_loading. That combination is deliberate: when
        # aiter really does pad it sets weight_padded on w2_weight, which makes
        # FusedMoE.use_padded_loading true, and on that branch the checkpoint
        # offset is shard_size * tp_rank with shard_size already padded. That
        # mis-slices for tp_rank > 0 for the routed experts too, so it is
        # pre-existing and out of scope here; it is also why a padded path at
        # TP > 1 has no test. test_padded_loading_agrees_with_the_plain_path
        # covers use_padded_loading at TP1, where the offset is zero.
        pad = MX_BLOCK_SIZE
        full_gate = _fake_packed(INTERMEDIATE, HIDDEN // 2, offset=0)
        full_down = _fake_packed(HIDDEN, INTERMEDIATE // 2, offset=0)
        for tp_rank in (0, 7):
            layer = _StubFusedMoE(moe_tp_size=8, intermediate_pad=pad)
            self.assertTrue(layer.load("w1", tp_rank=tp_rank))
            self.assertTrue(layer.load("w2", tp_rank=tp_rank))

            rows = layer.intermediate_per_partition
            cols = rows // 2
            weight = layer.w13_weight.data[SHARED_SLOT]
            w2 = layer.w2_weight.data[SHARED_SLOT]
            with self.subTest(tp_rank=tp_rank):
                torch.testing.assert_close(
                    weight[:rows], full_gate[rows * tp_rank : rows * (tp_rank + 1)]
                )
                self.assertTrue((weight[rows : rows + pad] == 0).all())
                torch.testing.assert_close(
                    w2[:, :cols], full_down[:, cols * tp_rank : cols * (tp_rank + 1)]
                )
                self.assertTrue((w2[:, cols:] == 0).all())

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
        # Even at TP8 the scheme sees the full checkpoint tensor; the narrowing
        # happens afterwards, on the packed result. That is what makes a rank's
        # bytes match an offline-quantized checkpoint of the same shard.
        layer = _StubFusedMoE(moe_tp_size=8)
        layer.load("w1", tp_rank=7)
        layer.load("w2", tp_rank=7)
        self.assertEqual(
            layer.scheme.seen, [(INTERMEDIATE, HIDDEN), (HIDDEN, INTERMEDIATE)]
        )


if __name__ == "__main__":
    unittest.main()
