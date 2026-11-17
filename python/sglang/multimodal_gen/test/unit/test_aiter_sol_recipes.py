# SPDX-License-Identifier: Apache-2.0
"""The aiter_sol backend: its recipe table, config resolution, and dispatch.

Routing, pooling and dispatch must all read the same block tile, and the recipe
table decides which of aiter's two Sol-Attn entry points a row can use. Both are
silently wrong when broken -- a mask cut for one geometry indexes real keys at
another, and the raw entry point drops a block-scaled K's scale -- so they are
pinned here rather than left to a generation to discover.

Runs without a GPU and without aiter: the aiter surface is replaced by stand-ins
that record what they are handed. The two cases that need real aiter say so.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime import server_args as server_args_module
from sglang.multimodal_gen.runtime.layers.attention.backends import (
    aiter_sol,
    aiter_sol_dispatch,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.server_args import set_global_server_args
from sglang.test.test_utils import CustomTestCase

HEADS = 2
HEAD_DIM = 128
# 16 blocks of the stand-in 128-token KV tile, which is the point below which
# the backend warns that routing has too few blocks to say anything.
SEQUENCE = 2048
SOFTMAX_SCALE = HEAD_DIM**-0.5
BLOCK_TILE = (128, 128)

_ROUTING_KEYS = (
    "kv_block_indices",
    "lut_start",
    "lut_count",
    "mean_k",
    "mean_v",
    "block_bitmap",
    "mean_k_scale",
    "mean_v_scale",
)


def _passthrough(tensor, *args):
    """Stand in for a quantizer: the operand and a descale placeholder."""
    return tensor, tensor


class _SolTestCase(CustomTestCase):
    """The recipe table and the dispatch logic, over a recorded aiter surface."""

    def setUp(self):
        super().setUp()
        self.sol_calls = []
        self.packed_calls = []
        self.prepare_calls = []
        self.fused_calls = []
        self.ragged = True

        self.addCleanup(set_global_server_args, server_args_module._global_server_args)
        set_global_server_args(SimpleNamespace(attention_backend_config=None))

        formats = SimpleNamespace(
            BF16="BF16", INT8="INT8", MXFP6="MXFP6", MXFP4="MXFP4"
        )
        scale_modes = SimpleNamespace(
            NONE="NONE",
            F32_PER_TENSOR="F32_PER_TENSOR",
            F32_PER_CHANNEL="F32_PER_CHANNEL",
            E8M0_PER_1X32="E8M0_PER_1X32",
        )
        packs = SimpleNamespace(DEFAULT="DEFAULT", V_FOR_FP6_P="V_FOR_FP6_P")

        self._patch("AITER_SOL_AVAILABLE", True)
        self._patch("FUSED_ROUTING_AVAILABLE", True)
        self._patch("_AiterAttentionFormat", formats)
        self._patch("_AiterAttentionScaleMode", scale_modes)
        self._patch("_AiterAttentionPack", packs)
        self._patch("_aiter_native_fp8_format", lambda: "FP8_E4M3")
        self._patch("_aiter_sol_mode", 2)
        self._patch("_aiter_operands", lambda *values: tuple(values))
        self._patch("_aiter_block_tile", lambda operands, mode: BLOCK_TILE)
        self._patch(
            "_aiter_kv_tile_for_q_tile",
            lambda q_tile, operands, mode: (
                BLOCK_TILE[1] if q_tile == BLOCK_TILE[0] else 0
            ),
        )
        self._patch("_aiter_block_tiles", lambda operands, mode: (BLOCK_TILE,))
        self._patch("_aiter_ragged_kv", lambda operands, mode, block_tile: self.ragged)
        self._patch("_aiter_q_multiplier", lambda softmax_scale: softmax_scale)
        self._patch("_aiter_mxfp4_k_view", lambda k, descale: k)
        self._patch("_aiter_mxfp4_v_view", lambda v, descale, seqlen: v)
        self._patch("_aiter_mxfp6_k_view", lambda k, descale, *shape: (k, descale))
        self._patch("_aiter_mha_v4_sol", self._record_sol)
        self._patch("_aiter_mha_v4_packed", self._record_packed)
        self._patch("_aiter_sol_prepare", self._record_prepare)
        self._patch("sol_attn_routing_fused", self._record_fused)
        self._patch(
            "_QUANTIZERS",
            {name: _passthrough for name in aiter_sol_dispatch._QUANTIZERS}
            | {row.quantize_q: _passthrough for row in aiter_sol_dispatch._RECIPE_ROWS}
            | {row.quantize_k: _passthrough for row in aiter_sol_dispatch._RECIPE_ROWS}
            | {row.quantize_v: _passthrough for row in aiter_sol_dispatch._RECIPE_ROWS},
        )

    def _key_seen(self):
        """The key operand the dispatched call received, whichever path it took."""
        (call,) = self.sol_calls or self.packed_calls
        return call["operands"][1]

    def _beta_seen(self):
        """The beta the dispatched call received, whichever path it took."""
        (call,) = self.sol_calls or self.fused_calls or self.prepare_calls
        return call["beta"]

    def _patch(self, name, value):
        patcher = patch.object(aiter_sol_dispatch, name, value)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _record_sol(self, q, k, v, q_format, k_format, v_format, **kwargs):
        self.sol_calls.append(
            {"operands": (q, k, v), "formats": (q_format, k_format, v_format), **kwargs}
        )
        return torch.zeros_like(q)

    def _record_packed(self, q, k, v, *descales_and_formats, **kwargs):
        self.packed_calls.append(
            {"operands": (q, k, v), "positional": descales_and_formats, **kwargs}
        )
        return torch.zeros_like(q)

    def _record_fused(self, q, k, v, beta, recipe, block_tile=None):
        self.fused_calls.append(
            {"q": q, "k": k, "v": v, "beta": beta, "recipe": recipe}
        )
        return self._stand_in_routing(q[0], k[0], *BLOCK_TILE)

    def _record_prepare(self, q, k, v, beta, block_m, block_n, **kwargs):
        self.prepare_calls.append(
            {"q": q, "k": k, "v": v, "beta": beta, "tile": (block_m, block_n), **kwargs}
        )
        return self._stand_in_routing(q, k, block_m, block_n)

    @staticmethod
    def _stand_in_routing(q, k, block_m, block_n):
        """The routing dictionary shape, with every block selected."""
        blocks = -(-k.shape[1] // block_n)
        q_tiles = -(-q.shape[1] // block_m)
        mask = torch.ones(q.shape[0], q.shape[2], q_tiles, blocks, dtype=torch.bool)
        routing = {key: None for key in _ROUTING_KEYS}
        routing["block_attn_mask"] = mask
        return routing

    def _operands(self, seqlen=SEQUENCE, dtype=torch.bfloat16):
        return tuple(
            torch.zeros(1, seqlen, HEADS, HEAD_DIM, dtype=dtype) for _ in range(3)
        )

    def _build(self, config=None, **overrides):
        """Build the impl the way a run does, selecting through server args."""
        aiter_sol.get_global_server_args().attention_backend_config = config or {}
        kwargs = {
            "num_heads": HEADS,
            "head_size": HEAD_DIM,
            "softmax_scale": SOFTMAX_SCALE,
            **overrides,
        }
        with patch.object(aiter_sol, "check_sol_attn_build"):
            return aiter_sol.AITERSolImpl(**kwargs)


class TestSolRecipeTable(_SolTestCase):
    """Which aiter entry point each row may take, and how V is stored."""

    def test_only_an_addressable_block_scaled_k_leaves_the_raw_path(self):
        # mha_v4_sol routes internally and takes no K scale, so a row whose K
        # carries an E8M0 image that routing must read cannot use it. A packed
        # K is sub-byte and tile-ordered, so its routing reads the BF16 source
        # instead and the raw path stays available.
        packed_only = {
            name
            for name, row in aiter_sol_dispatch._RECIPES.items()
            if not row.routes_through_raw
        }
        self.assertEqual(packed_only, {"mxfp8"})

    def test_mx_v_rows_are_stored_in_the_fp6_p_order(self):
        # The pack is what tells the kernel how to read V; getting it from the
        # V format rather than the packer name would put f8f6 (MXFP6 V via the
        # FP6-P packer) in the wrong order.
        fp6_p = {
            name
            for name, row in aiter_sol_dispatch._RECIPES.items()
            if row.v_pack == "V_FOR_FP6_P"
        }
        self.assertEqual(fp6_p, {"f8f6", "f6f6", "f6f4", "mxfp4"})

    def test_every_row_names_a_quantizer_that_exists(self):
        # A typo in a table entry would otherwise surface as a KeyError at the
        # first generation that selects the recipe.
        for row in aiter_sol_dispatch._RECIPE_ROWS:
            with self.subTest(recipe=row.name):
                for key in (row.quantize_q, row.quantize_k, row.quantize_v):
                    self.assertIn(key, aiter_sol_dispatch._QUANTIZERS)


class TestSolBlockTileAgreement(_SolTestCase):
    """Routing, pooling and dispatch must be handed the same geometry."""

    def test_routing_and_launch_use_one_block_tile(self):
        # A LUT holds block numbers, not their size, so a mask routed at one
        # tile and read at another attends to the wrong keys with every index
        # still in range. Nothing downstream can detect that.
        self._build({"recipe": "mxfp8"}).forward(*self._operands(), None)
        (prepare,) = self.prepare_calls
        (packed,) = self.packed_calls
        self.assertEqual(prepare["tile"], BLOCK_TILE)
        self.assertEqual(packed["block_tile"], BLOCK_TILE)

    def test_an_explicit_block_size_reaches_both(self):
        tile = (64, 64)
        self._patch("_aiter_block_tile", lambda operands, mode: (1, 1))
        impl = self._build({"recipe": "mxfp8", "block_size": "64x64"})
        impl.forward(*self._operands(), None)
        (prepare,) = self.prepare_calls
        (packed,) = self.packed_calls
        self.assertEqual(prepare["tile"], tile)
        self.assertEqual(packed["block_tile"], tile)


class TestSolDispatchPath(_SolTestCase):
    def test_a_raw_capable_recipe_skips_the_python_routing(self):
        # mxfp4 routes through raw and packs its operands, so the fused
        # kernel cannot pool it.
        self._build({"recipe": "mxfp4"}).forward(*self._operands(), None)
        self.assertEqual(len(self.sol_calls), 1)
        self.assertEqual(self.prepare_calls, [])
        self.assertEqual(self.packed_calls, [])

    def test_a_served_recipe_routes_through_the_fused_kernel_by_default(self):
        # The default decides what every run without an explicit key gets.
        self._build({"recipe": "fp8"}).forward(*self._operands(), None)
        self.assertEqual(len(self.fused_calls), 1)
        self.assertEqual(self.sol_calls, [])

    def test_mxfp8_is_quantized_and_routed_here(self):
        self._build({"recipe": "mxfp8"}).forward(*self._operands(), None)
        self.assertEqual(self.sol_calls, [])
        self.assertEqual(len(self.prepare_calls), 1)
        self.assertEqual(len(self.packed_calls), 1)

    def test_asking_for_the_head_cost_forces_the_packed_path(self):
        # The raw entry point keeps its routing internal, so a caller that
        # needs the per-head block count cannot take it even on a row that
        # otherwise would.
        query, key, value = self._operands()
        _, head_cost = aiter_sol_dispatch.sol_attn_bshd(
            query, key, value, recipe="fp8", return_head_cost=True
        )
        self.assertEqual(self.sol_calls, [])
        self.assertEqual(len(self.packed_calls), 1)
        self.assertEqual(head_cost.shape, (HEADS,))

    def test_fused_routing_leaves_the_raw_entry_point(self):
        # Routing built outside aiter has to be handed in, which only the
        # packed entry point accepts.
        self._build({"recipe": "fp8"}).forward(*self._operands(), None)
        self.assertEqual(self.sol_calls, [])
        self.assertEqual(self.prepare_calls, [])
        self.assertEqual(len(self.fused_calls), 1)
        self.assertEqual(len(self.packed_calls), 1)

    def test_fused_routing_falls_back_on_a_block_scaled_row(self):
        # The fused kernel routes on the stored codes, which a block-granular
        # scale makes meaningless, so such a row keeps aiter's own routing.
        self._build({"recipe": "mxfp8"}).forward(*self._operands(), None)
        self.assertEqual(self.fused_calls, [])
        self.assertEqual(len(self.prepare_calls), 1)

    def test_a_block_scaled_operand_hands_its_scale_to_the_routing(self):
        # Block-scaled codes are not proportional to their values, so routing
        # and pooling need the scale; a per-tensor descale survives pooling and
        # must be left out or the operand pools twice.
        self._build({"recipe": "mxfp8"}).forward(*self._operands(), None)
        (prepare,) = self.prepare_calls
        self.assertIsNotNone(prepare["k_scale"])
        self.assertIsNotNone(prepare["q_scale"])
        self.assertIsNone(prepare["v_scale"])


class TestSolRoutingMode(_SolTestCase):
    """Which routing path a recipe takes."""

    def test_the_fused_kernel_serves_the_rows_with_a_poolable_scale(self):
        # A packed operand is sub-byte and tile-ordered and a block-granular
        # scale varies along the pooled axis, so neither can be pooled from
        # the stored codes.
        served = {
            name
            for name, row in aiter_sol_dispatch._RECIPES.items()
            if aiter_sol_dispatch._fused_routing_serves(row)
        }
        self.assertEqual(served, {"bf16", "bf16fp8", "fp8", "i8fp8"})

    def test_an_aiter_without_the_helpers_keeps_its_own_routing(self):
        # The fused path reads aiter internals. When they are not there the
        # backend still runs, on aiter's routing.
        self._patch("FUSED_ROUTING_AVAILABLE", False)
        self.assertEqual(
            aiter_sol_dispatch.sol_attn_routing_mode("bf16"),
            "eager",
        )
        self._build({"recipe": "bf16"}).forward(*self._operands(), None)
        self.assertEqual(self.fused_calls, [])
        self.assertEqual(len(self.sol_calls), 1)

    def test_the_mode_follows_the_recipe(self):
        # The mode is what the log line reports, so it has to name the path the
        # call will really take.
        self.assertEqual(aiter_sol_dispatch.sol_attn_routing_mode("bf16"), "fused")
        self.assertEqual(aiter_sol_dispatch.sol_attn_routing_mode("mxfp8"), "eager")


class TestSolKeyPadding(_SolTestCase):
    def test_key_seqlen_is_dropped_before_the_tile_pad(self):
        # Trimming after padding would stack the caller's pad and the tile pad,
        # leaving a whole block of zero keys that still draws softmax weight.
        self.ragged = False
        query, key, value = self._operands()
        aiter_sol_dispatch.sol_attn_bshd(
            query, key, value, recipe="fp8", key_seqlen=SEQUENCE - 300
        )
        self.assertEqual(self._key_seen().shape[1], SEQUENCE - 256)

    def test_a_ragged_row_is_not_padded_at_all(self):
        self.ragged = True
        query, key, value = self._operands(seqlen=SEQUENCE - 1)
        aiter_sol_dispatch.sol_attn_bshd(query, key, value, recipe="fp8")
        self.assertEqual(self._key_seen().shape[1], SEQUENCE - 1)


class TestSolBetaSchedule(_SolTestCase):
    def test_a_ramp_hits_both_endpoints(self):
        schedule = aiter_sol._BetaSchedule.from_spec("1.0:0.0")
        self.assertAlmostEqual(schedule.beta_at(0, 40), 1.0)
        self.assertAlmostEqual(schedule.beta_at(39, 40), 0.0)
        self.assertAlmostEqual(schedule.beta_at(20, 41), 0.5)

    def test_a_ramp_clamps_past_the_last_step(self):
        # A pipeline that runs one extra forward must not extrapolate the ramp
        # past its endpoint and route against a threshold nobody asked for.
        schedule = aiter_sol._BetaSchedule.from_spec("1.0:0.0")
        self.assertAlmostEqual(schedule.beta_at(99, 40), 0.0)

    def test_a_single_step_takes_the_first_endpoint(self):
        schedule = aiter_sol._BetaSchedule.from_spec("1.0:0.0")
        self.assertAlmostEqual(schedule.beta_at(0, 1), 1.0)

    def test_a_per_step_list_indexes_and_clamps(self):
        schedule = aiter_sol._BetaSchedule.from_spec("0.9,0.5,0.1")
        self.assertAlmostEqual(schedule.beta_at(1, 3), 0.5)
        self.assertAlmostEqual(schedule.beta_at(7, 3), 0.1)

    def test_one_value_is_a_flat_schedule(self):
        schedule = aiter_sol._BetaSchedule.from_spec("0.25")
        self.assertAlmostEqual(schedule.beta_at(0, 40), 0.25)
        self.assertAlmostEqual(schedule.beta_at(39, 40), 0.25)

    def test_an_empty_schedule_is_rejected(self):
        with self.assertRaises(ValueError):
            aiter_sol._BetaSchedule.from_spec(" , ")

    def test_the_scheduled_beta_reaches_the_kernel(self):
        impl = self._build({"recipe": "fp8", "beta_schedule": "1.0:0.0"})
        batch = SimpleNamespace(sampling_params=SimpleNamespace(num_inference_steps=5))
        with set_forward_context(
            current_timestep=4, attn_metadata=None, forward_batch=batch
        ):
            impl.forward(*self._operands(), None)
        self.assertAlmostEqual(self._beta_seen(), 0.0)

    def test_the_constant_beta_stands_outside_a_denoising_loop(self):
        # Stages set a forward context without a request, and direct callers
        # set none at all; neither may crash the backend.
        impl = self._build({"recipe": "fp8", "beta_schedule": "1.0:0.0", "beta": 0.25})
        impl.forward(*self._operands(), None)
        self.assertAlmostEqual(self._beta_seen(), 0.25)


class TestSolConfig(_SolTestCase):
    def test_the_defaults_are_the_documented_ones(self):
        impl = self._build()
        self.assertEqual(impl.recipe, aiter_sol._DEFAULT_RECIPE)
        self.assertEqual(impl.beta, aiter_sol._DEFAULT_BETA)
        self.assertIsNone(impl.beta_schedule)
        self.assertIsNone(impl.block_tile)

    def test_an_unknown_recipe_names_the_supported_ones(self):
        # The message is the only place a user learns the valid names.
        with self.assertRaises(ValueError) as raised:
            self._build({"recipe": "f4f4"})
        for name in aiter_sol_dispatch.SOL_ATTN_RECIPES:
            self.assertIn(name, str(raised.exception))

    def test_a_recipe_name_is_case_insensitive(self):
        self.assertEqual(self._build({"recipe": "MxFp8"}).recipe, "mxfp8")

    def test_block_size_accepts_both_separators(self):
        self.assertEqual(self._build({"block_size": "64x64"}).block_tile, (64, 64))
        self.assertEqual(self._build({"block_size": "64,64"}).block_tile, (64, 64))

    def test_a_malformed_block_size_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "q_tile"):
            self._build({"block_size": "128"})


class TestSolRejectedCalls(_SolTestCase):
    def test_rejects_a_head_dim_other_than_128(self):
        with self.assertRaisesRegex(
            aiter_sol_dispatch.SolAttnUnsupported, "head_dim == 128"
        ):
            self._build(head_size=64)

    def test_rejects_causal_masking(self):
        # The pooled correction assumes every skipped block is fully
        # attendable, so there is no causal row to fall back to.
        with self.assertRaisesRegex(aiter_sol_dispatch.SolAttnUnsupported, "causal"):
            self._build(causal=True)

    def test_rejects_a_dtype_other_than_bf16(self):
        impl = self._build()
        with self.assertRaises(aiter_sol_dispatch.SolAttnUnsupported):
            impl.forward(*self._operands(dtype=torch.float16), None)

    def test_rejects_a_pack_of_several_sequences(self):
        # Sol-Attn has no key-padding mask and its pooled blocks would straddle
        # any boundary other than the end of the sequence, so a pack of two
        # real segments must be refused rather than attended across the seam.
        impl = self._build()
        query, key, value = (
            tensor.squeeze(0) for tensor in self._operands(seqlen=SEQUENCE)
        )
        with self.assertRaisesRegex(
            aiter_sol_dispatch.SolAttnUnsupported, "trailing pad"
        ):
            impl.forward_varlen(
                query,
                key,
                value,
                cu_seqlens=None,
                max_seqlen=SEQUENCE // 2,
                cu_seqlens_host=(0, SEQUENCE // 2, SEQUENCE - 128, SEQUENCE),
            )

    def test_rejects_a_pack_whose_keys_have_their_own_bounds(self):
        # Asymmetric key ranges are a second sequence on the key side, which
        # the single-segment path cannot express.
        impl = self._build()
        query, key, value = (
            tensor.squeeze(0) for tensor in self._operands(seqlen=SEQUENCE)
        )
        with self.assertRaisesRegex(
            aiter_sol_dispatch.SolAttnUnsupported, "trailing pad"
        ):
            impl.forward_varlen(
                query,
                key,
                value,
                cu_seqlens=None,
                max_seqlen=SEQUENCE // 2,
                cu_seqlens_host=(0, SEQUENCE // 2, SEQUENCE),
                cu_seqlens_k=torch.tensor([0, SEQUENCE // 2]),
            )


class TestSolTrailingPad(_SolTestCase):
    def test_a_trailing_pad_is_served_and_the_pad_rows_are_zeroed(self):
        # The live prefix is attended on its own and the pad rows are written
        # as zeros, so a caller that reads the whole buffer never sees the
        # junk an unmasked kernel would leave there.
        impl = self._build()
        query, key, value = (
            tensor.squeeze(0) for tensor in self._operands(seqlen=SEQUENCE)
        )
        used = SEQUENCE // 2
        output = impl.forward_varlen(
            query,
            key,
            value,
            cu_seqlens=None,
            max_seqlen=used,
            cu_seqlens_host=(0, used, SEQUENCE),
        )
        self.assertEqual(self._key_seen().shape[1], used)
        self.assertEqual(output.shape, (SEQUENCE, HEADS, HEAD_DIM))
        self.assertTrue(torch.all(output[used:] == 0))


class TestSolAgainstRealAiter(CustomTestCase):
    """The two facts only aiter can confirm."""

    @unittest.skipUnless(
        aiter_sol_dispatch.AITER_SOL_AVAILABLE,
        "requires SGLANG_USE_AITER=1 and an aiter that ships mha_v4_sol",
    )
    def test_every_recipe_resolves_against_aiter(self):
        # Formats, scale modes and packs are resolved by name, so a member
        # aiter renames or drops is an AttributeError here rather than at the
        # first generation that selects the recipe.
        for row in aiter_sol_dispatch._RECIPE_ROWS:
            with self.subTest(recipe=row.name):
                aiter_sol_dispatch._recipe_operands(row)

    @unittest.skipUnless(
        aiter_sol_dispatch.AITER_SOL_AVAILABLE and torch.cuda.is_available(),
        "requires a GPU and an aiter that ships mha_v4_sol",
    )
    def test_the_fused_routing_reproduces_aiter_routing(self):
        # The fused kernel replaces aiter's routing, so it has to select the
        # same blocks, not merely a similar number of them: one block of
        # difference sends a query tile to the pooled branch for keys it should
        # have read exactly.
        torch.manual_seed(0)
        batch, seqlen, heads, beta = 1, 4096, 2, 0.5
        # Clustered, because routing on noise selects every block and tests
        # nothing. Each operand follows a few shared centers.
        centers = torch.randn(8, HEAD_DIM, device="cuda", dtype=torch.bfloat16)

        def clustered():
            pick = torch.randint(
                0, centers.shape[0], (batch, seqlen, heads), device="cuda"
            )
            noise = torch.randn(
                batch, seqlen, heads, HEAD_DIM, device="cuda", dtype=torch.bfloat16
            )
            return centers[pick] + 0.25 * noise

        query, key, value = clustered(), clustered(), clustered()
        recipe = aiter_sol_dispatch._RECIPES["bf16"]
        operands = aiter_sol_dispatch._quantize(
            recipe, query, key, value, SOFTMAX_SCALE
        )
        reference = aiter_sol_dispatch.sol_attn_routing_for(*operands, beta, recipe)
        fused = aiter_sol_dispatch.sol_attn_routing_fused(*operands, beta, recipe)

        self.assertTrue(
            torch.equal(reference["block_attn_mask"], fused["block_attn_mask"])
        )
        self.assertTrue(torch.equal(reference["lut_count"], fused["lut_count"]))
        self.assertTrue(torch.equal(reference["block_bitmap"], fused["block_bitmap"]))
        # A routing that selects everything, or nothing, would pass the
        # comparisons above without exercising the threshold.
        density = fused["block_attn_mask"].float().mean().item()
        self.assertGreater(density, 0.0)
        self.assertLess(density, 1.0)

    @unittest.skipUnless(
        aiter_sol_dispatch.AITER_SOL_AVAILABLE,
        "requires SGLANG_USE_AITER=1 and an aiter that ships mha_v4_sol",
    )
    def test_every_recipe_names_a_bound_quantizer(self):
        for row in aiter_sol_dispatch._RECIPE_ROWS:
            with self.subTest(recipe=row.name):
                for key in (row.quantize_q, row.quantize_k, row.quantize_v):
                    self.assertIn(key, aiter_sol_dispatch._QUANTIZERS)


if __name__ == "__main__":
    unittest.main()
