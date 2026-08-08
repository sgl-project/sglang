import logging
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.models.deepseek_v4 import (
    _COMPRESSOR_FUSABLE_LEAVES,
    _COMPRESSOR_SHARD_IDS,
    _COMPRESSOR_SHARD_ORDER,
    _WQKV_A_FUSABLE_LEAVES,
    _WQKV_A_SHARD_IDS,
    _WQKV_A_SHARD_ORDER,
    DeepseekV4ForCausalLM,
    _classify_fused_shard,
    _fused_module_prefixes,
    _block_scale_concat_is_exact,
    _unfusable_layer_leaves,
    _pop_fused_weight,
    _reject_unfusable_leaf,
    _summarize_unplaced_weights,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ATTN_PREFIX = "model.layers.3.self_attn"
COMPRESSOR_PREFIX = "model.layers.3.self_attn.compressor"
DENSE_LAYER = "model.layers.0.self_attn"
PACKED_LAYER = "model.layers.1.self_attn"


def run_load_weights(params, weights, fuse_wqa_wkv=True):
    """Drive the real DeepseekV4ForCausalLM.load_weights over a name stream.

    ``params`` stands in for ``named_parameters()``; everything load_weights
    needs beyond the weight-placement logic is stubbed out.
    """
    model = SimpleNamespace(
        config=SimpleNamespace(num_hidden_layers=8, n_routed_experts=1),
        quant_config=None,
        num_fused_shared_experts=0,
        pp_group=SimpleNamespace(is_first_rank=True, is_last_rank=True),
        model=SimpleNamespace(),
        named_parameters=lambda: list(params.items()),
        remap_weight_name_to_dpsk_hf_format=(
            DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format
        ),
        post_load_weights=lambda **kwargs: None,
        _prewarm_mhc_pre_kernels=lambda: None,
    )
    with patch.object(envs.SGLANG_OPT_FUSE_WQA_WKV, "get", return_value=fuse_wqa_wkv):
        DeepseekV4ForCausalLM.load_weights(model, list(weights))


def fuse_wqa_wkv_stream(names_and_weights):
    """Mimic the wq_a+wkv arm of DeepseekV4ForCausalLM.load_weights.

    Returns the fused parameters keyed by target parameter name, plus the names
    the arm did not claim (which load_weights hands to the generic path).
    """
    cache = {}
    fused = {}
    unclaimed = []
    for name, loaded_weight in names_and_weights:
        classified = _classify_fused_shard(name, _WQKV_A_SHARD_IDS)
        if classified is None:
            unclaimed.append(name)
            continue
        shard_id, parent, leaf = classified
        if leaf not in _WQKV_A_FUSABLE_LEAVES:
            _reject_unfusable_leaf(
                name,
                leaf,
                _WQKV_A_FUSABLE_LEAVES,
                "Set SGLANG_OPT_FUSE_WQA_WKV=0 to load this checkpoint with "
                "separate wq_a and wkv projections.",
            )
        param_name = f"{parent}.wqkv_a.{leaf}"
        fused_weight = _pop_fused_weight(
            cache, param_name, shard_id, loaded_weight, _WQKV_A_SHARD_ORDER
        )
        if fused_weight is not None:
            fused[param_name] = fused_weight
    assert not cache, cache.keys()
    return fused, unclaimed


class TestWqaWkvFusionClassification(unittest.TestCase):
    def test_dense_stream_fuses_on_dim0(self):
        wq_a = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4)
        wkv = torch.arange(2 * 4, dtype=torch.float32).reshape(2, 4) + 100
        wq_a_scale = torch.arange(2 * 1, dtype=torch.float32).reshape(2, 1)
        wkv_scale = torch.arange(1 * 1, dtype=torch.float32).reshape(1, 1) + 50

        fused, unclaimed = fuse_wqa_wkv_stream(
            [
                (f"{ATTN_PREFIX}.wq_a.weight", wq_a),
                (f"{ATTN_PREFIX}.wkv.weight", wkv),
                (f"{ATTN_PREFIX}.wq_a.weight_scale_inv", wq_a_scale),
                (f"{ATTN_PREFIX}.wkv.weight_scale_inv", wkv_scale),
            ]
        )

        self.assertEqual(unclaimed, [])
        self.assertEqual(
            sorted(fused),
            [f"{ATTN_PREFIX}.wqkv_a.weight", f"{ATTN_PREFIX}.wqkv_a.weight_scale_inv"],
        )
        torch.testing.assert_close(
            fused[f"{ATTN_PREFIX}.wqkv_a.weight"], torch.cat([wq_a, wkv], dim=0)
        )
        torch.testing.assert_close(
            fused[f"{ATTN_PREFIX}.wqkv_a.weight_scale_inv"],
            torch.cat([wq_a_scale, wkv_scale], dim=0),
        )

    def test_shard_order_is_q_then_kv_regardless_of_arrival_order(self):
        wq_a = torch.ones(8, 4)
        wkv = torch.zeros(2, 4)

        fused, _ = fuse_wqa_wkv_stream(
            [
                (f"{ATTN_PREFIX}.wkv.weight", wkv),
                (f"{ATTN_PREFIX}.wq_a.weight", wq_a),
            ]
        )

        torch.testing.assert_close(
            fused[f"{ATTN_PREFIX}.wqkv_a.weight"], torch.cat([wq_a, wkv], dim=0)
        )

    def test_packed_stream_is_refused_instead_of_dropped(self):
        # auto_round / gptq packing: the checkpoint carries .qweight, .qzeros
        # and .scales instead of a dense .weight.
        for leaf in ("qweight", "qzeros", "scales"):
            name = f"{ATTN_PREFIX}.wq_a.{leaf}"
            with self.subTest(leaf=leaf):
                with self.assertRaises(ValueError) as ctx:
                    fuse_wqa_wkv_stream([(name, torch.zeros(2, 2))])
                message = str(ctx.exception)
                self.assertIn(name, message)
                self.assertIn(leaf, message)
                self.assertIn("SGLANG_OPT_FUSE_WQA_WKV=0", message)

    def test_packed_wkv_is_refused(self):
        name = f"{ATTN_PREFIX}.wkv.qweight"
        with self.assertRaises(ValueError) as ctx:
            fuse_wqa_wkv_stream([(name, torch.zeros(2, 2))])
        self.assertIn(name, str(ctx.exception))

    def test_unrelated_names_are_left_to_the_generic_path(self):
        names = [
            f"{ATTN_PREFIX}.wq_b.weight",
            f"{ATTN_PREFIX}.wo_a.weight",
            "model.embed_tokens.weight",
            "lm_head.weight",
        ]
        fused, unclaimed = fuse_wqa_wkv_stream([(n, torch.zeros(1)) for n in names])
        self.assertEqual(fused, {})
        self.assertEqual(unclaimed, names)


class TestCompressorFusionClassification(unittest.TestCase):
    def test_dense_stream_fuses_kv_then_wgate(self):
        cache = {}
        kv = torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3)
        wgate = torch.arange(2 * 3, dtype=torch.float32).reshape(2, 3) + 100

        results = []
        for name, loaded_weight in [
            (f"{COMPRESSOR_PREFIX}.wgate.weight", wgate),
            (f"{COMPRESSOR_PREFIX}.wkv.weight", kv),
        ]:
            shard_id, key, leaf = _classify_fused_shard(name, _COMPRESSOR_SHARD_IDS)
            self.assertEqual(key, COMPRESSOR_PREFIX)
            self.assertIn(leaf, _COMPRESSOR_FUSABLE_LEAVES)
            results.append(
                _pop_fused_weight(
                    cache,
                    f"{key}.wkv_gate.{leaf}",
                    shard_id,
                    loaded_weight,
                    _COMPRESSOR_SHARD_ORDER,
                )
            )

        self.assertIsNone(results[0])
        torch.testing.assert_close(results[1], torch.cat([kv, wgate], dim=0))
        self.assertEqual(cache, {})

    def test_unrecognized_leaf_raises_named_error(self):
        name = f"{COMPRESSOR_PREFIX}.wkv.qweight"
        shard_id, key, leaf = _classify_fused_shard(name, _COMPRESSOR_SHARD_IDS)
        self.assertEqual(shard_id, "kv")
        self.assertNotIn(leaf, _COMPRESSOR_FUSABLE_LEAVES)
        with self.assertRaises(ValueError) as ctx:
            _reject_unfusable_leaf(
                name,
                leaf,
                _COMPRESSOR_FUSABLE_LEAVES,
                "The compressor wkv+wgate fusion supports dense checkpoints only.",
            )
        self.assertIn(name, str(ctx.exception))
        self.assertIn("qweight", str(ctx.exception))

    def test_unrecognized_segment_is_not_classified(self):
        # ".compressor.w" also prefix-matches names the fusion cannot handle.
        self.assertIsNone(
            _classify_fused_shard(
                f"{COMPRESSOR_PREFIX}.wsomething.weight", _COMPRESSOR_SHARD_IDS
            )
        )


class TestLoadWeightsFusedProjection(unittest.TestCase):
    """End-to-end over the real load_weights branch chain."""

    def test_dense_stream_places_the_fused_parameter(self):
        wq_a = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4)
        wkv = torch.arange(2 * 4, dtype=torch.float32).reshape(2, 4) + 100
        params = {
            f"{ATTN_PREFIX}.wqkv_a.weight": torch.nn.Parameter(torch.zeros(10, 4))
        }

        run_load_weights(
            params,
            [
                (f"{ATTN_PREFIX}.wq_a.weight", wq_a),
                (f"{ATTN_PREFIX}.wkv.weight", wkv),
            ],
        )

        torch.testing.assert_close(
            params[f"{ATTN_PREFIX}.wqkv_a.weight"].data,
            torch.cat([wq_a, wkv], dim=0),
        )

    def test_packed_stream_raises_instead_of_skipping(self):
        # The reporter's auto_round checkpoint (packing_format
        # auto_round:auto_gptq) ships these names. They used to fall through to
        # the generic path, log "not found in params_dict" and be dropped.
        params = {
            f"{ATTN_PREFIX}.wqkv_a.qweight": torch.nn.Parameter(torch.zeros(4, 10))
        }
        weights = [
            (f"{ATTN_PREFIX}.wq_a.qweight", torch.zeros(4, 8)),
            (f"{ATTN_PREFIX}.wkv.qweight", torch.zeros(4, 2)),
        ]

        with self.assertRaises(ValueError) as ctx:
            run_load_weights(params, weights)

        message = str(ctx.exception)
        self.assertIn(f"{ATTN_PREFIX}.wq_a.qweight", message)
        self.assertIn("SGLANG_OPT_FUSE_WQA_WKV=0", message)

    def test_compressor_dense_stream_places_the_fused_parameter(self):
        kv = torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3)
        wgate = torch.arange(2 * 3, dtype=torch.float32).reshape(2, 3) + 100
        params = {
            f"{COMPRESSOR_PREFIX}.wkv_gate.weight": torch.nn.Parameter(
                torch.zeros(6, 3)
            )
        }

        run_load_weights(
            params,
            [
                (f"{COMPRESSOR_PREFIX}.wkv.weight", kv),
                (f"{COMPRESSOR_PREFIX}.wgate.weight", wgate),
            ],
        )

        torch.testing.assert_close(
            params[f"{COMPRESSOR_PREFIX}.wkv_gate.weight"].data,
            torch.cat([kv, wgate], dim=0),
        )

    def test_compressor_packed_stream_raises_named_error(self):
        params = {
            f"{COMPRESSOR_PREFIX}.wkv_gate.qweight": torch.nn.Parameter(
                torch.zeros(3, 6)
            )
        }
        weights = [
            (f"{COMPRESSOR_PREFIX}.wkv.qweight", torch.zeros(3, 4)),
            (f"{COMPRESSOR_PREFIX}.wgate.qweight", torch.zeros(3, 2)),
        ]

        with self.assertRaises(ValueError) as ctx:
            run_load_weights(params, weights)

        self.assertIn(f"{COMPRESSOR_PREFIX}.wkv.qweight", str(ctx.exception))

    def test_packed_stream_loads_unfused_when_fusion_is_off(self):
        qweight = torch.ones(4, 8)
        params = {
            f"{ATTN_PREFIX}.wq_a.qweight": torch.nn.Parameter(torch.zeros(4, 8)),
        }

        run_load_weights(
            params,
            [(f"{ATTN_PREFIX}.wq_a.qweight", qweight)],
            fuse_wqa_wkv=False,
        )

        torch.testing.assert_close(params[f"{ATTN_PREFIX}.wq_a.qweight"].data, qweight)


def layer_with_params(*names):
    """A stand-in for a built linear carrying exactly ``names``.

    Used for the schemes whose real quantization method cannot be constructed
    without a device (compressed tensors probes the device capability before it
    picks a scheme). The parameter names are the ones those methods register.
    """
    layer = torch.nn.Module()
    for name in names:
        layer.register_parameter(name, torch.nn.Parameter(torch.zeros(2, 2)))
    return layer


class TestBlockScaleConcatPrecondition(unittest.TestCase):
    """The fusion concatenates block scales on dim 0 like the weights.

    That reproduces the fused parameter's scale layout only while every shard
    but the last is a whole number of blocks.
    """

    def test_no_block_size_means_nothing_to_check(self):
        self.assertTrue(_block_scale_concat_is_exact((1024, 512), None))
        self.assertTrue(_block_scale_concat_is_exact((1000, 500), []))

    def test_published_v4_geometry_is_exact(self):
        # q_lora_rank 1024, head_dim 448 + 64, against a 128-row block.
        self.assertTrue(_block_scale_concat_is_exact((1024, 512), [128, 128]))

    def test_two_partial_shards_that_share_a_block_are_not_exact(self):
        # 1000 % 128 = 104, 520 % 128 = 8; both partial and 112 <= 128, so the
        # two shards pad to a scale row each where the fused layout shares one.
        self.assertFalse(_block_scale_concat_is_exact((1000, 520), [128, 128]))

    def test_a_partial_shard_beside_a_whole_one_is_still_exact(self):
        self.assertTrue(_block_scale_concat_is_exact((1000, 512), [128, 128]))
        self.assertTrue(_block_scale_concat_is_exact((1024, 500), [128, 128]))

    def test_remainders_that_overflow_a_block_are_still_exact(self):
        # 104 + 88 > 128: the seam already costs a whole row either way.
        self.assertTrue(_block_scale_concat_is_exact((1000, 600), [128, 128]))

    def test_the_mismatch_is_a_row_count_not_a_shape_error(self):
        """Why this is a precondition and not an assertion downstream.

        The concatenated scales have MORE rows than the fused parameter
        declares, and the fusion never compares the two, so a misaligned
        geometry shifts every scale past the seam instead of raising.
        """
        block, shards = 128, (1000, 520)
        per_shard = sum(-(-n // block) for n in shards)
        fused = -(-sum(shards) // block)
        self.assertEqual((per_shard, fused), (13, 12))


class TestPackedLayerFallback(unittest.TestCase):
    def test_unquantized_replicated_linear_is_fusable(self):
        layer = ReplicatedLinear(8, 4, bias=False, quant_config=None)
        self.assertEqual(_unfusable_layer_leaves(layer, _WQKV_A_FUSABLE_LEAVES), ())

    def test_packed_layer_is_not_fusable(self):
        layer = layer_with_params("qweight", "qzeros", "scales")
        self.assertEqual(
            _unfusable_layer_leaves(layer, _WQKV_A_FUSABLE_LEAVES),
            ("qweight", "qzeros", "scales"),
        )

    def test_per_tensor_fp8_layer_is_not_fusable(self):
        """`weight` is present, so a `weight`-only test would leave the fusion on.

        `Fp8LinearMethod` registers `weight_scale` and `input_scale` beside the
        weight; the fusion has no shard for either, so the layer has to build
        the separate projections.
        """
        layer = layer_with_params("weight", "weight_scale", "input_scale")
        self.assertEqual(
            _unfusable_layer_leaves(layer, _WQKV_A_FUSABLE_LEAVES),
            ("input_scale", "weight_scale"),
        )

    def test_compressed_tensors_packed_layer_is_not_fusable(self):
        layer = layer_with_params(
            "weight_packed", "weight_scale", "weight_shape", "weight_zero_point"
        )
        self.assertEqual(
            _unfusable_layer_leaves(layer, _WQKV_A_FUSABLE_LEAVES),
            ("weight_packed", "weight_scale", "weight_shape", "weight_zero_point"),
        )

    def test_block_scaled_fp8_layer_stays_fusable(self):
        """The path this fusion exists for must not be disabled by the widening."""
        layer = layer_with_params("weight", "weight_scale_inv")
        self.assertEqual(_unfusable_layer_leaves(layer, _WQKV_A_FUSABLE_LEAVES), ())

    def test_the_weight_only_predicate_accepted_per_tensor_fp8(self):
        """The predicate this replaced, kept executable.

        `hasattr(layer, "weight")` answers True for per-tensor fp8, which is why
        that checkpoint reached `_reject_unfusable_leaf` at load time instead of
        building the separate projections.
        """
        layer = layer_with_params("weight", "weight_scale", "input_scale")
        self.assertTrue(hasattr(layer, "weight"))
        self.assertTrue(_unfusable_layer_leaves(layer, _WQKV_A_FUSABLE_LEAVES))

    def test_per_tensor_fp8_checkpoint_loads_through_the_separate_projections(self):
        """End to end: every scale reaches its own parameter.

        Before the build-side check covered it, the layer kept `wqkv_a`, the
        `weight` shards fused, and the first scale tensor raised -- a checkpoint
        that loads unfused became one that does not load.
        """
        leaves = ("weight", "weight_scale", "input_scale")
        params = {
            f"{ATTN_PREFIX}.{proj}.{leaf}": torch.nn.Parameter(torch.zeros(4, 8))
            for proj in ("wq_a", "wkv")
            for leaf in leaves
        }
        weights = [
            (f"{ATTN_PREFIX}.{proj}.{leaf}", torch.full((4, 8), float(i)))
            for i, (proj, leaf) in enumerate(
                (projection, name) for projection in ("wq_a", "wkv") for name in leaves
            )
        ]

        run_load_weights(params, weights, fuse_wqa_wkv=True)

        for name, expected in weights:
            torch.testing.assert_close(params[name].data, expected)

    def test_loader_skips_fusion_when_the_model_has_no_wqkv_a(self):
        # MqaAttentionBase built the unfused pair because the quantization
        # method produced a packed layer, so load_weights must not try to fuse
        # even though SGLANG_OPT_FUSE_WQA_WKV is on.
        qweight = torch.ones(4, 8)
        params = {
            f"{ATTN_PREFIX}.wq_a.qweight": torch.nn.Parameter(torch.zeros(4, 8)),
        }

        run_load_weights(
            params,
            [(f"{ATTN_PREFIX}.wq_a.qweight", qweight)],
            fuse_wqa_wkv=True,
        )

        torch.testing.assert_close(params[f"{ATTN_PREFIX}.wq_a.qweight"].data, qweight)


class TestMixedPrecisionCheckpoint(unittest.TestCase):
    """Per-layer quantization: some layers dense and fused, others packed.

    A checkpoint that excludes individual attention blocks from quantization
    (`quantization_config.extra_config` with `bits: 16` entries) builds a model
    in which layer 0 keeps `wqkv_a` and layer 1 does not. The fusion decision
    has to follow the layer under load, not the model as a whole.
    """

    def mixed_params(self):
        return {
            # Layer 0 stayed dense, so MqaAttentionBase kept the fusion.
            f"{DENSE_LAYER}.wqkv_a.weight": torch.nn.Parameter(torch.zeros(10, 4)),
            # Layer 1 is packed, so MqaAttentionBase dropped wqkv_a and built
            # the separate projections.
            f"{PACKED_LAYER}.wq_a.qweight": torch.nn.Parameter(torch.zeros(4, 8)),
            f"{PACKED_LAYER}.wq_a.scales": torch.nn.Parameter(torch.zeros(1, 8)),
            f"{PACKED_LAYER}.wkv.qweight": torch.nn.Parameter(torch.zeros(4, 2)),
            f"{PACKED_LAYER}.wkv.scales": torch.nn.Parameter(torch.zeros(1, 2)),
        }

    def test_prefixes_report_the_layers_that_kept_the_fusion(self):
        prefixes = _fused_module_prefixes(self.mixed_params(), "wqkv_a")

        self.assertEqual(prefixes, {DENSE_LAYER})

    def test_prefixes_cover_the_mtp_naming_space(self):
        # load_weights rewrites the MTP layer prefix to "model.decoder", which
        # carries no layer index.
        params = {
            "model.decoder.self_attn.wqkv_a.weight": torch.nn.Parameter(
                torch.zeros(2, 2)
            )
        }

        self.assertEqual(
            _fused_module_prefixes(params, "wqkv_a"), {"model.decoder.self_attn"}
        )

    def test_mixed_checkpoint_loads_every_layer(self):
        wq_a = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4)
        wkv = torch.arange(2 * 4, dtype=torch.float32).reshape(2, 4) + 100
        packed = {
            f"{PACKED_LAYER}.wq_a.qweight": torch.ones(4, 8),
            f"{PACKED_LAYER}.wq_a.scales": torch.full((1, 8), 2.0),
            f"{PACKED_LAYER}.wkv.qweight": torch.full((4, 2), 3.0),
            f"{PACKED_LAYER}.wkv.scales": torch.full((1, 2), 4.0),
        }
        params = self.mixed_params()

        run_load_weights(
            params,
            [
                (f"{DENSE_LAYER}.wq_a.weight", wq_a),
                (f"{DENSE_LAYER}.wkv.weight", wkv),
                *packed.items(),
            ],
        )

        # The dense layer is fused exactly as on a uniform dense checkpoint.
        torch.testing.assert_close(
            params[f"{DENSE_LAYER}.wqkv_a.weight"].data, torch.cat([wq_a, wkv], dim=0)
        )
        # The packed layer loads through its own separate projections.
        for name, weight in packed.items():
            torch.testing.assert_close(params[name].data, weight)

    def test_mixed_checkpoint_emits_no_unplaced_summary(self):
        params = self.mixed_params()

        with self.assertLogs("sglang.srt.models.deepseek_v4", level="WARNING") as logs:
            logging.getLogger("sglang.srt.models.deepseek_v4").warning("probe")
            run_load_weights(
                params,
                [
                    (f"{DENSE_LAYER}.wq_a.weight", torch.zeros(8, 4)),
                    (f"{DENSE_LAYER}.wkv.weight", torch.zeros(2, 4)),
                    (f"{PACKED_LAYER}.wq_a.qweight", torch.zeros(4, 8)),
                    (f"{PACKED_LAYER}.wq_a.scales", torch.zeros(1, 8)),
                    (f"{PACKED_LAYER}.wkv.qweight", torch.zeros(4, 2)),
                    (f"{PACKED_LAYER}.wkv.scales", torch.zeros(1, 2)),
                ],
            )

        self.assertEqual(
            [record for record in logs.output if "not placed" in record], []
        )
        self.assertEqual(
            [record for record in logs.output if "not found in params_dict" in record],
            [],
        )

    def test_packed_tensor_for_a_fused_layer_is_still_refused(self):
        # The layer kept wqkv_a, so a packed checkpoint tensor for it is a real
        # disagreement between model and checkpoint and must not be joined.
        params = self.mixed_params()

        with self.assertRaises(ValueError) as ctx:
            run_load_weights(
                params,
                [(f"{DENSE_LAYER}.wq_a.qweight", torch.zeros(4, 8))],
            )

        message = str(ctx.exception)
        self.assertIn(f"{DENSE_LAYER}.wq_a.qweight", message)
        self.assertIn("SGLANG_OPT_FUSE_WQA_WKV=0", message)


class TestUnplacedWeightSummary(unittest.TestCase):
    def test_summary_counts_and_collapses_layer_indices(self):
        names = [
            f"model.layers.{layer}.self_attn.{module}.{leaf}"
            for layer in range(43)
            for module in ("wq_a", "wkv")
            for leaf in ("qweight", "qzeros", "scales")
        ]

        summary = _summarize_unplaced_weights(names)

        self.assertIn("258 checkpoint weights were not placed", summary)
        self.assertIn("model.layers.*.self_attn.wq_a (129)", summary)
        self.assertIn("model.layers.*.self_attn.wkv (129)", summary)

    def test_summary_lists_at_most_top_prefixes(self):
        names = [f"module_{i}.leaf" for i in range(20)]
        summary = _summarize_unplaced_weights(names, top=2)
        self.assertIn("20 checkpoint weights were not placed", summary)
        self.assertEqual(summary.count("(1)"), 2)

    def test_load_weights_emits_one_summary(self):
        params = {f"{ATTN_PREFIX}.wq_b.weight": torch.nn.Parameter(torch.zeros(2, 2))}
        weights = [
            (f"model.layers.{layer}.self_attn.q_norm.stray", torch.zeros(2))
            for layer in range(3)
        ]

        with self.assertLogs("sglang.srt.models.deepseek_v4", level="WARNING") as logs:
            run_load_weights(params, weights)

        summaries = [
            record
            for record in logs.output
            if "checkpoint weights were not placed" in record
        ]
        self.assertEqual(len(summaries), 1)
        self.assertIn("3 checkpoint weights were not placed", summaries[0])
        self.assertIn("model.layers.*.self_attn.q_norm (3)", summaries[0])


class TestFusedShardBuffer(unittest.TestCase):
    def test_duplicate_shard_is_rejected(self):
        cache = {}
        _pop_fused_weight(cache, "p", "q", torch.zeros(1), _WQKV_A_SHARD_ORDER)
        with self.assertRaises(AssertionError):
            _pop_fused_weight(cache, "p", "q", torch.zeros(1), _WQKV_A_SHARD_ORDER)

    def test_incomplete_bucket_stays_in_cache(self):
        cache = {}
        self.assertIsNone(
            _pop_fused_weight(cache, "p", "q", torch.zeros(1), _WQKV_A_SHARD_ORDER)
        )
        self.assertEqual(list(cache), ["p"])


if __name__ == "__main__":
    unittest.main()
