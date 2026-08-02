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
    _is_dense_linear,
    _pop_fused_weight,
    _reject_unfusable_leaf,
    _summarize_unplaced_weights,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ATTN_PREFIX = "model.layers.3.self_attn"
COMPRESSOR_PREFIX = "model.layers.3.self_attn.compressor"


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


class TestPackedLayerFallback(unittest.TestCase):
    def test_unquantized_replicated_linear_is_dense(self):
        layer = ReplicatedLinear(8, 4, bias=False, quant_config=None)
        self.assertTrue(_is_dense_linear(layer))

    def test_packed_layer_is_not_dense(self):
        layer = torch.nn.Module()
        for packed_name in ("qweight", "qzeros", "scales"):
            layer.register_parameter(packed_name, torch.nn.Parameter(torch.zeros(2, 2)))
        self.assertFalse(_is_dense_linear(layer))

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
