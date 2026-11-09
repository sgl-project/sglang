import unittest

from sglang.srt.hardware_backend.npu.dsa_int8_indexer import (
    parse_modelslim_int8_indexer_layers,
    resolve_npu_int8_indexer_layer_ids,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _indexer_key(layer_id: int, suffix: str = "quant_type") -> str:
    return f"model.layers.{layer_id}.self_attn.indexer.{suffix}"


def _resolve(**overrides):
    kwargs = dict(
        is_npu=True,
        index_head_dim=128,
        kv_cache_is_fp8=False,
        enable_hierarchical_cache=False,
        enable_hisparse=False,
        enable_sparse_kv_offload=False,
        candidate_layer_ids=tuple(range(8)),
        modelslim_quant_description=None,
    )
    kwargs.update(overrides)
    return resolve_npu_int8_indexer_layer_ids(**kwargs)


class TestParseModelSlimInt8IndexerLayers(unittest.TestCase):
    def test_only_int8_dynamic_layers_are_selected(self):
        description = {
            _indexer_key(1): "INT8_DYNAMIC",
            _indexer_key(2): "BF16",
            _indexer_key(5): "INT8_DYNAMIC",
        }
        self.assertEqual(
            parse_modelslim_int8_indexer_layers(description), frozenset({1, 5})
        )

    def test_no_per_layer_keys_means_all_layers(self):
        description = {"indexer_quant_type": "INT8_DYNAMIC"}
        self.assertIsNone(parse_modelslim_int8_indexer_layers(description))

    def test_weight_scheme_keys_alone_select_nothing(self):
        # Only the indexer's weight scheme is described per layer, so no layer
        # was calibrated for an INT8 cache.
        description = {
            "indexer_quant_type": "INT8_DYNAMIC",
            _indexer_key(3, "wq_b.weight"): "W8A8_DYNAMIC",
        }
        self.assertEqual(parse_modelslim_int8_indexer_layers(description), frozenset())

    def test_weight_scheme_keys_do_not_select_layers(self):
        # GLM-5.1 W4A8: wq_b is W8A8 on layers whose indexer cache stays bf16.
        description = {
            _indexer_key(4, "wq_b.weight"): "W8A8_DYNAMIC",
            _indexer_key(5): "INT8_DYNAMIC",
            _indexer_key(5, "wq_b.weight"): "W8A8_DYNAMIC",
            _indexer_key(78, "wq_b.weight"): "FLOAT",
        }
        self.assertEqual(
            parse_modelslim_int8_indexer_layers(description), frozenset({5})
        )

    def test_only_reverted_layers_select_nothing(self):
        self.assertEqual(
            parse_modelslim_int8_indexer_layers({_indexer_key(5): "BF16"}),
            frozenset(),
        )


class TestResolveNpuInt8IndexerLayerIds(unittest.TestCase):
    def test_all_candidates_without_modelslim_metadata(self):
        self.assertEqual(_resolve(), frozenset(range(8)))

    def test_modelslim_filter_intersects_candidates(self):
        description = {_indexer_key(i): "INT8_DYNAMIC" for i in (2, 3, 9)}
        description[_indexer_key(4)] = "BF16"
        self.assertEqual(
            _resolve(modelslim_quant_description=description), frozenset({2, 3})
        )

    def test_global_int8_metadata_falls_back_without_warning(self):
        with self.assertNoLogs(
            "sglang.srt.hardware_backend.npu.dsa_int8_indexer", level="WARNING"
        ):
            self.assertEqual(
                _resolve(
                    modelslim_quant_description={"indexer_quant_type": "INT8_DYNAMIC"}
                ),
                frozenset(range(8)),
            )

    def test_missing_calibration_warns(self):
        for description in (None, {}, {"indexer_quant_type": "BF16"}):
            with self.subTest(description=description):
                with self.assertLogs(
                    "sglang.srt.hardware_backend.npu.dsa_int8_indexer", level="WARNING"
                ) as logs:
                    self.assertEqual(
                        _resolve(modelslim_quant_description=description),
                        frozenset(range(8)),
                    )
                self.assertIn("--indexer-kv-cache-dtype int8", logs.output[0])

    def test_rank_with_no_selected_layers_is_valid(self):
        description = {_indexer_key(9): "INT8_DYNAMIC"}
        for candidates in ((4, 6, 7), ()):
            with self.subTest(candidates=candidates):
                self.assertEqual(
                    _resolve(
                        candidate_layer_ids=candidates,
                        modelslim_quant_description=description,
                    ),
                    frozenset(),
                )

    def test_speculative_decoding_is_rejected(self):
        """Reject MTP before local draft IDs can select target layer metadata."""
        with self.assertRaisesRegex(ValueError, "speculative decoding.*MTP"):
            _resolve(enable_speculative_decoding=True)

    def test_checkpoint_without_int8_layers_is_rejected(self):
        # Selecting nothing would disable the flag silently; the rank's own
        # candidates stay out of it so pipeline parallelism still works.
        description = {
            "indexer_quant_type": "W8A8_MXFP8",
            _indexer_key(2, "wq_b.weight"): "W8A8_DYNAMIC",
        }
        with self.assertRaises(ValueError):
            _resolve(modelslim_quant_description=description)

    def test_rejected_configurations(self):
        cases = {
            "not npu": dict(is_npu=False),
            "not dsa": dict(index_head_dim=None),
            "head dim": dict(index_head_dim=64),
            "fp8 kv": dict(kv_cache_is_fp8=True),
            "prefill cp": dict(enable_prefill_cp=True),
            "pp disaggregation": dict(enable_pp_disaggregation=True),
            "hicache": dict(enable_hierarchical_cache=True),
            "decode kv offload": dict(enable_decode_kv_offload=True),
            "decode host backup": dict(enable_decode_host_backup=True),
            "hisparse": dict(enable_hisparse=True),
            "sparse offload": dict(enable_sparse_kv_offload=True),
        }
        for name, overrides in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                _resolve(**overrides)


if __name__ == "__main__":
    unittest.main()
