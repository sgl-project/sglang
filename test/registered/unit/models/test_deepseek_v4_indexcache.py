import types
import unittest

from sglang.srt.models.deepseek_common.utils import (
    compute_dsv4_index_cache_descriptor,
    compute_dsv4_index_topk_flags,
    dsv4_index_cache_enabled,
    dsv4_index_cache_producer_layer_ids,
    get_dsv4_c4_layer_ids,
    validate_dsv4_index_cache_pd_compatibility,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _hf_config(index_topk_freq=1, index_topk_pattern=None):
    compress_ratios = [0, 0] + [4, 128] * 20 + [4]
    return types.SimpleNamespace(
        num_hidden_layers=43,
        compress_ratios=compress_ratios,
        index_topk=512,
        index_head_dim=128,
        index_n_heads=64,
        index_topk_freq=index_topk_freq,
        index_topk_pattern=index_topk_pattern,
    )


class TestDeepseekV4IndexCacheUtils(CustomTestCase):
    COMPRESS_RATIOS = [0, 0, 4, 128, 4, 128, 4, 128, 4, 0]
    C4_LAYERS = [2, 4, 6, 8]

    def _flags(self, *, freq=1, pattern=None):
        return [
            compute_dsv4_index_topk_flags(self.COMPRESS_RATIOS, layer_id, freq, pattern)
            for layer_id in self.C4_LAYERS
        ]

    def test_c4_layers_and_frequency(self):
        self.assertEqual(get_dsv4_c4_layer_ids(self.COMPRESS_RATIOS), self.C4_LAYERS)
        self.assertEqual(
            self._flags(freq=4),
            [(False, True), (True, True), (True, True), (True, False)],
        )

    def test_c4_and_full_layer_patterns(self):
        full_pattern = ["F"] * len(self.COMPRESS_RATIOS)
        full_pattern[4] = "S"
        full_pattern[6] = "S"
        expected = [(False, True), (True, True), (True, False), (False, False)]
        for pattern in ("FSSF", full_pattern):
            with self.subTest(pattern=pattern):
                self.assertEqual(self._flags(pattern=pattern), expected)

    def test_frequency_validation(self):
        for value in (0, -2, 2.5, True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError, "index_topk_freq must be a positive integer"
                ):
                    compute_dsv4_index_topk_flags(
                        self.COMPRESS_RATIOS, self.C4_LAYERS[0], value
                    )
        self.assertEqual(
            self._flags(freq=None),
            [(False, False), (False, False), (False, False), (False, False)],
        )

    def test_pattern_validation(self):
        invalid_cases = [
            ("FXSF", "only supports 'F'.*'S'"),
            ("FS", "length must either match"),
            ("SFFF", "first C4 layer must be 'F'"),
        ]
        full_pattern = ["F"] * len(self.COMPRESS_RATIOS)
        full_pattern[self.C4_LAYERS[0]] = "S"
        invalid_cases.append((full_pattern, "first C4 layer must be 'F'"))
        for pattern, error in invalid_cases:
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex(ValueError, error):
                    compute_dsv4_index_topk_flags(
                        self.COMPRESS_RATIOS,
                        self.C4_LAYERS[0],
                        index_topk_pattern=pattern,
                    )

    def test_descriptor_signature(self):
        sig_freq1, _ = compute_dsv4_index_cache_descriptor(
            _hf_config(index_topk_freq=1),
            fp4_indexer_enabled=True,
            page_size=64,
        )
        sig_freq4, _ = compute_dsv4_index_cache_descriptor(
            _hf_config(index_topk_freq=4),
            fp4_indexer_enabled=True,
            page_size=64,
        )
        self.assertEqual(sig_freq1, sig_freq4)

        for fp4, page_size in ((True, 128), (False, 64)):
            with self.subTest(fp4=fp4, page_size=page_size):
                signature, _ = compute_dsv4_index_cache_descriptor(
                    _hf_config(),
                    fp4_indexer_enabled=fp4,
                    page_size=page_size,
                )
                self.assertNotEqual(sig_freq1, signature)

    def test_producer_layers(self):
        cfg = _hf_config()
        cases = [(1, 21, False), (4, 6, True)]
        for freq, expected_count, enabled in cases:
            with self.subTest(freq=freq):
                producers = dsv4_index_cache_producer_layer_ids(
                    cfg.compress_ratios, freq
                )
                self.assertEqual(len(producers), expected_count)
                self.assertEqual(
                    dsv4_index_cache_enabled(cfg.compress_ratios, freq), enabled
                )

    def test_pd_compatibility(self):
        sig_freq1, prod_freq1 = compute_dsv4_index_cache_descriptor(
            _hf_config(index_topk_freq=1),
            fp4_indexer_enabled=True,
            page_size=64,
        )
        sig_freq4, prod_freq4 = compute_dsv4_index_cache_descriptor(
            _hf_config(index_topk_freq=4),
            fp4_indexer_enabled=True,
            page_size=64,
        )
        validate_dsv4_index_cache_pd_compatibility(
            prefill_layout_signature=sig_freq1,
            prefill_producer_layer_ids=prod_freq1,
            decode_layout_signature=sig_freq4,
            decode_producer_layer_ids=prod_freq4,
        )

        with self.assertRaisesRegex(RuntimeError, "producer coverage"):
            validate_dsv4_index_cache_pd_compatibility(
                prefill_layout_signature=sig_freq4,
                prefill_producer_layer_ids=prod_freq4,
                decode_layout_signature=sig_freq1,
                decode_producer_layer_ids=prod_freq1,
            )

        mismatch_sig, mismatch_prod = compute_dsv4_index_cache_descriptor(
            _hf_config(),
            fp4_indexer_enabled=True,
            page_size=128,
        )
        with self.assertRaisesRegex(RuntimeError, "layout mismatch"):
            validate_dsv4_index_cache_pd_compatibility(
                prefill_layout_signature=mismatch_sig,
                prefill_producer_layer_ids=mismatch_prod,
                decode_layout_signature=sig_freq1,
                decode_producer_layer_ids=prod_freq1,
            )


if __name__ == "__main__":
    unittest.main()
