"""CPU coverage for QSA sparse prefill launch geometry selection."""

import os
import unittest
from unittest import mock

from sglang.srt.layers.attention.qsa import sparse_attn
from sglang.srt.layers.attention.qsa.sparse_attn import _SM120_PREFILL_CONFIGS
from sglang.srt.utils import common as utils_common
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestQsaSparsePrefillConfig(unittest.TestCase):
    @staticmethod
    def _clear_arch_caches():
        utils_common.is_cuda.cache_clear()
        sparse_attn.is_sm120_supported.cache_clear()
        sparse_attn.is_sm121.cache_clear()
        sparse_attn._get_prefill_device_configs.cache_clear()
        sparse_attn._triton_supports_sm120_prefill.cache_clear()
        sparse_attn._sm120_multiprocessor_count.cache_clear()

    def setUp(self):
        self.current_device = mock.patch.object(
            sparse_attn.torch.cuda, "current_device", return_value=0
        )
        self.current_device.start()
        self._clear_arch_caches()

    def tearDown(self):
        self._clear_arch_caches()
        self.current_device.stop()

    @staticmethod
    def _config(
        total_q,
        group_size,
        num_requests=1,
        head_dim=256,
        *,
        kernel="ordinary",
        topk=2051,
        num_kv_heads=1,
        max_q=None,
    ):
        return sparse_attn._get_prefill_config(
            total_q,
            group_size,
            num_requests,
            head_dim,
            kernel=kernel,
            topk=topk,
            num_kv_heads=num_kv_heads,
            max_q=total_q if max_q is None else max_q,
        )

    def test_sm120_geometry(self):
        expected = {
            3: (16, 16, 1, 2),
            **_SM120_PREFILL_CONFIGS,
        }
        with (
            mock.patch.object(sparse_attn, "is_sm120_supported", return_value=True),
            mock.patch.object(sparse_attn, "is_sm121", return_value=False),
            mock.patch.object(
                sparse_attn, "_triton_supports_sm120_prefill", return_value=True
            ),
            mock.patch.object(
                sparse_attn, "_sm120_multiprocessor_count", return_value=188
            ),
            mock.patch.object(
                sparse_attn.torch.cuda,
                "get_device_name",
                return_value="NVIDIA L20",
            ),
        ):
            for group_size, geometry in expected.items():
                with self.subTest(group_size=group_size):
                    self.assertEqual(
                        self._config(8192, group_size),
                        geometry,
                    )
            fallback_by_total_q = {
                32: (16, 32, 8, 2),
                64: (16, 64, 8, 2),
                128: (16, 64, 4, 2),
                256: (16, 32, 4, 2),
                511: (16, 32, 4, 2),
            }
            for group_size in (6, 12):
                total_q_expected = {
                    **fallback_by_total_q,
                    512: (16, 32, 4, 2),
                    1024: (16, 16, 1, 2),
                    8192: _SM120_PREFILL_CONFIGS[group_size],
                }
                for total_q, geometry in total_q_expected.items():
                    with self.subTest(total_q=total_q, group_size=group_size):
                        self.assertEqual(
                            self._config(total_q, group_size),
                            geometry,
                        )
            self.assertEqual(self._config(8192, 24), (32, 16, 1, 2))
            with self.subTest(gate="head-dim"):
                self.assertEqual(
                    self._config(8192, 6, head_dim=128),
                    sparse_attn._get_table_prefill_config(8192, 6),
                )
            for group_size in (6, 12):
                with self.subTest(group_size=group_size, gate="per-request"):
                    self.assertEqual(
                        self._config(
                            1024,
                            group_size,
                            16,
                            kernel="chunk",
                            max_q=64,
                        ),
                        sparse_attn._get_table_prefill_config(1024, group_size),
                    )
                    self.assertEqual(
                        self._config(
                            1024,
                            group_size,
                            2,
                            kernel="chunk",
                            max_q=512,
                        ),
                        _SM120_PREFILL_CONFIGS[group_size],
                    )
                    self.assertEqual(
                        self._config(
                            1024,
                            group_size,
                            2,
                            kernel="chunk",
                            max_q=1024,
                        ),
                        _SM120_PREFILL_CONFIGS[group_size],
                    )
                    self.assertEqual(
                        self._config(
                            1023,
                            group_size,
                            2,
                            kernel="chunk",
                            max_q=512,
                        ),
                        sparse_attn._get_table_prefill_config(1023, group_size),
                    )

    def test_sm120_sm_count_gate(self):
        with (
            mock.patch.object(sparse_attn, "is_sm120_supported", return_value=True),
            mock.patch.object(sparse_attn, "is_sm121", return_value=False),
            mock.patch.object(
                sparse_attn, "_triton_supports_sm120_prefill", return_value=True
            ),
            mock.patch.object(
                sparse_attn.torch.cuda,
                "get_device_name",
                return_value="NVIDIA L20",
            ),
        ):
            with mock.patch.object(
                sparse_attn, "_sm120_multiprocessor_count", return_value=169
            ):
                self.assertEqual(
                    self._config(8192, 6),
                    sparse_attn._get_table_prefill_config(8192, 6),
                )
                with mock.patch.dict(
                    os.environ, {"SGLANG_QSA_PREFILL_GEOMETRY": "tuned"}
                ):
                    self.assertEqual(
                        self._config(8192, 6),
                        _SM120_PREFILL_CONFIGS[6],
                    )
            with mock.patch.object(
                sparse_attn, "_sm120_multiprocessor_count", return_value=170
            ):
                self.assertEqual(
                    self._config(8192, 6),
                    sparse_attn._get_table_prefill_config(8192, 6),
                )

    def test_sm120_measured_envelope(self):
        with (
            mock.patch.object(sparse_attn, "is_sm120_supported", return_value=True),
            mock.patch.object(sparse_attn, "is_sm121", return_value=False),
            mock.patch.object(
                sparse_attn, "_triton_supports_sm120_prefill", return_value=True
            ),
            mock.patch.object(
                sparse_attn, "_sm120_multiprocessor_count", return_value=188
            ),
            mock.patch.object(
                sparse_attn.torch.cuda,
                "get_device_name",
                return_value="NVIDIA L20",
            ),
        ):
            defaults = {
                "total_q": 8192,
                "group_size": 6,
                "num_requests": 1,
                "head_dim": 256,
                "kernel": "ordinary",
                "topk": 2051,
                "num_kv_heads": 1,
                "max_q": 8192,
            }
            cases = {
                "ordinary-minimum": ({"total_q": 8191, "max_q": 8191}, 8191, 6),
                "topk": ({"topk": 2048}, 8192, 6),
                "group-6-kv-heads": ({"num_kv_heads": 2}, 8192, 6),
                "group-12-kv-heads": (
                    {"group_size": 12, "num_kv_heads": 3},
                    8192,
                    12,
                ),
                "chunk-average": (
                    {
                        "kernel": "chunk",
                        "total_q": 1023,
                        "num_requests": 2,
                        "max_q": 512,
                    },
                    1023,
                    6,
                ),
                "chunk-grid": (
                    {
                        "kernel": "chunk",
                        "total_q": 8207,
                        "num_requests": 16,
                        "max_q": 8192,
                    },
                    8207,
                    6,
                ),
            }
            for geometry in ("auto", "tuned"):
                with mock.patch.dict(
                    os.environ, {"SGLANG_QSA_PREFILL_GEOMETRY": geometry}
                ):
                    for gate, (changes, total_q, group_size) in cases.items():
                        with self.subTest(geometry=geometry, gate=gate):
                            arguments = {**defaults, **changes}
                            self.assertEqual(
                                sparse_attn._get_prefill_config(**arguments),
                                sparse_attn._get_table_prefill_config(
                                    total_q, group_size
                                ),
                            )

    def test_sm120_triton_version_gate(self):
        with (
            mock.patch.object(sparse_attn, "is_sm120_supported", return_value=True),
            mock.patch.object(sparse_attn, "is_sm121", return_value=False),
            mock.patch.object(
                sparse_attn, "_sm120_multiprocessor_count", return_value=188
            ),
            mock.patch.object(
                sparse_attn.torch.cuda,
                "get_device_name",
                return_value="NVIDIA L20",
            ),
        ):
            for version, expected in (
                ("3.6.0", (16, 16, 1, 2)),
                ("3.7.0", _SM120_PREFILL_CONFIGS[6]),
            ):
                with (
                    self.subTest(version=version),
                    mock.patch.object(sparse_attn.triton, "__version__", version),
                ):
                    self._clear_arch_caches()
                    self.assertEqual(self._config(8192, 6), expected)

    def test_sm120_geometry_override(self):
        with (
            mock.patch.object(sparse_attn, "is_sm120_supported", return_value=True),
            mock.patch.object(sparse_attn, "is_sm121", return_value=False),
            mock.patch.object(
                sparse_attn, "_triton_supports_sm120_prefill", return_value=True
            ),
            mock.patch.object(
                sparse_attn, "_sm120_multiprocessor_count", return_value=188
            ),
            mock.patch.object(
                sparse_attn.torch.cuda,
                "get_device_name",
                return_value="NVIDIA L20",
            ),
        ):
            for value, expected in (
                (" auto ", _SM120_PREFILL_CONFIGS[6]),
                ("table", (16, 16, 1, 2)),
            ):
                with (
                    self.subTest(value=value),
                    mock.patch.dict(os.environ, {"SGLANG_QSA_PREFILL_GEOMETRY": value}),
                ):
                    self._clear_arch_caches()
                    self.assertEqual(self._config(8192, 6), expected)

    def test_geometry_override_is_read_per_call(self):
        with (
            mock.patch.object(sparse_attn, "is_sm120_supported", return_value=True),
            mock.patch.object(sparse_attn, "is_sm121", return_value=False),
            mock.patch.object(
                sparse_attn, "_triton_supports_sm120_prefill", return_value=True
            ),
            mock.patch.object(
                sparse_attn, "_sm120_multiprocessor_count", return_value=188
            ),
            mock.patch.object(
                sparse_attn.torch.cuda,
                "get_device_name",
                return_value="NVIDIA L20",
            ),
            mock.patch.dict(os.environ, {"SGLANG_QSA_PREFILL_GEOMETRY": "auto"}),
        ):
            self.assertEqual(
                self._config(8192, 6),
                _SM120_PREFILL_CONFIGS[6],
            )
            os.environ["SGLANG_QSA_PREFILL_GEOMETRY"] = "table"
            self.assertEqual(
                self._config(8192, 6),
                (16, 16, 1, 2),
            )

    def test_invalid_geometry_override(self):
        with mock.patch.dict(os.environ, {"SGLANG_QSA_PREFILL_GEOMETRY": "invalid"}):
            with self.assertRaisesRegex(
                ValueError,
                "SGLANG_QSA_PREFILL_GEOMETRY must be one of auto, table, tuned; got 'invalid'",
            ):
                self._config(8192, 6)

    def test_sm120_predicate_wiring(self):
        with (
            mock.patch.object(
                sparse_attn.torch.cuda, "is_available", return_value=True
            ),
            mock.patch.object(
                sparse_attn.torch.cuda,
                "get_device_capability",
                return_value=(12, 0),
            ),
            mock.patch.object(sparse_attn.torch.version, "cuda", "13.0"),
            mock.patch.object(
                sparse_attn, "_triton_supports_sm120_prefill", return_value=True
            ),
            mock.patch.object(
                sparse_attn, "_sm120_multiprocessor_count", return_value=188
            ),
        ):
            self._clear_arch_caches()
            self.assertEqual(
                self._config(8192, 6),
                _SM120_PREFILL_CONFIGS[6],
            )
            self.assertEqual(
                self._config(8192, 12),
                _SM120_PREFILL_CONFIGS[12],
            )

    def test_current_device_used_for_hardware_queries(self):
        properties = mock.Mock(multi_processor_count=188)
        with (
            mock.patch.object(sparse_attn, "is_sm120_supported", return_value=True),
            mock.patch.object(sparse_attn, "is_sm121", return_value=False),
            mock.patch.object(
                sparse_attn, "_triton_supports_sm120_prefill", return_value=True
            ),
            mock.patch.object(sparse_attn.torch.cuda, "current_device", return_value=1),
            mock.patch.object(
                sparse_attn.torch.cuda,
                "get_device_properties",
                return_value=properties,
            ) as get_properties,
            mock.patch.object(
                sparse_attn.torch.cuda,
                "get_device_name",
                return_value="NVIDIA H20",
            ) as get_device_name,
        ):
            self.assertEqual(self._config(8192, 6), _SM120_PREFILL_CONFIGS[6])
            with mock.patch.dict(os.environ, {"SGLANG_QSA_PREFILL_GEOMETRY": "table"}):
                self.assertEqual(self._config(128, 6), (16, 32, 4, 2))
        get_properties.assert_called_once_with(1)
        get_device_name.assert_called_once_with(1)

    def test_prefill_geometry_invariants(self):
        for group_size, config in _SM120_PREFILL_CONFIGS.items():
            with self.subTest(group_size=group_size):
                self.assertGreaterEqual(config.block_m, group_size)
        with mock.patch.object(
            sparse_attn.torch.cuda,
            "get_device_name",
            return_value="NVIDIA L20",
        ):
            self.assertEqual(
                sparse_attn._get_table_prefill_config(8192, 6),
                (16, 16, 1, 2),
            )

    def test_non_sm120_uses_fallback_table(self):
        with (
            mock.patch.object(sparse_attn, "is_sm120_supported", return_value=False),
            mock.patch.object(
                sparse_attn.torch.cuda,
                "get_device_name",
                return_value="NVIDIA L20",
            ),
        ):
            self.assertEqual(self._config(8192, 6), (16, 16, 1, 2))

    def test_sm121_uses_fallback_table(self):
        with (
            mock.patch.object(sparse_attn, "is_sm120_supported", return_value=True),
            mock.patch.object(sparse_attn, "is_sm121", return_value=True),
            mock.patch.object(
                sparse_attn.torch.cuda,
                "get_device_name",
                return_value="NVIDIA L20",
            ),
        ):
            self.assertEqual(self._config(8192, 6), (16, 16, 1, 2))


if __name__ == "__main__":
    unittest.main()
