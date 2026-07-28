import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.hisparse_hook import (
    apply_runtime_sparse_eager_defaults,
    use_runtime_sparse_cuda_graph,
)
from sglang.srt.mem_cache.sparsity import parse_runtime_sparse_config
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _args(config, *, page_size=16):
    return SimpleNamespace(
        enable_hisparse=True,
        hisparse_config=json.dumps(config),
        page_size=page_size,
    )


class TestRuntimeSparseConfig(unittest.TestCase):
    def _parse(self, config, *, page_size=16):
        with patch(
            "sglang.srt.arg_groups.overrides.attention_backends_of",
            return_value=("fa3", "fa3"),
        ):
            return parse_runtime_sparse_config(_args(config, page_size=page_size))

    def test_parses_quest_runtime_options_and_rejects_page_mismatch(self):
        config = self._parse(
            {
                "algorithm": "quest",
                "backend": "fa3",
                "page_size": 16,
                "sparsity_ratio": 0.125,
                "num_recent_pages": 4,
            }
        )
        self.assertEqual(
            (config.algorithm, config.backend, config.page_size),
            ("quest", "fa3", 16),
        )
        self.assertEqual(config.sparse_extra_config["sparsity_ratio"], 0.125)
        self.assertEqual(config.sparse_extra_config["num_recent_pages"], 4)

        with self.assertRaisesRegex(ValueError, "page_size.*match"):
            self._parse(
                {"algorithm": "quest", "backend": "fa3", "page_size": 8},
                page_size=16,
            )

    @patch(
        "sglang.srt.arg_groups.hisparse_hook.use_runtime_sparse_attention",
        return_value=True,
    )
    def test_graph_provider_controls_decode_while_prefill_remains_eager(
        self, _mock_runtime
    ):
        for provider_ready, expected_decode in (
            (False, Backend.DISABLED),
            (True, Backend.BREAKABLE),
        ):
            with self.subTest(provider_ready=provider_ready), patch(
                "sglang.srt.arg_groups.hisparse_hook.use_runtime_sparse_cuda_graph",
                return_value=provider_ready,
            ):
                args = SimpleNamespace(
                    cuda_graph_config=SimpleNamespace(
                        prefill=SimpleNamespace(backend=Backend.BREAKABLE),
                        decode=SimpleNamespace(backend=Backend.BREAKABLE),
                    ),
                    _cuda_graph_config_locked=set(),
                )
                apply_runtime_sparse_eager_defaults(args)
                self.assertEqual(
                    args.cuda_graph_config.prefill.backend, Backend.DISABLED
                )
                self.assertEqual(args.cuda_graph_config.decode.backend, expected_decode)

    @patch(
        "sglang.srt.arg_groups.hisparse_hook.use_runtime_sparse_attention",
        return_value=True,
    )
    def test_missing_or_incapable_graph_provider_stays_eager(self, _mock_runtime):
        args = _args(
            {
                "algorithm": "quest",
                "backend": "fa3",
                "page_size": 16,
                "enable_cuda_graph_retrieval": True,
            }
        )
        with patch("sglang.srt.arg_groups.hisparse_hook.find_spec", return_value=None):
            self.assertFalse(use_runtime_sparse_cuda_graph(args))

        provider = SimpleNamespace(
            is_runtime_sparse_cuda_graph_available=lambda _coordinator: False
        )
        with (
            patch(
                "sglang.srt.arg_groups.overrides.attention_backends_of",
                return_value=("fa3", "fa3"),
            ),
            patch(
                "sglang.srt.arg_groups.hisparse_hook.find_spec", return_value=object()
            ),
            patch(
                "sglang.srt.arg_groups.hisparse_hook.import_module",
                return_value=provider,
            ),
            patch(
                "sglang.srt.mem_cache.sparsity.factory._create_sparse_algorithm",
                return_value=object(),
            ),
        ):
            self.assertFalse(use_runtime_sparse_cuda_graph(args))
