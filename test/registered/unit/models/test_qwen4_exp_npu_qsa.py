"""CPU checks for the explicit, bounded NPU dense-QSA fallback."""

import sys
import unittest
from contextlib import contextmanager
from types import SimpleNamespace as NS
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.qsa import config as qsa_config
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.qwen4_exp_cpu_test_utils import load
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def checkpoint():
    return NS(
        architectures=["Qwen4ExpForConditionalGeneration"],
        indexer_n_heads=8,
        indexer_kv_heads=1,
        indexer_head_dim=128,
        indexer_budget=2048,
        indexer_compress_ratio=4,
    )


@contextmanager
def config_module(*, npu=True, enabled=True):
    with (
        patch.object(qsa_config, "is_npu", return_value=npu),
        envs.SGLANG_ENABLE_QWEN4_NPU_DENSE_FALLBACK.override(enabled),
    ):
        yield qsa_config


class TestNPUDenseQSA(CustomTestCase):
    def test_active_profile_requires_npu_and_explicit_opt_in(self):
        for npu in (True, False):
            for enabled in (True, False):
                with (
                    self.subTest(npu=npu, enabled=enabled),
                    config_module(npu=npu, enabled=enabled) as code,
                ):
                    config = checkpoint()
                    raw_profile = code.parse_qsa_profile(config)
                    self.assertEqual(raw_profile.budget, 2048)
                    self.assertEqual(
                        code.get_active_qsa_profile(config),
                        None if npu and enabled else raw_profile,
                    )

    def test_budget_boundary(self):
        with config_module() as code:
            config = NS(text_config=checkpoint())
            for length in (512, 1024, 2048):
                code.validate_qsa_dense_fallback(config, length)
            with self.assertRaisesRegex(ValueError, "2048"):
                code.validate_qsa_dense_fallback(config, 2049)

    def test_other_platforms_and_models_unchanged(self):
        with config_module() as code:
            code.validate_qsa_dense_fallback(NS(), 262144)
            self.assertIsNone(code.get_active_qsa_profile(NS()))
        for options in ({"enabled": False}, {"npu": False}):
            with config_module(**options) as code:
                code.validate_qsa_dense_fallback(checkpoint(), 262144)

    def test_model_config_checks_resolved_context_including_default(self):
        code = load(
            "srt/configs/model_config.py",
            {"ModelConfig": {"_derive_context_length"}},
            {"get_context_length": lambda config: 262144},
        )
        model = code.ModelConfig()
        model.is_draft_model = False
        model.hf_config = checkpoint()
        model.hf_text_config = model.hf_config
        with config_module():
            for length in (None, 2049):
                with self.subTest(length=length), self.assertRaises(ValueError):
                    model._derive_context_length(length)
            model._derive_context_length(1024)
        self.assertEqual(model.hf_config.context_len, 1024)

    def test_dense_fallback_does_not_allocate_qsa_cells(self):
        code = load(
            "srt/model_executor/pool_configurator.py",
            {"DefaultPoolConfigurator": {"_compute_qsa_cell_size"}},
        )
        with (
            config_module(),
            patch.dict(
                sys.modules,
                {
                    "sglang.srt.mem_cache.qsa_kv_pool": NS(
                        QSATokenToKVPool=object, resolve_qsa_indexer_dtype=None
                    ),
                },
            ),
        ):
            self.assertEqual(
                code.DefaultPoolConfigurator._compute_qsa_cell_size(
                    hf_config=checkpoint(), num_layers=12
                ),
                0,
            )


if __name__ == "__main__":
    unittest.main()
