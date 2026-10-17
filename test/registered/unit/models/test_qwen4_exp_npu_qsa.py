"""CPU checks for the explicit, bounded NPU dense-QSA fallback."""

import sys
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from qwen4_exp_cpu_test_utils import load

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def config_module(npu=True, enabled=True):
    return load(
        "srt/layers/attention/qsa/config.py",
        {"qsa_dense_fallback": None, "validate_qsa_dense_fallback": None},
        {
            "is_npu": lambda: npu,
            "envs": NS(SGLANG_QWEN4_NPU_DENSE_FALLBACK=NS(get=lambda: enabled)),
            "parse_qsa_profile": lambda config: config,
        },
    )


class TestNPUDenseQSA(CustomTestCase):
    def test_requires_npu_and_explicit_opt_in(self):
        for npu in (True, False):
            for enabled in (True, False):
                with self.subTest(npu=npu, enabled=enabled):
                    self.assertEqual(
                        config_module(npu, enabled).qsa_dense_fallback(),
                        npu and enabled,
                    )

    def test_budget_boundary(self):
        code = config_module()
        profile = NS(budget=2048)
        for length in (512, 1024, 2048):
            code.validate_qsa_dense_fallback(profile, length)
        with self.assertRaisesRegex(ValueError, "2048"):
            code.validate_qsa_dense_fallback(profile, 2049)

    def test_other_platforms_and_models_unchanged(self):
        config_module().validate_qsa_dense_fallback(None, 262144)
        config_module(enabled=False).validate_qsa_dense_fallback(
            NS(budget=2048), 262144
        )
        config_module(npu=False).validate_qsa_dense_fallback(NS(budget=2048), 262144)

    def test_model_config_checks_resolved_context_including_default(self):
        code = load(
            "srt/configs/model_config.py",
            {"ModelConfig": {"_derive_context_length"}},
            {"get_context_length": lambda config: 262144},
        )
        model = code.ModelConfig()
        model.is_draft_model = False
        model.hf_config = NS(budget=2048)
        model.hf_text_config = model.hf_config
        with patch.dict(
            sys.modules,
            {
                "sglang.srt.layers.attention.qsa.config": config_module(),
            },
        ):
            for length in (None, 2049):
                with self.subTest(length=length), self.assertRaises(ValueError):
                    model._derive_context_length(length)
            model._derive_context_length(1024)
        self.assertEqual(model.hf_config.context_len, 1024)

    def test_dense_fallback_does_not_allocate_qsa_cells(self):
        # Production code returns before importing/using QSA indexer allocation.
        config = "sglang.srt.layers.attention.qsa.config"
        pool = "sglang.srt.mem_cache.qsa_kv_pool"
        parse = Mock(side_effect=AssertionError("QSA allocation used in dense mode"))
        code = load(
            "srt/model_executor/pool_configurator.py",
            {"DefaultPoolConfigurator": {"_compute_qsa_cell_size"}},
        )
        with patch.dict(
            sys.modules,
            {
                config: NS(parse_qsa_profile=parse, qsa_dense_fallback=lambda: True),
                pool: NS(QSATokenToKVPool=object, resolve_qsa_indexer_dtype=Mock()),
            },
        ):
            self.assertEqual(
                code.DefaultPoolConfigurator._compute_qsa_cell_size(
                    hf_config=object(), num_layers=12
                ),
                0,
            )
        parse.assert_not_called()


if __name__ == "__main__":
    unittest.main()
