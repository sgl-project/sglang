import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.deepseek_v4_hook import validate_deepseek_v41_features
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _config(**overrides):
    values = {
        "pp_virtual_stages": 2,
        "pp_size": 4,
        "tp_size": 2,
        "attn_cp_size": 1,
        "dp_size": 1,
        "dcp_size": 1,
        "enable_prefill_cp": False,
        "enable_dp_attention": False,
        "speculative_algorithm": None,
        "disaggregation_mode": "prefill",
        "disaggregation_transfer_backend": "mooncake",
        "language_only": False,
        "language_model_only": False,
        "enable_hierarchical_cache": False,
        "pp_async_batch_depth": 0,
        "disable_overlap_schedule": False,
        "enable_encoder_swa_bounded_replay": False,
        "enable_decoder_swa_bounded_replay": False,
        "cuda_graph_config": SimpleNamespace(
            prefill=SimpleNamespace(backend=Backend.DISABLED)
        ),
        "enable_mixed_chunk": False,
        "enable_two_batch_overlap": False,
        "enable_hisparse": False,
        "dsv4_attn_backend": "triton",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _validate(config):
    with (
        patch(
            "sglang.srt.arg_groups.deepseek_v4_hook.resolving_view",
            return_value=config,
        ),
        patch(
            "sglang.srt.arg_groups.deepseek_v4_hook.model_config_of",
            return_value=SimpleNamespace(
                hf_config=SimpleNamespace(model_type="deepseek_v41")
            ),
        ),
        patch(
            "sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate."
            "is_unified_kv_triton",
            return_value=False,
        ),
    ):
        validate_deepseek_v41_features(SimpleNamespace())


class TestDeepSeekV41VPPGate(unittest.TestCase):
    def test_pp4_tp2_ordinary_prefill_is_supported(self):
        _validate(_config(pp_virtual_stages=1))

    def test_ordinary_pp_rejects_unvalidated_pp_tp_layout(self):
        with self.assertRaisesRegex(ValueError, "PP4 x TP2"):
            _validate(_config(pp_virtual_stages=1, pp_size=2, tp_size=4))

    def test_pp4_tp2_vpp2_without_cp_is_supported(self):
        _validate(_config())

    def test_pp2_tp4_vpp2_without_cp_is_supported(self):
        _validate(_config(pp_size=2, tp_size=4))

    def test_pp4_tp2_vpp2_supports_multimodal_loading(self):
        _validate(_config(language_only=False, language_model_only=False))

    def test_pp4_tp2_vpp2_supports_decoder_swa_bounded_replay(self):
        _validate(_config(enable_decoder_swa_bounded_replay=True))

    def test_vpp2_rejects_prefill_cp(self):
        for config in (
            _config(enable_prefill_cp=True),
            _config(attn_cp_size=2),
        ):
            with self.subTest(config=config):
                with self.assertRaisesRegex(ValueError, "context parallelism"):
                    _validate(config)

    def test_vpp2_requires_overlap_scheduler(self):
        with self.assertRaisesRegex(ValueError, "disabled overlap scheduling"):
            _validate(_config(disable_overlap_schedule=True))

    def test_vpp2_rejects_unvalidated_pp_tp_layout(self):
        with self.assertRaisesRegex(ValueError, "PP4 x TP2 or PP2 x TP4"):
            _validate(_config(pp_size=2, tp_size=2))


if __name__ == "__main__":
    unittest.main()
