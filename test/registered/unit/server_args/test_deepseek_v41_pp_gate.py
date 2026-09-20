import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.deepseek_v4_hook import validate_deepseek_v41_features
from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _config(**overrides):
    values = {
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
        "enable_hierarchical_cache": False,
        "disable_radix_cache": True,
        "pp_async_batch_depth": 0,
        "disable_flashinfer_autotune": True,
        "enable_encoder_swa_bounded_replay": False,
        "enable_decoder_swa_bounded_replay": True,
        "enable_mixed_chunk": False,
        "enable_two_batch_overlap": False,
        "enable_hisparse": False,
        "dsv4_attn_backend": "auto",
        "cuda_graph_config": SimpleNamespace(
            prefill=SimpleNamespace(backend=Backend.DISABLED)
        ),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _validate(config, *, host_engram=True):
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
        envs.SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE.override(host_engram),
    ):
        validate_deepseek_v41_features(SimpleNamespace())
        return envs.SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE.get()


class TestDeepSeekV41PPGate(unittest.TestCase):
    def test_pp4_tp2_disaggregated_prefill_is_supported(self):
        _validate(_config())

    def test_topologies_and_cache_modes(self):
        for pp, tp in ((4, 2), (2, 4), (8, 1)):
            for mode in ("null", "prefill", "decode"):
                for hicache in (False, True):
                    with self.subTest(pp=pp, tp=tp, mode=mode, hicache=hicache):
                        _validate(
                            _config(
                                pp_size=pp,
                                tp_size=tp,
                                disaggregation_mode=mode,
                                disable_radix_cache=False,
                                enable_hierarchical_cache=hicache,
                            )
                        )

    def test_execution_options_do_not_restrict_pp_deployment(self):
        self.assertTrue(
            _validate(
                _config(disable_flashinfer_autotune=False, pp_async_batch_depth=2),
                host_engram=False,
            )
        )


if __name__ == "__main__":
    unittest.main()
