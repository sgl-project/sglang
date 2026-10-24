import unittest
from types import SimpleNamespace
from unittest.mock import patch

from transformers import LlamaConfig

from sglang.srt.configs import (
    FalconH1Config,
    FalconMambaConfig,
    InklingModelConfig,
    Lfm2Config,
    Mamba2Config,
    MambaConfig,
    NemotronHConfig,
    ZayaConfig,
)
from sglang.srt.configs.granitemoehybrid import GraniteMoeHybridConfig
from sglang.srt.configs.hybrid_arch import mamba2_scan_config
from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestMambaChunkAlignment(CustomTestCase):
    def test_deterministic_alignment_combines_attention_and_scan_grids(self):
        config = GraniteMoeHybridConfig(
            num_hidden_layers=1, layer_types=["mamba"], mamba_chunk_size=256
        )
        cases = (
            (config, "triton", True, 256),
            (config, "flashinfer", True, 4096),
            (config, "triton", False, None),
            (LlamaConfig(), "triton", True, None),
            (MambaConfig(), "triton", True, None),
            (FalconMambaConfig(), "triton", True, None),
            (Lfm2Config(), "triton", True, None),
        )
        for model, backend, deterministic, expected in cases:
            with self.subTest(
                backend=backend, deterministic=deterministic, model=model
            ):
                scheduler = Scheduler.__new__(Scheduler)
                scheduler.model_config = SimpleNamespace(hf_config=model)
                execution = SimpleNamespace(
                    deterministic=SimpleNamespace(
                        enable_deterministic_inference=deterministic
                    )
                )
                with (
                    patch(
                        "sglang.srt.managers.scheduler.get_exec", return_value=execution
                    ),
                    patch(
                        "sglang.srt.managers.scheduler.attention_backends",
                        return_value=(backend, backend),
                    ),
                ):
                    scheduler.init_deterministic_inference_config()
                self.assertEqual(scheduler.truncation_align_size, expected)

    def test_scan_config_excludes_models_that_only_share_the_cache(self):
        for config, expected in (
            (FalconH1Config(), True),
            (NemotronHConfig(), True),
            (Mamba2Config(), True),
            (GraniteMoeHybridConfig(num_hidden_layers=1, layer_types=["mamba"]), True),
            (
                GraniteMoeHybridConfig(num_hidden_layers=1, layer_types=["attention"]),
                False,
            ),
            (MambaConfig(), False),
            (FalconMambaConfig(), False),
            (Lfm2Config(), False),
            (InklingModelConfig(), False),
            (ZayaConfig(), False),
            (LlamaConfig(), False),
        ):
            with self.subTest(model=type(config).__name__):
                model = SimpleNamespace(hf_config=config, is_draft_model=False)
                self.assertEqual(mamba2_scan_config(model) is config, expected)


if __name__ == "__main__":
    unittest.main()
