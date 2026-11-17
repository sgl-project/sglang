"""Unit tests for srt/models/minicpmv.py"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from transformers import Qwen2Config, SiglipVisionConfig

from sglang.srt.models.minicpmv import MiniCPMV
from sglang.srt.runtime_context import get_parallel, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.parallel_groups import publish
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestMiniCPMVVersionDispatch(CustomTestCase):
    def setUp(self):
        super().setUp()
        reset_context()
        self.addCleanup(reset_context)
        # CPU CI does not enable the CPU engine; keep RoPE on the native path.
        self.enterContext(
            patch("sglang.srt.layers.rotary_embedding.base._is_cpu", True)
        )
        publish(ServerArgs(model_path="dummy", device="cuda", tp_size=1), role="test")
        get_parallel().override_permanently(
            pp_group=SimpleNamespace(
                is_first_rank=True, is_last_rank=True, rank_in_group=0, world_size=1
            )
        )

    def test_versionless_config_is_rejected_before_construction(self):
        """A config without ``version`` resolves to a MiniCPM-V generation that
        has no implementation; it is refused up front with the supported list,
        not after a model of another generation was built for it."""
        config = Qwen2Config(
            hidden_size=128,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            vocab_size=64,
            max_position_embeddings=64,
            query_num=4,
            drop_vision_last_layer=False,
            vision_config=SiglipVisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=28,
                patch_size=14,
            ),
        )
        self.assertFalse(hasattr(config, "version"))
        with self.assertRaisesRegex(ValueError, "only supports versions"):
            MiniCPMV(config)


if __name__ == "__main__":
    unittest.main()
