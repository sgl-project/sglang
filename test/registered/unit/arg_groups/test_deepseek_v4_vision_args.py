"""Configuration limits for atomic DeepSeek-V4 image prefill."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import sglang.srt.arg_groups.deepseek_v4_hook as vision_hook
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, stage="base-a", runner_config="cpu")


class TestVisionConfiguration(CustomTestCase):
    def validate(
        self, *, chunk, page=256, cp=False, vision=True, dynamic=False, hip=False
    ):
        cfg = SimpleNamespace(
            enable_prefill_cp=cp,
            enable_dsa_prefill_context_parallel=False,
            page_size=page,
            chunked_prefill_size=chunk,
            enable_dynamic_chunking=dynamic,
        )
        hf = SimpleNamespace(
            architectures=["DeepseekV4ForCausalLM"],
            vision_n_layers=int(vision),
            vision_max_n_token=384,
        )
        with (
            patch.object(vision_hook, "resolving_view", return_value=cfg),
            patch.object(
                vision_hook,
                "model_config_of",
                return_value=SimpleNamespace(hf_config=hf),
            ),
            patch.object(
                vision_hook, "get_platform", return_value=SimpleNamespace(is_hip=hip)
            ),
        ):
            vision_hook.validate_deepseek_v4_vision(object())

    def test_minimum_chunk_accounts_for_page_alignment(self):
        with self.assertRaisesRegex(ValueError, "at least 768"):
            self.validate(chunk=512)
        self.validate(chunk=768)
        self.validate(chunk=-1)
        self.validate(chunk=384, page=1)

    def test_text_model_is_unaffected(self):
        self.validate(chunk=16, cp=True, vision=False)

    def test_dynamic_budget_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "dynamic chunk sizes"):
            self.validate(chunk=1024, dynamic=True)

    def test_unified_kv_storage_is_rejected(self):
        with (
            patch.object(
                vision_hook.envs.SGLANG_HACK_FLASHMLA_BACKEND,
                "get",
                return_value="unified_kv_triton",
            ),
            self.assertRaisesRegex(ValueError, "complete image blocks"),
        ):
            self.validate(chunk=1024, hip=True)

    def test_unsupported_cp_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "context parallelism"):
            self.validate(chunk=1024, cp=True)


if __name__ == "__main__":
    unittest.main()
