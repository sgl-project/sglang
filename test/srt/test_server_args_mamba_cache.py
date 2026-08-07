import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.overrides import _mamba_radix_cache_resolution


class TestMambaCacheServerArgs(unittest.TestCase):
    def _make_view(self, arch):
        return SimpleNamespace(
            page_size=128,
            disable_radix_cache=False,
            disable_overlap_schedule=False,
            mamba_radix_cache_strategy="no_buffer",
            attention_backend="ascend",
            linear_attn_backend="triton",
            get_model_config=lambda: SimpleNamespace(
                hf_config=SimpleNamespace(architectures=[arch])
            ),
        )

    @patch("sglang.srt.arg_groups.overrides.is_npu", return_value=True)
    def test_qwen35_text_arches_handle_mamba_cache_page_size(self, _mock_npu):
        for arch in [
            "Qwen3_5ForCausalLM",
            "Qwen3_5MoeForCausalLM",
            "Qwen3_5ForCausalLMMTP",
            "Qwen3_5ForConditionalGeneration",
            "Qwen3_5MoeForConditionalGeneration",
        ]:
            with self.subTest(arch=arch):
                declared = _mamba_radix_cache_resolution(self._make_view(arch))

                self.assertTrue(declared["uses_mamba_radix_cache"])
                self.assertEqual(declared["mamba_radix_cache_strategy"], "extra_buffer")


if __name__ == "__main__":
    unittest.main()
