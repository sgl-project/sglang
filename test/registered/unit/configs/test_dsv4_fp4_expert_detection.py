"""DeepSeek-V4 routed-expert layout detection must work before the weights are
downloaded (ROCM-32676): with a cold Hugging Face cache an FP8 checkpoint was
taken for MXFP4 and crashed in the fused-MoE weight loader."""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_EXPERT_KEY = "model.layers.3.ffn.experts.7.w1.weight"
_INDEX = {
    "weight_map": {
        "model.embed_tokens.weight": "a.safetensors",
        _EXPERT_KEY: "b.safetensors",
    }
}


class TestDsv4Fp4ExpertDetection(CustomTestCase):
    def _detect_cold_cache(self, expert_dtype, *, offline=False):
        from sglang.srt.configs import deepseek_v4
        from sglang.srt.model_loader import weight_utils

        with tempfile.TemporaryDirectory() as tmp:
            index_path = os.path.join(tmp, "model.safetensors.index.json")
            with open(index_path, "w") as f:
                json.dump(_INDEX, f)
            header = SimpleNamespace(
                tensors={_EXPERT_KEY: SimpleNamespace(dtype=expert_dtype)}
            )
            with (
                mock.patch("sglang.srt.utils.find_local_repo_dir", return_value=None),
                mock.patch.object(
                    weight_utils, "hf_hub_download", return_value=index_path
                ) as dl,
                mock.patch.object(
                    weight_utils.huggingface_hub,
                    "parse_safetensors_file_metadata",
                    return_value=header,
                ) as meta,
                mock.patch.object(
                    weight_utils.huggingface_hub.constants, "HF_HUB_OFFLINE", offline
                ),
            ):
                result = deepseek_v4.try_detect_fp4_experts("org/DeepSeek-V4-Flash-FP8")
            return result, dl, meta

    def test_cold_cache_fp8_checkpoint_is_not_fp4(self):
        """Nothing cached: the remote shard header (F8_E4M3) decides, not the mxfp4 default."""
        result, _, meta = self._detect_cold_cache("F8_E4M3")
        self.assertIs(result, False)
        meta.assert_called_once()
        self.assertEqual(meta.call_args.args[1], "b.safetensors")

    def test_cold_cache_mxfp4_checkpoint_is_fp4(self):
        result, _, _ = self._detect_cold_cache("I8")
        self.assertIs(result, True)

    def test_offline_cold_cache_stays_undetected(self):
        """Offline, nothing can be read: None, so the caller keeps the env default."""
        result, dl, meta = self._detect_cold_cache("F8_E4M3", offline=True)
        self.assertIsNone(result)
        dl.assert_not_called()
        meta.assert_not_called()


if __name__ == "__main__":
    unittest.main()
