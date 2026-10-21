"""XPU platform-gate coverage for the Qwen4-Exp compressed QSA indexer dtype.

Mirrors test/registered/unit/test_model_overrides.py's golden-resolution
fixture (``_construct``/``_resolved``/``_IsolatedPublish``), scoped to the
``--qsa-indexer-dtype fp8_e4m3`` + XPU platform branch: the CUDA SM90/SM100
and non-accelerated fallback branches stay in test_model_overrides.py.
"""

import json
import os
import shutil
import tempfile

from sglang.srt.runtime_context import get_context, override_platform, reset_context
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=5, suite="stage-a-test-1-gpu-xpu")


class TestQwen4FP8IndexerDtypeXpuGate(CustomTestCase):
    """``--qsa-indexer-dtype fp8_e4m3`` also runs on XPU (not just CUDA
    SM90/SM100), provided the compressed QSA indexer is configured."""

    _MINI_CONFIG = {
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "vocab_size": 512,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-5,
        "torch_dtype": "bfloat16",
        "kv_lora_rank": 32,
        "qk_nope_head_dim": 16,
        "qk_rope_head_dim": 8,
        "v_head_dim": 16,
    }

    def setUp(self):
        super().setUp()
        self._saved_server_args = get_context()._server_args

    def tearDown(self):
        reset_context()
        if self._saved_server_args is not None:
            get_context()._server_args = self._saved_server_args
        super().tearDown()

    def _resolved(self, server_args, field):
        from sglang.srt.arg_groups.overrides import resolution_result

        return resolution_result(server_args, field)

    def _construct(self, arch, model_type, config_extra=None, **server_kwargs):
        from sglang.srt.server_args import ServerArgs

        # Golden resolution must be host-independent: accelerator-less CI
        # runners resolve only the base platform, where get_device() raises.
        server_kwargs.setdefault("device", "cuda")
        config = dict(self._MINI_CONFIG, architectures=[arch], model_type=model_type)
        config.update(config_extra or {})
        config_dir = tempfile.mkdtemp(prefix="golden_override_")
        self.addCleanup(shutil.rmtree, config_dir, ignore_errors=True)
        with open(os.path.join(config_dir, "config.json"), "w") as f:
            json.dump(config, f)
        server_args = ServerArgs(model_path=config_dir, **server_kwargs)
        server_args.resolve_once()
        return server_args

    def test_qwen4_fp8_indexer_dtype_xpu_platform_gate(self):
        qwen4 = ("Qwen4ExpForConditionalGeneration", "qwen4_exp")
        compressed = {
            "indexer_n_heads": 4,
            "indexer_kv_heads": 1,
            "indexer_head_dim": 128,
            "indexer_budget": 2048,
            "indexer_compress_ratio": 4,
        }
        with override_platform(
            is_cuda=False, is_xpu=True, is_sm90=False, is_sm100=False
        ):
            # Compressed addressing forces page_size=64; XPU does not support
            # the mamba-cache extra_buffer strategy, so no_buffer's
            # page_size=1 requirement must be sidestepped explicitly here.
            sa = self._construct(
                *qwen4,
                config_extra=compressed,
                qsa_indexer_dtype="fp8_e4m3",
                disable_radix_cache=True,
            )
            self.assertEqual(self._resolved(sa, "qsa_indexer_dtype"), "fp8_e4m3")
            # No compressed indexer fields: no QSA profile, nothing to store in fp8.
            with self.assertRaisesRegex(ValueError, "compressed QSA indexer"):
                self._construct(*qwen4, qsa_indexer_dtype="fp8_e4m3")


if __name__ == "__main__":
    import unittest

    unittest.main()
