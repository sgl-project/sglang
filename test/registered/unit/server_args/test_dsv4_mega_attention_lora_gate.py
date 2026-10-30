"""DeepSeek-V4.1 keeps LoRA off the FlashMLA MegaAttention backend.

BUG REGRESSION. MegaAttention computes wo_a inside its kernel from a copy of
the base weight and reads a wo_b attribute that LoRA wrappers do not carry.
`auto` still picked it on SM100 with fp8 KV and no attention TP when LoRA was
on, so a wo_b adapter crashed the first decode and a wo_a adapter was silently
dropped.

    python -m pytest test/registered/unit/server_args/test_dsv4_mega_attention_lora_gate.py -v
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import msgspec

from sglang.srt.arg_groups import deepseek_v4_hook
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

_ENV_GATE = "sglang.srt.layers.attention.dsv4.unified_kv_gate"


def _v41_args(**fields):
    sa = ServerArgs(model_path="local-deepseek-v41", attention_backend="dsv4")
    sa._model_config = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="deepseek_v41")
    )
    for name, value in fields.items():
        msgspec.Struct.__setattr__(sa, name, value)
    return sa


def _auto_backend(**fields):
    sa = _v41_args(
        device="cuda", kv_cache_dtype="fp8_e4m3", disaggregation_mode="null", **fields
    )
    with (
        patch("sglang.srt.utils.common.is_sm100_supported", return_value=True),
        patch.object(
            deepseek_v4_hook, "_attention_is_tensor_parallel", return_value=False
        ),
    ):
        return deepseek_v4_hook._default_dsv4_attention_backend(sa)


class TestMegaAttentionLoraGate(unittest.TestCase):
    def test_auto_keeps_lora_on_flashmla(self):
        self.assertEqual(_auto_backend(enable_lora=True), "flashmla")
        # --lora-paths has not been turned into enable_lora yet at this point
        self.assertEqual(
            _auto_backend(enable_lora=None, lora_paths=["a=/tmp/a"]), "flashmla"
        )

    def test_auto_still_picks_mega_without_lora(self):
        """An always-true LoRA predicate would drop MegaAttention for every launch."""
        self.assertEqual(
            _auto_backend(enable_lora=None, lora_paths=None), "flashmla_mega"
        )
        self.assertEqual(
            _auto_backend(enable_lora=False, lora_paths=["a=/tmp/a"]),
            "flashmla_mega",
        )

    def test_explicit_mega_with_lora_is_refused(self):
        sa = _v41_args(dsv4_attn_backend="flashmla_mega", enable_lora=True)
        with (
            patch(f"{_ENV_GATE}.is_unified_kv_triton", return_value=False),
            self.assertRaisesRegex(
                ValueError, "LoRA with --dsv4-attn-backend flashmla_mega"
            ),
        ):
            deepseek_v4_hook.validate_deepseek_v41_features(sa)


if __name__ == "__main__":
    unittest.main()
