import importlib.util
import sys
import unittest
from pathlib import Path

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.attention_methods.mla_attention import (
    MLAAttentionCase,
    run_mla_attention_case,
)

# tokenspeed_mla is a CuTe DSL backend for Blackwell (SM100). It additionally
# enforces:
#   - data_type == torch.float8_e4m3fn (kv_cache_dtype=fp8_e4m3)
#   - page_size in {32, 64}
# See python/sglang/srt/layers/attention/tokenspeed_mla_backend.py and
# is_tokenspeed_mla_available() in python/sglang/srt/utils/common.py.
#
# The shared MLAAttentionCase fixture (common/attention_methods/mla_attention.py)
# hard-codes kv_cache_dtype="auto" and aligns kv_cache_dtype with model dtype
# (fp16/bf16). Wiring an FP8 KV cache requires fixture changes (separate
# kv_cache_dtype plumbing on MockMLAModelRunner, FP8 KV pool setup, quantize/
# dequantize parity in the reference path) that are out of scope here.
_MIN_SM = 100


def _supported() -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "CUDA is required"
    if importlib.util.find_spec("tokenspeed_mla") is None:
        return False, "tokenspeed_mla python package is not installed"
    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor
    if sm < _MIN_SM:
        return (
            False,
            f"tokenspeed_mla requires SM {_MIN_SM // 10}.{_MIN_SM % 10}+ (Blackwell), "
            f"got SM {major}.{minor}",
        )
    # Even with Blackwell + tokenspeed_mla, the backend rejects construction
    # unless kv_cache_dtype=fp8_e4m3. The current shared MLA fixture does not
    # support FP8 KV cache; flip this once that fixture variant lands.
    return (
        False,
        "tokenspeed_mla requires FP8 KV cache fixture (not in scope): "
        "shared MLAAttentionCase fixture hard-codes kv_cache_dtype=auto and "
        "model dtype for the KV pool; add an fp8_e4m3 KV cache variant before "
        "enabling.",
    )


_SUPPORTED, _SKIP_REASON = _supported()


MLA_SHAPE_KWARGS = dict(
    kv_lora_rank=512,
    qk_rope_head_dim=64,
    hidden_size=1024,
    max_context_len=256,
)


@unittest.skipIf(not _SUPPORTED, _SKIP_REASON)
class TestTokenspeedMLAAttentionBackendCorrectness(CustomTestCase):
    CASES = (
        MLAAttentionCase(
            name="mla_extend_zero_prefix_exact_tokenspeed_page",
            backend="tokenspeed_mla",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=64,
            prefix_lens=(0,),
            extend_lens=(64,),
        ),
    )

    def test_projected_mla_attention_cases(self):
        for case in self.CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_mla_attention_case(self, case, **MLA_SHAPE_KWARGS)


if __name__ == "__main__":
    unittest.main()
