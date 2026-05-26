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

# Cutlass MLA requires Blackwell (SM 10.0). PAGE_SIZE is fixed to 128 in the
# backend (see python/sglang/srt/layers/attention/cutlass_mla_backend.py).
_MIN_SM = 100

MLA_SHAPE_KWARGS = dict(
    kv_lora_rank=512,
    qk_rope_head_dim=64,
    hidden_size=1024,
    max_context_len=256,
)


def _supported() -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "CUDA is required"
    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor
    if sm < _MIN_SM:
        return (
            False,
            f"cutlass_mla requires SM {_MIN_SM // 10}.{_MIN_SM % 10}+, "
            f"got SM {major}.{minor}",
        )
    return True, ""


_SUPPORTED, _SKIP_REASON = _supported()


@unittest.skipIf(not _SUPPORTED, _SKIP_REASON)
class TestCutlassMLAAttentionBackendCorrectness(CustomTestCase):
    CASES = (
        MLAAttentionCase(
            name="mla_extend_zero_prefix_exact_cutlass_page",
            backend="cutlass_mla",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=128,
            prefix_lens=(0,),
            extend_lens=(128,),
        ),
    )

    def test_projected_mla_attention_cases(self):
        for case in self.CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_mla_attention_case(self, case, **MLA_SHAPE_KWARGS)


if __name__ == "__main__":
    unittest.main()
