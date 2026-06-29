import sys
import unittest
from pathlib import Path

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sglang.test.kits.attention_unittest.attention_methods.mla_attention import (
    MLAAttentionCase,
    run_mla_attention_case,
)

# Cutlass MLA requires exactly Blackwell SM 10.0. The sgl-kernel
# `cutlass_mla_decode` checks `sm_version == 100` (major*10+minor), so
# SM 10.3 (GB300) reports sm_version=103 and is rejected by the kernel.
# PAGE_SIZE is fixed to 128 in the backend.
_REQUIRED_SM_MAJOR = 10
_REQUIRED_SM_MINOR = 0

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
    if major != _REQUIRED_SM_MAJOR or minor != _REQUIRED_SM_MINOR:
        return (
            False,
            f"cutlass_mla requires exactly SM {_REQUIRED_SM_MAJOR}.{_REQUIRED_SM_MINOR} "
            f"(B200 Blackwell); got SM {major}.{minor}",
        )
    return True, ""


_SUPPORTED, _SKIP_REASON = _supported()


from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="4-gpu-b200")
register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")


@unittest.skipIf(not _SUPPORTED, _SKIP_REASON)
class TestCutlassMLAAttentionBackendCorrectness(CustomTestCase):
    # CutlassMLABackend only overrides `forward_decode`; EXTEND falls through
    # to the FlashInferMLAAttnBackend parent and bypasses cutlass code
    # entirely. Use DECODE so the test actually exercises the cutlass kernel
    # on Blackwell. Page size is fixed to PAGE_SIZE=128 (server_args.py
    # forces this for cutlass_mla).
    CASES = (
        MLAAttentionCase(
            name="mla_decode_cutlass_page_boundary",
            backend="cutlass_mla",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            page_size=128,
            prefix_lens=(126, 127, 128),
        ),
        MLAAttentionCase(
            name="mla_decode_cutlass_bsz1_nonzero_prefix",
            backend="cutlass_mla",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            page_size=128,
            prefix_lens=(63,),
        ),
        MLAAttentionCase(
            name="mla_decode_cutlass_above_page",
            backend="cutlass_mla",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            page_size=128,
            prefix_lens=(128, 129, 130),
        ),
        MLAAttentionCase(
            name="mla_decode_cutlass_multi_page",
            backend="cutlass_mla",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            page_size=128,
            prefix_lens=(127, 200, 255),
        ),
    )

    def test_projected_mla_attention_cases(self):
        for case in self.CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_mla_attention_case(self, case, **MLA_SHAPE_KWARGS)


if __name__ == "__main__":
    unittest.main()
