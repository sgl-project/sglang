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

# trtllm_mla goes through FlashInfer's XQA MLA path. Per PLAN.md and the
# project's is_sm120_supported helper (device_capability_majors=[12]), the
# decode path requires SM120a / SM121a (Blackwell variants), i.e. major==12.
# The backend itself has no hard gate — failure surfaces inside FlashInfer at
# kernel-dispatch time — so we mirror is_sm120_supported here.
_REQUIRED_MAJOR = 12

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
    if major != _REQUIRED_MAJOR:
        return (
            False,
            f"trtllm_mla requires SM 12.0a / 12.1a (FlashInfer XQA MLA), "
            f"got SM {major}.{minor}",
        )
    return True, ""


_SUPPORTED, _SKIP_REASON = _supported()


from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b", runner_config="4-gpu-b200")
register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-large")


@unittest.skipIf(not _SUPPORTED, _SKIP_REASON)
class TestTRTLLMMLAAttentionBackendCorrectness(CustomTestCase):
    # trtllm_mla allows page_size in {32, 64} (server_args.py:2790-2794).
    # Cover both, with extend + decode + ragged + page-boundary layouts.
    CASES = (
        # ----- page_size=64 -----
        MLAAttentionCase(
            name="mla_extend_trtllm_zero_prefix_exact_page_64",
            backend="trtllm_mla",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=64,
            prefix_lens=(0,),
            extend_lens=(64,),
        ),
        MLAAttentionCase(
            name="mla_extend_trtllm_zero_prefix_below_page_64",
            backend="trtllm_mla",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=64,
            prefix_lens=(0,),
            extend_lens=(63,),
        ),
        MLAAttentionCase(
            name="mla_extend_trtllm_zero_prefix_above_page_64",
            backend="trtllm_mla",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=64,
            prefix_lens=(0,),
            extend_lens=(65,),
        ),
        MLAAttentionCase(
            name="mla_extend_trtllm_prefix_exact_page_64",
            backend="trtllm_mla",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=64,
            prefix_lens=(64,),
            extend_lens=(4,),
        ),
        MLAAttentionCase(
            name="mla_extend_trtllm_cross_page_boundary_64",
            backend="trtllm_mla",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=64,
            prefix_lens=(63,),
            extend_lens=(2,),
        ),
        MLAAttentionCase(
            name="mla_extend_trtllm_ragged_page_boundary_64",
            backend="trtllm_mla",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=64,
            prefix_lens=(0, 32, 64),
            extend_lens=(63, 32, 1),
        ),
        MLAAttentionCase(
            name="mla_decode_trtllm_page_boundary_64",
            backend="trtllm_mla",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            page_size=64,
            prefix_lens=(62, 63, 64),
        ),
        MLAAttentionCase(
            name="mla_decode_trtllm_bsz1_nonzero_prefix_64",
            backend="trtllm_mla",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            page_size=64,
            prefix_lens=(31,),
        ),
        # ----- page_size=32 -----
        MLAAttentionCase(
            name="mla_extend_trtllm_zero_prefix_exact_page_32",
            backend="trtllm_mla",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=32,
            prefix_lens=(0,),
            extend_lens=(32,),
        ),
        MLAAttentionCase(
            name="mla_extend_trtllm_cross_page_boundary_32",
            backend="trtllm_mla",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=32,
            prefix_lens=(31,),
            extend_lens=(2,),
        ),
        MLAAttentionCase(
            name="mla_decode_trtllm_page_boundary_32",
            backend="trtllm_mla",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            page_size=32,
            prefix_lens=(30, 31, 32),
        ),
    )

    def test_projected_mla_attention_cases(self):
        for case in self.CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_mla_attention_case(self, case, **MLA_SHAPE_KWARGS)


if __name__ == "__main__":
    unittest.main()
