import unittest

import torch

from sglang.kernels.ops.attention.dsv4.candidate_blocks import candidate_block_logits
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4.indexer import select_candidate_blocks
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _deepselect_sm90_available() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        return False
    try:
        import deep_select  # noqa: F401
    except ImportError:
        return False
    return True


@unittest.skipUnless(
    _deepselect_sm90_available(), "requires the optional DeepSelect SM90 package"
)
class TestDeepSelectCandidateBlocks(CustomTestCase):
    def test_candidate_publication_matches_torch(self):
        torch.manual_seed(913)
        rows, width, block_size, topk_blocks = 6, 131079, 8, 513
        logits = torch.randn(rows, width + 11, device="cuda")[:, :width]
        lengths = torch.tensor(
            [0, 1, 17, width // 2, width - 3, width],
            device="cuda",
            dtype=torch.int32,
        )

        with envs.SGLANG_OPT_DSV41_DEEPSELECT_CANDIDATE_TOPK.override(False):
            ref_logits, ref_keep = candidate_block_logits(
                logits,
                lengths,
                topk_blocks=topk_blocks,
                block_size=block_size,
                published=None,
            )
        with envs.SGLANG_OPT_DSV41_DEEPSELECT_CANDIDATE_TOPK.override(True):
            got_logits, got_keep = candidate_block_logits(
                logits,
                lengths,
                topk_blocks=topk_blocks,
                block_size=block_size,
                published=None,
            )

        torch.testing.assert_close(got_logits, ref_logits, rtol=0, atol=0)
        torch.testing.assert_close(got_keep, ref_keep, rtol=0, atol=0)
        with envs.SGLANG_OPT_DSV41_DEEPSELECT_CANDIDATE_TOPK.override(True):
            selector_keep = select_candidate_blocks(
                ref_logits,
                lengths[:, None],
                topk_blocks=topk_blocks,
                block_size=block_size,
            )
        torch.testing.assert_close(selector_keep, ref_keep, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
