"""Compare reduced DFlash verification blocks with the sparse attention reference."""

import unittest

import torch

from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.dsa_attention import (
    DSA_PAGE_SIZE,
    DSA_SPARSE_ATOL,
    DSA_SPARSE_FP8_ATOL,
    DSA_SPARSE_FP8_RTOL,
    DSA_SPARSE_RTOL,
    DSAAttentionCase,
    build_dsa_sparse_attention_fixture,
    expected_dsa_sparse_fixture_output,
    run_dsa_sparse_fixture_eager,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="4-gpu-b200")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestDSADFlashVerifyWidth(CustomTestCase):
    def test_eager_verification_with_a_reduced_block(self):
        for fp8_kv in (False, True):
            for width in (1, 12):
                with self.subTest(fp8_kv=fp8_kv, width=width):
                    case = DSAAttentionCase(
                        name=f"dflash_live_width_{width}",
                        backend="dsa",
                        forward_mode=ForwardMode.TARGET_VERIFY,
                        num_heads=4,
                        num_kv_heads=1,
                        page_size=DSA_PAGE_SIZE,
                        prefix_lens=(2048, 4096),
                        extend_lens=(width, width),
                    )
                    fixture = build_dsa_sparse_attention_fixture(
                        self, case, fp8_kv_cache=fp8_kv
                    )
                    self.addCleanup(fixture.runner._server_args_override.restore)
                    get_context().override(
                        "dflash_live_verify_width",
                        speculative_algorithm="DFLASH",
                        speculative_num_draft_tokens=12,
                    )
                    fixture.backend.speculative_num_draft_tokens = 12
                    batch = fixture.forward_batch
                    # TARGET_VERIFY carries prefix lengths; the backend adds
                    # the live query block, whose cache slots the kit populated.
                    batch.seq_lens_cpu = torch.tensor(
                        case.prefix_lens, dtype=torch.int32
                    )
                    batch.seq_lens = batch.seq_lens_cpu.to(batch.input_ids.device)
                    batch.seq_lens_sum = sum(case.prefix_lens)
                    fixture.runner.runner_ids.bind_own_plan(batch)
                    batch.spec_algorithm = SpeculativeAlgorithm.DFLASH
                    batch.spec_info = DFlashVerifyInput(
                        draft_token=batch.input_ids,
                        positions=batch.positions,
                        draft_token_num=width,
                    )
                    actual = run_dsa_sparse_fixture_eager(fixture, self)
                    expected = expected_dsa_sparse_fixture_output(fixture)
                    torch.testing.assert_close(
                        actual,
                        expected,
                        atol=DSA_SPARSE_FP8_ATOL if fp8_kv else DSA_SPARSE_ATOL,
                        rtol=DSA_SPARSE_FP8_RTOL if fp8_kv else DSA_SPARSE_RTOL,
                    )
                    self.assertEqual(fixture.backend.speculative_num_draft_tokens, 12)


if __name__ == "__main__":
    unittest.main()
