import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.kernels.ops.speculative.draft_extend import fused_draft_extend_prolog
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.eagle_info import EagleDraftExtendInput
from sglang.srt.speculative.eagle_worker_common import prepare_for_draft_extend
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=5, suite="jit-kernel-unit-test-amd")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
class TestDraftExtendProlog(CustomTestCase):
    def test_matches_eager_metadata_construction(self):
        cases = [
            ([2048], 6, 0, False, torch.int64),
            ([0, 5, 17], 1, 0, False, torch.int32),
            ([0, 2, 9], 6, 4, True, torch.int64),
            ([], 6, 0, False, torch.int64),
        ]
        for (
            seq_len_values,
            num_draft_tokens,
            front_offset,
            use_positions,
            seq_lens_dtype,
        ) in cases:
            with self.subTest(
                seq_len_values=seq_len_values,
                num_draft_tokens=num_draft_tokens,
                front_offset=front_offset,
                use_positions=use_positions,
                seq_lens_dtype=seq_lens_dtype,
            ):
                seq_lens = torch.tensor(
                    seq_len_values, dtype=seq_lens_dtype, device="cuda"
                )
                original_seq_lens = seq_lens.clone()
                window_size = num_draft_tokens + front_offset
                supplied_positions = (
                    torch.full(
                        (len(seq_len_values) * window_size,),
                        -7,
                        dtype=torch.int64,
                        device="cuda",
                    )
                    if use_positions
                    else None
                )
                expected_supplied_positions = (
                    supplied_positions.clone()
                    if supplied_positions is not None
                    else None
                )

                output = fused_draft_extend_prolog(
                    seq_lens,
                    num_draft_tokens,
                    front_offset=front_offset,
                    positions=supplied_positions,
                )

                expected_prefix = torch.clamp(seq_lens - front_offset, min=0).to(
                    torch.int32
                )
                expected_extend = torch.full_like(
                    expected_prefix, window_size, dtype=torch.int32
                )
                expected_start = (
                    torch.arange(len(seq_len_values), dtype=torch.int32, device="cuda")
                    * window_size
                )
                if supplied_positions is not None:
                    expected_positions = expected_supplied_positions
                else:
                    expected_positions = (
                        expected_prefix.to(torch.int64)[:, None]
                        + torch.arange(window_size, dtype=torch.int64, device="cuda")[
                            None, :
                        ]
                    ).reshape(-1)

                torch.testing.assert_close(output.prefix_lens, expected_prefix)
                torch.testing.assert_close(output.extend_seq_lens, expected_extend)
                torch.testing.assert_close(output.extend_start_loc, expected_start)
                torch.testing.assert_close(output.positions, expected_positions)
                torch.testing.assert_close(
                    output.post_extend_seq_lens, seq_lens + num_draft_tokens
                )
                torch.testing.assert_close(seq_lens, original_seq_lens)
                if supplied_positions is not None:
                    self.assertEqual(
                        output.positions.data_ptr(), supplied_positions.data_ptr()
                    )

    def test_prepare_adopts_atomic_layout_and_skips_position_kernel(self):
        seq_lens = torch.tensor([19, 37], dtype=torch.int64, device="cuda")
        original_seq_lens = seq_lens.clone()
        batch = MagicMock()
        batch.forward_mode = ForwardMode.DECODE
        batch.seq_lens = seq_lens
        batch.seq_lens_cpu = None
        batch.seq_lens_sum = None
        batch.input_ids = torch.empty(0, dtype=torch.int64, device="cuda")
        batch.req_pool_indices = torch.arange(2, device="cuda")
        batch.out_cache_loc = torch.arange(12, device="cuda")
        batch.reqs = []
        batch.sampling_info = None
        batch.return_hidden_states = False
        batch.is_prefill_only = False
        batch.extend_input_logprob_token_ids = None
        batch.global_num_tokens = None
        batch.dllm_config = None
        batch.model_config.vocab_size = 1024

        model_runner = MagicMock()
        model_runner.device = torch.device("cuda")
        model_runner.spec_algorithm.is_standalone.return_value = True
        model_runner.server_args.attention_backend = "triton"
        model_runner.server_args.enable_lora = False
        model_runner.ngram_embedding_manager.enabled = False
        model_runner.model_config.model_is_mrope = False
        model_runner.dcp_size = 1
        cuda_graph_runner = MagicMock()
        cuda_graph_runner.can_run_graph.return_value = True
        spec_info = EagleDraftExtendInput()
        predict = torch.arange(12, dtype=torch.int64, device="cuda")

        with (
            patch(
                "sglang.srt.model_executor.forward_batch_info.compute_position",
                side_effect=AssertionError("position kernel must be skipped"),
            ),
            patch(
                "sglang.srt.model_executor.forward_batch_info.enable_num_token_non_padded",
                return_value=False,
            ),
            patch("sglang.srt.speculative.eagle_worker_common.maybe_detect_oob"),
        ):
            forward_batch = prepare_for_draft_extend(
                spec_info,
                batch,
                predict,
                6,
                model_runner,
                cuda_graph_runner,
                return_hidden_states_before_norm=False,
            )

        layout = spec_info.precomputed_extend_layout
        self.assertIsNotNone(layout)
        self.assertEqual(spec_info.positions.data_ptr(), layout.positions.data_ptr())
        self.assertEqual(
            forward_batch.positions.data_ptr(), layout.positions.data_ptr()
        )
        self.assertEqual(
            forward_batch.extend_start_loc.data_ptr(),
            layout.extend_start_loc.data_ptr(),
        )
        torch.testing.assert_close(
            forward_batch.extend_prefix_lens, original_seq_lens.to(torch.int32)
        )
        torch.testing.assert_close(
            forward_batch.extend_seq_lens,
            torch.full((2,), 6, dtype=torch.int32, device="cuda"),
        )
        torch.testing.assert_close(batch.seq_lens, original_seq_lens)
        torch.testing.assert_close(forward_batch.seq_lens, original_seq_lens + 6)


if __name__ == "__main__":
    unittest.main()
