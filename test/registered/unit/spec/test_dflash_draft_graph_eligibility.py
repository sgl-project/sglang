"""The scheduler's graph admission must reach the DFLASH draft runner."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative import dflash_worker_v2
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.speculative.draft_worker_common import make_draft_block_spec_info
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDFlashDraftGraphEligibility(unittest.TestCase):
    def test_scheduler_admission_reaches_the_draft_forward_batch(self):
        worker = object.__new__(DFlashWorkerV2)
        worker.device = "cpu"
        worker.block_size = 2
        worker.draft_model = SimpleNamespace()
        embedding = torch.nn.Embedding(16, 4)
        target_model = SimpleNamespace(
            get_input_embeddings=lambda: embedding, lm_head=torch.nn.Linear(4, 16)
        )
        iteration_ids = SimpleNamespace(bind=Mock())
        target_translator = SimpleNamespace(plan=Mock(return_value=iteration_ids))
        draft_translator = SimpleNamespace()
        worker.model_runner = SimpleNamespace(
            runner_ids=target_translator,
            model=target_model,
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.arange(16).view(1, 16)
            ),
        )
        worker._target_worker = SimpleNamespace(model_runner=worker.model_runner)
        # Stop at the actual runner boundary, before target verification.
        worker.draft_model_runner = SimpleNamespace(
            runner_ids=draft_translator,
            forward=Mock(side_effect=StopIteration),
        )
        worker.draft_owns_attention = False
        worker._draft_block_ids_buf = None
        worker._draft_block_spec_info = make_draft_block_spec_info(
            draft_token_num=2, device="cpu"
        )
        worker._block_pos_offsets = torch.arange(2)
        worker._use_triton_prepare_block = False
        worker._mask_token_id = 2
        worker._full_embed_gpu = None
        worker._noise_embed_scale = 1.0
        worker.use_compact_draft_cache = False
        worker.selector = None
        worker.lilicorr = None

        draft_input = DFlashDraftInputV2.create_idle_input(device="cpu")
        draft_input.bonus_tokens = torch.tensor([1])
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            is_extend_in_batch=False,
            sampling_info=None,
            spec_info=draft_input,
            seq_lens=torch.tensor([3]),
            seq_lens_cpu=torch.tensor([3]),
            req_pool_indices=torch.tensor([0]),
        )
        for allowed in (False, True):
            batch.can_run_decode_cuda_graph = allowed
            with (
                self.subTest(allowed=allowed),
                patch.object(
                    dflash_worker_v2, "draft_tp_context", return_value=nullcontext()
                ),
                patch.object(torch.Tensor, "record_stream"),
                patch.object(torch, "get_device_module", return_value=Mock()),
                patch.object(
                    dflash_worker_v2,
                    "assign_extend_cache_locs_func",
                    return_value=torch.tensor([3, 4]),
                ),
                patch.object(
                    dflash_worker_v2, "enable_num_token_non_padded", return_value=False
                ),
                self.assertRaises(StopIteration),
            ):
                worker.forward_batch_generation(batch)
            forward_batch = worker.draft_model_runner.forward.call_args.args[0]
            iteration_ids.bind.assert_called_with(forward_batch, draft_translator)
            self.assertIs(forward_batch.can_run_decode_cuda_graph, allowed)
            self.assertEqual(forward_batch.forward_mode, ForwardMode.TARGET_VERIFY)
            torch.testing.assert_close(forward_batch.input_ids, torch.tensor([1, 2]))


if __name__ == "__main__":
    unittest.main()
