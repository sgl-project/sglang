"""Zero-step DECODE state/seed contracts; GPU serving coverage is separate."""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.speculative.frozen_kv_mtp_worker_v2 import (
    FrozenKVMTPDraftWorker,
    FrozenKVMTPWorkerV2,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")

MODULE = "sglang.srt.speculative.frozen_kv_mtp_worker_v2"


class TestFrozenZeroDecode(unittest.TestCase):
    def test_war_fence_uses_target_for_both_zero_and_positive_steps(self):
        target = object()
        for steps in (0, 2):
            worker = NS(
                speculative_num_steps=steps,
                _target_worker=NS(model_runner=target),
                _draft_worker=NS(draft_runner=object()),
            )
            self.assertIs(
                FrozenKVMTPWorkerV2.last_shared_read_runner.fget(worker), target
            )

    def test_capture_mode_and_fallback_gates(self):
        worker = NS(speculative_num_steps=0)
        batch = NS(
            forward_mode=ForwardMode.DECODE,
            has_grammar=False,
            return_logprob=False,
            hisparse_coordinator=None,
        )
        with patch(
            f"{MODULE}.envs.SGLANG_FROZEN_KV_ZERO_DECODE.get", return_value=True
        ):
            self.assertEqual(
                FrozenKVMTPWorkerV2._target_graph_capture_kwargs(worker, 0),
                dict(
                    capture_forward_mode=ForwardMode.DECODE,
                    capture_hidden_mode=CaptureHiddenMode.FULL,
                ),
            )
            self.assertEqual(
                FrozenKVMTPWorkerV2._target_graph_capture_kwargs(worker, 2), {}
            )
            self.assertTrue(FrozenKVMTPWorkerV2._use_zero_decode(worker, batch))
            for key, value in (
                ("has_grammar", True),
                ("return_logprob", True),
                ("hisparse_coordinator", object()),
                ("forward_mode", ForwardMode.IDLE),
            ):
                old = getattr(batch, key)
                setattr(batch, key, value)
                self.assertFalse(FrozenKVMTPWorkerV2._use_zero_decode(worker, batch))
                setattr(batch, key, old)
        with patch(
            f"{MODULE}.envs.SGLANG_FROZEN_KV_ZERO_DECODE.get", return_value=False
        ):
            self.assertFalse(FrozenKVMTPWorkerV2._use_zero_decode(worker, batch))
            self.assertEqual(
                FrozenKVMTPWorkerV2._target_graph_capture_kwargs(worker, 0), {}
            )

    def test_decode_graph_cannot_replay_verify_even_at_width_one(self):
        runner = NS(
            capture_forward_mode_override=ForwardMode.DECODE,
            capture_forward_mode=ForwardMode.DECODE,
        )
        self.assertFalse(
            DecodeCudaGraphRunner.can_run_graph(
                runner, NS(forward_mode=ForwardMode.TARGET_VERIFY)
            )
        )
        self.assertIsNone(DecodeCudaGraphRunner.get_spec_info(runner, 8))

    def test_zero_round_preserves_frontier_and_owns_seed_hidden(self):
        for cpu_lengths in (True, False):
            with self.subTest(cpu_lengths=cpu_lengths):
                self.check_two_rounds(cpu_lengths)

    def check_two_rounds(self, cpu_lengths):
        hidden_buffer = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        seed_builder = NS(topk=1)
        seed_builder._build_seed_draft_input = lambda tokens, hidden: (
            FrozenKVMTPDraftWorker._build_seed_draft_input(seed_builder, tokens, hidden)
        )
        target = NS(model_runner=NS())
        seen = []

        def forward(*, batch, forward_batch):
            self.assertIsNone(batch)
            self.assertIsNone(forward_batch.spec_info)
            self.assertEqual(forward_batch.forward_mode, ForwardMode.DECODE)
            seen.append(forward_batch)
            return GenerationBatchResult(
                next_token_ids=forward_batch.input_ids + 10,
                logits_output=NS(hidden_states=hidden_buffer),
                can_run_cuda_graph=True,
            )

        target.forward_batch_generation = forward
        worker = NS(
            device="cpu",
            req_to_token_pool=NS(req_to_token=object()),
            _target_worker=target,
            draft_worker=seed_builder,
        )
        old_seed = NS(bonus_tokens=torch.tensor([5, 6, 7]))
        batch = NS(
            forward_mode=ForwardMode.DECODE,
            seq_lens=torch.tensor([1023, 1024, 2047], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([1023, 1024, 2047]) if cpu_lengths else None,
            req_pool_indices=torch.tensor([4, 2, 9]),
            spec_info=old_seed,
        )
        original_lengths = batch.seq_lens.clone()
        publish = Mock()
        locations = Mock(return_value=torch.tensor([31, 32, 63]))
        with (
            patch(f"{MODULE}.assign_extend_cache_locs_uniform_func", locations),
            patch(
                f"{MODULE}.torch.get_device_module",
                return_value=NS(current_stream=lambda: None),
            ),
            patch(f"{MODULE}.record_stream_each"),
            patch(f"{MODULE}.ForwardBatch.init_new", side_effect=lambda b, *a, **kw: b),
        ):
            result = FrozenKVMTPWorkerV2._forward_zero_decode(worker, batch, publish)
            torch.testing.assert_close(batch.seq_lens, original_lengths)
            self.assertIs(batch.spec_info, old_seed)
            self.assertIs(locations.call_args.kwargs["start_offset"], batch.seq_lens)
            self.assertEqual(locations.call_args.kwargs["draft_token_num"], 1)
            torch.testing.assert_close(result.new_seq_lens, original_lengths + 1)
            self.assertIs(publish.call_args.args[0], result.new_seq_lens)
            self.assertIs(result.extra_keep_alive_refs[0], seen[0])
            self.assertEqual(result.get_num_proposed_drafts(3), 0)
            self.assertEqual(result.accept_lens.tolist(), [1, 1, 1])
            self.assertEqual(
                result.next_draft_input.bonus_tokens.tolist(), [15, 16, 17]
            )
            first_hidden = result.next_draft_input.hidden_states.clone()
            # Simulate graph replay overwriting the shared model output.
            hidden_buffer.add_(100)
            torch.testing.assert_close(
                result.next_draft_input.hidden_states, first_hidden
            )
            batch.seq_lens = result.new_seq_lens
            batch.spec_info = result.next_draft_input
            if cpu_lengths:
                batch.seq_lens_cpu += 1
            second = FrozenKVMTPWorkerV2._forward_zero_decode(worker, batch)
            self.assertEqual(
                second.next_draft_input.bonus_tokens.tolist(), [25, 26, 27]
            )
            torch.testing.assert_close(
                second.next_draft_input.hidden_states, first_hidden + 100
            )
            torch.testing.assert_close(second.new_seq_lens, original_lengths + 2)
            self.assertEqual(second.next_draft_input.topk_p.shape, (3, 1))
            if cpu_lengths:
                self.assertEqual(seen[-1].seq_lens_cpu.tolist(), [1025, 1026, 2049])
            else:
                self.assertIsNone(seen[-1].seq_lens_cpu)


if __name__ == "__main__":
    unittest.main()
