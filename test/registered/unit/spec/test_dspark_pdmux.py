"""DSpark injects layerwise target captures once, on the final prefill slice."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestDSparkPDMux(unittest.TestCase):
    def test_verify_plans_into_selected_decode_backend(self):
        prefill, decode = Mock(), Mock()
        runner = SimpleNamespace(
            attn_backend=prefill,
            get_decode_attn_backend=lambda: decode,
            decode_cuda_graph_runner=None,
        )
        verify = DFlashVerifyInput(torch.tensor([1, 2]), torch.tensor([3, 4]), 2)
        batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)
        persistent = object()
        with (
            patch(
                "sglang.srt.speculative.dflash_info.ForwardBatch.init_new",
                return_value=persistent,
            ),
            patch("sglang.srt.speculative.spec_utils.prepare_mamba_track_for_verify"),
        ):
            result, graph = verify.prepare_for_verify(
                batch, SimpleNamespace(model_runner=runner)
            )
        self.assertIs(result, persistent)
        self.assertFalse(graph)
        decode.init_forward_metadata.assert_called_once_with(persistent)
        prefill.init_forward_metadata.assert_not_called()

    def make_worker(self, idle=False):
        worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        worker.model_runner = SimpleNamespace(
            model_config=SimpleNamespace(num_hidden_layers=2)
        )
        worker._verify_planner = SimpleNamespace(note_non_decode_step=Mock())
        worker._observers = SimpleNamespace(note_prefill_step=Mock())
        worker._finalize_prefill = Mock(return_value="injected")
        worker._decode_idle_result = Mock(return_value="idle")
        batch = SimpleNamespace(
            split_index=0,
            split_forward_batch=SimpleNamespace(split_index=0),
            forward_mode=ForwardMode.IDLE if idle else ForwardMode.SPLIT_PREFILL,
        )
        intermediate = SimpleNamespace(logits_output=None)
        # The reference worker returns final IDLE results with no logits directly.
        final = SimpleNamespace(logits_output=None if idle else object())

        def target(current, **kwargs):
            current.split_forward_batch.split_index += 1
            return (
                intermediate if current.split_forward_batch.split_index == 1 else final
            )

        worker._target_worker = SimpleNamespace(
            forward_batch_split_prefill=Mock(side_effect=target)
        )
        return worker, batch, intermediate, final

    def run_split(self, idle=False):
        worker, batch, intermediate, final = self.make_worker(idle)
        prefill, decode = Mock(), Mock()
        with (
            patch(
                "sglang.srt.multiplex.pdmux_context.get_current_stream_idx",
                return_value=0,
            ),
            patch(
                "sglang.srt.multiplex.pdmux_context.get_stream_groups",
                return_value=[(prefill, decode)],
            ),
        ):
            self.assertIs(worker.forward_batch_split_prefill(batch), intermediate)
            worker._finalize_prefill.assert_not_called()
            prefill.wait_stream.assert_not_called()
            decode.wait_event.assert_not_called()
            batch.split_index = 1
            self.assertEqual(
                worker.forward_batch_split_prefill(batch),
                final if idle else "injected",
            )
        worker._verify_planner.note_non_decode_step.assert_called_once_with()
        worker._observers.note_prefill_step.assert_called_once_with()
        prefill.wait_stream.assert_not_called()
        prefill.record_event.assert_not_called()
        decode.wait_event.assert_not_called()
        self.assertEqual(
            worker.target_worker.forward_batch_split_prefill.call_args.kwargs,
            {"capture_hidden_mode": CaptureHiddenMode.FULL},
        )
        if idle:
            worker._decode_idle_result.assert_not_called()
            worker._finalize_prefill.assert_not_called()
        else:
            worker._finalize_prefill.assert_called_once_with(
                batch, final, on_publish=None
            )

    def test_final_slice_injects_without_extra_stream_fences(self):
        self.run_split()

    def test_final_idle_without_logits_skips_injection(self):
        self.run_split(idle=True)

    def test_stream_switch_updates_target_and_dspark_draft(self):
        worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        worker.model_runner = SimpleNamespace(update_decode_attn_backend=Mock())
        worker.draft_model_runner = SimpleNamespace(update_decode_attn_backend=Mock())
        worker.update_pdmux_decode_attn_backend(2)
        worker.model_runner.update_decode_attn_backend.assert_called_once_with(2)
        worker.draft_model_runner.update_decode_attn_backend.assert_called_once_with(2)

    def test_unadapted_target_is_rejected_before_draft_construction(self):
        target = SimpleNamespace(
            device="cuda", model_runner=SimpleNamespace(model=object())
        )
        with (
            patch(
                "sglang.srt.speculative.dspark_components.dspark_worker_v2.get_disagg",
                return_value=SimpleNamespace(enable_pdmux=True),
            ),
            patch(
                "sglang.srt.speculative.dspark_components.dspark_worker_v2.get_schedule",
                return_value=SimpleNamespace(page_size=256),
            ),
            patch(
                "sglang.srt.speculative.dspark_components.dspark_worker_v2.get_parallel",
                return_value=SimpleNamespace(
                    pp_group=SimpleNamespace(is_last_rank=True)
                ),
            ),
            self.assertRaisesRegex(NotImplementedError, "auxiliary hidden states"),
        ):
            DSparkWorkerV2(SimpleNamespace(), 0, 0, target)

    def test_ordinary_heterogeneous_dp_prefill_coordination_is_preserved(self):
        worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        worker._hosts_draft = True
        worker.enable_dp_spec_prefill_coordination = True
        worker._proposer = SimpleNamespace(query_token_num=2)
        worker.verify_num_draft_tokens = 3
        worker._forward_dp_spec_prefill_coordination = Mock(return_value="coordinated")
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            is_extend_in_batch=True,
            dp_spec_prefill_coordination_metadata=(
                [5, 1],
                [1, 1],
                torch.tensor([True, False]),
            ),
        )
        proxy = object()
        self.assertEqual(
            worker.forward_batch_generation(batch, pp_proxy_tensors=proxy),
            "coordinated",
        )
        call = worker._forward_dp_spec_prefill_coordination.call_args
        self.assertIs(call.args[0], batch)
        self.assertTrue(call.args[1].heterogeneous)
        self.assertIs(call.args[-1], proxy)

    def test_tail_capture_selects_injection_positions_and_swa_owners(self):
        worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        worker._target_hidden_projection_enabled = False
        worker._tp_sync = Mock()
        worker._kv_injector = Mock()
        worker.model_runner = SimpleNamespace(prefill_attention_backend_str="dsv4")
        batch = SimpleNamespace(
            forward_mode=ForwardMode.SPLIT_PREFILL,
            seq_lens=torch.tensor([7, 11]),
            extend_lens=[3, 2],
            prefix_lens=[4, 9],
            req_pool_indices=torch.tensor([6, 8]),
            out_cache_loc=torch.arange(20, 25),
        )
        indices = torch.tensor([1, 2, 4])
        hidden = torch.arange(12).reshape(3, 4).float()
        logits = SimpleNamespace(
            hidden_states=hidden, hidden_states_token_indices=indices
        )
        result = SimpleNamespace(
            logits_output=logits, next_token_ids=torch.tensor([12, 13])
        )
        publish = Mock()
        with (
            patch(
                "sglang.srt.speculative.dspark_components.dspark_worker_v2.is_unified_kv_triton",
                return_value=True,
            ),
            patch(
                "sglang.srt.speculative.dspark_components.dspark_worker_v2.compute_position",
                return_value=(torch.tensor([4, 5, 6, 9, 10]), None),
            ),
        ):
            self.assertIs(worker._finalize_prefill(batch, result, publish), result)
        args = worker._kv_injector.inject_target_hidden.call_args.kwargs
        self.assertIs(args["target_hidden"], hidden)
        torch.testing.assert_close(args["cache_loc"], torch.tensor([21, 22, 24]))
        torch.testing.assert_close(args["positions"], torch.tensor([5, 6, 10]))
        torch.testing.assert_close(args["state_slot"], torch.tensor([6, 6, 8]))
        torch.testing.assert_close(args["final_pos"], torch.tensor([6, 6, 10]))
        self.assertIs(result.new_seq_lens, batch.seq_lens)
        self.assertIs(result.next_draft_input.new_seq_lens, batch.seq_lens)
        publish.assert_called_once_with(batch.seq_lens)
        self.assertIsNone(logits.hidden_states)
        self.assertIsNone(logits.hidden_states_token_indices)


if __name__ == "__main__":
    unittest.main()
