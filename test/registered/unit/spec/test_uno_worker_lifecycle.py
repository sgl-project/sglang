"""UNO's optional dependencies and shared-model LoRA state boundaries."""

import runpy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.speculative import uno_info
from sglang.srt.speculative.uno_cuda_graph_runner import UnoDecodeCudaGraphRunner
from sglang.srt.speculative.uno_info import UnoDraftInput
from sglang.srt.speculative.uno_worker_v2 import UnoWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestUnoWorkerLifecycle(CustomTestCase):
    def test_linear_worker_import_does_not_require_cuda_tree_module(self):
        worker_path = Path(uno_info.__file__).with_name("uno_worker_v2.py")
        with patch.dict(sys.modules, {"sglang.srt.speculative.uno_tree": None}):
            namespace = runpy.run_path(str(worker_path))
        self.assertIn("UnoWorkerV2", namespace)

    @staticmethod
    def _worker():
        worker = UnoWorkerV2.__new__(UnoWorkerV2)
        manager = SimpleNamespace(batch_info=None)
        manager.reset_lora_batch = Mock(
            side_effect=lambda: setattr(manager, "batch_info", None)
        )
        manager.prepare_lora_token_segments = Mock(
            side_effect=lambda **kwargs: setattr(manager, "batch_info", kwargs)
        )
        worker.lora_manager = manager
        worker.model_runner = SimpleNamespace(decode_cuda_graph_runner=None)
        worker.tree_mode = False
        worker.num_speculative_proposals = 3
        worker.uno_lora_id = "uno_adapter"
        return worker

    def test_eager_draft_routes_rows_then_clears_adapter(self):
        worker = self._worker()
        batch = SimpleNamespace(
            batch_size=2,
            forward_mode=SimpleNamespace(is_cuda_graph=lambda: False),
        )
        expected = object()

        def forward(*args, **kwargs):
            self.assertEqual(
                worker.lora_manager.batch_info,
                {
                    "lora_ids": [None, "uno_adapter", None, "uno_adapter"],
                    "segment_lens": [1, 3, 1, 3],
                },
            )
            return expected

        worker._run_target_block = Mock(side_effect=forward)
        self.assertIs(worker._run_draft_block(batch), expected)
        self.assertIsNone(worker.lora_manager.batch_info)

    def test_failed_draft_and_routing_setup_clear_adapter(self):
        for failure in ("routing", "forward"):
            with self.subTest(failure=failure):
                worker = self._worker()
                batch = SimpleNamespace(
                    batch_size=1,
                    forward_mode=SimpleNamespace(is_cuda_graph=lambda: False),
                )

                def fail(*args, **kwargs):
                    worker.lora_manager.batch_info = "partially prepared draft"
                    raise RuntimeError("draft failed")

                worker._run_target_block = Mock(side_effect=fail)
                if failure == "routing":
                    worker.lora_manager.prepare_lora_token_segments.side_effect = fail
                with self.assertRaisesRegex(RuntimeError, "draft failed"):
                    worker._run_draft_block(batch)
                self.assertIsNone(worker.lora_manager.batch_info)

    def test_graph_draft_reuses_captured_routing(self):
        worker = self._worker()
        worker.model_runner.decode_cuda_graph_runner = SimpleNamespace(
            can_run_graph=lambda batch: True
        )
        batch = SimpleNamespace(
            batch_size=2,
            forward_mode=SimpleNamespace(is_cuda_graph=lambda: True),
        )
        worker._run_target_block = Mock(return_value="graph result")
        self.assertEqual(worker._run_draft_block(batch), "graph result")
        worker.lora_manager.prepare_lora_token_segments.assert_not_called()
        self.assertIsNone(worker.lora_manager.batch_info)

    def test_greedy_decode_preserves_ar_tokens_and_committed_kv(self):
        # Model each KV row as an integer contribution to a causal toy model.
        # Adapted draft rows deliberately dirty KV; verification must overwrite
        # every committed row and leave only the final seed without clean KV.
        vocab, capacity, batch_size = 19, 128, 2

        def next_token(prefix):
            weights = torch.arange(1, prefix.numel() + 1)
            return int((prefix * weights).sum() + 3) % vocab

        for width in (1, 4):
            for accepted_proposals in range(width):
                with self.subTest(width=width, accepted=accepted_proposals):
                    worker = self._worker()
                    worker.device = "cpu"
                    worker.forward_width = width
                    worker.tail_width = width + 1
                    worker.num_speculative_proposals = width - 1
                    worker._tail_offsets = torch.arange(width + 1)
                    table = torch.arange(batch_size * capacity).reshape(
                        batch_size, capacity
                    )
                    worker.model_runner.req_to_token_pool = SimpleNamespace(
                        req_to_token=table
                    )
                    worker.model_runner.model_config = SimpleNamespace(vocab_size=vocab)
                    worker.model_runner.kv_index_translator = SimpleNamespace(
                        bind_own_plan=Mock()
                    )
                    kv = torch.full((batch_size * capacity,), -1, dtype=torch.int64)
                    histories = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6, 7])]
                    for row, history in enumerate(histories):
                        kv[table[row, : len(history)]] = history
                    seed = torch.tensor([next_token(history) for history in histories])
                    lengths = torch.tensor([len(history) for history in histories])
                    state = UnoDraftInput(seed, lengths.clone(), width)
                    batch = SimpleNamespace(
                        seq_lens=lengths.clone(),
                        seq_lens_cpu=lengths.clone(),
                        spec_info=state,
                        req_pool_indices=torch.arange(batch_size),
                        sampling_info=SimpleNamespace(is_all_greedy=True),
                    )
                    gold = []

                    def forward(*, batch, forward_batch, is_verify):
                        fb = forward_batch
                        logits = torch.full((batch_size * width, vocab), -100.0)
                        adapted_draft = worker.lora_manager.batch_info is not None
                        for row in range(batch_size):
                            for column in range(width):
                                flat = row * width + column
                                position = int(fb.positions[flat])
                                loc = fb.out_cache_loc[flat]
                                self.assertEqual(int(loc), int(table[row, position]))
                                is_adapted = adapted_draft and column > 0
                                kv[loc] = fb.input_ids[flat] + (
                                    vocab if is_adapted else 0
                                )
                                if is_adapted:
                                    token = gold[row][column]
                                    if column > accepted_proposals:
                                        token = (token + 1) % vocab
                                else:
                                    token = next_token(kv[table[row, : position + 1]])
                                logits[flat, token] = 100.0
                        return SimpleNamespace(
                            logits_output=SimpleNamespace(next_token_logits=logits),
                            can_run_cuda_graph=False,
                            routed_experts_output=None,
                            indexer_topk_output=None,
                        )

                    worker._target_worker = SimpleNamespace(
                        forward_batch_generation=forward
                    )
                    for _ in range(3):
                        gold.clear()
                        for row, history in enumerate(histories):
                            current = torch.cat(
                                (history, batch.spec_info.bonus_tokens[row : row + 1])
                            )
                            tokens = []
                            for _ in range(width + 1):
                                token = next_token(current)
                                tokens.append(token)
                                current = torch.cat((current, torch.tensor([token])))
                            gold.append(tokens)
                        result = worker._forward_decode(batch, on_publish=None)
                        outputs = result.next_token_ids.reshape(batch_size, width + 1)
                        for row in range(batch_size):
                            count = int(result.accept_lens[row])
                            self.assertEqual(count, accepted_proposals + 2)
                            self.assertEqual(
                                outputs[row, :count].tolist(), gold[row][:count]
                            )
                            histories[row] = torch.cat(
                                (
                                    histories[row],
                                    batch.spec_info.bonus_tokens[row : row + 1],
                                    outputs[row, : count - 1],
                                )
                            )
                            self.assertEqual(
                                len(histories[row]), int(result.new_seq_lens[row])
                            )
                            torch.testing.assert_close(
                                kv[table[row, : len(histories[row])]], histories[row]
                            )
                        self.assertIsNone(worker.lora_manager.batch_info)
                        batch.seq_lens = result.new_seq_lens
                        batch.seq_lens_cpu = result.new_seq_lens.clone()
                        batch.spec_info = result.next_draft_input

    def test_failed_draft_graph_setup_and_capture_clear_adapter(self):
        for tree_draft in (False, True):
            for failure in ("routing", "capture"):
                with self.subTest(tree_draft=tree_draft, failure=failure):
                    runner = UnoDecodeCudaGraphRunner.__new__(UnoDecodeCudaGraphRunner)
                    runner._tree_draft_mode = tree_draft
                    runner._tree_mode = False
                    runner._lora_state = Mock()
                    if failure == "routing":
                        runner._lora_state.capture_draft.side_effect = RuntimeError(
                            "capture failed"
                        )
                    with (
                        patch.object(
                            UnoDecodeCudaGraphRunner.__bases__[0],
                            "capture_one_shape",
                            side_effect=RuntimeError("capture failed"),
                        ),
                        self.assertRaisesRegex(RuntimeError, "capture failed"),
                    ):
                        runner.capture_one_shape(2, Mock(), variant_label="lora")
                    runner._lora_state.capture_draft.assert_called_once_with(2)
                    runner._lora_state.reset.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
