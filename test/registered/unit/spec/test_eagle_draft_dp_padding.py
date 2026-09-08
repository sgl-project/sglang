import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardMode,
    _should_materialize_idle_eagle_megamoe_dummy,
)
from sglang.srt.model_executor.runner.eager_runner import EagerRunner
from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA
from sglang.srt.speculative.eagle_worker_v2 import (
    _draft_extend_terminal_select_index,
    _slice_draft_output_to_local_tokens,
)
from sglang.srt.speculative.spec_info import SpecInputType
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestEagleDraftDPPadding(unittest.TestCase):
    def test_draft_extend_terminal_row_identity_is_shared(self):
        """Token, hidden, position, KV slot, logits, and seed select one row."""
        width = 4
        accept_lens = torch.tensor([1, 3, 4], dtype=torch.int64)
        prefix_lens = torch.tensor([10, 20, 30], dtype=torch.int64)
        req_pool_indices = torch.tensor([2, 0, 3], dtype=torch.int64)
        select = _draft_extend_terminal_select_index(accept_lens, width)

        row = torch.arange(accept_lens.numel() * width).reshape(-1)
        request = torch.arange(accept_lens.numel()).repeat_interleave(width)
        offset = torch.arange(width).repeat(accept_lens.numel())
        positions = prefix_lens.repeat_interleave(width) + offset

        # Model the compacted verify output and every draft-extend row-domain
        # consumer with distinct values derived from the same dense row id.
        input_ids = 10_000 + row
        hidden_states = torch.stack((request, positions), dim=1)
        logits = 20_000 + row
        dsa_seed = torch.stack((30_000 + row, 40_000 + row), dim=1)
        req_to_token = torch.empty((4, 64), dtype=torch.int64)
        for req in range(req_to_token.shape[0]):
            req_to_token[req] = req * 100_000 + torch.arange(64)
        out_cache_loc = req_to_token[
            req_pool_indices.repeat_interleave(width), positions
        ]

        expected_offset = accept_lens - 1
        expected_position = prefix_lens + expected_offset
        expected_row = torch.arange(accept_lens.numel()) * width + expected_offset
        self.assertEqual(select.dtype, torch.int64)
        torch.testing.assert_close(select, expected_row)
        torch.testing.assert_close(request[select], torch.arange(3))
        torch.testing.assert_close(positions[select], expected_position)
        torch.testing.assert_close(
            out_cache_loc[select],
            req_to_token[req_pool_indices, expected_position],
        )
        torch.testing.assert_close(input_ids[select], 10_000 + expected_row)
        torch.testing.assert_close(hidden_states[select, 0], torch.arange(3))
        torch.testing.assert_close(hidden_states[select, 1], expected_position)
        torch.testing.assert_close(logits[select], 20_000 + expected_row)
        torch.testing.assert_close(dsa_seed[select, 0], 30_000 + expected_row)

    def test_draft_extend_terminal_select_rejects_invalid_domain(self):
        with self.assertRaisesRegex(ValueError, "must be 1-D"):
            _draft_extend_terminal_select_index(torch.ones((2, 1)), 4)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            _draft_extend_terminal_select_index(torch.ones(2), 0)

    def test_idle_eagle_megamoe_materializes_collective_valid_dummy(self):
        algorithm = SimpleNamespace(is_eagle=lambda: True)
        draft_info = SimpleNamespace(spec_input_type=SpecInputType.EAGLE_DRAFT)
        backend = SimpleNamespace(is_megamoe=lambda: True)
        supported_runner = SimpleNamespace(
            model=SimpleNamespace(supports_symmetric_spec_megamoe_dummy=True)
        )

        with patch(
            "sglang.srt.layers.moe.utils.get_moe_a2a_backend",
            return_value=backend,
        ):
            self.assertTrue(
                _should_materialize_idle_eagle_megamoe_dummy(
                    model_runner=supported_runner,
                    forward_mode=ForwardMode.IDLE,
                    spec_algorithm=algorithm,
                    spec_info=draft_info,
                    dp_padding_mode=DpPaddingMode.MAX_LEN,
                    num_tokens=1,
                )
            )
            for mode, padding, tokens in (
                (ForwardMode.DECODE, DpPaddingMode.MAX_LEN, 1),
                (ForwardMode.IDLE, DpPaddingMode.SUM_LEN, 1),
                (ForwardMode.IDLE, DpPaddingMode.MAX_LEN, 0),
            ):
                self.assertFalse(
                    _should_materialize_idle_eagle_megamoe_dummy(
                        model_runner=supported_runner,
                        forward_mode=mode,
                        spec_algorithm=algorithm,
                        spec_info=draft_info,
                        dp_padding_mode=padding,
                        num_tokens=tokens,
                    )
                )
            self.assertFalse(
                _should_materialize_idle_eagle_megamoe_dummy(
                    model_runner=SimpleNamespace(model=SimpleNamespace()),
                    forward_mode=ForwardMode.IDLE,
                    spec_algorithm=algorithm,
                    spec_info=draft_info,
                    dp_padding_mode=DpPaddingMode.MAX_LEN,
                    num_tokens=1,
                )
            )
            self.assertFalse(
                _should_materialize_idle_eagle_megamoe_dummy(
                    model_runner=supported_runner,
                    forward_mode=ForwardMode.IDLE,
                    spec_algorithm=algorithm,
                    spec_info=SimpleNamespace(
                        spec_input_type=SpecInputType.EAGLE_DRAFT_EXTEND
                    ),
                    dp_padding_mode=DpPaddingMode.MAX_LEN,
                    num_tokens=1,
                )
            )

    def test_idle_megamoe_prepare_and_post_forward_round_trip(self):
        spec_info = SimpleNamespace(
            is_draft_input=lambda: True,
            spec_input_type=SpecInputType.EAGLE_DRAFT,
            num_tokens_per_req=1,
            topk_p=torch.empty((0, 1)),
            topk_index=torch.empty((0, 1), dtype=torch.int64),
            draft_probs=None,
            num_correct_drafts=None,
            hidden_states=torch.empty((0, 8)),
        )
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.IDLE,
            batch_size=0,
            input_ids=torch.empty(0, dtype=torch.int64),
            req_pool_indices=torch.empty(0, dtype=torch.int64),
            seq_lens=torch.empty(0, dtype=torch.int64),
            seq_lens_sum=0,
            out_cache_loc=torch.empty(0, dtype=torch.int64),
            positions=torch.empty(0, dtype=torch.int64),
            seq_lens_cpu=torch.empty(0, dtype=torch.int64),
            spec_info=spec_info,
            spec_algorithm=SimpleNamespace(is_eagle=lambda: True),
            global_num_tokens_cpu=[1, 0],
            global_num_tokens_gpu=torch.tensor([1, 0]),
            global_num_tokens_for_logprob_cpu=[1, 0],
            num_token_non_padded=torch.tensor(0),
            num_token_non_padded_cpu=0,
        )
        model_runner = SimpleNamespace(
            model=SimpleNamespace(supports_symmetric_spec_megamoe_dummy=True),
            model_config=SimpleNamespace(hf_config=SimpleNamespace()),
            is_draft_worker=True,
            enable_elastic_ep=False,
            attn_backend=SimpleNamespace(
                get_cpu_graph_seq_len_fill_value=lambda: 1,
                get_cuda_graph_seq_len_fill_value=lambda: 1,
            ),
        )
        parallel = SimpleNamespace(attn_tp_size=1, attn_dp_rank=1)
        execution = SimpleNamespace(
            graph=SimpleNamespace(
                cuda_graph_config=SimpleNamespace(prefill=SimpleNamespace(bs=[]))
            )
        )
        backend = SimpleNamespace(is_megamoe=lambda: True)

        with patch("sglang.srt.model_executor.forward_batch_info._is_cpu", True), patch(
            "sglang.srt.model_executor.forward_batch_info.get_parallel",
            return_value=parallel,
        ), patch(
            "sglang.srt.model_executor.forward_batch_info.get_exec",
            return_value=execution,
        ), patch(
            "sglang.srt.model_executor.forward_batch_info.set_dp_buffer_len"
        ), patch(
            "sglang.srt.model_executor.forward_batch_info.set_is_extend_in_batch"
        ), patch(
            "sglang.srt.model_executor.forward_batch_info.mambaish_config",
            return_value=None,
        ), patch(
            "sglang.srt.layers.dp_attention.DpPaddingMode.get_dp_padding_mode",
            return_value=DpPaddingMode.MAX_LEN,
        ), patch(
            "sglang.srt.layers.cp.padding.get_cp_padding_align_size",
            return_value=1,
        ), patch(
            "sglang.srt.layers.cp.utils.enable_cp_v2", return_value=False
        ), patch(
            "sglang.srt.layers.moe.utils.get_moe_a2a_backend",
            return_value=backend,
        ), patch(
            "sglang.srt.batch_overlap.two_batch_overlap.TboForwardBatchPreparer.prepare"
        ):
            forward_batch.prepare_mlp_sync_batch(model_runner)

        self.assertTrue(forward_batch.symmetric_spec_megamoe_dummy)
        self.assertEqual(forward_batch.forward_mode, ForwardMode.IDLE)
        self.assertEqual(forward_batch.batch_size, 1)
        self.assertEqual(forward_batch.input_ids.shape[0], 1)
        self.assertEqual(forward_batch.num_token_non_padded_cpu, 1)
        self.assertEqual(forward_batch.num_token_non_padded.item(), 1)
        self.assertEqual(spec_info.hidden_states.shape[0], 1)

        logits_output = SimpleNamespace(
            next_token_logits=torch.randn(1, 16), hidden_states=torch.randn(1, 8)
        )
        forward_batch.post_forward_mlp_sync_batch(logits_output)

        self.assertEqual(forward_batch.batch_size, 0)
        self.assertEqual(logits_output.next_token_logits.shape[0], 0)
        self.assertEqual(logits_output.hidden_states.shape[0], 0)
        self.assertEqual(spec_info.hidden_states.shape[0], 0)
        self.assertEqual(forward_batch.out_cache_loc.shape[0], 0)
        self.assertEqual(forward_batch.positions.shape[0], 0)
        self.assertEqual(forward_batch.seq_lens.shape[0], 0)
        self.assertEqual(forward_batch.req_pool_indices.shape[0], 0)
        self.assertEqual(forward_batch.seq_lens_cpu.shape[0], 0)

    def test_megamoe_dummy_bypasses_attention_but_preserves_rows(self):
        hidden_states = torch.randn(1, 8)
        forward_batch = SimpleNamespace(symmetric_spec_megamoe_dummy=True)

        state = DeepseekV2AttentionMLA.forward_prepare(
            SimpleNamespace(),
            positions=torch.zeros(1, dtype=torch.int64),
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            zero_allocator=MagicMock(),
        )

        self.assertIs(state[0], hidden_states)
        self.assertIsNone(state[1])
        self.assertIs(state[2], forward_batch)
        self.assertIsNone(state[3])

    def test_idle_runner_does_not_plan_fake_kv_for_megamoe_dummy(self):
        forward_batch = SimpleNamespace(batch_size=1, symmetric_spec_megamoe_dummy=True)
        loaded_batch = SimpleNamespace(
            input_ids=torch.zeros(1, dtype=torch.int64),
            positions=torch.zeros(1, dtype=torch.int64),
            symmetric_spec_megamoe_dummy=True,
        )
        attn_backend = MagicMock()
        model = SimpleNamespace(forward=MagicMock(return_value=object()))
        model_runner = SimpleNamespace(
            attn_backend=attn_backend,
            model=model,
            device_timer=None,
            _pp_kwargs=lambda _proxy: {},
        )
        runner = SimpleNamespace(
            model_runner=model_runner,
            enable_pdmux=False,
            load_batch=MagicMock(return_value=loaded_batch),
        )

        with patch(
            "sglang.srt.model_executor.runner.eager_runner.device_timer_ctx",
            return_value=nullcontext(),
        ):
            result = EagerRunner._execute_idle(runner, forward_batch)

        runner.load_batch.assert_called_once_with(forward_batch, None)
        attn_backend.init_forward_metadata.assert_not_called()
        self.assertIsNone(attn_backend.forward_metadata)
        model.forward.assert_called_once_with(
            loaded_batch.input_ids, loaded_batch.positions, loaded_batch
        )
        self.assertIs(result, model.forward.return_value)

    def test_discards_dp_padding_rows(self):
        logits = torch.arange(24, dtype=torch.float32).reshape(3, 8)
        hidden_states = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        positions = torch.tensor([7, 100, 100])

        local_logits, local_hidden_states, local_positions = (
            _slice_draft_output_to_local_tokens(
                logits, hidden_states, positions, num_local_tokens=1
            )
        )

        self.assertEqual(local_logits.shape, (1, 8))
        self.assertEqual(local_hidden_states.shape, (1, 4))
        self.assertEqual(local_positions.tolist(), [7])
        local_positions.add_(1)
        self.assertEqual(positions.tolist(), [8, 100, 100])

    def test_rejects_missing_local_rows(self):
        with self.assertRaisesRegex(RuntimeError, "next_token_logits has 0 rows"):
            _slice_draft_output_to_local_tokens(
                torch.empty((0, 8)),
                torch.empty((1, 4)),
                torch.tensor([7]),
                num_local_tokens=1,
            )


if __name__ == "__main__":
    unittest.main()
