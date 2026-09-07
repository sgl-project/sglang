"""CPU tests for whole-request-group padding in hybrid target verify."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.srt.model_executor import forward_batch_info as fbi
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _spec(width: int = 6, *, ragged=False, draft=False):
    return SimpleNamespace(
        num_tokens_per_req=width,
        ragged_verify_layout=object() if ragged else None,
        is_draft_input=lambda: draft,
    )


def _model_runner(*, hybrid=True):
    runner = MagicMock()
    runner.model_config = object()
    runner.is_draft_worker = False
    runner.enable_elastic_ep = False
    runner.attn_tp_sequence_sharded.return_value = False
    runner.attn_backend.get_cpu_graph_seq_len_fill_value.return_value = 1
    runner.attn_backend.get_cuda_graph_seq_len_fill_value.return_value = 1
    runner._hybrid_config = object() if hybrid else None
    return runner


def _batch(
    *,
    mode=ForwardMode.TARGET_VERIFY,
    batch_size: int,
    width: int = 6,
    global_num_tokens=None,
    spec=None,
    is_extend_in_batch: bool = False,
):
    num_tokens = batch_size * width if spec is not None else batch_size
    if global_num_tokens is None:
        global_num_tokens = [num_tokens]
    return ForwardBatch(
        forward_mode=mode,
        batch_size=batch_size,
        input_ids=torch.arange(num_tokens),
        req_pool_indices=torch.arange(batch_size),
        seq_lens=torch.arange(10, 10 + batch_size, dtype=torch.int64),
        out_cache_loc=torch.arange(num_tokens),
        seq_lens_sum=int(torch.arange(10, 10 + batch_size).sum()),
        positions=torch.arange(num_tokens),
        seq_lens_cpu=torch.arange(10, 10 + batch_size, dtype=torch.int64),
        lora_ids=[None] * batch_size,
        spec_info=spec,
        is_extend_in_batch=is_extend_in_batch,
        global_num_tokens_cpu=list(global_num_tokens),
        global_num_tokens_gpu=torch.tensor(global_num_tokens, dtype=torch.int64),
        global_num_tokens_for_logprob_cpu=[0] * len(global_num_tokens),
        global_num_tokens_for_logprob_gpu=torch.zeros(
            len(global_num_tokens), dtype=torch.int64
        ),
    )


def _prepare(
    fb,
    runner,
    *,
    attn_tp_size: int = 8,
    attn_dp_rank: int = 0,
    attn_dp_size: int = 1,
    cp_align: int = 1,
    cp_v2: bool = False,
):
    calls = {}
    exec_ctx = SimpleNamespace(
        graph=SimpleNamespace(
            cuda_graph_config=SimpleNamespace(prefill=SimpleNamespace(bs=[]))
        )
    )
    with ExitStack() as stack:
        stack.enter_context(
            get_parallel().override(
                attn_tp_size=attn_tp_size,
                attn_tp_rank=0,
                attn_dp_rank=attn_dp_rank,
            )
        )
        stack.enter_context(
            patch(
                "sglang.srt.layers.dp_attention.get_attention_dp_size",
                return_value=attn_dp_size,
            )
        )
        stack.enter_context(
            patch(
                "sglang.srt.layers.moe.utils.get_moe_a2a_backend",
                return_value=MoeA2ABackend.NONE,
            )
        )
        stack.enter_context(patch.object(fbi, "get_exec", return_value=exec_ctx))
        stack.enter_context(patch.object(fbi, "_is_cpu", True))
        stack.enter_context(
            patch.object(
                fbi,
                "mambaish_config",
                side_effect=lambda model_config: getattr(
                    runner, "_hybrid_config", None
                ),
            )
        )
        stack.enter_context(
            patch(
                "sglang.srt.layers.cp.padding.get_cp_padding_align_size",
                return_value=cp_align,
            )
        )
        stack.enter_context(
            patch("sglang.srt.layers.cp.utils.enable_cp_v2", return_value=cp_v2)
        )
        stack.enter_context(
            patch(
                "sglang.srt.batch_overlap.two_batch_overlap.TboForwardBatchPreparer.prepare"
            )
        )
        stack.enter_context(
            patch.object(
                fbi,
                "set_dp_buffer_len",
                side_effect=lambda *args: calls.setdefault("dp_buffer", args),
            )
        )
        stack.enter_context(patch.object(fbi, "set_is_extend_in_batch"))
        fb.prepare_mlp_sync_batch(runner)
    return calls["dp_buffer"]


class TestHybridVerifyGroupPadding(CustomTestCase):
    def test_hybrid_target_verify_aligns_to_complete_request_groups(self):
        for batch_size, expected_tokens, expected_bs in [
            (33, 216, 36),
            (36, 216, 36),
            (38, 240, 40),
            (39, 240, 40),
        ]:
            with self.subTest(batch_size=batch_size):
                fb = _batch(batch_size=batch_size, spec=_spec())

                dp_buffer = _prepare(fb, _model_runner())

                self.assertEqual(fb.global_num_tokens_cpu, [expected_tokens])
                self.assertEqual(fb.input_ids.shape[0], expected_tokens)
                self.assertEqual(fb.positions.shape[0], expected_tokens)
                self.assertEqual(fb.out_cache_loc.shape[0], expected_tokens)
                self.assertEqual(fb.batch_size, expected_bs)
                self.assertEqual(fb.req_pool_indices.shape[0], expected_bs)
                self.assertEqual(fb.seq_lens.shape[0], expected_bs)
                self.assertEqual(fb._original_batch_size, batch_size)
                self.assertEqual(fb._original_num_tokens, batch_size * 6)
                self.assertEqual(dp_buffer[0], expected_tokens)
                self.assertEqual(dp_buffer[1], expected_tokens)
                self.assertTrue(dp_buffer[2])

    def test_target_verify_post_forward_restores_known_output_domain(self):
        fb = _batch(batch_size=38, spec=_spec())
        _prepare(fb, _model_runner())
        logits_output = SimpleNamespace(
            next_token_logits=torch.randn(240, 16),
            hidden_states=torch.randn(240, 4),
        )

        fb.post_forward_mlp_sync_batch(logits_output)

        self.assertEqual(fb.batch_size, 38)
        self.assertEqual(logits_output.next_token_logits.shape[0], 228)
        self.assertEqual(logits_output.hidden_states.shape[0], 228)
        # Target-verify callers consume the sliced outputs above. The prepared
        # input metadata remains padded today; keep that as an explicit
        # diagnostic rather than a required contract without a caller proving it.
        self.assertEqual(fb.positions.shape[0], 240)
        self.assertEqual(fb.req_pool_indices.shape[0], 40)
        self.assertEqual(fb.seq_lens.shape[0], 40)
        self.assertEqual(fb.seq_lens_cpu.shape[0], 40)

    def test_context_parallel_alignment_is_composed_with_request_width(self):
        fb = _batch(batch_size=33, spec=_spec())

        _prepare(fb, _model_runner(), cp_align=40)

        self.assertEqual(fb.global_num_tokens_cpu, [240])
        self.assertEqual(fb.input_ids.shape[0], 240)
        self.assertEqual(fb.batch_size, 40)

    def test_dp_max_len_pads_active_and_idle_ranks_to_same_group_boundary(self):
        global_tokens = [33 * 6, 0]
        active = _batch(batch_size=33, spec=_spec(), global_num_tokens=global_tokens)
        idle = _batch(
            mode=ForwardMode.IDLE,
            batch_size=0,
            spec=_spec(),
            global_num_tokens=global_tokens,
        )

        active_dp = _prepare(active, _model_runner(), attn_dp_rank=0, attn_dp_size=2)
        idle_dp = _prepare(idle, _model_runner(), attn_dp_rank=1, attn_dp_size=2)

        self.assertEqual(active.dp_padding_mode, DpPaddingMode.MAX_LEN)
        self.assertEqual(idle.dp_padding_mode, DpPaddingMode.MAX_LEN)
        self.assertEqual(active.global_num_tokens_cpu, [216, 216])
        self.assertEqual(idle.global_num_tokens_cpu, [216, 216])
        self.assertEqual(active.input_ids.shape[0], 216)
        self.assertEqual(idle.input_ids.shape[0], 216)
        self.assertEqual(active.batch_size, 36)
        self.assertEqual(idle.batch_size, 36)
        self.assertEqual(idle.forward_mode, ForwardMode.TARGET_VERIFY)
        self.assertEqual(active_dp[0], 432)
        self.assertEqual(idle_dp[0], 432)
        self.assertEqual(active_dp[1], 216)
        self.assertEqual(idle_dp[1], 216)

    def test_dp_sum_len_preserves_rank_local_group_boundaries(self):
        global_tokens = [1 * 6, 33 * 6, 0]
        active = _batch(
            batch_size=33,
            spec=_spec(),
            global_num_tokens=global_tokens,
        )
        idle = _batch(
            mode=ForwardMode.IDLE,
            batch_size=0,
            spec=_spec(),
            global_num_tokens=global_tokens,
        )

        active_dp = _prepare(active, _model_runner(), attn_dp_rank=1, attn_dp_size=3)
        idle_dp = _prepare(idle, _model_runner(), attn_dp_rank=2, attn_dp_size=3)

        self.assertEqual(active.dp_padding_mode, DpPaddingMode.SUM_LEN)
        self.assertEqual(idle.dp_padding_mode, DpPaddingMode.SUM_LEN)
        self.assertEqual(active.global_num_tokens_cpu, [24, 216, 0])
        self.assertEqual(idle.global_num_tokens_cpu, [24, 216, 0])
        self.assertEqual(active.input_ids.shape[0], 216)
        self.assertEqual(active.batch_size, 36)
        self.assertEqual(idle.input_ids.shape[0], 0)
        self.assertEqual(idle.batch_size, 0)
        self.assertEqual(active_dp[0], 240)
        self.assertEqual(idle_dp[0], 240)
        self.assertEqual(active_dp[1], 216)
        self.assertEqual(idle_dp[1], 0)

    def test_non_hybrid_non_verify_and_ragged_keep_token_padding_behavior(self):
        cases = [
            (
                "non_hybrid",
                _model_runner(hybrid=False),
                _spec(),
                ForwardMode.TARGET_VERIFY,
                232,
                38,
            ),
            ("decode", _model_runner(), None, ForwardMode.DECODE, 40, 40),
            (
                "ragged",
                _model_runner(),
                _spec(ragged=True),
                ForwardMode.TARGET_VERIFY,
                232,
                38,
            ),
        ]
        for label, runner, spec, mode, expected_tokens, expected_bs in cases:
            with self.subTest(label=label):
                fb = _batch(batch_size=38, spec=spec, mode=mode)

                _prepare(fb, runner)

                self.assertEqual(fb.global_num_tokens_cpu, [expected_tokens])
                self.assertEqual(fb.input_ids.shape[0], expected_tokens)
                self.assertEqual(fb.batch_size, expected_bs)


if __name__ == "__main__":
    unittest.main()
