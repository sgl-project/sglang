"""Decoder-tail graphs on DP speculative/prefill coordination steps."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.runner.row_bucket_graph_runner import _Graph
from sglang.srt.models import deepseek_v4_replay_tail
from sglang.srt.models.deepseek_v4_mhc import HcState
from sglang.srt.models.deepseek_v4_replay_tail import DecoderReplayTail
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")

# Stands in for SWA_WINDOW (128): a peer prefilling one request has one window of
# tail rows, so its bucket is one row step.
ROW_STEP = 8
PEER_TAIL_ROWS = ROW_STEP
PEER_PREFILL_TOKENS = 32768


def _late_layers(x: torch.Tensor) -> torch.Tensor:
    # Row-wise stand-in for the late layers.
    return x * 2 + 1


def _record(body, rows, local, batch, start, free_before):
    # A plain CUDA graph in place of the distributed breakable recording.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        body()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outputs = body()
    torch.cuda.current_stream().wait_stream(stream)
    return _Graph(graph, local, outputs, (batch,))


def _batch(forward_mode, num_rows, *, coordinated):
    ids = torch.zeros(num_rows, dtype=torch.int64, device="cuda")
    batch = ForwardBatch(
        forward_mode=forward_mode,
        batch_size=max(num_rows, 1),
        input_ids=ids,
        req_pool_indices=ids,
        seq_lens=ids,
        out_cache_loc=ids,
        seq_lens_sum=0,
    )
    batch.is_extend_in_batch = True
    batch.dp_spec_prefill_coordination_applied = coordinated
    # MegaMoE with a DP LM head keeps local-only counts.
    batch.global_num_tokens_cpu = [num_rows]
    batch.global_num_tokens_gpu = torch.tensor(
        [num_rows], dtype=torch.int32, device="cuda"
    )
    batch.global_dp_buffer_len = num_rows
    batch.dp_padding_mode = DpPaddingMode.SUM_LEN
    batch.global_num_token_non_padded_cpu = num_rows
    return batch


def _late_layer_rows(tail, forward_batch, x):
    """This rank's late-layer output, dispatched as DeepseekV4Model dispatches a rank
    without a prefill tail: the tail plan, then run_idle or the eager layers."""
    use_tail_graph, dp_rows = tail.plan(
        None,
        forward_batch=forward_batch,
        num_tokens=x.shape[0],
        tail_in_graph=False,
        late_dp_counts=None,
    )
    if not use_tail_graph:
        return False, _late_layers(x)
    ids = forward_batch.input_ids
    residual, _, _ = tail.run_idle(
        state=HcState(x),
        positions=ids,
        input_ids=ids,
        input_ids_global=None,
        forward_batch=forward_batch,
        capture_dspark=False,
        dp_rows=dp_rows,
    )
    return True, residual


class TestDecoderTailDpSpecCoordination(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")

    def setUp(self):
        super().setUp()
        # The tail graphs start after a forward of one full prefill chunk, whose
        # size the runner reads off the schedule bag.
        override = get_context().override_server_args(
            chunked_prefill_size=PEER_PREFILL_TOKENS
        )
        override.install()
        self.addCleanup(override.restore)

    def _run(self, forward_mode, num_rows, *, coordinated):
        tail = DecoderReplayTail(
            model=SimpleNamespace(late_layer_vl_routing=False),
            layer_ids=range(1),
            row_step=ROW_STEP,
            max_rows=4 * ROW_STEP,
        )
        batch = _batch(forward_mode, num_rows, coordinated=coordinated)
        x = torch.randn(num_rows, 1, 4, device="cuda")
        megamoe = SimpleNamespace(is_none=lambda: False)
        with (
            # DP rank 1 of TP2 / DP2 attention / EP2 MoE.
            get_parallel().override(
                tp_size=2,
                tp_rank=1,
                attn_dp_size=2,
                attn_dp_rank=1,
                attn_tp_size=1,
                attn_tp_rank=0,
                attn_cp_size=1,
                attn_cp_rank=0,
                moe_ep_size=2,
                moe_dp_size=1,
                moe_tp_size=1,
            ),
            # Peer 0 prefills one request (one bucket of tail rows); this rank has
            # no prefill tail. Every rank reports its tail rows in this gather.
            patch.object(
                deepseek_v4_replay_tail,
                "scattered_dp_counts",
                return_value=(PEER_PREFILL_TOKENS, [PEER_TAIL_ROWS, 0]),
            ),
            patch.object(
                deepseek_v4_replay_tail, "get_moe_a2a_backend", return_value=megamoe
            ),
            patch.object(
                deepseek_v4_replay_tail, "check_cuda_graph_backend", return_value=True
            ),
            forward_context(ForwardContext(attn_backend=None), forward_batch=batch),
            patch.object(
                tail,
                "_layer_stack",
                side_effect=lambda state, **kwargs: (
                    _late_layers(state.streams),
                    None,
                    [],
                ),
            ),
            patch.object(
                tail, "_break_context", side_effect=lambda batch: nullcontext()
            ),
            patch.object(tail.runner, "_record", side_effect=_record),
        ):
            return x, *_late_layer_rows(tail, batch, x)

    def test_coordinated_verify_rank_keeps_its_rows(self):
        """A verify rank paired with a prefilling peer has rows but no tail. Its late
        layers must run on those rows, not on the idle replay of the peer's bucket
        (which returned PEER_TAIL_ROWS stale rows: the "Expected 132, got 128" crash
        of DFlash verify on DeepSeek-V4.1)."""
        for num_rows in (2 * 5, 5):  # above and below the peer's bucket
            with self.subTest(num_rows=num_rows):
                x, _, out = self._run(
                    ForwardMode.TARGET_VERIFY, num_rows, coordinated=True
                )
                torch.testing.assert_close(out, _late_layers(x), rtol=0, atol=0)

    def test_idle_rank_still_replays_the_peer_bucket(self):
        """An idle rank on an uncoordinated step keeps the shared tail graph: its peers
        capture and replay that bucket in step with it."""
        x, used_graph, out = self._run(ForwardMode.IDLE, 0, coordinated=False)
        self.assertTrue(used_graph)
        self.assertEqual(out.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
