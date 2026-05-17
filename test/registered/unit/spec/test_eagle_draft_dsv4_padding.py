import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.deepseek_v4_backend import (
    DeepseekV4MultiStepBackend,
    _split_eagle_draft_out_cache_loc_by_step,
)
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class _NoopDeepepAdapter:
    def replay(self):
        pass


class _RecordingDraftBackend:
    def __init__(self, *, raise_on_init: bool = False):
        self.calls = []
        self.raise_on_init = raise_on_init

    def init_forward_metadata_replay_cuda_graph(self, forward_batch, bs):
        self.calls.append(
            {
                "bs": bs,
                "batch_size": forward_batch.batch_size,
                "seq_lens": forward_batch.seq_lens.clone(),
                "req_pool_indices": forward_batch.req_pool_indices.clone(),
                "out_cache_loc": forward_batch.out_cache_loc.clone(),
                "positions": forward_batch.positions.clone(),
            }
        )
        if self.raise_on_init:
            raise RuntimeError("metadata init failed")


class _RecordingDSV4StepBackend:
    def __init__(self):
        self.calls = []
        self._replay_forward_batch = None

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs,
        req_pool_indices,
        seq_lens,
        seq_lens_sum,
        encoder_lens,
        forward_mode,
        spec_info,
        seq_lens_cpu,
    ):
        del encoder_lens, forward_mode, spec_info
        self.calls.append(
            {
                "bs": bs,
                "req_pool_indices": req_pool_indices.clone(),
                "seq_lens": seq_lens.clone(),
                "seq_lens_sum": seq_lens_sum,
                "seq_lens_cpu": seq_lens_cpu.clone(),
                "out_cache_loc": self._replay_forward_batch.out_cache_loc.clone(),
            }
        )


def _make_runner(draft_attn_backend):
    def _noop_replay(forward_batch):
        del forward_batch

    runner = object.__new__(EAGLEDraftCudaGraphRunner)
    runner.deepep_adapter = _NoopDeepepAdapter()
    runner.buffers = SimpleNamespace(
        seq_lens=torch.empty(4, dtype=torch.int32),
        out_cache_loc=torch.empty(12, dtype=torch.int64),
        positions=torch.empty(4, dtype=torch.int64),
        topk_p=torch.empty((4, 1), dtype=torch.float32),
        topk_index=torch.empty((4, 1), dtype=torch.int64),
        hidden_states=None,
        req_pool_indices=torch.empty(4, dtype=torch.int64),
        seq_lens_cpu=torch.empty(4, dtype=torch.int32),
        global_num_tokens_gpu=None,
        global_num_tokens_for_logprob_gpu=None,
    )
    runner.capture_bs = [1, 2, 4, 8]
    runner.num_tokens_per_bs = 1
    runner.speculative_num_steps = 3
    runner.require_mlp_tp_gather = False
    runner.require_gathered_buffer = False
    runner.seq_len_fill_value = 1
    runner.draft_attn_backend = draft_attn_backend
    runner.model_runner = SimpleNamespace(
        model_config=SimpleNamespace(vocab_size=32000)
    )
    runner.output_buffers = {
        4: (
            torch.arange(4, dtype=torch.int64),
            torch.arange(10, 14, dtype=torch.int64),
            torch.arange(20, 24, dtype=torch.int64),
        )
    }
    runner._replay = _noop_replay
    return runner


def _make_forward_batch():
    return SimpleNamespace(
        batch_size=3,
        seq_lens=torch.tensor([5, 7, 11], dtype=torch.int32),
        req_pool_indices=torch.tensor([101, 102, 103], dtype=torch.int64),
        out_cache_loc=torch.tensor(
            [10, 11, 12, 20, 21, 22, 30, 31, 32], dtype=torch.int64
        ),
        positions=torch.tensor([5, 7, 11], dtype=torch.int64),
        seq_lens_cpu=torch.tensor([5, 7, 11], dtype=torch.int32),
        seq_lens_sum=23,
        spec_info=SimpleNamespace(
            topk_p=torch.ones((3, 1), dtype=torch.float32),
            topk_index=torch.zeros((3, 1), dtype=torch.int64),
            hidden_states=None,
        ),
    )


class TestEagleDraftDSV4Padding(unittest.TestCase):
    def test_replay_pads_out_cache_loc_for_metadata(self):
        draft_attn_backend = _RecordingDraftBackend()
        runner = _make_runner(draft_attn_backend)
        forward_batch = _make_forward_batch()

        out = EAGLEDraftCudaGraphRunner.replay(runner, forward_batch)

        self.assertEqual(
            [x.tolist() for x in out],
            [[0, 1, 2], [10, 11, 12], [20, 21, 22]],
        )
        call = draft_attn_backend.calls[0]
        self.assertEqual(call["bs"], 4)
        self.assertEqual(call["batch_size"], 4)
        self.assertEqual(call["seq_lens"].tolist(), [5, 7, 11, 1])
        self.assertEqual(call["req_pool_indices"].tolist(), [101, 102, 103, 0])
        self.assertEqual(
            call["out_cache_loc"].tolist(),
            [10, 11, 12, 20, 21, 22, 30, 31, 32, 0, 0, 0],
        )
        self.assertEqual(call["positions"].tolist(), [5, 7, 11, 0])

    def test_replay_restores_forward_batch_after_metadata_error(self):
        draft_attn_backend = _RecordingDraftBackend(raise_on_init=True)
        runner = _make_runner(draft_attn_backend)
        forward_batch = _make_forward_batch()
        original_batch_size = forward_batch.batch_size
        original_seq_lens = forward_batch.seq_lens
        original_req_pool_indices = forward_batch.req_pool_indices
        original_out_cache_loc = forward_batch.out_cache_loc
        original_positions = forward_batch.positions
        original_seq_lens_cpu = forward_batch.seq_lens_cpu

        with self.assertRaisesRegex(RuntimeError, "metadata init failed"):
            EAGLEDraftCudaGraphRunner.replay(runner, forward_batch)

        self.assertEqual(forward_batch.batch_size, original_batch_size)
        self.assertIs(forward_batch.seq_lens, original_seq_lens)
        self.assertIs(forward_batch.req_pool_indices, original_req_pool_indices)
        self.assertIs(forward_batch.out_cache_loc, original_out_cache_loc)
        self.assertIs(forward_batch.positions, original_positions)
        self.assertIs(forward_batch.seq_lens_cpu, original_seq_lens_cpu)
        self.assertEqual(
            draft_attn_backend.calls[0]["out_cache_loc"].tolist(),
            [10, 11, 12, 20, 21, 22, 30, 31, 32, 0, 0, 0],
        )

    def test_dsv4_multistep_replay_passes_per_step_cache_locs(self):
        backend = object.__new__(DeepseekV4MultiStepBackend)
        backend.topk = 1
        backend.speculative_num_steps = 3
        backend.attn_backends = [_RecordingDSV4StepBackend() for _ in range(3)]
        forward_batch = _make_forward_batch()
        original_out_cache_loc = torch.tensor(
            [10, 11, 12, 20, 21, 22, 30, 31, 32, 0, 0, 0],
            dtype=torch.int64,
        )
        forward_batch.batch_size = 4
        forward_batch.seq_lens = torch.tensor([5, 7, 11, 1], dtype=torch.int32)
        forward_batch.req_pool_indices = torch.tensor(
            [101, 102, 103, 0], dtype=torch.int64
        )
        forward_batch.seq_lens_cpu = torch.tensor([5, 7, 11, 1], dtype=torch.int32)
        forward_batch.out_cache_loc = original_out_cache_loc

        DeepseekV4MultiStepBackend.init_forward_metadata_replay_cuda_graph(
            backend, forward_batch, 4
        )

        self.assertEqual(
            backend.attn_backends[0].calls[0]["out_cache_loc"].tolist(),
            [10, 20, 30, 0],
        )
        self.assertEqual(
            backend.attn_backends[1].calls[0]["out_cache_loc"].tolist(),
            [11, 21, 31, 0],
        )
        self.assertEqual(backend.attn_backends[2].calls, [])
        self.assertIs(forward_batch.out_cache_loc, original_out_cache_loc)
        for step_backend in backend.attn_backends:
            self.assertIsNone(step_backend._replay_forward_batch)

    def test_split_draft_cache_locs_uses_step_major_order(self):
        out_cache_loc = torch.tensor(
            [
                10,
                11,
                12,
                20,
                21,
                22,
                30,
                31,
                32,
                0,
                0,
                0,
            ],
            dtype=torch.int64,
        )

        split = _split_eagle_draft_out_cache_loc_by_step(
            out_cache_loc,
            batch_size=4,
            topk=1,
            speculative_num_steps=3,
        )

        torch.testing.assert_close(
            split,
            torch.tensor(
                [
                    [10, 20, 30, 0],
                    [11, 21, 31, 0],
                    [12, 22, 32, 0],
                ],
                dtype=torch.int64,
            ),
        )

    def test_split_draft_cache_locs_rejects_unpadded_input(self):
        out_cache_loc = torch.arange(9, dtype=torch.int64)

        with self.assertRaisesRegex(ValueError, "must be padded"):
            _split_eagle_draft_out_cache_loc_by_step(
                out_cache_loc,
                batch_size=4,
                topk=1,
                speculative_num_steps=3,
            )


if __name__ == "__main__":
    unittest.main()
