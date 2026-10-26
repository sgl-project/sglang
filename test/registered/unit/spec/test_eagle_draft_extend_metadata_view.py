"""CPU regression coverage for draft-extend replay metadata views.

Both draft-extend CUDA graph runners reuse buffers sized for the largest
captured batch.  Replay metadata must expose only the selected graph bucket,
while retaining the synthetic rows between the live batch and that bucket.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    MambaAttnBackendBase,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
    EAGLEDraftExtendCudaGraphRunner,
)
from sglang.srt.speculative.eagle_info import EagleDraftExtendInput
from sglang.srt.speculative.multi_layer_eagle_draft_extend_cuda_graph_runner import (
    MultiLayerEagleDraftExtendCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CAPTURE_BATCH_SIZES = [1, 4, 8]
BUCKET_BS = 4
MAX_BS = 8
TOKENS_PER_REQUEST = 3
MAX_NUM_TOKENS = MAX_BS * TOKENS_PER_REQUEST
SEQ_LEN_FILL_VALUE = 1
TAIL_SENTINEL = 77


class _Event:
    def record(self):
        pass


class _ReplayMetadataSpy:
    """Snapshot the replay view, then exercise the real backend entry point."""

    def __init__(self):
        self.observations = []
        self.replay_calls = []
        self._mamba_backend = object.__new__(MambaAttnBackendBase)
        self._mamba_backend._replay_metadata = self._record_replay_metadata

    def init_forward_metadata_out_graph(self, forward_batch, in_capture=False):
        self.observations.append(
            SimpleNamespace(
                batch_size=forward_batch.batch_size,
                num_padding=getattr(forward_batch, "num_padding", None),
                req_pool_indices=forward_batch.req_pool_indices.clone(),
                seq_lens=forward_batch.seq_lens.clone(),
                seq_lens_sum=forward_batch.seq_lens_sum,
                seq_lens_cpu=(
                    None
                    if forward_batch.seq_lens_cpu is None
                    else forward_batch.seq_lens_cpu.clone()
                ),
            )
        )
        MambaAttnBackendBase.init_forward_metadata_out_graph(
            self._mamba_backend, forward_batch, in_capture=in_capture
        )

    def _record_replay_metadata(
        self,
        bs,
        req_pool_indices,
        forward_mode,
        spec_info,
        seq_lens_cpu,
        *,
        num_padding=None,
        in_capture=False,
        mamba_track_indices=None,
    ):
        self.replay_calls.append(
            SimpleNamespace(
                batch_size=bs,
                num_padding=num_padding,
                req_pool_indices=req_pool_indices.clone(),
                seq_lens_cpu=(None if seq_lens_cpu is None else seq_lens_cpu.clone()),
            )
        )
        return SimpleNamespace()


class TestEagleDraftExtendMetadataView(CustomTestCase):
    def _case_values(self, raw_bs):
        if raw_bs == 2:
            return [10, 11], [7, 9]
        if raw_bs == BUCKET_BS:
            return [10, 11, 12, 13], [7, 9, 11, 13]
        raise AssertionError(f"Unsupported test batch size: {raw_bs}")

    def _new_buffers(self):
        return SimpleNamespace(
            input_ids=torch.full((MAX_NUM_TOKENS,), -1, dtype=torch.int64),
            req_pool_indices=torch.full((MAX_BS,), TAIL_SENTINEL, dtype=torch.int32),
            out_cache_loc=torch.full((MAX_NUM_TOKENS,), -1, dtype=torch.int64),
            positions=torch.full((MAX_NUM_TOKENS,), -1, dtype=torch.int64),
            hidden_states=None,
            seq_lens=torch.full((MAX_BS,), TAIL_SENTINEL, dtype=torch.int32),
            seq_lens_cpu=torch.full((MAX_BS,), TAIL_SENTINEL, dtype=torch.int64),
            extend_seq_lens=torch.full(
                (MAX_BS,), TOKENS_PER_REQUEST, dtype=torch.int32
            ),
            num_correct_drafts=torch.zeros(MAX_BS, dtype=torch.int32),
            num_accept_tokens=torch.zeros(MAX_BS, dtype=torch.int32),
            global_num_tokens_gpu=None,
            global_num_tokens_for_logprob_gpu=None,
        )

    def _new_spec_info(self, raw_bs):
        return EagleDraftExtendInput(
            hidden_states=None,
            num_correct_drafts=torch.full(
                (raw_bs,), TOKENS_PER_REQUEST - 1, dtype=torch.int32
            ),
            num_accept_tokens=torch.full(
                (raw_bs,), TOKENS_PER_REQUEST, dtype=torch.int32
            ),
        )

    def _run_single_layer(self, raw_bs, *, cpu_mirror):
        seq_lens, req_pool_indices = self._case_values(raw_bs)
        num_tokens = raw_bs * TOKENS_PER_REQUEST
        backend = _ReplayMetadataSpy()
        runner = EAGLEDraftExtendCudaGraphRunner.__new__(
            EAGLEDraftExtendCudaGraphRunner
        )
        runner.deepep_adapter = SimpleNamespace(replay=lambda: None)
        runner.buffers = self._new_buffers()
        runner.capture_bs = CAPTURE_BATCH_SIZES
        runner.captured_req_width = TOKENS_PER_REQUEST
        runner.seq_len_fill_value = SEQ_LEN_FILL_VALUE
        runner.extend_seq_lens_cpu = [TOKENS_PER_REQUEST] * MAX_BS
        runner.require_mlp_tp_gather = False
        runner.require_gathered_buffer = False
        runner.forward_mode = ForwardMode.DRAFT_EXTEND_V2
        runner.draft_extend_attn_backend = backend
        runner.device_module = SimpleNamespace(Event=_Event)
        runner.model_runner = SimpleNamespace(
            device_timer=None,
            shared_read_done_event=None,
        )
        runner._replay_graph = lambda shape_key, forward_batch: SimpleNamespace(
            next_token_logits=torch.zeros(BUCKET_BS * TOKENS_PER_REQUEST, 2),
            hidden_states=torch.zeros(BUCKET_BS * TOKENS_PER_REQUEST, 2),
        )

        forward_batch = SimpleNamespace(
            batch_size=raw_bs,
            input_ids=torch.arange(num_tokens, dtype=torch.int64),
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
            seq_lens_cpu=(
                torch.tensor(seq_lens, dtype=torch.int64) if cpu_mirror else None
            ),
            seq_lens_sum=sum(seq_lens),
            out_cache_loc=torch.arange(num_tokens, dtype=torch.int64),
            positions=torch.arange(num_tokens, dtype=torch.int64),
            req_pool_indices=torch.tensor(req_pool_indices, dtype=torch.int32),
            extend_seq_lens=None,
            extend_seq_lens_cpu=None,
            spec_info=self._new_spec_info(raw_bs),
        )
        runner.execute(forward_batch)
        return backend

    def _run_multi_layer(self, raw_bs, *, cpu_mirror):
        seq_lens, req_pool_indices = self._case_values(raw_bs)
        num_padding = BUCKET_BS - raw_bs
        padded_seq_lens = seq_lens + [SEQ_LEN_FILL_VALUE] * num_padding
        padded_req_pool_indices = req_pool_indices + [0] * num_padding

        backend = _ReplayMetadataSpy()
        buffers = self._new_buffers()
        buffers.seq_lens[:BUCKET_BS].copy_(
            torch.tensor(padded_seq_lens, dtype=torch.int32)
        )
        buffers.req_pool_indices[:BUCKET_BS].copy_(
            torch.tensor(padded_req_pool_indices, dtype=torch.int32)
        )
        buffers.seq_lens_cpu[:BUCKET_BS].copy_(
            torch.tensor(padded_seq_lens, dtype=torch.int64)
        )

        runner = MultiLayerEagleDraftExtendCudaGraphRunner.__new__(
            MultiLayerEagleDraftExtendCudaGraphRunner
        )
        runner.deepep_adapter = SimpleNamespace(replay=lambda: None)
        runner.buffers = buffers
        runner.captured_req_width = TOKENS_PER_REQUEST
        runner.forward_mode = ForwardMode.DRAFT_EXTEND_V2
        runner.metadata_captured_in_graph = False
        runner.step = 0
        runner.raw_bs = raw_bs
        runner.eagle_worker = SimpleNamespace(draft_extend_attn_backend_list=[backend])
        runner.model_runner = SimpleNamespace(device_timer=None)
        runner._make_graph_key = lambda bs: bs
        runner._replay_graph = lambda shape_key, forward_batch: SimpleNamespace()

        spec_info = EagleDraftExtendInput(
            hidden_states=None,
            num_correct_drafts=torch.full(
                (BUCKET_BS,), TOKENS_PER_REQUEST - 1, dtype=torch.int32
            ),
            num_accept_tokens=torch.full(
                (BUCKET_BS,), TOKENS_PER_REQUEST, dtype=torch.int32
            ),
        )
        runner.replay(
            BUCKET_BS,
            sum(padded_seq_lens),
            spec_info,
            buffers.seq_lens_cpu if cpu_mirror else None,
        )
        return backend

    def _assert_contract(self, backend, raw_bs, *, cpu_mirror):
        seq_lens, req_pool_indices = self._case_values(raw_bs)
        num_padding = BUCKET_BS - raw_bs
        expected_seq_lens = seq_lens + [SEQ_LEN_FILL_VALUE] * num_padding
        expected_req_pool_indices = req_pool_indices + [0] * num_padding
        expected_cpu_lens = expected_seq_lens if cpu_mirror else None

        self.assertEqual(len(backend.observations), 1)
        observation = backend.observations[0]
        self.assertEqual(
            {
                "batch_size": observation.batch_size,
                "num_padding": observation.num_padding,
                "req_pool_indices": observation.req_pool_indices.tolist(),
                "seq_lens": observation.seq_lens.tolist(),
                "seq_lens_sum": observation.seq_lens_sum,
                "seq_lens_cpu": (
                    None
                    if observation.seq_lens_cpu is None
                    else observation.seq_lens_cpu.tolist()
                ),
            },
            {
                "batch_size": BUCKET_BS,
                "num_padding": num_padding,
                "req_pool_indices": expected_req_pool_indices,
                "seq_lens": expected_seq_lens,
                "seq_lens_sum": sum(expected_seq_lens),
                "seq_lens_cpu": expected_cpu_lens,
            },
        )

        self.assertEqual(len(backend.replay_calls), 1)
        replay_call = backend.replay_calls[0]
        self.assertEqual(replay_call.batch_size, BUCKET_BS)
        self.assertEqual(replay_call.num_padding, num_padding)
        self.assertEqual(
            replay_call.req_pool_indices.tolist(), expected_req_pool_indices
        )
        if expected_cpu_lens is None:
            self.assertIsNone(replay_call.seq_lens_cpu)
        else:
            self.assertEqual(replay_call.seq_lens_cpu.tolist(), expected_cpu_lens)

    def test_single_layer_padded_replay_uses_bucket_view(self):
        self._assert_contract(
            self._run_single_layer(2, cpu_mirror=True),
            2,
            cpu_mirror=True,
        )

    def test_multi_layer_padded_replay_uses_bucket_view(self):
        self._assert_contract(
            self._run_multi_layer(2, cpu_mirror=True),
            2,
            cpu_mirror=True,
        )

    def test_single_layer_exact_bucket_reports_zero_padding(self):
        self._assert_contract(
            self._run_single_layer(BUCKET_BS, cpu_mirror=True),
            BUCKET_BS,
            cpu_mirror=True,
        )

    def test_multi_layer_exact_bucket_reports_zero_padding(self):
        self._assert_contract(
            self._run_multi_layer(BUCKET_BS, cpu_mirror=True),
            BUCKET_BS,
            cpu_mirror=True,
        )

    def test_single_layer_preserves_absent_cpu_mirror(self):
        self._assert_contract(
            self._run_single_layer(2, cpu_mirror=False),
            2,
            cpu_mirror=False,
        )

    def test_multi_layer_preserves_absent_cpu_mirror(self):
        self._assert_contract(
            self._run_multi_layer(2, cpu_mirror=False),
            2,
            cpu_mirror=False,
        )


if __name__ == "__main__":
    unittest.main()
