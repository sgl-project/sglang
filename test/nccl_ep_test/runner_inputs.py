"""Model-free fixture for SGLang's real decode load_batch/execute path.

Only model initialization, attention metadata, and the outer capture loop are
experiment scaffolding. DecodeInputBuffers, registry filling, bucket selection,
hidden-mode recapture, replay_session, and execute are the production code.
The backend and forward callback can also be the real NCCL EP implementation.
"""

from types import SimpleNamespace

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.cuda_graph_buffer_registry import (
    build_decode_registry,
    build_eager_registry,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.model_executor.runner.eager_runner import EagerRunner
from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
    FullCudaGraphBackend,
)
from sglang.srt.model_executor.runner_utils.buffers import DecodeInputBuffers
from sglang.srt.model_executor.runner_utils.deepep_adapter import (
    DeepEPCudaGraphRunnerAdapter,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


class MetadataObserver:
    """Attention is outside this MoE experiment; observe its actual replay view."""

    use_captured_forward_metadata_for_breakable_cuda_graph = False

    def __init__(self):
        self.views = []

    def init_forward_metadata_out_graph(self, view):
        self.views.append(view)

    def init_forward_metadata(self, view):
        self.views.append(view)


class SyntheticDecodeRunner(DecodeCudaGraphRunner):
    def __init__(
        self,
        forward,
        tp_group,
        *,
        buckets=(8, 16, 32),
        backend_factory=None,
        share_inputs=False
    ):
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.device_module = torch.cuda
        self.model_runner = SimpleNamespace(
            tp_group=tp_group,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            is_draft_worker=False,
            hisparse_coordinator=None,
            device_timer=None,
        )
        self.capture_bs = sorted(set(buckets))
        self.max_bs = self.max_num_token = max(self.capture_bs)
        self.captured_req_width = 1
        self.capture_forward_mode = ForwardMode.DECODE
        self.capture_hidden_mode = CaptureHiddenMode.NULL
        self.enable_return_hidden_states = False
        self.ragged_verify_mode = False
        self.require_mlp_tp_gather = self.require_mlp_sync = False
        self.enable_pdmux = self.enable_two_batch_overlap = False
        self.is_encoder_decoder = self.is_dllm = self.disable_padding = False
        self.seq_len_fill_value = 1
        self.attn_backend = MetadataObserver()
        self.deepep_adapter = DeepEPCudaGraphRunnerAdapter()
        self.buffers = DecodeInputBuffers.create(
            device=self.device,
            max_bs=self.max_bs,
            max_num_token=self.max_num_token,
            hidden_size=2048,
            next_token_logits_buffer=torch.empty(self.max_bs, 1, device=self.device),
            dtype=torch.bfloat16,
            dp_size=1,
            pp_size=1,
            is_encoder_decoder=False,
            require_mlp_tp_gather=False,
            seq_len_fill_value=1,
            encoder_len_fill_value=0,
            num_tokens_per_req=1,
            cache_loc_dtype=torch.int64,
            enable_mamba_track=False,
        )
        if share_inputs:
            self.buffers.share_buffers()
        self.buffer_registry = build_decode_registry(
            device=self.device,
            max_bs=self.max_bs,
            max_num_token=self.max_num_token,
            seq_len_fill_value=1,
            cache_loc_dtype=torch.int64,
            enable_num_token_non_padded=True,
            share_pool=False,
            source=self.buffers,
        )
        self.backend = (backend_factory or FullCudaGraphBackend)(self)
        self.synthetic_forward = forward
        self.capture_generations = 0
        self.capture()

    def capture(self):
        # The callback supplies synthetic experts instead of initializing a
        # language model. The backend still records actual CUDA Graphs.
        self.stream = torch.cuda.Stream()
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream), self.backend.capture_session(self.stream):
            for bucket in reversed(self.capture_bs):
                self.buffers.num_token_non_padded.fill_(bucket)
                batch = self.static_batch(bucket)
                self.backend.capture_one(
                    self._make_graph_key(bucket),
                    lambda batch=batch: self.synthetic_forward(batch),
                )
        torch.cuda.current_stream().wait_stream(self.stream)
        self.capture_generations += 1

    def static_batch(self, bucket):
        buffers = self.buffers
        return ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=bucket,
            input_ids=buffers.input_ids[:bucket],
            req_pool_indices=buffers.req_pool_indices[:bucket],
            seq_lens=buffers.seq_lens[:bucket],
            out_cache_loc=buffers.out_cache_loc[:bucket],
            seq_lens_sum=bucket,
            positions=buffers.positions[:bucket],
            num_token_non_padded=buffers.num_token_non_padded,
            capture_hidden_mode=self.capture_hidden_mode,
        )


class SyntheticEagerRunner(EagerRunner):
    """Model-free constructor; inherited execute/load_batch use the real registry."""

    def __init__(self, forward, capacity):
        self.enable_pdmux = False
        self.model_runner = SimpleNamespace(
            device_timer=None,
            attn_backend=MetadataObserver(),
            model=SimpleNamespace(
                forward=lambda ids, positions, batch, **kwargs: forward(batch)
            ),
            _pp_kwargs=lambda tensors: {},
            _extend_forward_kwargs=lambda batch, tensors: {},
            prefill_cuda_graph_runner=None,
        )
        self._eager_registry = build_eager_registry(
            device=torch.device("cuda", torch.cuda.current_device()),
            max_bs=capacity,
            max_num_token=capacity,
            cache_loc_dtype=torch.int64,
        )


def input_batch(values, *, valid_rows=None, hidden_mode=CaptureHiddenMode.NULL):
    values = torch.as_tensor(values, dtype=torch.int64, device="cuda")
    count = len(values)
    return ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        batch_size=count,
        input_ids=values,
        req_pool_indices=torch.arange(count, device="cuda"),
        seq_lens=torch.ones(count, dtype=torch.int64, device="cuda"),
        out_cache_loc=torch.arange(count, device="cuda"),
        seq_lens_sum=count,
        seq_lens_cpu=torch.ones(count, dtype=torch.int64),
        positions=torch.arange(count, device="cuda"),
        num_token_non_padded=torch.tensor(
            [count if valid_rows is None else valid_rows],
            dtype=torch.int32,
            device="cuda",
        ),
        capture_hidden_mode=hidden_mode,
    )


def exercise_inputs(*, recapture=False):
    """Local ordinary CUDA Graph proof through the actual decode runner."""
    from sglang.srt.layers.moe.utils import MoeA2ABackend
    from sglang.srt.runtime_context import get_flags

    get_flags().moe.a2a_backend = MoeA2ABackend.NONE

    def forward(batch):
        row = torch.arange(batch.batch_size, device="cuda")
        output = torch.where(
            row < batch.num_token_non_padded,
            batch.input_ids * 2 + batch.positions,
            -torch.ones_like(batch.input_ids),
        )
        return LogitsProcessorOutput(next_token_logits=output[:, None])

    runner = SyntheticDecodeRunner(
        forward, SimpleNamespace(barrier=lambda: None), buckets=(8, 16)
    )
    inputs = (
        input_batch([1, 2, 3, 4, 5]),
        input_batch(list(range(9)), valid_rows=0),
        input_batch(
            [1, 2, 3, 4, 5],
            hidden_mode=CaptureHiddenMode.FULL if recapture else CaptureHiddenMode.NULL,
        ),
    )
    expected = ([2, 5, 8, 11, 14], [-1] * 9, [2, 5, 8, 11, 14])
    addresses = (
        runner.buffers.input_ids.data_ptr(),
        runner.buffers.positions.data_ptr(),
    )
    selected, valid = [], []
    try:
        for batch, wanted in zip(inputs, expected):
            # A changed hidden mode triggers production load_batch's recapture.
            # can_run_graph would normally select another runner for this
            # request; call execute directly to cover its recapture contract.
            actual = runner.execute(batch).next_token_logits
            torch.testing.assert_close(
                actual.cpu()[:, 0], torch.tensor(wanted), rtol=0, atol=0
            )
            selected.append(runner.attn_backend.views[-1].batch_size)
            valid.append(int(runner.buffers.num_token_non_padded.item()))
            assert addresses == (
                runner.buffers.input_ids.data_ptr(),
                runner.buffers.positions.data_ptr(),
            )
        assert not runner.can_run_graph(input_batch(list(range(17))))
        return {
            "selected_buckets": selected,
            "valid_rows": valid,
            "recapture_generations": runner.capture_generations,
            "ep_tested": False,
            "model_loaded": False,
            "runner_constructor": "synthetic fixture; real load_batch/execute/backend",
        }
    finally:
        torch.cuda.synchronize()
        runner.backend.cleanup()
