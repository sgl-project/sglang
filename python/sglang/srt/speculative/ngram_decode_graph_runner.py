from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import sglang.srt.model_executor.cuda_graph_runner as cgr
from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

if TYPE_CHECKING:
    from sglang.srt.speculative.ngram_worker import NGRAMWorker


class NgramDecodeCudaGraphRunner(CudaGraphRunner):
    """
    Cuda graph runner for decode batches using the dedicated attention backend.
    We patch graph runner with target_decode_attn_backend from ngram worker.
    The KV cache management including req_to_token_pool and token_to_kv_pool
    is shared between specdecode and non-specdecode.
    """

    def __init__(self, ngram_worker: "NGRAMWorker"):
        self.ngram_worker = ngram_worker
        self._target_model_runner = ngram_worker.target_worker.model_runner
        self._decode_attn_backend = ngram_worker.target_decode_attn_backend
        if self._decode_attn_backend is None:
            raise RuntimeError(
                "target_decode_attn_backend is required for decode graph"
            )

        with self._override_batch_sizes(), self._use_decode_attn_backend(
            self._target_model_runner
        ):
            backup_spec_algorithm = self._target_model_runner.spec_algorithm
            self._target_model_runner.spec_algorithm = SpeculativeAlgorithm.NONE
            super().__init__(self._target_model_runner)
            self._target_model_runner.spec_algorithm = backup_spec_algorithm

    @contextmanager
    def _use_decode_attn_backend(self, model_runner=None):
        runner = model_runner or self.model_runner
        original_backend = runner.attn_backend
        runner.attn_backend = self._decode_attn_backend
        try:
            yield
        finally:
            runner.attn_backend = original_backend

    @contextmanager
    def _override_batch_sizes(self):
        original = cgr.get_batch_sizes_to_capture

        def filtered(model_runner):
            backup_capture_bs = model_runner.server_args.cuda_graph_bs
            model_runner.server_args.cuda_graph_bs = (
                model_runner.server_args.capture_bs_for_decode
            )
            capture_bs, compile_bs = original(model_runner)
            threshold = model_runner.server_args.speculative_batch_size_threshold
            if threshold:
                capture_bs = [bs for bs in capture_bs if bs > threshold]
                if not capture_bs:
                    raise ValueError(
                        "speculative_batch_size_threshold filters out all capture sizes"
                    )
                compile_bs = [bs for bs in compile_bs if bs in capture_bs]
            model_runner.server_args.cuda_graph_bs = backup_capture_bs
            return capture_bs, compile_bs

        cgr.get_batch_sizes_to_capture = filtered
        try:
            yield
        finally:
            cgr.get_batch_sizes_to_capture = original

    def capture_one_batch_size(self, *args, **kwargs):
        with self._use_decode_attn_backend():
            return super().capture_one_batch_size(*args, **kwargs)

    def replay_prepare(self, *args, **kwargs):
        with self._use_decode_attn_backend():
            return super().replay_prepare(*args, **kwargs)
