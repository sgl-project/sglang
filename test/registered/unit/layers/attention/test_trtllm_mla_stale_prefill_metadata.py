"""trtllm_mla keeps forward_prefill_metadata as persistent instance state.

Prefill CUDA-graph capture leaves it with fallback_to_flashinfer_impl=True; if
the decode runner's TARGET_VERIFY / DRAFT_EXTEND capture then consults it, the
flashinfer-wrapper attention gets recorded into the captured graph while
replay-prep later feeds it trtllm-gen-style metadata — the replayed graph is
born corrupted and hits an async illegal memory access (#31198, #28386).

Locks the two guards: the CUDA-graph metadata path must clear the stale state
(mirroring the eager-path clear in init_forward_metadata), and forward_extend's
prefill-fallback short-circuit must never apply to decode-family extend modes.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.attention.trtllm_mla_backend import (
    TRTLLMMLABackend,
    TRTLLMMLAPrefillMetadata,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _backend_with_stale_prefill_metadata() -> TRTLLMMLABackend:
    backend = TRTLLMMLABackend.__new__(TRTLLMMLABackend)
    backend.forward_prefill_metadata = TRTLLMMLAPrefillMetadata(
        max_seq_len=8,
        cum_seq_lens=torch.zeros(3, dtype=torch.int32),
        seq_lens=torch.full((2,), 8, dtype=torch.int32),
        fallback_to_flashinfer_impl=True,
    )
    return backend


def _capture_forward_batch(mode: ForwardMode) -> SimpleNamespace:
    return SimpleNamespace(
        forward_mode=mode,
        batch_size=2,
        positions=torch.zeros(4, dtype=torch.int64),
        seq_lens=torch.full((2,), 8, dtype=torch.int32),
        req_pool_indices=torch.zeros(2, dtype=torch.int32),
    )


@pytest.mark.parametrize(
    "mode",
    [ForwardMode.TARGET_VERIFY, ForwardMode.DECODE, ForwardMode.DRAFT_EXTEND_V2],
)
def test_capture_metadata_clears_stale_prefill_fallback(mode):
    backend = _backend_with_stale_prefill_metadata()
    with (
        patch.object(TRTLLMMLABackend, "_init_cuda_graph_metadata"),
        patch.object(TRTLLMMLABackend, "_apply_cuda_graph_metadata"),
    ):
        backend.init_forward_metadata_out_graph(
            _capture_forward_batch(mode), in_capture=True
        )
    assert backend.forward_prefill_metadata is None


def test_replay_prep_does_not_clear_live_prefill_metadata():
    # The non-capture (replay-prep) path runs every service step, where live
    # prefill metadata legitimately coexists with verify replays under the
    # overlap scheduler — it must NOT clear.
    backend = _backend_with_stale_prefill_metadata()
    with patch.object(TRTLLMMLABackend, "_apply_cuda_graph_metadata"):
        backend.init_forward_metadata_out_graph(
            _capture_forward_batch(ForwardMode.TARGET_VERIFY), in_capture=False
        )
    assert backend.forward_prefill_metadata is not None
