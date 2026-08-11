import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention import tokenspeed_mla_backend as backend_module
from sglang.srt.layers.attention.tokenspeed_mla_backend import TokenspeedMLABackend
from sglang.srt.layers.attention.trtllm_mla_backend import TRTLLMMLADecodeMetadata
from sglang.srt.layers.dcp.layout import get_dcp_lens
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import is_xpu
from sglang.test.ci.ci_register import register_cuda_ci, register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="4-gpu-b200")
register_xpu_ci(est_time=60, suite="stage-b-test-1-gpu-xpu")

NUM_DRAFT_TOKENS = 8
DCP_SIZE = 4
DCP_RANK = 2

# The metadata buffers are plain int32 tensors and the DCP length math is pure
# index arithmetic, so this test is device-agnostic; only the buffers' device
# needs to follow the platform. _fill_dcp_block_kv_indices is mocked out, so no
# device kernel is launched.
_DEVICE = "cuda" if torch.cuda.is_available() else ("xpu" if is_xpu() else None)


def _make_backend(bs: int):
    backend = object.__new__(TokenspeedMLABackend)
    backend.num_draft_tokens = NUM_DRAFT_TOKENS
    metadata = TRTLLMMLADecodeMetadata(
        block_kv_indices=torch.full((bs, 4), -1, dtype=torch.int32, device=_DEVICE),
        seq_lens_k=torch.zeros(bs, dtype=torch.int32, device=_DEVICE),
        global_seq_lens_k=torch.zeros(bs, dtype=torch.int32, device=_DEVICE),
    )
    backend.decode_cuda_graph_metadata = {bs: metadata}
    return backend, metadata


def _apply(backend, *, bs: int, seq_lens: torch.Tensor, forward_mode):
    parallel = SimpleNamespace(dcp_enabled=True, dcp_size=DCP_SIZE, dcp_rank=DCP_RANK)
    with (
        patch.object(backend_module, "get_parallel", return_value=parallel),
        patch.object(backend, "_fill_dcp_block_kv_indices") as fill,
    ):
        backend._apply_cuda_graph_metadata(
            bs=bs,
            req_pool_indices=torch.arange(bs, dtype=torch.int32, device=_DEVICE),
            seq_lens=seq_lens,
            forward_mode=forward_mode,
        )
    return fill


@unittest.skipUnless(_DEVICE is not None, "DCP metadata buffers need CUDA or XPU")
class TestTokenspeedMLADCPMetadata(CustomTestCase):
    def test_target_verify_splits_global_and_local_lengths(self):
        bs = 3
        backend, metadata = _make_backend(bs)
        prefix_lens = torch.tensor([10, 20, 30], dtype=torch.int32, device=_DEVICE)

        fill = _apply(
            backend,
            bs=bs,
            seq_lens=prefix_lens,
            forward_mode=ForwardMode.TARGET_VERIFY,
        )

        expected_global = prefix_lens + NUM_DRAFT_TOKENS
        expected_local = get_dcp_lens(expected_global, DCP_SIZE, DCP_RANK).to(
            torch.int32
        )
        torch.testing.assert_close(metadata.global_seq_lens_k, expected_global)
        torch.testing.assert_close(metadata.seq_lens_k, expected_local)
        fill.assert_called_once()
        torch.testing.assert_close(fill.call_args.args[2], expected_local)

    def test_decode_does_not_add_draft_tokens(self):
        bs = 3
        backend, _ = _make_backend(bs)
        seq_lens = torch.tensor([10, 20, 30], dtype=torch.int32, device=_DEVICE)

        fill = _apply(
            backend, bs=bs, seq_lens=seq_lens, forward_mode=ForwardMode.DECODE
        )

        expected_local = get_dcp_lens(seq_lens, DCP_SIZE, DCP_RANK).to(torch.int32)
        fill.assert_called_once()
        torch.testing.assert_close(fill.call_args.args[2], expected_local)


if __name__ == "__main__":
    unittest.main()
