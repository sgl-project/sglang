"""The mamba plan-stream verify fixup hook must refresh draft-produced tree links.

Under ``SGLANG_ENABLE_OVERLAP_PLAN_STREAM``, ``_replay_metadata`` copies
``spec_info.retrieve_next_token`` / ``retrieve_next_sibling`` into the captured
per-bs buffers on the plan stream, racing the draft's ``build_tree`` on the
compute stream. ``update_verify_buffers_to_fill_after_draft`` is the post-join
fixup that re-copies them into the ``cuda_graph_bs`` buffers; chain mode
(``topk == 1``), the eager path (``cuda_graph_bs is None``), and dummy/capture
runs (``retrieve_next_token is None``) must stay no-ops.

Pure host-side buffer logic — CPU tensors, no CUDA required.
"""

import unittest

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    MambaAttnBackendBase,
)
from sglang.srt.speculative.eagle_info import EagleVerifyInput
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=2, stage="base-b", runner_config="1-gpu-large")

_STALE = -555
_MAX_BS = 4
_DRAFT_TOKEN_NUM = 8


def _make_backend(topk: int) -> MambaAttnBackendBase:
    """Bare backend with only the fields the hook reads (no ModelRunner)."""
    backend = object.__new__(MambaAttnBackendBase)
    backend.topk = topk
    # Mirror init_cuda_graph_state's per-bs buffer shapes: (bs, draft_token_num).
    backend.retrieve_next_token_list = [
        torch.full((bs, _DRAFT_TOKEN_NUM), _STALE, dtype=torch.int32)
        for bs in range(1, _MAX_BS + 1)
    ]
    backend.retrieve_next_sibling_list = [
        torch.full((bs, _DRAFT_TOKEN_NUM), _STALE, dtype=torch.int32)
        for bs in range(1, _MAX_BS + 1)
    ]
    return backend


def _make_verify_input(bs_without_pad: int, base: int) -> EagleVerifyInput:
    """EagleVerifyInput carrying just the tree-link tensors the hook consumes."""
    spec_info = object.__new__(EagleVerifyInput)
    numel = bs_without_pad * _DRAFT_TOKEN_NUM
    spec_info.retrieve_next_token = (
        torch.arange(base, base + numel, dtype=torch.int32)
    ).reshape(bs_without_pad, _DRAFT_TOKEN_NUM)
    spec_info.retrieve_next_sibling = (
        torch.arange(base + numel, base + 2 * numel, dtype=torch.int32)
    ).reshape(bs_without_pad, _DRAFT_TOKEN_NUM)
    return spec_info


class TestVerifyBufferFixupHook(CustomTestCase):
    def test_refresh_overwrites_stale_links(self):
        backend = _make_backend(topk=2)
        cuda_graph_bs = _MAX_BS
        bs_without_pad = _MAX_BS - 1  # one padded row
        spec_info = _make_verify_input(bs_without_pad=bs_without_pad, base=100)

        backend.update_verify_buffers_to_fill_after_draft(
            spec_info=spec_info, cuda_graph_bs=cuda_graph_bs
        )

        for buf_list, fresh in (
            (backend.retrieve_next_token_list, spec_info.retrieve_next_token),
            (backend.retrieve_next_sibling_list, spec_info.retrieve_next_sibling),
        ):
            buf = buf_list[cuda_graph_bs - 1]
            self.assertTrue(
                torch.equal(buf[:bs_without_pad], fresh),
                "fresh tree links not copied into the captured buffer",
            )
            # The padded tail row is not covered by the copy.
            self.assertTrue(
                (buf[bs_without_pad:] == _STALE).all(),
                "rows beyond bs_without_pad must not be written",
            )
            # Buffers of other captured batch sizes stay untouched.
            for other_bs in range(1, _MAX_BS):
                self.assertTrue(
                    (buf_list[other_bs - 1] == _STALE).all(),
                    f"buffer for bs={other_bs} must stay untouched",
                )

    def test_chain_topk1_is_noop(self):
        backend = _make_backend(topk=1)
        spec_info = _make_verify_input(bs_without_pad=2, base=100)
        backend.update_verify_buffers_to_fill_after_draft(
            spec_info=spec_info, cuda_graph_bs=2
        )
        self.assertTrue((backend.retrieve_next_token_list[1] == _STALE).all())
        self.assertTrue((backend.retrieve_next_sibling_list[1] == _STALE).all())

    def test_eager_path_is_noop(self):
        backend = _make_backend(topk=2)
        spec_info = _make_verify_input(bs_without_pad=2, base=100)
        backend.update_verify_buffers_to_fill_after_draft(
            spec_info=spec_info, cuda_graph_bs=None
        )
        self.assertTrue((backend.retrieve_next_token_list[1] == _STALE).all())
        self.assertTrue((backend.retrieve_next_sibling_list[1] == _STALE).all())

    def test_dummy_run_none_links_is_noop(self):
        backend = _make_backend(topk=2)
        spec_info = object.__new__(EagleVerifyInput)
        spec_info.retrieve_next_token = None  # dummy / capture run
        spec_info.retrieve_next_sibling = None
        backend.update_verify_buffers_to_fill_after_draft(
            spec_info=spec_info, cuda_graph_bs=2
        )
        self.assertTrue((backend.retrieve_next_token_list[1] == _STALE).all())
        self.assertTrue((backend.retrieve_next_sibling_list[1] == _STALE).all())

    def test_non_eagle_spec_input_is_noop(self):
        backend = _make_backend(topk=2)
        backend.update_verify_buffers_to_fill_after_draft(
            spec_info=None, cuda_graph_bs=2
        )
        self.assertTrue((backend.retrieve_next_token_list[1] == _STALE).all())


if __name__ == "__main__":
    unittest.main()
