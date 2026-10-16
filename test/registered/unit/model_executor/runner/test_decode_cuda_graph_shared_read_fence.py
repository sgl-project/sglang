from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import (
    AttentionBackend,
    SharedReadEnds,
)
from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend
from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.speculative.draft_extend_read_fence import (
    DraftExtendFencePolicy,
    DraftExtendReadFence,
)
from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
    EAGLEDraftExtendCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

DECODE = ForwardMode.DECODE


def _runner(*, has_marker: bool = False):
    runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
    runner.model_runner = SimpleNamespace(shared_read_done_event=None)
    runner.in_graph_metadata_prep_done = object() if has_marker else None
    return runner


def _backend(declared: SharedReadEnds):
    # Spec'd against the real ABC so a rename fails here, not at runtime.
    backend = create_autospec(AttentionBackend, instance=True)
    backend.shared_read_ends.return_value = declared
    return backend


@pytest.mark.parametrize(
    "declared, has_marker, expected",
    [
        # The backend's declaration decides where the record lands.
        (SharedReadEnds.IN_REPLAY, True, SharedReadEnds.IN_REPLAY),
        # No external-event support (e.g. HIP): IN_REPLAY cannot be recorded
        # in-graph, and a PRE_REPLAY fence would not cover the replay reads.
        # The sound fallback is POST_REPLAY.
        (SharedReadEnds.IN_REPLAY, False, SharedReadEnds.POST_REPLAY),
        # Only an in-graph declaration is demoted; the rest pass through.
        (SharedReadEnds.POST_REPLAY, False, SharedReadEnds.POST_REPLAY),
    ],
)
def test_resolve_shared_read_ends(declared, has_marker, expected):
    runner = _runner(has_marker=has_marker)
    backend = _backend(declared)

    assert runner._resolve_shared_read_ends(backend, DECODE) is expected
    backend.shared_read_ends.assert_called_once_with(DECODE)


def test_publish_read_done():
    runner = _runner(has_marker=True)
    recorded = []
    runner.device_module = SimpleNamespace(
        Event=lambda: SimpleNamespace(record=lambda: recorded.append("record"))
    )

    runner._publish_read_done(in_graph=True)
    # In-graph: hand over the graph-recorded marker, do not record a new event.
    marker = runner.in_graph_metadata_prep_done
    assert runner.model_runner.shared_read_done_event is marker
    assert recorded == []

    runner._publish_read_done(in_graph=False)
    assert recorded == ["record"]
    assert runner.model_runner.shared_read_done_event is not marker


DRAFT_EXTEND = ForwardMode.DRAFT_EXTEND_V2


def _extend_runner(*, declared: SharedReadEnds, fence, has_marker: bool):
    runner = EAGLEDraftExtendCudaGraphRunner.__new__(EAGLEDraftExtendCudaGraphRunner)
    runner.model_runner = SimpleNamespace(shared_read_done_event=None)
    runner.in_graph_metadata_prep_done = object() if has_marker else None
    runner.forward_mode = DRAFT_EXTEND
    runner.draft_extend_attn_backend = _backend(declared)
    runner.read_fence = fence
    return runner


class _UnguardedAllocator(BaseTokenToKVPoolAllocator):
    def __init__(self, *, need_sort):
        super().__init__(
            size=64,
            page_size=4,
            dtype=torch.float16,
            device="cpu",
            kvcache=None,
            need_sort=need_sort,
        )

    def alloc(self, need_size):
        raise NotImplementedError

    def clear(self):
        pass

    def free(self, free_index):
        raise NotImplementedError


def _allocator(*, need_sort, guarded=True):
    if not guarded:
        return _UnguardedAllocator(need_sort=need_sort)
    return PagedTokenToKVPoolAllocator(
        size=64,
        page_size=4,
        dtype=torch.float16,
        device="cpu",
        kvcache=None,
        need_sort=need_sort,
    )


def _fence(*, ctl_path=None, allow_pre_replay=False):
    # Tests drive the control-file poll by hand instead of the background thread.
    return DraftExtendReadFence(
        capture_marker=True,
        allow_pre_replay=allow_pre_replay,
        ctl_path=ctl_path,
        start_poller=False,
    )


@pytest.mark.parametrize("declared", list(SharedReadEnds))
def test_draft_extend_read_ends_without_fence_is_stock(declared):
    # Switches off: the backend's declaration passes through untouched.
    runner = _extend_runner(declared=declared, fence=None, has_marker=False)
    assert runner._draft_extend_shared_read_ends() is declared


def test_draft_extend_fence_disabled_by_default():
    with (
        envs.SGLANG_Q35MF_PDDEC_FENCE.override(False),
        envs.SGLANG_Q35MF_PDDEC_FENCE_PREREPLAY_UNSAFE.override(False),
        envs.SGLANG_Q35MF_PDDEC_FENCE_AB_CTL.override(""),
    ):
        assert (
            DraftExtendReadFence.from_env(allocator=_allocator(need_sort=True)) is None
        )
        backend = TRTLLMHAAttnBackend.__new__(TRTLLMHAAttnBackend)
        assert backend.shared_read_ends(DRAFT_EXTEND) is SharedReadEnds.UNKNOWN


def test_trtllm_mha_declares_in_replay_draft_extend_only_when_enabled():
    backend = TRTLLMHAAttnBackend.__new__(TRTLLMHAAttnBackend)
    with envs.SGLANG_Q35MF_PDDEC_FENCE.override(True):
        assert backend.shared_read_ends(DRAFT_EXTEND) is SharedReadEnds.IN_REPLAY
        # Other modes keep their declarations.
        assert backend.shared_read_ends(ForwardMode.EXTEND) is SharedReadEnds.PRE_REPLAY
        assert backend.shared_read_ends(DECODE) is SharedReadEnds.IN_REPLAY
        assert (
            backend.shared_read_ends(ForwardMode.TARGET_VERIFY)
            is SharedReadEnds.IN_REPLAY
        )


@pytest.mark.parametrize(
    "has_marker, expected",
    [
        (True, SharedReadEnds.IN_REPLAY),
        # No captured marker: only a post-replay event covers the graph's reads.
        (False, SharedReadEnds.POST_REPLAY),
    ],
)
def test_draft_extend_in_graph_policy(has_marker, expected):
    fence = _fence()
    runner = _extend_runner(
        declared=SharedReadEnds.IN_REPLAY, fence=fence, has_marker=has_marker
    )
    assert runner._draft_extend_shared_read_ends() is expected
    key = "in_graph" if has_marker else "in_graph_unavailable"
    assert fence.policy_round_ct[key] == 1


def test_draft_extend_in_graph_policy_keeps_unknown_backend_coarse():
    # A backend that does not declare in-replay reads keeps the whole-forward fence.
    fence = _fence()
    runner = _extend_runner(
        declared=SharedReadEnds.UNKNOWN, fence=fence, has_marker=True
    )
    assert runner._draft_extend_shared_read_ends() is SharedReadEnds.UNKNOWN
    assert fence.policy_round_ct["in_graph_unavailable"] == 1


def test_draft_extend_ab_ctl_switches_policy_between_rounds(tmp_path):
    ctl = tmp_path / "fence_policy"
    fence = _fence(ctl_path=str(ctl), allow_pre_replay=True)
    runner = _extend_runner(
        declared=SharedReadEnds.IN_REPLAY, fence=fence, has_marker=True
    )

    # Missing control file: stock, i.e. no event and the whole-forward fence.
    fence._poll_ctl_once()
    assert runner._draft_extend_shared_read_ends() is SharedReadEnds.UNKNOWN

    ctl.write_text("in_graph\n")
    # Rounds only see what the poller last read.
    assert runner._draft_extend_shared_read_ends() is SharedReadEnds.UNKNOWN
    fence._poll_ctl_once()
    assert runner._draft_extend_shared_read_ends() is SharedReadEnds.IN_REPLAY

    ctl.write_text("pre_replay")
    fence._poll_ctl_once()
    assert runner._draft_extend_shared_read_ends() is SharedReadEnds.PRE_REPLAY

    ctl.write_text("bogus")
    fence._poll_ctl_once()
    assert runner._draft_extend_shared_read_ends() is SharedReadEnds.UNKNOWN

    assert fence.snapshot() == {
        "policy": DraftExtendFencePolicy.STOCK.value,
        "round_ct": 5,
        "stock": 3,
        "in_graph": 1,
        "pre_replay": 1,
        "in_graph_unavailable": 0,
    }


def test_draft_extend_pre_replay_needs_unsafe_switch(tmp_path):
    ctl = tmp_path / "fence_policy"
    ctl.write_text("pre_replay")
    fence = _fence(ctl_path=str(ctl), allow_pre_replay=False)
    runner = _extend_runner(
        declared=SharedReadEnds.IN_REPLAY, fence=fence, has_marker=True
    )
    fence._poll_ctl_once()
    assert runner._draft_extend_shared_read_ends() is SharedReadEnds.UNKNOWN


def test_draft_extend_static_policies_from_env():
    with (
        envs.SGLANG_Q35MF_PDDEC_FENCE.override(True),
        envs.SGLANG_Q35MF_PDDEC_FENCE_PREREPLAY_UNSAFE.override(False),
        envs.SGLANG_Q35MF_PDDEC_FENCE_AB_CTL.override(""),
    ):
        fence = DraftExtendReadFence.from_env(allocator=_allocator(need_sort=False))
        assert fence.capture_marker
        assert fence.policy_for_round() is DraftExtendFencePolicy.IN_GRAPH
    with (
        envs.SGLANG_Q35MF_PDDEC_FENCE.override(False),
        envs.SGLANG_Q35MF_PDDEC_FENCE_PREREPLAY_UNSAFE.override(True),
        envs.SGLANG_Q35MF_PDDEC_FENCE_AB_CTL.override(""),
    ):
        fence = DraftExtendReadFence.from_env(allocator=_allocator(need_sort=False))
        # The measurement arm needs no marker.
        assert not fence.capture_marker
        assert fence.policy_for_round() is DraftExtendFencePolicy.PRE_REPLAY


def test_draft_extend_fence_off_under_pd_without_reuse_guard():
    # PD transfers into reused pages are not stream-ordered: no guard, no fine fence.
    with (
        envs.SGLANG_Q35MF_PDDEC_FENCE.override(True),
        envs.SGLANG_Q35MF_PDDEC_FENCE_PREREPLAY_UNSAFE.override(False),
        envs.SGLANG_Q35MF_PDDEC_FENCE_AB_CTL.override(""),
    ):
        from_env = DraftExtendReadFence.from_env
        assert from_env(allocator=_allocator(need_sort=True, guarded=False)) is None
        assert from_env(allocator=_allocator(need_sort=True)) is not None
        assert (
            from_env(allocator=_allocator(need_sort=False, guarded=False)) is not None
        )


@pytest.mark.parametrize(
    "ends, has_marker, guarded",
    [
        (SharedReadEnds.IN_REPLAY, True, True),
        (SharedReadEnds.PRE_REPLAY, False, True),
        # These fences already wait for the end of the replay.
        (SharedReadEnds.IN_REPLAY, False, False),
        (SharedReadEnds.POST_REPLAY, False, False),
        (SharedReadEnds.UNKNOWN, True, False),
    ],
)
def test_draft_extend_kv_writes_need_guard_only_when_fence_ends_early(
    ends, has_marker, guarded
):
    runner = _extend_runner(declared=ends, fence=None, has_marker=has_marker)
    runner.device_module = SimpleNamespace(
        Event=lambda: SimpleNamespace(record=lambda: None)
    )
    assert (runner._kv_writes_done_event(ends) is not None) is guarded


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
