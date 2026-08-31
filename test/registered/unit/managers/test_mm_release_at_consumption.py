"""Multimodal release-at-consumption lifecycle (opt-in).

With SGLANG_ENABLE_MM_RELEASE_AT_CONSUMPTION=1, image features are dropped
once their token spans are fully inside the KV cache (the radix tree retains
the prompt KV for retract re-prefill), instead of being retained per TP rank
until request completion.
"""

from __future__ import annotations

import gc
import os
import sys
from array import array
from http import HTTPStatus
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

if TYPE_CHECKING:
    from sglang.srt.managers.io_struct import AbortReq
    from sglang.srt.managers.schedule_batch import MultimodalDataItem, Req


def _types():
    from sglang.srt.managers.mm_utils import (
        ShmPointerMMData,
        release_consumed_mm_features,
    )
    from sglang.srt.managers.schedule_batch import (
        Modality,
        MultimodalDataItem,
        MultimodalInputs,
        Req,
    )

    return (
        ShmPointerMMData,
        release_consumed_mm_features,
        Modality,
        MultimodalDataItem,
        MultimodalInputs,
        Req,
    )


def _image_item(offsets: list[tuple[int, int]], feature: object) -> MultimodalDataItem:
    _, _, Modality, MultimodalDataItem, _, _ = _types()
    return MultimodalDataItem(
        modality=Modality.IMAGE,
        offsets=offsets,
        feature=cast(torch.Tensor, feature),
        hash=1,
        pad_value=1,
    )


def _req(rid: str, input_len: int = 3) -> Req:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    return Req(rid, "", array("q", range(input_len)), SamplingParams())


def test_release_and_reprefill_boundaries() -> None:
    _, release, _, _, MultimodalInputs, _ = _types()
    base = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    items = [
        _image_item([(0, 1)], base[0:2]),
        _image_item([(2, 3)], [base[2:3], base[3:4]]),  # list-backed feature
        _image_item([(4, 5)], base[4:6]),
    ]
    mm_inputs = MultimodalInputs(mm_items=items)

    # Chunk [0, 4): items ending inside are released, the straddler is kept.
    release([mm_inputs], [0], [4])
    assert items[0].feature is None and items[0].feature_released
    assert items[1].feature is None and items[1].feature_released
    assert isinstance(items[2].feature, torch.Tensor)
    assert not items[2].feature_released

    # Re-prefill guard: safe while the cached prefix covers released spans.
    assert not mm_inputs.has_released_items_beyond_prefix(4)
    assert mm_inputs.has_released_items_beyond_prefix(3)


def test_release_policy_requires_opt_in_radix_and_pp1() -> None:
    from sglang.srt.environ import envs
    from sglang.srt.managers.mm_utils import should_release_mm_features

    def args(**kw: object) -> SimpleNamespace:
        return SimpleNamespace(**{"disable_radix_cache": False, "pp_size": 1, **kw})

    assert not should_release_mm_features(args())  # opt-in flag unset
    with envs.SGLANG_ENABLE_MM_RELEASE_AT_CONSUMPTION.override(True):
        assert should_release_mm_features(args())
        assert not should_release_mm_features(args(disable_radix_cache=True))
        assert not should_release_mm_features(args(pp_size=2))


def test_released_feature_cache_miss_aborts_retryable_503() -> None:
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.runtime_context import get_context

    _, _, _, _, MultimodalInputs, _ = _types()
    override = get_context().override_server_args()
    override.install()
    try:
        scheduler = object.__new__(Scheduler)
        scheduler.enable_hicache_storage = False
        scheduler.enable_hierarchical_cache = False
        scheduler.tree_cache = None
        retired: list[object] = []
        scheduler.beam_coordinator = SimpleNamespace(retire_group=retired.append)
        sent: list[tuple[object, object]] = []
        object.__setattr__(
            scheduler,
            "ipc_channels",
            SimpleNamespace(
                send_to_tokenizer=SimpleNamespace(
                    send_output=lambda out, req=None: sent.append((out, req))
                )
            ),
        )

        req = _req("rid-mm-abort")
        released = _image_item([(0, 99)], torch.zeros(4, 2))
        released.release_feature(consumed=True)
        pending = _image_item([(100, 199)], torch.zeros(4, 2))
        req.multimodal_inputs = MultimodalInputs(mm_items=[released, pending])
        req.reset_for_retract()

        scheduler._abort_mm_req_with_released_features(req)

        abort_output, abort_target = sent[0]
        abort_req = cast("AbortReq", abort_output)
        assert abort_target is req
        assert abort_req.finished_reason is not None
        assert (
            abort_req.finished_reason["status_code"] == HTTPStatus.SERVICE_UNAVAILABLE
        )
        # Terminal cleanup: remaining features dropped, beam group retired.
        assert pending.feature is None
        assert retired == [req]
    finally:
        override.restore()


def test_terminal_release_spares_session_features() -> None:
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.runtime_context import get_context

    _, _, _, _, MultimodalInputs, _ = _types()
    override = get_context().override_server_args()
    override.install()
    try:
        scheduler = object.__new__(Scheduler)
        scheduler.enable_hicache_storage = False
        scheduler.enable_priority_scheduling = False
        scheduler.max_queued_requests = 0
        scheduler.waiting_queue = []
        object.__setattr__(
            scheduler,
            "ipc_channels",
            SimpleNamespace(
                send_to_tokenizer=SimpleNamespace(
                    send_output=lambda out, req=None: None
                )
            ),
        )

        session_req = _req("rid-session", input_len=1)
        session_req.session = object()  # later turns share these mm_inputs
        session_item = _image_item([(0, 0)], torch.zeros(1, 2))
        session_req.multimodal_inputs = MultimodalInputs(mm_items=[session_item])
        assert scheduler._abort_on_queued_limit(session_req)
        assert session_item.feature is not None

        plain_req = _req("rid-plain", input_len=1)
        plain_item = _image_item([(0, 0)], torch.zeros(1, 2))
        plain_req.multimodal_inputs = MultimodalInputs(mm_items=[plain_item])
        assert scheduler._abort_on_queued_limit(plain_req)
        assert plain_item.feature is None
    finally:
        override.restore()


@pytest.mark.skipif(sys.platform != "linux", reason="requires /dev/shm and /proc")
def test_shm_release_keeps_views_alive() -> None:
    ShmPointerMMData, _, _, _, _, _ = _types()
    src = torch.arange(64, dtype=torch.float32)
    sender = ShmPointerMMData(src)
    receiver = ShmPointerMMData.__new__(ShmPointerMMData)
    receiver.__setstate__(sender.__getstate__())
    assert receiver.tensor is not None
    view = receiver.tensor[8:16]
    name = sender.shm_name

    receiver.release()

    assert not os.path.exists(f"/dev/shm/{name}")
    # The view must still read the original data (no use-after-unmap).
    torch.testing.assert_close(view, src[8:16])
    del view, receiver
    gc.collect()
    with open("/proc/self/maps") as maps:
        assert name not in maps.read()
