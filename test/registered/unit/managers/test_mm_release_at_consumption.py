"""Multimodal release-at-consumption lifecycle (opt-in).

With SGLANG_ENABLE_MM_RELEASE_AT_CONSUMPTION=1, image features are dropped
once their token spans are fully inside the KV cache (the radix tree retains
the prompt KV for retract re-prefill), instead of being retained per TP rank
until request completion.
"""

from __future__ import annotations

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
    from sglang.srt.managers.mm_utils import release_consumed_mm_features
    from sglang.srt.managers.schedule_batch import (
        Modality,
        MultimodalDataItem,
        MultimodalInputs,
        Req,
    )

    return (
        release_consumed_mm_features,
        Modality,
        MultimodalDataItem,
        MultimodalInputs,
        Req,
    )


def _image_item(offsets: list[tuple[int, int]], feature: object) -> MultimodalDataItem:
    _, Modality, MultimodalDataItem, _, _ = _types()
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
    release, _, _, MultimodalInputs, _ = _types()
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


def _abort_scheduler():
    from sglang.srt.managers.scheduler import Scheduler

    scheduler = object.__new__(Scheduler)
    finished: list[object] = []
    scheduler.tree_cache = SimpleNamespace(
        finish=lambda handle, outcome: finished.append(handle)
    )
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
    return scheduler, sent, retired, finished


def _assert_retryable_abort(sent: list[tuple[object, object]], req: Req) -> None:
    abort_output, abort_target = sent[0]
    abort_req = cast("AbortReq", abort_output)
    assert abort_target is req
    assert abort_req.finished_reason is not None
    assert abort_req.finished_reason["status_code"] == HTTPStatus.SERVICE_UNAVAILABLE


def test_released_feature_cache_miss_aborts_retryable_503() -> None:
    from sglang.srt.runtime_context import get_context

    _, _, _, MultimodalInputs, _ = _types()
    override = get_context().override_server_args()
    override.install()
    try:
        scheduler, sent, retired, finished = _abort_scheduler()

        req = _req("rid-mm-abort")
        released = _image_item([(0, 99)], torch.zeros(4, 2))
        released.release_feature(consumed=True)
        pending = _image_item([(100, 199)], torch.zeros(4, 2))
        req.multimodal_inputs = MultimodalInputs(mm_items=[released, pending])
        req.reset_for_retract()

        scheduler._abort_mm_req_with_released_features(req)

        _assert_retryable_abort(sent, req)
        # Terminal cleanup: remaining features dropped, beam group retired.
        assert pending.feature is None
        assert retired == [req]
        assert finished == [req.cache_request_handle]
    finally:
        override.restore()


def test_session_turn_with_evicted_released_history_aborts_503() -> None:
    from sglang.srt.runtime_context import get_context

    release, _, _, MultimodalInputs, _ = _types()
    override = get_context().override_server_args()
    override.install()
    try:
        scheduler, sent, retired, _ = _abort_scheduler()

        history = _image_item([(0, 9)], torch.zeros(4, 2))
        release([MultimodalInputs(mm_items=[history])], [0], [10])
        assert history.feature_released

        req = _req("rid-session-turn", input_len=20)
        req.session = SimpleNamespace(streaming=False)
        new_media = _image_item([(12, 15)], torch.zeros(4, 2))
        req.multimodal_inputs = MultimodalInputs(mm_items=[history, new_media])
        req._session_mm_items = [new_media]
        assert req.multimodal_inputs.has_released_items_beyond_prefix(0)

        scheduler._abort_mm_req_with_released_features(req)

        _assert_retryable_abort(sent, req)
        assert new_media.feature is None
        assert req.multimodal_inputs is None
        assert retired == [req]
    finally:
        override.restore()


def test_retraction_holds_prompt_kv_after_media_release() -> None:
    from unittest.mock import MagicMock

    from sglang.srt.managers.schedule_batch import ReqKvInfo, release_req
    from sglang.srt.mem_cache.base_prefix_cache import (
        EvictParams,
        InsertParams,
        MatchPrefixParams,
    )
    from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
    from sglang.srt.runtime_context import get_context

    _, _, _, MultimodalInputs, _ = _types()
    prompt = array("q", range(10))
    override = get_context().override_server_args()
    override.install()
    try:
        allocator = MagicMock(device=torch.device("cpu"), page_size=1)
        tree = RadixCache.create_simulated(mock_allocator=allocator)
        tree.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.zeros(1, 16, dtype=torch.int64),
            free=lambda req: setattr(req.kv, "req_pool_idx", None),
        )

        def retract(released: bool) -> Req:
            tree.insert(InsertParams(key=RadixKey(prompt), value=torch.arange(10)))
            node = tree.match_prefix(
                MatchPrefixParams(key=RadixKey(prompt))
            ).last_device_node
            req = _req("rid-retract", input_len=10)
            req.kv = ReqKvInfo(
                req_pool_idx=0,
                kv_committed_len=10,
                kv_allocated_len=10,
                cache_protected_len=10,
            )
            req.lock = tree.lock(node)
            item = _image_item([(2, 5)], torch.zeros(4, 2))
            if released:
                item.release_feature(consumed=True)
            req.multimodal_inputs = MultimodalInputs(mm_items=[item])
            assert release_req(
                req=req,
                remaing_req_count=0,
                req_to_token_pool=tree.req_to_token_pool,
                token_to_kv_pool_allocator=allocator,
                tree_cache=tree,
                hisparse_coordinator=None,
                requeue=True,
            )
            tree.evict(EvictParams(num_tokens=100))
            return req

        def cached_len() -> int:
            match = tree.match_prefix(MatchPrefixParams(key=RadixKey(prompt)))
            return int(match.device_prefix_len)

        # Released media can only come back from the cached prompt.
        req = retract(released=True)
        assert cached_len() == 10
        req.release_mm_prefix_hold(tree)
        tree.evict(EvictParams(num_tokens=100))
        assert cached_len() == 0

        req = retract(released=False)
        assert req.mm_prefix_hold is None
        assert cached_len() == 0
    finally:
        override.restore()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
