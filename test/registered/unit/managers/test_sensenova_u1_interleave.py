# SPDX-License-Identifier: Apache-2.0

from array import array
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    FINISH_LENGTH,
    FINISH_MATCHED_TOKEN,
    Req,
    ReqKvInfo,
    ScheduleBatch,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.sensenova_u1_interleave import (
    SenseNovaU1InterleaveController,
)
from sglang.srt.models.neo_chat_limits import (
    U1_EXACT_TEXT_CUSTOM_PARAM,
    U1_FLOW_BATCH_ISOLATION_PARAM,
    U1_FLOW_CUSTOM_PARAM,
    U1_FLOW_RADIX_PREFIX_LIMIT_PARAM,
    U1_INTERLEAVE_CUSTOM_PARAM,
    U1_MAX_INTERLEAVE_IMAGES,
    normalize_u1_interleave_request,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

IMG_START = 10
IMG_CONTEXT = 11
IMG_END = 12
EOS = 99


class _FakeTokenizer:
    eos_token_id = EOS


class _FakeAllocator:
    page_size = 1

    def free_segment(self, *_args, **_kwargs):
        return None


class _FakeReqPool:
    def __init__(self):
        self.req_to_token = torch.arange(4096, dtype=torch.int32).reshape(4, 1024)

    def free(self, req):
        req.req_pool_idx = None


class _FakeTreeCache:
    def __init__(self):
        self.req_to_token_pool = _FakeReqPool()
        self.token_to_kv_pool_allocator = _FakeAllocator()
        self.cached_unfinished = []
        self.cached_finished = []

    @staticmethod
    def supports_mamba():
        return False

    def cache_unfinished_req(self, req, **_kwargs):
        fill_ids = list(req.get_fill_ids())
        self.cached_unfinished.append((req.rid, fill_ids))
        req.prefix_indices = (
            self.req_to_token_pool.req_to_token[req.req_pool_idx, : len(fill_ids)]
            .to(torch.int64)
            .clone()
        )
        req.cache_protected_len = len(fill_ids)

    def cache_finished_req(self, req, is_insert=True, *, kv_len_to_handle):
        self.cached_finished.append((req.rid, is_insert, kv_len_to_handle))


class _FakeOutputStreamer:
    def __init__(self):
        self.calls = []

    def stream_output(self, reqs, return_logprob, skip_req=None):
        self.calls.append((list(reqs), return_logprob, skip_req))


class _FakeScheduler:
    def __init__(self, *, max_req_len=4096, admit_internal=True):
        self.enable_overlap = False
        self.spec_algorithm = SimpleNamespace(is_none=lambda: True)
        self.disaggregation_mode = DisaggregationMode.NULL
        self.ps = SimpleNamespace(pp_size=1)
        self.enable_hisparse = False
        self.tree_cache = _FakeTreeCache()
        self.model_config = SimpleNamespace(
            hf_config=SimpleNamespace(
                patch_size=16,
                downsample_ratio=0.5,
            )
        )
        self.max_req_len = max_req_len
        self.waiting_queue = []
        self.output_streamer = _FakeOutputStreamer()
        self.admit_internal = admit_internal

    def _add_internal_request_to_queue(self, req):
        if not self.admit_internal:
            return False
        self.waiting_queue.append(req)
        return True

    def _resume_interleave_parent(self, req):
        self.waiting_queue.append(req)

    def _release_sensenova_u1_exact_parent_kv(self, req):
        self.tree_cache.cache_finished_req(
            req,
            is_insert=False,
            kv_len_to_handle=req.kv_committed_len,
        )
        self.tree_cache.req_to_token_pool.free(req)
        req.kv = None


def _normalized_spec(*, max_images=2, seed=7):
    return normalize_u1_interleave_request(
        {
            "width": 64,
            "height": 64,
            "num_steps": 2,
            "max_images": max_images,
            "seed": seed,
        },
        input_token_count=2,
        max_new_tokens=32,
        context_len=4096,
        img_start_token_id=IMG_START,
        img_context_token_id=IMG_CONTEXT,
        img_end_token_id=IMG_END,
    )


def _parent_req(*, max_images=2, seed=7):
    sampling = SamplingParams(
        max_new_tokens=32,
        temperature=0,
        stop_token_ids={IMG_START},
        custom_params={
            U1_INTERLEAVE_CUSTOM_PARAM: _normalized_spec(
                max_images=max_images,
                seed=seed,
            )
        },
    )
    req = Req(
        rid="parent",
        origin_input_text="prompt",
        origin_input_ids=array("q", [1, 2]),
        sampling_params=sampling,
        eos_token_ids={EOS},
        vocab_size=128,
    )
    req.tokenizer = _FakeTokenizer()
    req.req_pool_idx = 1
    req.output_ids.extend([3, IMG_START])
    req.kv_committed_len = req.seqlen - 1
    req.kv = ReqKvInfo(
        kv_allocated_len=req.kv_committed_len,
        swa_evicted_seqlen=0,
    )
    req.finished_reason = FINISH_MATCHED_TOKEN(matched=IMG_START)
    req.finished_len = len(req.output_ids)
    req.already_computed = len(req.origin_input_ids)
    return req


def _park_parent(*, max_images=2, seed=7, scheduler=None):
    scheduler = scheduler or _FakeScheduler()
    controller = SenseNovaU1InterleaveController(scheduler)
    parent = _parent_req(max_images=max_images, seed=seed)
    assert controller.register_parent(parent) is None
    assert controller.maybe_park_parent(parent)
    prefix_child = scheduler.waiting_queue[-1]
    assert prefix_child.rid == "parent::sensenova_u1_prefix:0"
    assert controller.is_internal_child(prefix_child)
    prefix_child.finished_reason = FINISH_LENGTH(length=1)
    scheduler.waiting_queue.clear()
    controller.complete_child(prefix_child)
    child = scheduler.waiting_queue[-1]
    return scheduler, controller, parent, child


def _complete_child(controller, child, *, value=0.0):
    child.finished_reason = FINISH_LENGTH(length=1)
    child.customized_info = {
        "sensenova_u1_flow_image_tensor": [
            torch.full((1, 3, 64, 64), value, dtype=torch.float16)
        ]
    }
    controller.complete_child(child)


def _patch_release_kv_cache(monkeypatch):
    def release(req, tree_cache, is_insert=True):
        tree_cache.cache_finished_req(
            req,
            is_insert=is_insert,
            kv_len_to_handle=req.kv_committed_len,
        )
        tree_cache.req_to_token_pool.free(req)
        req.kv = None

    monkeypatch.setattr(
        "sglang.srt.managers.scheduler_components."
        "sensenova_u1_interleave.release_kv_cache",
        release,
    )


def test_u1_interleave_normalization_reserves_image_context() -> None:
    spec = normalize_u1_interleave_request(
        {"width": 64, "height": 64, "max_images": 2},
        input_token_count=10,
        max_new_tokens=20,
        context_len=40,
        img_start_token_id=IMG_START,
        img_context_token_id=IMG_CONTEXT,
        img_end_token_id=IMG_END,
    )
    assert spec["image_tokens"] == 4
    assert spec["image_span_tokens"] == 5
    assert spec["turn_seeds"] == [0, 1]

    with pytest.raises(ValueError, match="context window"):
        normalize_u1_interleave_request(
            {"width": 64, "height": 64, "max_images": 2},
            input_token_count=10,
            max_new_tokens=20,
            context_len=39,
            img_start_token_id=IMG_START,
            img_context_token_id=IMG_CONTEXT,
            img_end_token_id=IMG_END,
        )


def test_u1_interleave_normalization_bounds_image_turns() -> None:
    with pytest.raises(ValueError, match="max_images exceeds"):
        normalize_u1_interleave_request(
            {"max_images": U1_MAX_INTERLEAVE_IMAGES + 1},
            input_token_count=1,
            max_new_tokens=1,
            context_len=10000,
            img_start_token_id=IMG_START,
            img_context_token_id=IMG_CONTEXT,
            img_end_token_id=IMG_END,
        )


def test_u1_interleave_normalization_wraps_deterministic_turn_seeds() -> None:
    spec = normalize_u1_interleave_request(
        {"max_images": 2, "seed": 2**63 - 1},
        input_token_count=1,
        max_new_tokens=1,
        context_len=10000,
        img_start_token_id=IMG_START,
        img_context_token_id=IMG_CONTEXT,
        img_end_token_id=IMG_END,
    )

    assert spec["turn_seeds"] == [2**63 - 1, 0]


def test_u1_exact_text_request_skips_shared_radix_insert() -> None:
    scheduler = _FakeScheduler()
    controller = SenseNovaU1InterleaveController(scheduler)
    sampling = SamplingParams(
        max_new_tokens=4,
        temperature=0,
        ignore_eos=True,
        custom_params={
            U1_EXACT_TEXT_CUSTOM_PARAM: {
                "decode_steps": 4,
                "img_start_token_id": IMG_START,
                "eos_token_ids": [],
                "compiled_add_rms": True,
                "lm_head_linear": True,
            }
        },
    )
    req = Req(
        rid="exact-text",
        origin_input_text=None,
        origin_input_ids=array("q", [1, 2]),
        sampling_params=sampling,
        eos_token_ids={EOS},
        vocab_size=128,
    )

    assert controller.register_parent(req) is None
    assert req.batch_isolation_key == "sensenova_u1_exact_text:exact-text"
    assert req.radix_cache_prefix_limit == 0
    assert req.skip_radix_cache_insert
    assert req.extra_key == "sensenova_u1_exact_text:exact-text"
    assert req._compute_max_prefix_len(len(req.origin_input_ids)) == 0
    assert (
        req.sampling_params.custom_params[U1_FLOW_BATCH_ISOLATION_PARAM]
        == "sensenova_u1_exact_text:exact-text"
    )
    assert req.sampling_params.custom_params[U1_FLOW_RADIX_PREFIX_LIMIT_PARAM] == 0


def test_u1_exact_text_rejects_work_beyond_sampling_budget() -> None:
    scheduler = _FakeScheduler()
    controller = SenseNovaU1InterleaveController(scheduler)
    req = Req(
        rid="invalid-exact-text",
        origin_input_text=None,
        origin_input_ids=array("q", [1, 2]),
        sampling_params=SamplingParams(
            max_new_tokens=4,
            temperature=0,
            ignore_eos=True,
            custom_params={
                U1_EXACT_TEXT_CUSTOM_PARAM: {
                    "decode_steps": 5,
                    "img_start_token_id": IMG_START,
                    "eos_token_ids": [],
                    "compiled_add_rms": True,
                    "lm_head_linear": True,
                }
            },
        ),
        eos_token_ids={EOS},
        vocab_size=128,
    )

    assert (
        controller.register_parent(req)
        == "SenseNova U1 exact text decode_steps must equal max_new_tokens"
    )
    assert not req.skip_radix_cache_insert


def test_u1_interleave_bounds_primed_prefix_hints() -> None:
    controller = SenseNovaU1InterleaveController(_FakeScheduler())
    limit = controller._MAX_PRIMED_FLOW_PREFIXES

    for index in range(limit + 1):
        controller._mark_flow_prefix_primed(f"prefix-{index}")

    assert len(controller._primed_flow_prefixes) == limit
    assert "prefix-0" not in controller._primed_flow_prefixes
    assert f"prefix-{limit}" in controller._primed_flow_prefixes


def test_u1_interleave_lifecycle_parks_flows_reencodes_and_resumes() -> None:
    scheduler, controller, parent, child = _park_parent(max_images=2)

    assert parent.finished_reason is None
    assert controller.is_parked(parent)
    assert scheduler.tree_cache.cached_unfinished == []
    assert parent.req_pool_idx is None
    assert parent.kv is None
    assert child.rid == "parent::sensenova_u1_flow:0"
    assert controller.is_internal_child(child)
    assert child.origin_input_ids.tolist() == [1, 2, 3, IMG_START] + [EOS] * 4
    flow = child.sampling_params.custom_params[U1_FLOW_CUSTOM_PARAM]
    assert flow["image_start"] == 4
    assert flow["image_t_index"] == 4
    assert flow["seed"] == 7
    assert child.batch_isolation_key.startswith("sensenova_u1_interleave_flow:")
    assert child.radix_cache_prefix_limit == 4
    assert child.sampling_params.stop_strs == []
    assert child.sampling_params.stop_regex_strs == []
    assert parent.batch_isolation_key == "sensenova_u1_exact_text:interleave"

    scheduler.waiting_queue.clear()
    _complete_child(controller, child)

    assert scheduler.waiting_queue == [parent]
    assert not controller.is_parked(parent)
    assert parent.output_ids.tolist() == [
        3,
        IMG_START,
        IMG_CONTEXT,
        IMG_CONTEXT,
        IMG_CONTEXT,
        IMG_CONTEXT,
        IMG_END,
    ]
    assert parent.sampling_params.max_new_tokens == 37
    assert parent.mm_image_tokens == 4
    assert len(parent.multimodal_inputs.mm_items) == 1
    assert parent.multimodal_inputs.mm_items[0].offsets == [(4, 7)]
    assert parent.multimodal_inputs.mm_items[0].pad_value == IMG_CONTEXT
    assert parent.multimodal_inputs.mrope_positions.shape[0] == 3
    assert parent.already_computed == 0
    assert parent.radix_cache_prefix_limit == 0
    exact = parent.sampling_params.custom_params[U1_EXACT_TEXT_CUSTOM_PARAM]
    assert exact["decode_steps"] == 29
    image_values = parent.customized_info["sensenova_u1_interleave_image_b64"]
    assert image_values[1] is not None
    assert all(value is None for i, value in enumerate(image_values) if i != 1)


def test_u1_interleave_resume_recomputes_exact_parent_prefix() -> None:
    scheduler, controller, parent, child = _park_parent()
    scheduler.waiting_queue.clear()
    _complete_child(controller, child)

    assert parent.req_pool_idx is None
    assert parent.kv is None
    assert parent.already_computed == 0
    assert parent.radix_cache_prefix_limit == 0
    assert parent.skip_radix_cache_insert


def test_u1_interleave_parents_share_only_the_isolated_scheduling_lane() -> None:
    controller = SenseNovaU1InterleaveController(_FakeScheduler())
    first = _parent_req()
    second = _parent_req()
    second.rid = "neighbor"

    assert controller.register_parent(first) is None
    assert controller.register_parent(second) is None

    assert first.batch_isolation_key == second.batch_isolation_key
    assert first.batch_isolation_key == "sensenova_u1_exact_text:interleave"
    assert first.extra_key != second.extra_key
    assert first.radix_cache_prefix_limit == second.radix_cache_prefix_limit == 0
    assert first.skip_radix_cache_insert
    assert second.skip_radix_cache_insert


def test_u1_interleave_repeated_images_use_deterministic_turn_seeds() -> None:
    scheduler, controller, parent, child = _park_parent(max_images=2, seed=41)
    first_flow = child.sampling_params.custom_params[U1_FLOW_CUSTOM_PARAM]
    assert first_flow["seed"] == 41
    scheduler.waiting_queue.clear()
    _complete_child(controller, child, value=0.1)
    scheduler.waiting_queue.clear()
    mutable_spec = parent.sampling_params.custom_params[U1_INTERLEAVE_CUSTOM_PARAM]
    mutable_spec["seed"] = 100
    mutable_spec["turn_seeds"] = [100, 101]
    mutable_spec["max_images"] = 1
    state = controller._parents[parent.rid]
    assert state.spec is not mutable_spec
    assert state.spec["turn_seeds"] == [41, 42]
    assert state.spec["max_images"] == 2

    parent.output_ids.extend([20, IMG_START])
    parent.req_pool_idx = 1
    parent.kv_committed_len = parent.seqlen - 1
    parent.kv = ReqKvInfo(
        kv_allocated_len=parent.kv_committed_len,
        swa_evicted_seqlen=0,
    )
    parent.finished_reason = FINISH_MATCHED_TOKEN(matched=IMG_START)
    parent.finished_len = len(parent.output_ids)
    assert controller.maybe_park_parent(parent)
    second_prefix_child = scheduler.waiting_queue[-1]
    assert second_prefix_child.rid == "parent::sensenova_u1_prefix:1"
    assert second_prefix_child.extra_key != child.extra_key
    assert second_prefix_child.radix_cache_prefix_limit == 0
    second_prefix_child.finished_reason = FINISH_LENGTH(length=1)
    scheduler.waiting_queue.clear()
    controller.complete_child(second_prefix_child)
    second_child = scheduler.waiting_queue[-1]
    second_flow = second_child.sampling_params.custom_params[U1_FLOW_CUSTOM_PARAM]
    assert second_flow["seed"] == 42

    scheduler.waiting_queue.clear()
    _complete_child(controller, second_child, value=0.2)
    assert len(parent.multimodal_inputs.mm_items) == 2
    assert parent.sampling_params.max_new_tokens == 42
    assert parent.sampling_params.logit_bias[str(IMG_START)] == -100.0
    assert (
        parent.customized_info["sensenova_u1_interleave_image_index"].count(None)
        == len(parent.output_ids) - 2
    )


def test_u1_interleave_consumes_exact_text_segment() -> None:
    scheduler = _FakeScheduler()
    controller = SenseNovaU1InterleaveController(scheduler)
    parent = _parent_req(max_images=1)
    parent.output_ids = array("q")
    parent.finished_reason = None
    parent.finished_len = None
    assert controller.register_parent(parent) is None
    logits_output = SimpleNamespace(
        customized_info={
            "sensenova_u1_exact_text_tail": [[3, 4, IMG_START]],
            "sensenova_u1_exact_text_stats": [
                {
                    "generated_tokens": 4,
                    "graph_replayed": True,
                }
            ],
        }
    )
    parent.output_ids.append(2)

    accepted_len = controller.consume_exact_text_result(
        parent,
        0,
        logits_output,
    )

    assert accepted_len == 4
    assert parent.output_ids.tolist() == [2, 3, 4, IMG_START]
    assert "sensenova_u1_exact_text_tail" not in logits_output.customized_info
    assert "sensenova_u1_exact_text_stats" not in logits_output.customized_info


def test_u1_interleave_context_overflow_fails_closed() -> None:
    parent = _parent_req()
    scheduler = _FakeScheduler(
        max_req_len=parent.seqlen + _normalized_spec()["image_span_tokens"]
    )
    controller = SenseNovaU1InterleaveController(scheduler)
    assert controller.register_parent(parent) is None

    assert not controller.maybe_park_parent(parent)
    assert isinstance(parent.finished_reason, FINISH_ABORT)
    assert "context window" in parent.finished_reason.message
    assert scheduler.waiting_queue == []


def test_u1_interleave_internal_admission_failure_keeps_parent_finishable() -> None:
    scheduler = _FakeScheduler(admit_internal=False)
    controller = SenseNovaU1InterleaveController(scheduler)
    parent = _parent_req()
    assert controller.register_parent(parent) is None

    assert not controller.maybe_park_parent(parent)
    assert isinstance(parent.finished_reason, FINISH_ABORT)
    assert not controller.is_parked(parent)
    assert scheduler.waiting_queue == []


def test_u1_interleave_rejects_overlap_before_owning_parent_state() -> None:
    scheduler = _FakeScheduler()
    scheduler.enable_overlap = True
    controller = SenseNovaU1InterleaveController(scheduler)
    parent = _parent_req()

    error = controller.register_parent(parent)

    assert "--disable-overlap-schedule" in error
    assert parent.extra_key is None


def test_u1_interleave_cancellation_releases_parked_parent(monkeypatch) -> None:
    _patch_release_kv_cache(monkeypatch)
    scheduler, controller, parent, child = _park_parent()
    scheduler.waiting_queue.clear()

    direct = controller.before_abort(AbortReq(rid=parent.rid))

    assert direct == [parent]
    assert parent.req_pool_idx is None
    assert parent.kv is None
    assert controller.is_parked(parent)
    assert controller.is_internal_child(child)
    assert controller.complete_child(child) is None
    assert scheduler.waiting_queue == []


def test_u1_interleave_cancellation_while_prefix_child_running(
    monkeypatch,
) -> None:
    _patch_release_kv_cache(monkeypatch)
    scheduler = _FakeScheduler()
    controller = SenseNovaU1InterleaveController(scheduler)
    parent = _parent_req()
    assert controller.register_parent(parent) is None
    assert controller.maybe_park_parent(parent)
    prefix_child = scheduler.waiting_queue[-1]

    direct = controller.before_abort(AbortReq(rid=parent.rid))

    assert direct == [parent]
    assert controller.is_internal_child(prefix_child)
    assert controller.complete_child(prefix_child) is None


def test_u1_interleave_child_failure_finishes_original_parent_only(
    monkeypatch,
) -> None:
    _patch_release_kv_cache(monkeypatch)
    scheduler, controller, parent, child = _park_parent()

    controller.fail_internal_child(child, "internal queue timeout")

    assert parent.req_pool_idx is None
    assert parent.kv is None
    assert isinstance(parent.finished_reason, FINISH_ABORT)
    assert scheduler.output_streamer.calls[0][0] == [parent]


def test_u1_interleave_parked_req_filters_without_finalization() -> None:
    _, controller, parent, _ = _park_parent()
    batch = ScheduleBatch(reqs=[parent])

    batch.filter_batch()

    assert batch.reqs == []
    assert parent.finished_reason is None
    assert controller.is_parked(parent)


def test_u1_interleave_parked_req_is_removed_before_child_prefill() -> None:
    _, controller, parent, _ = _park_parent()
    batch = ScheduleBatch(reqs=[parent], batch_is_full=True)

    Scheduler._filter_parked_interleave_reqs(batch)

    assert batch.reqs == []
    assert not batch.batch_is_full
    assert parent.finished_reason is None
    assert controller.is_parked(parent)
