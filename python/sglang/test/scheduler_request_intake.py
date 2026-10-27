"""A request intake with inert collaborators, for tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.disaggregation.utils import TransferBackend
from sglang.srt.managers.request_lists import CoreLists
from sglang.srt.managers.scheduler_components.engine_state import LaunchCounters
from sglang.srt.managers.scheduler_components.mode_owners import ModeOwners
from sglang.srt.managers.scheduler_components.request_intake import RequestIntake
from sglang.srt.mem_cache.events import KvEventLoraNames
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


def make_request_intake(intake_class: type = RequestIntake, **deps) -> RequestIntake:
    """An intake of `intake_class` over empty core lists, with no mode owners
    and inert collaborators; `deps` replaces any constructor argument, service,
    request limit, or config value the intake reads (spec algorithm, transfer
    backend, sampling mask cap)."""
    args = dict(
        core_lists=CoreLists(),
        counters=LaunchCounters(),
        kv=SimpleNamespace(tree_cache=MagicMock(), token_to_kv_pool_allocator=None),
        capacity=SimpleNamespace(sliding_window_size=None),
        ipc_channels=MagicMock(),
        beam_coordinator=MagicMock(),
        session_controller=MagicMock(),
        grammar_manager=MagicMock(),
        request_retirer=MagicMock(),
        mode_owners=ModeOwners(),
        model_config=MagicMock(),
        tokenizer=None,
        mm_processor=None,
        workers=SimpleNamespace(
            tp_worker=SimpleNamespace(get_pad_input_ids_func=lambda: None)
        ),
        dp_tp_group=SimpleNamespace(cpu_group=None, rank_in_group=0, first_rank=0),
        kv_event_lora_names=KvEventLoraNames(),
    )
    limits = dict(
        max_req_len=4096,
        max_req_input_len=4095,
        max_queued_requests=None,
        chunked_prefill_size=None,
    )
    for name in limits:
        if name in deps:
            limits[name] = deps.pop(name)
    args["limits"] = SimpleNamespace(**limits)
    # Config the intake reads from the bags, pinned here as the tests expect.
    config = dict(
        spec_algorithm=SpeculativeAlgorithm.NONE,
        transfer_backend=TransferBackend.MOONCAKE,
        sampling_mask_max_tokens=64,
        enable_kv_cache_events=False,
    )
    for name in config:
        if name in deps:
            config[name] = deps.pop(name)
    # The services the intake writes to; a test passes either one alone.
    services = dict(
        metrics_sink=SimpleNamespace(enable_metrics=False, metrics_collector=None),
        output_streamer=MagicMock(),
        scheduler_stage_metrics=None,
    )
    for name in services:
        if name in deps:
            services[name] = deps.pop(name)
    args["services"] = SimpleNamespace(**services)
    args.update(deps)
    intake = intake_class(**args)
    for name, value in config.items():
        setattr(intake, name, value)
    return intake
