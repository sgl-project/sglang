"""Placement on parts where host and device share one memory pool (GB10, Jetson)."""

import torch

from sglang.multimodal_gen.runtime.managers.memory_managers import (
    auto_residency,
    host_memory_budget,
    layerwise_offload,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.auto_residency import (
    PAGEABLE_H2D_COST_MULTIPLIER,
    RankResidencyReport,
    ResidencyTarget,
    plan_auto_residency,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency import (
    LAYERWISE_OFFLOAD,
)
from sglang.multimodal_gen.runtime.platforms import current_platform

GIB_BYTES = 1024**3


def _share_pool(monkeypatch, shared: bool) -> None:
    monkeypatch.setattr(
        type(current_platform),
        "device_shares_host_memory",
        classmethod(lambda cls: shared),
    )


def test_shared_pool_hosting_keeps_every_mapped_layer_mapped():
    hosting = layerwise_offload._shared_pool_hosting(
        {0: 10, 1: 10, 2: 5}, {0: 10, 1: 4, 2: 0}
    )
    assert hosting == {0: "mapped", 1: "mapped", 2: "pageable"}


def test_no_pin_capacity_when_the_device_reads_host_pages(monkeypatch):
    _share_pool(monkeypatch, True)
    assert host_memory_budget.HostPinBudget.for_local_worker(2).spendable_bytes == 0
    _share_pool(monkeypatch, False)
    monkeypatch.setattr(
        host_memory_budget, "host_memory_available_bytes", lambda: 64 * GIB_BYTES
    )
    assert host_memory_budget.HostPinBudget.for_local_worker(1).spendable_bytes > 0


def test_pageable_penalty_disappears_on_a_shared_pool(monkeypatch):
    _share_pool(monkeypatch, True)
    assert auto_residency._pageable_h2d_cost_multiplier() == 1
    _share_pool(monkeypatch, False)
    assert (
        auto_residency._pageable_h2d_cost_multiplier() == PAGEABLE_H2D_COST_MULTIPLIER
    )


def test_cold_advice_is_harmless_on_an_ordinary_tensor():
    layerwise_offload._advise_mapped_source_cold(torch.zeros(4096, dtype=torch.uint8))
    layerwise_offload._advise_mapped_source_cold(torch.zeros(0))


def _pin_frontier() -> tuple[ResidencyTarget, ResidencyTarget]:
    pageable = ResidencyTarget(
        component_name="transformer",
        residency_mode=LAYERWISE_OFFLOAD,
        target_residency_mode=LAYERWISE_OFFLOAD,
        target_resident_weight_bytes=5 * GIB_BYTES,
        h2d_bytes_per_request=10 * GIB_BYTES,
        target_layerwise_resident_layers=(5,),
        target_layerwise_pinned_layers=((),),
        target_device_weight_bytes=5 * GIB_BYTES,
        current_placement=True,
    )
    pinned = ResidencyTarget(
        component_name="transformer",
        residency_mode=LAYERWISE_OFFLOAD,
        target_residency_mode=LAYERWISE_OFFLOAD,
        target_resident_weight_bytes=5 * GIB_BYTES,
        h2d_bytes_per_request=11 * GIB_BYTES,
        target_layerwise_resident_layers=(5,),
        target_layerwise_pinned_layers=((0,),),
        pinned_host_delta_bytes=2 * GIB_BYTES,
        target_device_weight_bytes=5 * GIB_BYTES,
        target_pinned_host_bytes=2 * GIB_BYTES,
    )
    return pageable, pinned


def _report(*, shared: bool, candidates) -> RankResidencyReport:
    pageable, pinned = candidates
    # 100 GiB budget, 89 GiB measured peak, 10 GiB reserve: one GiB of device
    # headroom, while the host side would allow ten GiB of pins.
    return RankResidencyReport(
        rank=0,
        budget_bytes=100 * GIB_BYTES,
        estimated_peak_bytes=89 * GIB_BYTES,
        host_pin_capacity_bytes=10 * GIB_BYTES,
        host_shares_device_pool=shared,
        candidates=[pageable, pinned],
        estimated_request_duration_ns=1_000_000_000,
        candidate_latency_savings_ns={
            pageable.option_key(): 100_000_000,
            pinned.option_key(): 110_000_000,
        },
    )


def test_pins_are_charged_to_device_phases_on_a_shared_pool():
    pageable, pinned = _pin_frontier()
    discrete = plan_auto_residency(
        reports=[_report(shared=False, candidates=(pageable, pinned))]
    )
    assert discrete.changes == [pinned]
    shared = plan_auto_residency(
        reports=[_report(shared=True, candidates=(pageable, pinned))]
    )
    assert shared.changes == []


class _FakeManager:
    def __init__(self, mapped: dict[int, int], *, resident_layers: int = 0):
        self.num_layers = 4
        self.residency_policy = "leading"
        self.resident_layers = resident_layers
        self.enabled = True
        self._mapped = mapped

    def mapped_layer_bytes(self) -> dict[int, int]:
        return self._mapped


def test_streamed_mapped_bytes_count_only_streamed_layers_with_a_mapping():
    manager = _FakeManager({0: 10, 1: 10, 2: 0, 3: 10})
    streamed = auto_residency._layerwise_streamed_mapped_bytes(
        managers=[manager], resident_layers=(0,), residency_policies=("leading",)
    )
    assert streamed == 30
    # A layer read once per request is a bounded re-read, not a pool claim.
    assert (
        auto_residency._layerwise_streamed_mapped_bytes(
            managers=[manager],
            resident_layers=(0,),
            residency_policies=("leading",),
            repeated=False,
        )
        == 0
    )
    assert (
        auto_residency._layerwise_streamed_mapped_bytes(
            managers=[manager],
            resident_layers=(0,),
            residency_policies=("leading",),
            layer_uses=((20, 20, 1, 1),),
        )
        == 20
    )
    # Leading residency keeps the first layers on the device; their mapped
    # bytes stop being page cache the stream depends on.
    assert (
        auto_residency._layerwise_streamed_mapped_bytes(
            managers=[manager], resident_layers=(2,), residency_policies=("leading",)
        )
        == 10
    )
    assert (
        auto_residency._layerwise_streamed_mapped_bytes(
            managers=[manager], resident_layers=(4,), residency_policies=("leading",)
        )
        == 0
    )


def test_estimated_manager_has_no_mapping():
    manager = auto_residency._EstimatedLayerwiseManager(
        layers_attr_str="blocks",
        layer_bytes={0: GIB_BYTES, 1: GIB_BYTES},
        prefetch_size=1,
        residency_policy="leading",
        pin_cpu_memory=False,
    )
    assert manager.mapped_layer_bytes() == {}


def test_no_resident_seed_on_a_shared_pool(monkeypatch):
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.managers.memory_managers import initial_residency

    _share_pool(monkeypatch, True)
    monkeypatch.setattr(
        initial_residency, "auto_residency_static_skip_reason", lambda args: None
    )
    monkeypatch.setattr(type(current_platform), "is_cuda", lambda self: True)
    seeded: list[str] = []
    server_args = SimpleNamespace(
        use_fsdp_inference=False,
        set_auto_residency_mode=lambda name, mode: seeded.append(name),
    )
    monkeypatch.setattr(
        initial_residency.current_platform,
        "get_available_gpu_memory",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("seed must not size the pool")
        ),
    )
    initial_residency.maybe_seed_initial_residency(server_args, inventory=[])
    assert seeded == []


def test_mapped_layers_carry_no_transfer_work_on_a_shared_pool(monkeypatch):
    class _Manager(_FakeManager):
        def layer_weight_bytes(self):
            return {0: 10, 1: 10, 2: 10, 3: 10}

    manager = _Manager({0: 10, 1: 10, 2: 4, 3: 0})
    _share_pool(monkeypatch, False)
    discrete = auto_residency._layerwise_transfer_work_bytes(
        managers=[manager],
        resident_layers=(0,),
        pinned_layers=((),),
        uses_per_streamed_layer=20,
    )
    _share_pool(monkeypatch, True)
    shared = auto_residency._layerwise_transfer_work_bytes(
        managers=[manager],
        resident_layers=(0,),
        pinned_layers=((),),
        uses_per_streamed_layer=20,
    )
    # Every layer streams 20 times; only the bytes off the mapping still cost.
    assert discrete == 20 * PAGEABLE_H2D_COST_MULTIPLIER * 40
    assert shared == 20 * (6 + 10)


def test_shared_pool_keeps_only_the_measured_and_streamed_layouts():
    kept = auto_residency._shared_pool_resident_targets(
        [(0, 0), (0, 10), (0, 25), (2, 50), (0, 50)], current_resident_layers=(0, 25)
    )
    assert kept == [(0, 0), (0, 25)]
    assert auto_residency._shared_pool_resident_targets([(0, 50), (0, 0)], None) == [
        (0, 0)
    ]


def test_populate_mapped_source_is_harmless_on_anonymous_memory():
    # Anonymous pages are already present; the advice must not raise or copy.
    layerwise_offload.populate_mapped_source([torch.zeros(1 << 16, dtype=torch.uint8)])
    layerwise_offload.populate_mapped_source([torch.zeros(0)])


def test_mapped_stream_cost_applies_when_the_cache_cannot_hold_the_cycle(monkeypatch):
    class _Manager(_FakeManager):
        def layer_weight_bytes(self):
            return {0: 10, 1: 10, 2: 10, 3: 10}

    manager = _Manager({0: 10, 1: 10, 2: 10, 3: 10})
    _share_pool(monkeypatch, True)
    free = auto_residency._layerwise_transfer_work_bytes(
        managers=[manager],
        resident_layers=(0,),
        pinned_layers=((),),
        uses_per_streamed_layer=19,
    )
    priced = auto_residency._layerwise_transfer_work_bytes(
        managers=[manager],
        resident_layers=(0,),
        pinned_layers=((),),
        uses_per_streamed_layer=19,
        mapped_stream_cost_multiplier=auto_residency.DISK_MISS_COST_MULTIPLIER,
    )
    assert free == 0
    assert priced == 19 * 24 * 40
    # Resident layers stop paying the per-step disk price.
    half = auto_residency._layerwise_transfer_work_bytes(
        managers=[manager],
        resident_layers=(2,),
        pinned_layers=((),),
        uses_per_streamed_layer=19,
        mapped_stream_cost_multiplier=auto_residency.DISK_MISS_COST_MULTIPLIER,
    )
    assert half == 19 * 24 * 20 + 1 * 24 * 20


def test_component_offload_moves_back_pageable_on_a_shared_pool(monkeypatch):
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.managers.memory_managers import (
        component_residency_strategies as strategies,
    )

    calls = []

    class _Module(torch.nn.Module):
        def to(self, *args, **kwargs):
            calls.append((args, kwargs))
            return self

    monkeypatch.setattr(
        strategies,
        "_module_reference_tensor",
        lambda module: SimpleNamespace(device=SimpleNamespace(type="cuda")),
    )
    use = SimpleNamespace(component_name="transformer")
    for shared, expected in ((True, False), (False, True)):
        _share_pool(monkeypatch, shared)
        strategy = strategies.ComponentOffloadStrategy()
        monkeypatch.setattr(strategy, "wait_for_use", lambda *a, **k: None)
        calls.clear()
        strategy.finish_use(_Module(), use, None)
        assert calls == [(("cpu",), {"non_blocking": expected})]


def _permanent_frontier(*, released_cache_bytes: int):
    streamed = ResidencyTarget(
        component_name="transformer",
        residency_mode=LAYERWISE_OFFLOAD,
        target_residency_mode=LAYERWISE_OFFLOAD,
        target_resident_weight_bytes=0,
        h2d_bytes_per_request=0,
        target_layerwise_resident_layers=(0,),
        target_layerwise_pinned_layers=((),),
        current_placement=True,
    )
    # A 57 GiB component whose 50 GiB of streamed layers sit in the page cache
    # of a shared pool: making it resident costs the pool 7 GiB net.
    permanent = ResidencyTarget(
        component_name="transformer",
        residency_mode=LAYERWISE_OFFLOAD,
        target_residency_mode=auto_residency.RESIDENT,
        target_resident_weight_bytes=57 * GIB_BYTES,
        target_device_weight_bytes=57 * GIB_BYTES,
        h2d_bytes_per_request=1000 * GIB_BYTES,
        target_layerwise_resident_layers=(50,),
        target_layerwise_pinned_layers=((),),
        permanent_residency=True,
        device_transition_delta_bytes=7 * GIB_BYTES,
        active_device_delta_bytes=7 * GIB_BYTES,
        inactive_device_delta_bytes=7 * GIB_BYTES,
        present_device_delta_bytes=7 * GIB_BYTES,
        released_cache_bytes=released_cache_bytes,
    )
    return streamed, permanent


def _two_phase_report(*, released_cache_bytes: int) -> RankResidencyReport:
    streamed, permanent = _permanent_frontier(released_cache_bytes=released_cache_bytes)
    # 43 GiB budget: the encoder phase never saw the transformer, so a permanent
    # placement is charged its footprint there -- 57 GiB unless the cache it
    # gives back is credited.
    return RankResidencyReport(
        rank=0,
        budget_bytes=43 * GIB_BYTES,
        estimated_peak_bytes=9 * GIB_BYTES,
        estimated_peak_bytes_by_phase={
            "encode": 3 * GIB_BYTES,
            "denoise": 9 * GIB_BYTES,
        },
        active_components_by_phase={
            "encode": ("text_encoder",),
            "denoise": ("transformer",),
        },
        used_components_by_phase={
            "encode": ("text_encoder",),
            "denoise": ("transformer",),
        },
        host_shares_device_pool=True,
        candidates=[streamed, permanent],
        estimated_request_duration_ns=300_000_000_000,
        candidate_latency_savings_ns={
            streamed.option_key(): 0,
            permanent.option_key(): 250_000_000_000,
        },
    )


def test_permanent_residency_is_credited_its_released_cache_in_every_phase():
    credited = plan_auto_residency(
        reports=[_two_phase_report(released_cache_bytes=50 * GIB_BYTES)]
    )
    assert [c.target_residency_mode for c in credited.changes] == [
        auto_residency.RESIDENT
    ]
    uncredited = plan_auto_residency(
        reports=[_two_phase_report(released_cache_bytes=0)]
    )
    assert uncredited.changes == []


class _PoolManager:
    """A layerwise manager described by numbers only: every layer mapped, no pins."""

    def __init__(self, layer_bytes: dict[int, int]):
        self.enabled = True
        self._configured = True
        self.num_layers = len(layer_bytes)
        self._layer_bytes = dict(layer_bytes)
        self.resident_layers = 0
        self.residency_policy = "leading"
        self.pin_cpu_memory = False
        self.layers_attr_str = "layers"

    def iter_cpu_weights(self):
        return iter(())

    def offloaded_weight_bytes(self):
        return sum(self._layer_bytes.values())

    def resident_weight_bytes(self, resident_layers=None, residency_policy=None):
        del residency_policy
        count = self.resident_layers if resident_layers is None else resident_layers
        return sum(self._layer_bytes[i] for i in range(min(count, self.num_layers)))

    def peak_managed_device_weight_bytes(
        self, resident_layers=None, residency_policy=None
    ):
        del residency_policy
        count = self.resident_layers if resident_layers is None else resident_layers
        return self.resident_weight_bytes(min(count + 2, self.num_layers))

    def layer_weight_bytes(self):
        return dict(self._layer_bytes)

    def layer_host_store_bytes(self):
        return {}

    def mapped_layer_bytes(self):
        return dict(self._layer_bytes)

    def pinned_host_weight_bytes(self):
        return 0

    def pinnable_layer_indices(self):
        return ()

    def pinned_layer_indices(self):
        return ()


class _PoolModule(layerwise_offload.LayerwiseOffloadableModuleMixin, torch.nn.Module):
    def __init__(self, layer_bytes: dict[int, int], full_bytes: int):
        torch.nn.Module.__init__(self)
        self.layerwise_offload_managers = [_PoolManager(layer_bytes)]
        self.fake_full_bytes = full_bytes


def test_shared_pool_promotes_the_dit_when_the_cycle_misses_the_cache(
    monkeypatch, caplog
):
    """The GB10 case: encoder 45 GiB, DiT 50.4 GiB (+7 unmanaged), VAE 4.5 GiB all
    streamed from a page cache that cannot hold the cycle; budget 43 GiB."""
    import logging

    from sglang.multimodal_gen.runtime.managers.memory_managers.auto_residency import (
        estimate_candidate_latency_savings_ns,
    )

    monkeypatch.setattr(
        auto_residency,
        "_module_weight_bytes",
        lambda module: getattr(module, "fake_full_bytes", 0),
    )
    gib = GIB_BYTES
    modules = {
        "text_encoder": _PoolModule({i: 45 * gib // 30 for i in range(30)}, 45 * gib),
        "transformer": _PoolModule({i: 50 * gib // 50 for i in range(50)}, 57 * gib),
        "video_vae": _PoolModule({i: 4 * gib // 8 for i in range(8)}, 4 * gib),
    }
    request_ns = 291 * 10**9
    candidates = auto_residency.collect_residency_targets(
        modules=modules,
        residency_mode_of=lambda name: LAYERWISE_OFFLOAD,
        explicit_residency_mode_of=lambda name: None,
        custom_strategy_names=(),
        num_inference_steps=50,
        allow_host_pin_reallocation=False,
        pin_cpu_memory=False,
        used_components=set(modules),
        request_duration_ns=request_ns,
        shared_memory_pool=True,
        mapped_stream_cost_multiplier=auto_residency.DISK_MISS_COST_MULTIPLIER,
    )
    stages = {
        "text_encoder": ("encode",),
        "transformer": ("denoise",),
        "video_vae": ("decode",),
    }
    savings = estimate_candidate_latency_savings_ns(
        candidates=candidates,
        request_duration_ns=request_ns,
        stage_duration_ns={
            "encode": 46 * 10**9,
            "denoise": 200 * 10**9,
            "decode": 42 * 10**9,
        },
        component_stages=stages,
    )
    report = RankResidencyReport(
        rank=0,
        budget_bytes=int(43.1 * gib),
        estimated_peak_bytes=int(9.4 * gib),
        estimated_peak_bytes_by_phase={
            "encode": 3 * gib,
            "denoise": int(9.4 * gib),
            "decode": 7 * gib,
        },
        active_components_by_phase={
            "encode": ("text_encoder",),
            "denoise": ("transformer",),
            "decode": ("video_vae",),
        },
        used_components_by_phase={
            "encode": ("text_encoder",),
            "denoise": ("transformer",),
            "decode": ("video_vae",),
        },
        host_shares_device_pool=True,
        device_transition_allocated_bytes=11 * gib,
        host_transition_headroom_bytes=80 * gib,
        estimated_request_duration_ns=request_ns,
        candidate_latency_savings_ns=savings,
        candidates=candidates,
    )
    caplog.set_level(logging.INFO, logger=auto_residency.__name__)
    plan = plan_auto_residency(reports=[report])
    chosen = {
        change.component_name: change.target_residency_mode for change in plan.changes
    }
    assert chosen.get("transformer") == auto_residency.RESIDENT, (
        chosen,
        plan.skip_reason,
        caplog.text,
    )


def test_single_iteration_probe_counts_repeated_stage_layers_per_iteration():
    """A 2-step probe of a pipeline that runs steps-1 iterations sees every DiT
    layer once. Those layers are per-iteration, not one-shot."""
    from sglang.multimodal_gen.runtime.managers.memory_managers.auto_residency import (
        WarmupMemoryRecord,
        estimate_layerwise_layer_uses,
    )

    stage = "MiniMaxH3DenoisingStage"
    record = WarmupMemoryRecord(
        width=864,
        height=480,
        num_frames=124,
        baseline_allocated_bytes=0,
        peak_allocated_bytes=1,
        succeeded=True,
        phase_used_components={
            f"7:{stage}:use:transformer": ("transformer",),
            "2:MiniMaxH3TextEncodingStage:use:text_encoder": ("text_encoder",),
        },
        layerwise_layer_uses={
            "transformer": {"blocks": (1, 1, 1)},
            "text_encoder": {"layers": (1, 1)},
        },
        layerwise_layer_uses_by_stage={
            stage: {"transformer": {"blocks": (1, 1, 1)}},
            "MiniMaxH3TextEncodingStage": {"text_encoder": {"layers": (1, 1)}},
        },
        num_inference_steps=2,
        stage_iterations={stage: (1, 19)},
    )
    uses = estimate_layerwise_layer_uses(
        records=[record],
        target_units=None,
        target_num_inference_steps=20,
    )
    assert uses["transformer"] == {"blocks": (19, 19, 19)}
    assert uses["text_encoder"] == {"layers": (1, 1)}
