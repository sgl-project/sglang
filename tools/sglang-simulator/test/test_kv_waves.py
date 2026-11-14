"""Tests for KV wave generation, scheduling, and network profiles."""

from pathlib import Path

import pytest
from sglang_simulator.simulation.sglang.kv_transfer import (
    KVTransferEstimator,
    load_network_profiles,
)
from sglang_simulator.simulation.sglang.kv_waves import (
    PrefillChunk,
    RequestWaves,
    Wave,
    plan_kv_transfers,
    resolve_wave_options,
    schedule_transfers,
    waves_from_chunks,
)

EXAMPLE_PROFILES = (
    Path(__file__).parents[1]
    / "examples"
    / "network_profiles"
    / "example_profiles.json"
)
# Made-up profiles in the shape of a measured fit.
PROFILES = {
    "RDMA": {
        "alpha_us": 100.0,
        "bandwidth_gbps": 10.0,
        "excursion": {
            "later_wave_exc_scale": 0.1,
            "bands": [
                {"max_bytes": 1_000_000, "probability": 0.2, "magnitude_s": {0.001: 1}},
                {"probability": 0.5, "magnitude_s": {0.002: 0.5, 0.02: 0.5}},
            ],
        },
    },
    "TCP": {"alpha_us": 1000.0, "bandwidth_gbps": 2.0},
}


def _network(name, **config):
    return {"network": name, "network_profiles": PROFILES, **config}


def schedule_edge(requests, estimator, **kwargs):
    """Schedule one transfer queue."""
    return schedule_transfers({None: requests}, estimator, **kwargs)


class _FixedEstimator:
    def __init__(self, service=1.0, excursion=0.0):
        self.service = service
        self.excursion = excursion

    def est_service_dur(self, nbytes: int) -> float:
        return self.service

    def est_excursion(self, nbytes: int, req_id: str, wave_index: int) -> float:
        return self.excursion


def _options(**config):
    return resolve_wave_options(
        config,
        chunked_prefill_size=8192,
        page_size=256,
        bytes_per_token=1.0,
    )


def _request(req_id: str, emits=(0.0,), prefill_end=0.0):
    return RequestWaves(
        req_id=req_id,
        waves=[Wave(req_id, index, emit, 1, 1) for index, emit in enumerate(emits)],
        prefill_end=prefill_end,
    )


def _waves(chunks, input_length, **config):
    return waves_from_chunks(
        chunks,
        req_id="r",
        created_time=0.0,
        input_length=input_length,
        options=_options(**config),
    )


def _draws(estimator, req_id="r"):
    return [
        estimator.est_excursion(32_000_000, req_id=req_id, wave_index=index)
        for index in range(3)
    ]


def test_chunking_follows_serving_and_defaults_are_neutral():
    assert _options().enabled is True
    assert _options().bootstrap_alpha_s == 0.0
    assert _options(chunked_prefill=False).enabled is False
    serving_disabled = resolve_wave_options(
        {"chunked_prefill": True},
        chunked_prefill_size=-1,
        page_size=256,
        bytes_per_token=1.0,
    )
    assert serving_disabled.enabled is False


def test_schedule_validation():
    with pytest.raises(ValueError, match="max_inflight_waves"):
        schedule_edge([], _FixedEstimator(), max_inflight_waves=0)
    for visible_scale in (-0.01, 1.01):
        with pytest.raises(ValueError, match="visible_scale"):
            schedule_edge([], _FixedEstimator(), visible_scale=visible_scale)


def test_width_boundary_and_fifo_order():
    requests = [_request(req_id) for req_id in ("a", "b", "c")]
    scheduled = schedule_edge(requests, _FixedEstimator(), max_inflight_waves=2)
    assert [scheduled[r].first_start for r in "abc"] == [0.0, 0.0, 1.0]


def test_same_request_waves_are_serial_without_reserving_another_slot():
    scheduled = schedule_edge(
        [_request("a", emits=(0.0, 0.0)), _request("b")],
        _FixedEstimator(),
        max_inflight_waves=2,
    )
    assert scheduled["a"].first_start == 0.0
    assert scheduled["a"].last_done == 2.0
    assert scheduled["a"].send_s == 2.0
    assert scheduled["b"].first_start == 0.0


def test_visible_scale_changes_only_visible_wait():
    estimator = _FixedEstimator(service=1.0, excursion=0.5)
    full = schedule_edge([_request("a")], estimator)["a"]
    scaled = schedule_edge([_request("a")], estimator, visible_scale=0.7)["a"]
    assert (full.service_s, full.excursion_s, full.send_s) == (1.0, 0.5, 1.5)
    assert (scaled.service_s, scaled.excursion_s, scaled.send_s) == (1.0, 0.5, 1.5)
    assert full.last_done == scaled.last_done == 1.5
    assert scaled.visible_s == pytest.approx(0.7 * full.visible_s)


class _LinearEstimator:
    """0.5 s per send plus 1 s per byte."""

    def est_service_dur(self, nbytes: int) -> float:
        return 0.5 + nbytes

    def est_excursion(self, nbytes: int, req_id: str, wave_index: int) -> float:
        return 0.0


def _flow(req_id, src, dst, emit=0.0):
    return RequestWaves(
        req_id=req_id,
        waves=[Wave(req_id, 0, emit, 1, 1)],
        prefill_end=0.0,
        src=src,
        dst=dst,
    )


@pytest.mark.parametrize(
    ("a", "b", "done"),
    [
        # Two queues leave one prefill node: each moves its byte at half rate.
        (("q0", "p0", "d0"), ("q1", "p0", "d1"), 2.5),
        # Two queues enter one decode node.
        (("q0", "p0", "d0"), ("q1", "p1", "d0"), 2.5),
        # Separate nodes do not interact.
        (("q0", "p0", "d0"), ("q1", "p1", "d1"), 1.5),
        # One queue is the fitted single-flow case: its slots never share.
        (("q0", "p0", "d0"), ("q0", "p0", "d0"), 1.5),
    ],
)
def test_queues_share_node_bandwidth_but_alpha_is_not_shared(a, b, done):
    queues = {}
    for req_id, (queue, src, dst) in zip("ab", (a, b)):
        queues.setdefault(queue, []).append(_flow(req_id, src, dst))
    scheduled = schedule_transfers(queues, _LinearEstimator(), max_inflight_waves=2)
    for req_id in "ab":
        assert scheduled[req_id].first_start == 0.0
        assert scheduled[req_id].last_done == pytest.approx(done)
        assert scheduled[req_id].send_s == pytest.approx(done)


def test_an_excursion_stretches_with_the_shared_bytes():
    class _Slow(_LinearEstimator):
        def est_excursion(self, nbytes, req_id, wave_index):
            return 0.5

    alone = schedule_transfers({"q0": [_flow("a", "p0", "d0")]}, _Slow())
    shared = schedule_transfers(
        {"q0": [_flow("a", "p0", "d0")], "q1": [_flow("b", "p0", "d1")]}, _Slow()
    )
    # alpha 0.5, then a byte and a 0.5 s excursion, at half rate when shared.
    assert alone["a"].last_done == 2.0
    assert shared["a"].last_done == pytest.approx(3.5)
    assert shared["a"].excursion_s == 0.5


def test_shared_rate_follows_the_queues_moving_at_each_moment():
    scheduled = schedule_transfers(
        {"q0": [_flow("a", "p0", "d0")], "q1": [_flow("b", "p0", "d1", emit=0.5)]},
        _LinearEstimator(),
    )
    # a moves alone over [0.5, 1.0), then shares with b until it finishes; b
    # moves the rest of its byte alone.
    assert scheduled["a"].last_done == pytest.approx(2.0)
    assert scheduled["b"].first_start == 0.5
    assert scheduled["b"].last_done == pytest.approx(2.5)


def test_page_alignment_and_whole_page_bytes():
    waves = _waves(
        [PrefillChunk(t=0.1, extend_end=8000), PrefillChunk(t=0.4, extend_end=8256)],
        8256,
    )
    assert [wave.tokens for wave in waves] == [7936, 320]
    # The partial last page still moves as a whole page.
    assert [wave.nbytes for wave in waves] == [7936, 512]


def test_prefill_prefix_hit_travels_with_the_first_wave():
    # 8192 cached tokens, then two EXTENDs of 8192 and 4096 new tokens.
    waves = _waves(
        [
            PrefillChunk(t=0.1, extend_end=16384),
            PrefillChunk(t=0.4, extend_end=20480),
        ],
        20480,
    )
    assert [wave.tokens for wave in waves] == [16384, 4096]


def test_bootstrap_gate_merges_early_chunk():
    waves = _waves(
        [PrefillChunk(t=0.1, extend_end=8192), PrefillChunk(t=0.4, extend_end=10240)],
        10240,
        bootstrap_alpha_s=0.28,
    )
    assert [(wave.emit, wave.tokens) for wave in waves] == [(0.4, 10240)]


@pytest.mark.parametrize("chunked_prefill", [True, False])
def test_prefill_finished_before_bootstrap_is_parked(chunked_prefill):
    waves = _waves(
        [PrefillChunk(t=0.10, extend_end=1024)],
        1024,
        bootstrap_alpha_s=0.28,
        chunked_prefill=chunked_prefill,
    )
    assert [(wave.emit, wave.tokens) for wave in waves] == [(0.28, 1024)]


def test_fully_cached_prompt_still_sends_one_wave():
    waves = _waves([PrefillChunk(t=0.4, extend_end=1024)], 1024)
    assert [wave.tokens for wave in waves] == [1024]


def test_disabled_chunking_degrades_to_one_wave():
    waves = _waves(
        [PrefillChunk(t=0.1, extend_end=8192), PrefillChunk(t=0.4, extend_end=10240)],
        10240,
        chunked_prefill=False,
    )
    assert [(wave.emit, wave.tokens) for wave in waves] == [(0.4, 10240)]


LAYOUT = {"page_size": 256, "bytes_per_token": 2.0, "chunked_prefill_size": 256}


def _plan(decode=1, layouts=(LAYOUT,), **config):
    """Each request: 512 tokens on prefill 0, finished at 1.0 s, to decode i."""
    routes = {f"r{index}": (0, index) for index in range(decode)}
    stats = {
        req_id: {
            "created_time": 0.0,
            "input_length": 512,
            "last_event_time": 1.0,
            "kv_prefill_chunks": [{"t": 1.0, "extend_end": 512}],
        }
        for req_id in routes
    }
    return plan_kv_transfers(
        stats,
        {**routes, "unfinished": (0, 0)},
        list(layouts),
        {"bandwidth_gbps": 1e-6, **config},
    )


def test_plan_sizes_waves_from_the_prefill_kv_layout():
    plan = _plan()
    # 512 tokens * 2 B at 1000 B/s, all after the prefill ends at 1.0 s.
    assert plan.transfers["r0"].visible_s == pytest.approx(1.024)
    assert plan.transfers["r0"].wave_tokens == [512]
    # A routed request without Prefill statistics is not scheduled.
    assert set(plan.transfers) == {"r0"}
    assert plan.num_queues == 1


def test_plan_without_a_kv_layout_is_an_error():
    with pytest.raises(RuntimeError, match="KV cache layout"):
        _plan(layouts=[None])
    with pytest.raises(ValueError, match="transfer_queues"):
        _plan(transfer_queues=0)


@pytest.mark.parametrize(
    ("transfer_queues", "waits"), [(4, [2.048, 2.048]), (1, [1.024, 2.048])]
)
def test_decode_instances_share_the_prefill_node_or_its_queue(transfer_queues, waits):
    plan = _plan(decode=2, max_inflight_waves=1, transfer_queues=transfer_queues)
    # Two decode instances land in separate prefill queues and send in
    # parallel at half rate each; with one queue, the second send waits for the
    # first. Either way the prefill node moves the same bytes by 2.048 s.
    assert plan.num_queues == min(transfer_queues, 2)
    assert [plan.transfers[f"r{i}"].visible_s for i in range(2)] == (
        pytest.approx(waits)
    )


def test_example_network_profiles_load_from_file():
    profiles = load_network_profiles(str(EXAMPLE_PROFILES))
    assert profiles
    for name, profile in profiles.items():
        assert profile["alpha_us"] > 0
        assert profile["bandwidth_gbps"] > 0
        # JSON keys are strings; the estimator reads them as seconds.
        config = {"network": name, "network_profiles": str(EXAMPLE_PROFILES)}
        estimator = KVTransferEstimator(config)
        assert estimator.est_service_dur(0) == pytest.approx(profile["alpha_us"] * 1e-6)
        assert estimator.est_excursion(32_000_000, req_id="r", wave_index=0) >= 0


def test_network_profiles_must_map_names_to_profiles():
    assert load_network_profiles(None) == {}
    with pytest.raises(ValueError, match="network_profiles"):
        load_network_profiles({"RDMA": 1.0})


def test_excursion_is_reproducible_per_request():
    first = KVTransferEstimator(_network("RDMA", seed=7))
    second = KVTransferEstimator(_network("RDMA", seed=7))
    # Another request drawing in between must not shift this request's stream.
    second.est_excursion(32_000_000, req_id="other", wave_index=0)
    assert _draws(first) == _draws(second)


def test_configured_bandwidth_wins_over_the_accelerator():
    # The profile's fitted rate, not the accelerator's nominal link speed.
    fitted = KVTransferEstimator(_network("RDMA"), accelerator_bandwidth=1)
    assert fitted.bandwidth == PROFILES["RDMA"]["bandwidth_gbps"] * 1e9
    # Without either key the model is the nominal bandwidth alone.
    nominal = KVTransferEstimator({}, accelerator_bandwidth=5e9)
    assert nominal.est_service_dur(5_000_000_000) == 1.0
    with pytest.raises(ValueError, match="bandwidth_gbps"):
        KVTransferEstimator({})


def test_unknown_network_is_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        KVTransferEstimator(_network("NOT_A_NETWORK"))
    # Nothing is bundled: a network name needs profiles to look it up in.
    with pytest.raises(ValueError, match="network_profiles"):
        KVTransferEstimator({"network": "RDMA"})


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"alpha_us": -1, "bandwidth_gbps": 1}, "alpha_us"),
        ({"alpha_us": 1, "bandwidth_gbps": 0}, "bandwidth"),
        (
            {
                "alpha_us": 1,
                "bandwidth_gbps": 1,
                "excursion": {"bands": [{"probability": 2, "magnitude_s": {0.1: 1}}]},
            },
            "probability",
        ),
    ],
)
def test_invalid_estimator_parameters_are_rejected(config, message):
    with pytest.raises(ValueError, match=message):
        KVTransferEstimator(config)
