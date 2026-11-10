"""Capture and export contracts for the collector's AIS observation export."""

import json
from statistics import pvariance
from types import SimpleNamespace as NS

import pytest
from sglang_simulator.tools.collect_baseline import hooks
from test_optional_aisimulate import run_isolated

from sglang.srt.model_executor.forward_mode import ForwardMode

EXTEND, DECODE, MIXED = ForwardMode.EXTEND, ForwardMode.DECODE, ForwardMode.MIXED


class FakeTime:
    """Deterministic clock: distinct increasing perf_counter values, so every
    step's wall_time is positive and distinct without sleeping."""

    def __init__(self):
        self.n = 0
        self.t = 0.0

    def time(self):
        return 1000.0

    def perf_counter(self):
        self.n += 1
        self.t += 0.001 * self.n
        return self.t


class _SeqLens(list):
    def tolist(self):
        return list(self)


def _req(rid, **kwargs):
    return NS(
        rid=rid,
        origin_input_ids=[1] * 8,
        cached_tokens_device=0,
        cached_tokens_host=0,
        cached_tokens_storage=0,
        output_ids=[],
        **kwargs,
    )


def _seqs(*lengths):
    return _SeqLens(lengths)


def _prefill_batch(*extends, prefixes=None):
    reqs = [_req(f"p{i}") for i in range(len(extends))]
    return NS(
        forward_mode=EXTEND,
        reqs=reqs,
        extend_lens=list(extends),
        prefix_lens=prefixes or [0] * len(extends),
    )


def _decode_batch(seqs, beam_tail=None, count=None):
    count = len(seqs) if count is None else count
    return NS(
        forward_mode=DECODE,
        reqs=[_req(f"d{i}") for i in range(count)],
        seq_lens_cpu=_seqs(*seqs),
        beam_tail=beam_tail,
    )


def _mixed_batch(extend=1, prefix=8, decode_seqs=(6,)):
    """Keep decode first, including when the prefill also has length one."""
    decode = [_req(f"d{i}") for i in range(len(decode_seqs))]
    return NS(
        forward_mode=MIXED,
        reqs=decode + [_req("p")],
        decoding_reqs=decode,
        seq_lens_cpu=_seqs(*decode_seqs, extend + prefix),
        extend_lens=[1] * len(decode) + [extend],
        prefix_lens=[s - 1 for s in decode_seqs] + [prefix],
    )


def install_collector(tmp_path, monkeypatch):
    """Install the collection hooks with export enabled, hermetic rank prefix
    and a deterministic FakeTime clock; return (scheduler, state). The fake
    scheduler mutates the batch during run_batch, so every test also proves
    the snapshot was taken before the forward pass."""
    monkeypatch.setenv(hooks.COLLECTION_OUTPUT_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(hooks.COLLECTION_EXPORT_AIS_OBSERVATIONS_ENV, "1")
    monkeypatch.setattr(hooks, "_rank_prefix", lambda: "TP0")
    monkeypatch.setattr(hooks, "_synchronize_device", lambda: None)
    monkeypatch.setattr(hooks, "time", FakeTime())

    class Scheduler:
        def run_batch(self, batch):
            batch.reqs.clear()
            return "result"

        def process_input_requests(self, recv_reqs):
            return None

        def process_batch_result(self, batch):
            return None

        def init_profiler(self):
            return None

    hooks.SchedulerCollectionHook.hook(Scheduler)
    state = hooks.CollectionState()
    monkeypatch.setattr(hooks, "COLLECTION_STATE", state)
    state.start()
    return Scheduler(), state


@pytest.fixture
def collector(tmp_path, monkeypatch):
    return install_collector(tmp_path, monkeypatch) + (tmp_path,)


def test_capture_contracts_per_mode(collector):
    """One representative batch per mode: single-token and uneven prefill,
    prior KV excluding the decode token, mixed membership by decoding_reqs
    (not by length), and beam-tail rows counted as computed work."""
    scheduler, state, _ = collector
    scheduler.run_batch(_prefill_batch(1, 512, prefixes=[0, 100]))
    scheduler.run_batch(_decode_batch([17, 13]))
    scheduler.run_batch(_mixed_batch())
    scheduler.run_batch(
        _decode_batch(
            [10, 12, 10, 10],
            beam_tail=NS(num_base_rows=2, entries=[NS(leader_idx=0, start=0, end=2)]),
            count=2,
        )
    )
    prefill, decode, mixed, beam = (
        obs[0]["scheduled_requests"] for obs in state.ais_observations
    )
    assert (prefill["num_prefill_requests"], prefill["sum_prefill_tokens"]) == (2, 513)
    assert prefill["sum_prefill_kv_tokens"] == 100  # only the uneven prefix row
    assert prefill["var_prefill_length"] == pvariance([1, 512])
    assert decode["num_decode_requests"] == 2
    assert decode["sum_decode_kv_tokens"] == 28  # (17-1) + (13-1)
    assert decode["var_decode_kv_tokens"] == 4
    assert (
        mixed["num_prefill_requests"],
        mixed["num_decode_requests"],
        mixed["sum_prefill_tokens"],
        mixed["sum_prefill_kv_tokens"],
        mixed["sum_decode_kv_tokens"],
    ) == (1, 1, 1, 8, 5)
    assert mixed["extend_lengths"] == [1, 1]  # scheduler order preserved
    assert mixed["past_kv_lengths"] == [5, 8]
    assert (beam["num_decode_requests"], beam["sum_decode_kv_tokens"]) == (4, 38)
    # request_infos stay tied to the batch's actual requests, not beam rows.
    assert [info["rid"] for info in state.schedule_batches[3]["request_infos"]] == [
        "d0",
        "d1",
    ]
    assert all(0 < obs[0]["wall_time"] for obs in state.ais_observations)


def test_unsupported_mode_fails_export_and_keeps_records(collector):
    scheduler, state, tmp_path = collector
    batch = _prefill_batch(1)
    batch.forward_mode = ForwardMode.TARGET_VERIFY
    batch.seq_lens_cpu = _seqs(4)
    scheduler.run_batch(batch)
    assert state.ais_observations == []
    assert len(state.schedule_batches) == 1  # the raw trace is unaffected
    with pytest.raises(RuntimeError, match="target_verify"):
        state.stop_and_export()
    assert state.active and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("wall_time", [0.0, float("nan")])
def test_invalid_wall_time_fails_export(collector, wall_time):
    _, state, _ = collector
    state.record_ais_observation({"scheduled_requests": {}}, wall_time)
    with pytest.raises(RuntimeError, match="wall_time"):
        state.stop_and_export()


def test_probe_idle_and_empty_steps_are_skipped(collector):
    scheduler, state, _ = collector
    scheduler.run_batch(_prefill_batch(100))
    probe = _mixed_batch()
    probe.reqs[0].rid = "HEALTH_CHECK_probe"
    scheduler.run_batch(probe)
    scheduler.run_batch(NS(forward_mode=ForwardMode.IDLE, reqs=[]))
    scheduler.run_batch(
        NS(forward_mode=EXTEND, reqs=[], extend_lens=[], prefix_lens=[])
    )
    assert len(state.ais_observations) == 1
    assert len(state.schedule_batches) == 3  # idle/empty rows stay in the trace


def test_export_shape_unit_and_default_off(collector, monkeypatch):
    scheduler, state, tmp_path = collector
    scheduler.run_batch(_prefill_batch(128))
    scheduler.run_batch(_decode_batch([1025]))
    names = {path.name for path in state.stop_and_export()}
    assert names == {
        "TP0.request.jsonl",
        "TP0.raw_request.jsonl",
        "TP0.schedule_batch.jsonl",
        "TP0.ais_observations.json",
    }
    observations = json.loads((tmp_path / "TP0.ais_observations.json").read_text())
    assert all(isinstance(obs, list) and len(obs) == 1 for obs in observations)
    rows = [
        json.loads(line)
        for line in (tmp_path / "TP0.schedule_batch.jsonl").read_text().splitlines()
    ]
    # wall_time equals the trace's iter_latency, in seconds.
    assert [obs[0]["wall_time"] for obs in observations] == [
        row["iter_latency"] for row in rows
    ]
    # Disabled export keeps the default file set, with no partial write.
    off = tmp_path / "off"
    off.mkdir()
    monkeypatch.setenv(hooks.COLLECTION_OUTPUT_DIR_ENV, str(off))
    monkeypatch.setenv(hooks.COLLECTION_EXPORT_AIS_OBSERVATIONS_ENV, "0")
    state.start()
    scheduler.run_batch(_prefill_batch(64))
    assert all(path.suffix == ".jsonl" for path in state.stop_and_export())


def test_existing_file_rolls_back_and_keeps_records(collector):
    scheduler, state, tmp_path = collector
    scheduler.run_batch(_prefill_batch(64))
    stale = tmp_path / "TP0.ais_observations.json"
    stale.write_text("[]")
    with pytest.raises(FileExistsError):
        state.stop_and_export()
    # Only the pre-existing file remains; records survive for a retry.
    assert {path.name for path in tmp_path.iterdir()} == {stale.name}
    assert stale.read_text() == "[]" and state.active
    stale.unlink()
    assert len(state.stop_and_export()) == 4
    assert not state.active and state.ais_observations == []


def test_collection_scope_guards(monkeypatch):
    from sglang.srt import runtime_context

    parallel = NS(tp_rank=0, pp_size=1, attn_dp_size=1, attn_cp_size=1)
    server_args = NS(speculative_algorithm=None)
    model_config = NS(is_multimodal=False)
    monkeypatch.setenv(hooks.COLLECTION_EXPORT_AIS_OBSERVATIONS_ENV, "1")
    monkeypatch.setattr(runtime_context, "get_parallel", lambda: parallel)
    monkeypatch.setattr(runtime_context, "get_server_args", lambda: server_args)
    monkeypatch.setattr(runtime_context, "process_model_config", lambda: model_config)
    hooks._validate_ais_export_request()  # TP-only, text-only passes.
    for name in ("pp_size", "attn_dp_size", "attn_cp_size"):
        setattr(parallel, name, 2)
        with pytest.raises(RuntimeError, match=name):
            hooks._validate_ais_export_request()
        setattr(parallel, name, 1)
    server_args.speculative_algorithm = "EAGLE"
    with pytest.raises(RuntimeError, match="speculative"):
        hooks._validate_ais_export_request()
    server_args.speculative_algorithm = None
    model_config.is_multimodal = True
    with pytest.raises(RuntimeError, match="text-only"):
        hooks._validate_ais_export_request()


def test_collector_imports_without_aisimulate_sdk():
    run_isolated(
        """
        import sglang_simulator.tools.collect_baseline.collect  # CLI entry point
        from sglang_simulator.simulation.sglang.batch_snapshot import (
            batch_to_fpm_metrics, capture_batch_entry,
        )
        assert "aisimulate" not in sys.modules
        assert "aisimulate_core" not in sys.modules
        """
    )
