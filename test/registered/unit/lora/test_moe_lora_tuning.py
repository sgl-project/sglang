"""CPU-only tuning contracts; CUDA and workloads are explicit test doubles."""

import copy
import hashlib
import importlib.util
import sys
import weakref
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

ROOT = Path(__file__).resolve().parents[4]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {name: module}):
        spec.loader.exec_module(module)
    return module


T = load("lora_tuning_cpu_test", "benchmark/kernels/lora_tuning.py")
P = load("moe_tuning_cpu_test", "benchmark/kernels/lora_moe/tune_plans.py")
register_cpu_ci = load(
    "tuning_ci_marker", "python/sglang/test/ci/ci_register.py"
).register_cpu_ci
register_cpu_ci(est_time=1, suite="base-a-test-cpu")
CASE = dict(
    name="cpu",
    hidden_size=2048,
    intermediate_size=512,
    num_local_experts=32,
    tokens=16,
    rank=32,
    top_k=8,
)


@pytest.mark.parametrize(
    "baseline,candidate,status",
    [
        ([100] * 3, [80] * 3, "WIN"),
        ([100] * 3, [200] * 3, "LOSS"),
        ([100] * 3, [99] * 3, "TIE"),
        ([100] * 3, [80, 80, 110], "INCONCLUSIVE"),
        ([80, 100, 120], [72, 90, 108], "TIE"),
        ([100] * 3, [80, 80, 100], "TIE"),
    ],
)
def test_compare_gates(baseline, candidate, status):
    result = T.compare(baseline, candidate)
    assert result["status"] == status
    assert result["worst_gain"] == min(b / c - 1 for b, c in zip(baseline, candidate))
    assert result["pairs"] == 3


@pytest.mark.parametrize(
    "bad",
    [[], [1] * 2, [1] * 4, [True] * 3, [0] * 3, [float("nan")] * 3, [float("inf")] * 3],
)
def test_compare_rejects_invalid_pairs(bad):
    with pytest.raises(ValueError):
        T.compare([1] * 3, bad)


def test_choose_preserves_results_and_uses_only_wins():
    result = T.choose([100] * 3, {"z": [80] * 3, "a": [80] * 3, "tail": [80, 80, 110]})
    assert result["winner"] == "a" and len(result["results"]) == 3
    assert T.choose([100] * 3, {"tie": [100] * 3})["winner"] is None
    with pytest.raises(ValueError):
        T.choose([], {})


@pytest.mark.parametrize("mode,warmup", [("eager", 2), ("graph", 2), ("graph", 0)])
def test_measure_executes_eager_or_captured_replay(mode, warmup):
    trace, cuda = [], Mock()
    outputs = []

    class Output:
        pass

    def call():
        trace.append("call")
        output = Output()
        outputs.append(weakref.ref(output))
        return output

    def replay():
        assert outputs[-1]() is not None

    @contextmanager
    def context(name):
        trace.append(name)
        yield

    cuda.stream.side_effect = lambda _: context("warm")
    cuda.graph.side_effect = lambda *a, **k: context("capture")
    cuda.Event.return_value.elapsed_time.return_value = 1
    cuda.CUDAGraph.return_value.replay.side_effect = replay
    fn = Mock(side_effect=call)
    with (
        patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=cuda)}),
        patch.object(T.time, "perf_counter_ns", side_effect=[0, 1_000_000] * 2),
    ):
        assert (
            T.measure(fn, mode=mode, warmup=warmup, repeats=2, iterations=4)
            == [250] * 2
        )
    graph, stream = cuda.CUDAGraph.return_value, cuda.Stream.return_value
    assert graph.replay.call_count == (9 if mode == "graph" else 0)
    assert fn.call_count == (
        warmup + max(warmup, 1) + 1 if mode == "graph" else warmup + 8
    )
    if mode == "graph":
        assert trace == ["call"] * warmup + ["warm"] + ["call"] * max(warmup, 1) + [
            "capture",
            "call",
        ]
        cuda.graph.assert_called_once_with(graph, stream=stream)
        stream.wait_stream.assert_called_once_with(cuda.current_stream.return_value)
        cuda.current_stream.return_value.wait_stream.assert_called_once_with(stream)


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_measure_rejects_invalid_observations(bad):
    cuda = SimpleNamespace(is_available=lambda: True, synchronize=lambda: None)
    with (
        patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=cuda)}),
        patch.object(T.time, "perf_counter_ns", side_effect=[0, bad]),
    ):
        with pytest.raises(ValueError):
            T.measure(lambda: None, mode="eager", warmup=0, repeats=1, iterations=1)


@pytest.mark.parametrize(
    "key,value", [("rank", 7), ("top_k", 33), ("quant", "nvfp4"), ("hidden_size", True)]
)
def test_case_rejects_unsupported_geometry(key, value):
    with pytest.raises(ValueError):
        P.Case.from_dict(CASE | {key: value})


@pytest.mark.parametrize(
    "tp,ep,expected",
    [(32, 8, (384, 8)), (8, 8, (1536, 8)), (6, 4, None), (40, 8, None), (6, 3, None)],
)
def test_geometry_total_tp_includes_ep(tp, ep, expected):
    path = SimpleNamespace(
        read_text=lambda: (
            '{"text_config":{"hidden_size":4096,"moe_intermediate_size":1536,"num_experts":64,"hidden_act":"silu"}}'
        )
    )
    if expected is None:
        with pytest.raises(ValueError):
            P.model_geometry(path, tp, ep)
    else:
        result = P.model_geometry(path, tp, ep)
        assert (result["intermediate_size"], result["num_local_experts"]) == expected
        assert result["hidden_size"] == 4096 and result["tp_size"] == tp


def test_candidates_are_deduplicated_one_axis_copies():
    original = {
        site: dict(num_warps=4, num_stages=2, BLOCK_SIZE_N=16, SPLIT_K=1)
        for site in ("gate_up_a", "down_a", "gate_up_b", "down_b")
    }
    before = copy.deepcopy(original)
    specs = P.candidate_specs(original)
    assert len(specs) == 17 and specs[0] == original
    for spec in specs[1:]:
        assert (
            sum(spec[s][k] != original[s][k] for s in original for k in original[s])
            == 1
        )
    specs[0]["gate_up_a"]["SPLIT_K"] = 9
    assert original == before and all(s["gate_up_a"]["SPLIT_K"] == 1 for s in specs[1:])


@pytest.mark.parametrize(
    "search,validation", [("WIN", "TIE"), ("LOSS", "WIN"), ("WIN", "INCONCLUSIVE")]
)
def test_override_requires_both_wins(search, validation):
    with pytest.raises(ValueError):
        P.study_override(
            P.Case.from_dict(CASE), {}, {}, {"status": search}, {"status": validation}
        )


@pytest.mark.parametrize(
    "scenario,expected",
    [
        ("tie", "RETAIN_BASELINE"),
        ("all_failed", "FAILED"),
        ("heldout_tie", "RETAIN_BASELINE"),
        ("win", "VALIDATED_WIN"),
        ("baseline_failed", "error"),
        ("validation_failed", "error"),
        ("changed_base", "error"),
        ("changed_source", "error"),
        ("retained_changed_source", "error"),
        ("changed_metadata", "error"),
        ("fatal", "error"),
    ],
)
def test_workflow_is_fail_closed(scenario, expected, monkeypatch):
    trace, emitted, seeds = [], [], []
    specs = [{"tile": i} for i in range(3)]

    class Workload:
        def __init__(self, case, seed):
            seeds.append(seed)
            self.seed = seed

        identity, incumbent = {"device": "fake"}, specs[0]
        selected = SimpleNamespace(
            plan=SimpleNamespace(
                gate_up_a=SimpleNamespace(family="grouped"),
                down_a=SimpleNamespace(family="grouped"),
                gate_up_b=object(),
                down_b=object(),
                finalize=SimpleNamespace(family="materialized"),
            )
        )

        def bind(self, spec):
            trace.append(("bind", self.seed, spec["tile"]))
            return (self.seed, spec["tile"])

        def check(self, call):
            trace.append(("check", *call))
            seed, tile = call
            if (
                (scenario == "baseline_failed" and tile == 0)
                or (scenario == "all_failed" and tile != 0)
                or (scenario == "validation_failed" and seed == 104729)
            ):
                raise ValueError("correctness failed")
            if scenario == "fatal" and tile == 1:
                raise RuntimeError("CUDA illegal memory access")

        def base_config(self):
            return {"base": self.seed if scenario == "changed_base" else 1}

    def paired(baseline, candidate, case, args):
        assert ("check", *baseline) in trace and ("check", *candidate) in trace
        trace.append(("timing", *candidate))
        tied = scenario in ("tie", "retained_changed_source") or (
            scenario == "heldout_tie" and candidate[0] == 104729
        )
        return dict(status="TIE" if tied else "WIN", median_gain=candidate[1] / 10)

    identities = [
        {"source": "before"},
        {
            "source": "after"
            if scenario in ("changed_source", "retained_changed_source")
            else "before"
        },
    ]
    metadata = [{}, {"version": "changed"} if scenario == "changed_metadata" else {}]
    monkeypatch.setitem(sys.modules, "workload", SimpleNamespace(Workload=Workload))
    monkeypatch.setattr(P, "verify_loaded_sources", Mock(side_effect=metadata))
    monkeypatch.setattr(P, "source_identity", Mock(side_effect=identities))
    monkeypatch.setattr(P, "candidate_specs", lambda *args: specs)
    monkeypatch.setattr(P, "paired", paired)
    error = expected == "error"
    with pytest.raises((ValueError, RuntimeError)) if error else nullcontext():
        result = P._run_case(
            P.Case.from_dict(CASE), SimpleNamespace(max_candidates=3), emitted.append
        )
        assert result["status"] == expected
        if expected == "VALIDATED_WIN":
            assert result["override"]["tiles"] == specs[2]
            assert result["override"]["production_table"] is False
            assert result["override"]["identity"]["sources"] == identities[0]
        else:
            assert result["override"] is None
    heldout = scenario not in (
        "tie",
        "retained_changed_source",
        "all_failed",
        "baseline_failed",
        "fatal",
    )
    assert seeds == ([1729, 104729] if heldout else [1729])
    assert trace[:2] == [("bind", 1729, 0), ("check", 1729, 0)]
    if scenario in ("baseline_failed", "all_failed", "fatal"):
        assert not any(t[0] == "timing" for t in trace)
    if scenario in ("validation_failed", "changed_base"):
        assert not any(t[0] == "timing" and t[1] == 104729 for t in trace)
    if scenario == "fatal":
        assert ("bind", 1729, 2) not in trace
        assert emitted[-1]["status"] == "FAILED"
    if scenario == "all_failed":
        assert [row["status"] for row in emitted] == ["CORRECT", "FAILED", "FAILED"]


def test_only_generated_version_may_be_external(tmp_path):
    path = tmp_path / "_version.py"
    path.write_text('__version__ = "test"\n')
    version = SimpleNamespace(__file__=str(path), __version__="test")
    with patch.object(P, "sys", SimpleNamespace(modules={"sglang._version": version})):
        assert P.verify_loaded_sources() == {
            "sglang._version": {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "version": "test",
            }
        }
        P.sys.modules["sglang.srt.lora.runner"] = version
        with pytest.raises(RuntimeError, match="outside this checkout"):
            P.verify_loaded_sources()


@pytest.mark.parametrize(
    "finalizer,expected",
    [("shared_one_pass", ["down_a"]), ("shared_token_delta", ["down_a", "down_b"])],
)
def test_only_consumed_moe_sites_are_candidates(finalizer, expected):
    plan = SimpleNamespace(
        gate_up_a=SimpleNamespace(family="token_dense"),
        down_a=SimpleNamespace(family="grouped"),
        gate_up_b=None,
        down_b=None,
        finalize=SimpleNamespace(family=finalizer),
    )
    sites = P.consumed_sites(plan)
    assert sites == expected
    incumbent = {
        name: dict(num_warps=4, num_stages=2, BLOCK_SIZE_N=16)
        for name in ("gate_up_a", "down_a", "gate_up_b", "down_b")
    }
    for candidate in P.candidate_specs(incumbent, sites):
        assert all(
            candidate[name] == incumbent[name]
            for name in incumbent
            if name not in sites
        )


def test_resident_cli_parallel_flags_are_not_silently_ignored(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tuner",
            "--cases",
            str(tmp_path / "missing.json"),
            "--out",
            str(tmp_path / "out"),
            "--tp-size",
            "4",
        ],
    )
    with pytest.raises(SystemExit) as error:
        P.main()
    assert error.value.code == 2 and not (tmp_path / "out").exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
