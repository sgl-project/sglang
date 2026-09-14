"""CPU gates for matched workloads, full-model reports and bounded commands."""

import json
import socket
import sqlite3
import sys
from dataclasses import asdict, replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from nccl_ep_test.followup_server import MODEL, REVISION
from nccl_ep_test.full_model_benchmark import (
    Workload,
    decode_step,
    load_benchmark_model,
    model_args,
)
from nccl_ep_test.performance_report import compare_logits, compare_runs, validate_pair
from nccl_ep_test.performance_suite import build_plan, execute
from nccl_ep_test.performance_trace import analyze, union_ns

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def pair(configuration="serial"):
    work = Workload(buckets=(8,), warmups=2, samples=2, rounds=1)
    tbo = "tbo" in configuration
    return [
        dict(
            rank=rank,
            passed=True,
            native_ep_tested=True,
            cleanup_completed=True,
            profiled=False,
            implementation="nccl_ep_full_model_decode_v1",
            source_head="head",
            configuration=configuration,
            model=MODEL,
            revision=REVISION,
            workload=asdict(work),
            workload_fingerprint=work.fingerprint(),
            model_shape=dict(layers=27, moe_layers=26),
            bindings={},
            environment={
                k: None
                for k in ("devices", "torch", "torch_cuda", "topology", "toolkit")
            },
            resolved_args=dict(
                enable_two_batch_overlap=tbo,
                enable_single_batch_overlap=False,
                ep_size=2,
                enable_eplb=False,
            ),
            records=[
                dict(
                    bucket=8,
                    round=0,
                    graph_passes=2,
                    tbo_passes=2 * tbo,
                    samples={
                        metric: [rank + 1, (rank + 1) * 2]
                        for metric in ("cuda_step_ms", "host_step_ms")
                    },
                )
            ],
        )
        for rank in (0, 1)
    ]


def test_workloads_keep_rank_specific_tokens_and_identical_kv_history():
    workload = Workload()
    assert workload.tokens(0, 8, 0) != workload.tokens(1, 8, 0)
    assert workload.tokens(0, 8, 0) != workload.tokens(0, 8, 1)
    assert workload.tokens(0, 8, 0) == workload.tokens(0, 32, 0)[:8]
    assert workload.fingerprint() != replace(workload, warmups=64).fingerprint()
    assert (
        workload.fingerprint()
        == Workload(**json.loads(json.dumps(asdict(workload)))).fingerprint()
    )
    for bad in ((1,), (8, 8), (128,), (3,)):
        with pytest.raises(ValueError):
            replace(workload, buckets=bad).validate()


def test_model_commands_keep_full_weights_capacity_and_graph_buckets():
    commands = [model_args(Workload(), config, 29619) for config in ("serial", "tbo")]
    assert commands[1] == commands[0] + ["--enable-two-batch-overlap"]
    for cmd in commands:
        assert "--enable-eplb" not in cmd and "--json-model-override-args" not in cmd
        assert cmd[cmd.index("--revision") + 1] == REVISION
        assert cmd[cmd.index("--cuda-graph-bs-decode") + 1 :][:3] == ["8", "32", "64"]
        assert int(cmd[cmd.index("--max-running-requests") + 1]) // 2 >= 64


def test_decode_driver_preserves_execution_evidence_at_runner_boundary(monkeypatch):
    from sglang.benchmark import one_batch
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    calls = []
    batch = SimpleNamespace(prepare_for_decode=lambda: calls.append("prepare"))
    fb = SimpleNamespace(can_run_tbo=True)
    monkeypatch.setattr(
        one_batch, "_maybe_prepare_mlp_sync_batch", lambda *a: calls.append("sync")
    )
    initialize = create_autospec(ForwardBatch.init_new, return_value=fb)
    monkeypatch.setattr(ForwardBatch, "init_new", initialize)
    logits = torch.ones(2, 8)
    runner = SimpleNamespace(
        forward=lambda b: SimpleNamespace(
            logits_output=SimpleNamespace(next_token_logits=logits), can_run_graph=True
        )
    )
    tokens = torch.tensor([2, 3])
    actual, graph, tbo = decode_step(tokens, batch, runner)
    assert calls == ["prepare", "sync"] and batch.input_ids is tokens
    initialize.assert_called_once_with(
        batch, runner, return_hidden_states_before_norm=False
    )
    assert actual is logits and graph and tbo


def test_model_loader_can_host_its_store_under_torchrun(monkeypatch, tmp_path):
    from dataclasses import make_dataclass

    import huggingface_hub

    from sglang.benchmark import one_batch

    frozen_args = make_dataclass("FrozenArgs", ["tokenizer_path"], frozen=True)
    server = frozen_args(str(tmp_path))

    def cached_snapshot(repo_id, *, revision, local_files_only):
        assert (repo_id, revision, local_files_only) == (MODEL, REVISION, True)
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", cached_snapshot)
    command = model_args(Workload(), "serial", 29619, resolve_tokenizer=True)
    assert command[command.index("--tokenizer-path") + 1] == str(tmp_path)
    # torchrun's agent serves MASTER_PORT, not the benchmark's independent port.
    monkeypatch.setenv("TORCHELASTIC_USE_AGENT_STORE", "True")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    def loader(server, ports, gpu_id, rank):
        assert server.tokenizer_path == str(tmp_path)
        rendezvous = torch.distributed.rendezvous(
            f"tcp://127.0.0.1:{ports.nccl_port}",
            rank=rank,
            world_size=1,
            timeout=timedelta(seconds=1),
        )
        store, _, _ = next(rendezvous)
        store.set("ready", "yes")
        assert store.get("ready") == b"yes"
        return "runner", "tokenizer"

    monkeypatch.setattr(one_batch, "load_model", loader)
    assert load_benchmark_model(server, port, 0) == ("runner", "tokenizer")


def test_rank_alignment_and_global_dp_throughput():
    summary = validate_pair(pair())
    values = summary["rows"][0]["metrics"]["cuda_step_ms"]
    assert values["median"] == 3
    assert values["aggregate_tokens_per_second"] == 16000 / 3


@pytest.mark.parametrize(
    "mutation",
    [
        "profile",
        "missing",
        "duplicate",
        "sample",
        "fallback",
        "nan",
        "revision",
        "shape",
        "sha",
        "cleanup",
        "fingerprint",
    ],
)
def test_invalid_reports_cannot_publish_performance(mutation):
    reports = pair("tbo")
    report = reports[1]
    if mutation == "profile":
        report["profiled"] = True
    elif mutation == "missing":
        report["records"] = []
    elif mutation == "duplicate":
        report["records"] *= 2
    elif mutation == "sample":
        report["records"][0]["samples"]["cuda_step_ms"].pop()
    elif mutation == "fallback":
        report["records"][0]["tbo_passes"] = 0
    elif mutation == "nan":
        report["records"][0]["samples"]["cuda_step_ms"][0] = float("nan")
    elif mutation == "revision":
        report["revision"] = "changed"
    elif mutation == "shape":
        report["model_shape"]["layers"] = 2
    elif mutation == "sha":
        report["source_head"] = "old"
    elif mutation == "cleanup":
        report["cleanup_completed"] = False
    else:
        report["workload_fingerprint"] = "wrong"
    with pytest.raises(ValueError):
        validate_pair(reports)


def test_logit_comparison_rejects_wrong_answers_and_missing_evidence(tmp_path):
    logits = {f"B8/round0/sample{s}": torch.tensor([[1.0, 2.0, 3.0]]) for s in (0, 1)}
    for config in ("serial", "tbo"):
        directory = tmp_path / config
        directory.mkdir()
        for rank, report in enumerate(pair(config)):
            (directory / f"model-rank{rank}.json").write_text(json.dumps(report))
            torch.save(logits, directory / f"logits-rank{rank}.pt")
    assert compare_runs(tmp_path / "serial", tmp_path / "tbo")["passed"]
    with pytest.raises(AssertionError):
        compare_logits(logits, {k: v + 1 for k, v in logits.items()})
    torch.save({}, tmp_path / "tbo" / "logits-rank1.pt")
    with pytest.raises(ValueError, match="Incomplete checkpoint"):
        compare_runs(tmp_path / "serial", tmp_path / "tbo")


def test_profile_uses_same_workload_and_capacity_as_timing(tmp_path):
    entries = build_plan(tmp_path)
    assert len({e["directory"] for e in entries}) == len(entries)
    models = [e for e in entries if e["phase"] == "model"]
    assert [e["configuration"] for e in models] == ["serial", "tbo", "tbo", "serial"]
    for entry in entries:
        command = entry["command"]
        pos = command.index("--buckets")
        assert command[pos + 1 : pos + 4] == ["8", "32", "64"]
        if entry["phase"] == "profile":
            assert "--capture-range=cudaProfilerApi" in command
            assert "--profile-bucket" in command
            assert "--cuda-graph-trace=node" in command


def test_failed_worker_and_existing_results_are_not_overwritten(tmp_path):
    entry = dict(
        directory=str(tmp_path / "failed"),
        command=[sys.executable, "-c", "raise SystemExit(3)"],
    )
    with pytest.raises(RuntimeError, match="exited 3"):
        execute(entry)
    record = json.loads((tmp_path / "failed" / "invocation.json").read_text())
    assert not record["passed"] and record["exit_code"] == 3
    with pytest.raises(FileExistsError):
        execute(entry)


def test_trace_requires_native_graph_work_on_both_ranks(tmp_path):
    path = tmp_path / "trace.sqlite"
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE StringIds (id INTEGER, value TEXT);
            CREATE TABLE NVTX_EVENTS (start INTEGER,end INTEGER,globalTid INTEGER,text TEXT,textId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER,end INTEGER,globalTid INTEGER,nameId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (deviceId INTEGER,start INTEGER,end INTEGER,shortName INTEGER,graphNodeId INTEGER);
        """)
        db.executemany(
            "INSERT INTO StringIds VALUES (?,?)",
            enumerate(
                [
                    "cudaGraphLaunch",
                    "nccl_ep_jit_ll_dispatch_kernel",
                    "fused_moe_kernel",
                    "nccl_ep_jit_ll_combine_kernel",
                    "cudaMalloc",
                ],
                start=1,
            ),
        )
        for rank in (0, 1):
            db.execute(
                "INSERT INTO NVTX_EVENTS VALUES (0,100,?,?,NULL)",
                (rank, f"nccl_ep_model/rank={rank}/step=0"),
            )
            db.execute(
                "INSERT INTO NVTX_EVENTS VALUES (20,25,?,'nccl_ep_performance/replay',NULL)",
                (rank,),
            )
            db.execute(
                "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (21,22,?,1)", (rank,)
            )
            db.executemany(
                "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,1)",
                [(rank, 30, 40, 2), (rank, 35, 60, 3), (rank, 60, 65, 4)],
            )
    result = analyze(path, expected_steps=1)
    assert result["passed"] and result["graph_launches"] == 2
    assert result["steps"][0]["kernel_busy_ns"] == 35
    assert result["steps"][0]["ep_other_kernel_overlap_ns"] == 5
    assert union_ns([(0, 10), (5, 15), (20, 25)]) == 20
    with sqlite3.connect(path) as db:
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (23,24,0,5)")
    with pytest.raises(ValueError, match="resource operations"):
        analyze(path, expected_steps=1)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
