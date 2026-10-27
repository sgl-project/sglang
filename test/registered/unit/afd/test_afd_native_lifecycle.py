"""Native runner ownership and exact-once aggregate shutdown contracts."""

from types import SimpleNamespace

import pytest

from sglang.srt.afd.connector import AFDConnector
from sglang.srt.afd.contracts import AFDRole
from sglang.srt.model_executor.model_runner_components import cuda_graph_setup
from sglang.srt.model_executor.model_runner_components.cuda_graph_setup import (
    capture_decode_graph,
)
from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
    FullCudaGraphBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_afd_startup_capture_uses_existing_eager_runner_and_native_memory_accounting(
    monkeypatch,
):
    eager = object()
    seen = []
    available = iter((12.0, 10.5))
    monkeypatch.setattr(
        cuda_graph_setup, "get_available_gpu_memory", lambda *args: next(available)
    )
    mr = SimpleNamespace(
        is_draft_worker=False,
        spec_algorithm=SimpleNamespace(is_speculative=lambda: False),
        device="cuda",
        gpu_id=0,
        afd_runtime=SimpleNamespace(capture_startup=seen.append),
    )
    capture = capture_decode_graph(model_runner=mr, eager_runner=eager)
    assert seen == [eager]
    assert capture.runner is None
    assert capture.memory_usage_gb == 1.5


@pytest.mark.parametrize("warmup", [-1, True, 0.5])
def test_standalone_graph_rejects_invalid_warmup(warmup):
    with pytest.raises(ValueError):
        FullCudaGraphBackend(device_module=object(), warmup_steps=warmup)


def test_standalone_graph_requires_one_context_and_safe_output_policy():
    with pytest.raises(ValueError):
        FullCudaGraphBackend()
    with pytest.raises(ValueError):
        FullCudaGraphBackend(
            device_module=object(), warmup_steps=0, reuse_output_buffer=True
        )
    runner = SimpleNamespace(device_module=object())
    with pytest.raises(ValueError):
        FullCudaGraphBackend(runner, device_module=runner.device_module)


@pytest.mark.parametrize("failed", [False, True])
def test_aggregate_close_is_exactly_once_even_when_peer_fails(failed):
    calls, emitted = [], []

    def exchange(*, usage):
        calls.append("exchange")
        if failed:
            raise TimeoutError("peer close deadline")
        return {"status": "CLOSED"}

    connector = AFDConnector(
        transport=SimpleNamespace(
            role=AFDRole.ATTENTION,
            validate_invariants=lambda: None,
            exchange_close=exchange,
            close=lambda: calls.append("transport") or {},
        ),
        graph_strategy=SimpleNamespace(close=lambda: calls.append("graph") or {}),
        usage_emitter=emitted.append,
    )
    if failed:
        with pytest.raises(RuntimeError, match="AGGREGATE_CLOSE_FAILED") as first:
            connector.close()
        with pytest.raises(RuntimeError) as second:
            connector.close()
        assert first.value is second.value
    else:
        receipt = connector.close()
        assert connector.close() is receipt
    assert calls == ["graph", "exchange", "transport"]
    assert len(emitted) == 1 and emitted[0]["usage_final"]
    assert emitted[0]["failures"] == (("close_exchange",) if failed else ())


@pytest.mark.parametrize("outcome", ["success", "exit", "timeout", "unreaped"])
def test_ffn_worker_failure_drains_only_owned_processes(monkeypatch, outcome):
    import multiprocessing
    import multiprocessing.connection

    from sglang.srt.afd import ffn_server
    from sglang.srt.afd.config import AFDConfig
    from sglang.srt.afd.contracts import AFDError

    workers = []

    class Worker:
        def __init__(self, *, target, args):
            self.pid = None
            self.exitcode = None
            self.name = f"lane{args[1]}"
            self.sentinel = len(workers)
            self.signals = []
            workers.append(self)

        def start(self):
            self.pid = 1

        def terminate(self):
            self.signals.append("terminate")

        def kill(self):
            self.signals.append("kill")

        def join(self, *, timeout):
            assert timeout >= 0
            if outcome != "unreaped":
                self.exitcode = 0

    def wait(pending, *, timeout):
        if outcome in ("timeout", "unreaped"):
            return []
        selected = workers[pending[0]]
        selected.exitcode = 7 if outcome == "exit" else 0
        return [selected.sentinel]

    monkeypatch.setattr(
        multiprocessing, "get_context", lambda _: SimpleNamespace(Process=Worker)
    )
    monkeypatch.setattr(multiprocessing.connection, "wait", wait)
    args = SimpleNamespace(
        afd_config=AFDConfig(lanes=4, close_timeout_seconds=1), nnodes=2, node_rank=1
    )
    assert ffn_server._node_split(args) == (2, 2, (2, 3))
    assert [ffn_server._local_slot(args, lane) for lane in (2, 3)] == [0, 1]
    with pytest.raises(AFDError, match="LANE_NOT_LOCAL"):
        ffn_server._local_slot(args, 0)
    if outcome == "success":
        ffn_server._run_lanes(args, owned=(2, 3))
    else:
        with pytest.raises(
            AFDError, match="LANE_EXITED" if outcome == "exit" else "DRAIN_TIMEOUT"
        ):
            ffn_server._run_lanes(args, owned=(2, 3))
    assert [w.name for w in workers] == ["lane2", "lane3"]
    if outcome in ("timeout", "unreaped"):
        assert all(w.signals[0] == "terminate" for w in workers)
    if outcome == "unreaped":
        assert all(w.signals == ["terminate", "kill"] for w in workers)
        with pytest.raises(AFDError, match="WORKERS_NOT_REAPED"):
            ffn_server._stop_workers(workers, seconds=0)
    else:
        assert all(w.exitcode is not None for w in workers)


def test_projection_gc_reclaims_unreachable_cycles_and_preserves_live_parameters():
    import weakref

    import torch

    from sglang.srt.afd.projection_gc import collect_projection_garbage

    live = torch.nn.Linear(2, 2)
    orphan = torch.nn.Linear(2, 2)
    orphan.cycle = orphan
    ref = weakref.ref(orphan)
    del orphan
    assert collect_projection_garbage(live, "off") is None
    receipt = collect_projection_garbage(live, "attention")
    assert ref() is None
    assert list(live.parameters())[0].shape == (2, 2)
    assert receipt["role"] == "attention" and receipt["device"] == "cpu"
    assert receipt["after"] == receipt["before"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-x"]))
