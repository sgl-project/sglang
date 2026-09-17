# SPDX-License-Identifier: Apache-2.0
"""Real diffusion UDS/IPC orchestration with a tiny, test-only CUDA component.

Only checkpoint resolution and distributed bootstrap are substituted. Requests,
generation admission, mapping, finalization ordering, and watchdogs are real.
No fault switches are added to production code.
"""

import multiprocessing as mp
import os
import signal
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.runtime.weight_cache.client import (
    WeightCacheClient,
    materialize_from_cache,
)
from sglang.multimodal_gen.runtime.weight_cache.daemon import DiffusionWeightCacheDaemon
from sglang.multimodal_gen.runtime.weight_cache.plan import CacheCompatibilityPlan
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=150, stage="base-b", runner_config="1-gpu-small")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA IPC")


class TinyComponent:
    name = "transformer"

    def __init__(self, connection=None, pause=None):
        self.connection = connection
        self.pause = pause

    def _pause(self, stage):
        if stage == self.pause:
            self.connection.send(stage)
            # Simulate blocked finalization/slow uncached loading while the
            # real watchdog must remain live. Parent never releases this wait.
            assert self.connection.recv() == "continue"

    def load_ordinary(self):
        model = torch.nn.Linear(4, 4, bias=False, device="cuda:0")
        with torch.no_grad():
            model.weight.fill_(2)
        return model.eval(), 0

    def build_meta(self):
        self._pause("before_fetch")
        return torch.nn.Linear(4, 4, bias=False, device="meta").eval()

    def finalize_after_import(self, model):
        self._pause("finalize")
        return model


class TinyPrepared:
    def __init__(self, connection=None, pause=None):
        self.cached_components = (TinyComponent(connection, pause),)

    def materialize(self, args, *, loaded_modules):
        self.cached_components[0]._pause("uncached_load")
        torch.testing.assert_close(
            loaded_modules["transformer"].weight.cpu(), torch.full((4, 4), 2.0)
        )
        return SimpleNamespace(memory_usages={}, model=loaded_modules["transformer"])


def _args(root):
    return SimpleNamespace(
        weight_cache_socket=str(Path(root) / "owner.sock"),
        weight_cache_timeout=5,
        weight_cache_max_deliveries=8,
        gpu_ids=None,
        base_gpu_id=0,
    )


def _owner(root, plan):
    from sglang.multimodal_gen.runtime.weight_cache import daemon

    os.environ["SGLANG_DIFFUSION_WEIGHT_CACHE_DIR"] = root
    owner = object.__new__(DiffusionWeightCacheDaemon)
    owner.args = _args(root)
    owner.prepared = TinyPrepared()
    owner.plan = plan
    owner.path = Path(owner.args.weight_cache_socket)
    owner.ready_path = owner.path.with_suffix(".ready")
    owner._initialize_control()
    with (
        patch.object(
            daemon,
            "bootstrap_diffusion_runtime",
            side_effect=lambda *a, **k: torch.cuda.set_device(0),
        ),
        patch.object(daemon, "compatibility_plan", return_value=plan),
    ):
        owner.run()


def _worker(root, plan, generation, connection, pause):
    from sglang.multimodal_gen.runtime.weight_cache import client

    args = _args(root)
    args._weight_cache_admission = (plan, generation)
    with patch.object(client, "compatibility_plan", return_value=plan):
        try:
            materialize_from_cache(TinyPrepared(connection, pause), args)
        except ValueError as error:
            connection.send(("rejected", str(error)))
            raise
    connection.send("pipeline_ready")
    connection.recv()


@pytest.fixture
def service():
    context = mp.get_context("spawn")
    children = []
    connections = []
    with tempfile.TemporaryDirectory(prefix="wc-fault-") as root:
        plan = CacheCompatibilityPlan.from_fields(
            rank={"device_uuid": str(torch.cuda.get_device_properties(0).uuid)},
            test="tiny-diffusion-service",
        )

        def start_owner():
            process = context.Process(target=_owner, args=(root, plan))
            process.start()
            children.append(process)
            ready = Path(root) / "owner.ready"
            deadline = time.monotonic() + 60
            while not ready.exists() or not ready.read_text().startswith(
                f"pid={process.pid}\n"
            ):
                assert process.is_alive(), process.exitcode
                assert time.monotonic() < deadline, "owner readiness timeout"
                time.sleep(0.1)
            return process

        def start_worker(generation, pause):
            parent, child = context.Pipe()
            worker = context.Process(
                target=_worker, args=(root, plan, generation, child, pause)
            )
            worker.start()
            child.close()
            children.append(worker)
            connections.append(parent)
            return worker, parent

        try:
            yield SimpleNamespace(
                root=root,
                plan=plan,
                args=_args(root),
                start_owner=start_owner,
                start_worker=start_worker,
            )
        finally:
            # Consumers first; no producer allocations are voluntarily freed
            # before their importing processes have exited.
            for process in reversed(children):
                if process.is_alive():
                    process.kill()
                process.join(10)
            for connection in connections:
                connection.close()


@pytest.mark.parametrize("stage", ["before_fetch", "finalize", "uncached_load"])
def test_owner_death_before_pipeline_ready_is_fail_stop(service, stage):
    owner = service.start_owner()
    with WeightCacheClient(service.plan, service.args) as client:
        generation, _ = client.manifest()
    worker, connection = service.start_worker(generation, stage)
    assert connection.poll(60), "worker never reached fault barrier"
    assert connection.recv() == stage
    owner.kill()
    owner.join(10)
    worker.join(10)
    assert worker.exitcode == -signal.SIGKILL, "watchdog did not kill blocked startup"
    # A successful return from materialize would publish this event.
    if connection.poll():
        with pytest.raises(EOFError):
            connection.recv()


def test_owner_replacement_between_manifest_and_worker_rejects_generation(service):
    first = service.start_owner()
    with WeightCacheClient(service.plan, service.args) as client:
        generation, _ = client.manifest()
    first.kill()
    first.join(10)
    second = service.start_owner()
    with WeightCacheClient(service.plan, service.args) as client:
        replacement, _ = client.manifest()
        assert replacement != generation
        # Replay old generation on a real authenticated connection: no exports.
        import uuid

        import msgspec

        with pytest.raises(RuntimeError, match="fetch generation mismatch"):
            client.request(
                "fetch_component",
                component="transformer",
                generation=msgspec.to_builtins(generation),
                request_id=uuid.uuid4().hex,
            )
    worker, connection = service.start_worker(generation, None)
    assert connection.poll(60)
    assert connection.recv() == (
        "rejected",
        "Weight-cache generation changed after launcher admission",
    )
    worker.join(60)
    assert worker.exitcode == 1  # strict admission exception, not readiness
    with WeightCacheClient(service.plan, service.args) as client:
        status = client.status()
        assert status["deliveries_reserved"] == 0
        assert status["active_consumers"] == 0
    assert second.is_alive()


def test_concurrent_worker_admission_and_status_during_meta_construction(service):
    owner = service.start_owner()
    with WeightCacheClient(service.plan, service.args) as client:
        generation, _ = client.manifest()
    workers = [service.start_worker(generation, "before_fetch") for _ in range(2)]
    for worker, connection in workers:
        assert connection.poll(60), "another client monopolized owner admission"
        assert connection.recv() == "before_fetch"
    with WeightCacheClient(service.plan, service.args) as client:
        assert client.status()["deliveries_reserved"] == 0
    for _, connection in workers:
        connection.send("continue")
    for _, connection in workers:
        assert connection.poll(60)
        assert connection.recv() == "pipeline_ready"
    with WeightCacheClient(service.plan, service.args) as client:
        status = client.status()
        assert status["active_consumers"] == 2
        assert status["deliveries_reserved"] == 2
        assert status["fetches_remaining"] == 6
    owner.terminate()
    owner.join(10)
    assert owner.exitcode == 0
    for worker, _ in workers:
        worker.join(10)
        assert not worker.is_alive()


if __name__ == "__main__":
    # Registered CI invokes files as `python3 file.py -f` (unittest convention).
    # Pass only pytest's own flags so the suite cannot silently run zero tests.
    raise SystemExit(pytest.main([__file__, "-v", "-x"]))
