"""Opt-in fault injection for externally launched CPU-buffer PD workers.

Prepend this directory to PYTHONPATH and set SGLANG_CPU_FAULT_DIR to a directory
shared by the worker processes on each host. Write ``allocate`` or ``post`` into its ``arm``
file to fail one matching operation per worker process per fault kind. Remove
``arm`` before recovery requests. Inspect the per-process JSONL event logs.
Only use on isolated test servers. No fault is injected without an arm file.
"""

import importlib.abc
import importlib.machinery
import json
import os
import sys
import threading
import time
from pathlib import Path

ROOT = os.environ.get("SGLANG_CPU_FAULT_DIR")


def install(module):
    root = Path(ROOT)
    root.mkdir(parents=True, exist_ok=True)
    seen = set()

    def event(event_name, **fields):
        with (root / f"{os.getpid()}.jsonl").open("a") as stream:
            stream.write(
                json.dumps({"time": time.time(), "event": event_name, **fields}) + "\n"
            )

    def armed(kind):
        try:
            value = (root / "arm").read_text().strip()
        except FileNotFoundError:
            return False
        if value != kind or kind in seen:
            return False
        seen.add(kind)
        event("injected", kind=kind)
        return True

    pool = module.CpuBufferPool
    allocate = pool.allocate

    def fault_allocate(self, size):
        if armed("allocate"):
            raise BufferError("Injected CPU buffer pool exhaustion")
        return allocate(self, size)

    pool.allocate = fault_allocate
    mixin = module.CpuBufferMixin
    send = mixin._send_cpu_chunk

    class AgentProxy:
        def __init__(self, agent):
            self.agent = agent

        def __getattr__(self, name):
            return getattr(self.agent, name)

        def transfer(self, handle):
            result = self.agent.transfer(handle)
            if armed("post"):
                # The write really was posted. The caller must drain its handle
                # before reusing the source or acknowledging the peer's abort.
                raise RuntimeError("Injected failure after NIXL write submission")
            return result

    def fault_send(self, *args, **kwargs):
        if not isinstance(self.agent, AgentProxy):
            self.agent = AgentProxy(self.agent)
        try:
            return send(self, *args, **kwargs)
        except Exception:
            with self._cpu_pool._lock:
                event(
                    "send_exception",
                    allocations=len(self._cpu_pool._ranges),
                    pending=len(self._cpu_pool._pending),
                )
            raise

    mixin._send_cpu_chunk = fault_send
    init = pool.__init__

    def monitored_init(self, *args, **kwargs):
        init(self, *args, **kwargs)

        def monitor():
            while True:
                with self._lock:
                    fields = dict(
                        allocations=len(self._ranges),
                        bytes=sum(self._ranges.values()),
                        pending=len(self._pending),
                    )
                event("pool", **fields)
                time.sleep(1)

        threading.Thread(target=monitor, daemon=True).start()

    pool.__init__ = monitored_init
    event("installed")


class FaultFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != "sglang.srt.disaggregation.nixl.cpu_buffer":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        original = spec.loader.exec_module

        def execute(module):
            original(module)
            install(module)

        spec.loader.exec_module = execute
        return spec


if ROOT:
    sys.meta_path.insert(0, FaultFinder())
