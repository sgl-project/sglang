"""HTTP/tokenizer workers must start with lazy CUDA module loading even when the
server runs with CUDA_MODULE_LOADING=EAGER: a spawned worker initializes CUDA
while importing sglang, and an eager init outlasts uvicorn's worker healthcheck.
"""

import multiprocessing
import os
import sys

import pytest

from sglang.srt.entrypoints import http_server
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _report_cuda_module_loading(queue):
    queue.put(os.environ.get("CUDA_MODULE_LOADING"))


@pytest.mark.parametrize(
    "before, after",
    [("EAGER", "LAZY"), ("eager", "LAZY"), ("LAZY", "LAZY"), (None, None)],
)
def test_http_workers_get_lazy_cuda_module_loading(monkeypatch, before, after):
    if before is None:
        monkeypatch.delenv("CUDA_MODULE_LOADING", raising=False)
    else:
        monkeypatch.setenv("CUDA_MODULE_LOADING", before)

    http_server._use_lazy_cuda_module_loading_for_http_workers()

    assert os.environ.get("CUDA_MODULE_LOADING") == after


def test_spawned_worker_inherits_lazy(monkeypatch):
    # uvicorn and granian start workers with the spawn context.
    monkeypatch.setenv("CUDA_MODULE_LOADING", "EAGER")
    ctx = multiprocessing.get_context("spawn")

    queue = ctx.Queue()
    proc = ctx.Process(target=_report_cuda_module_loading, args=(queue,))
    proc.start()
    assert queue.get(timeout=60) == "EAGER"
    proc.join(timeout=60)

    http_server._use_lazy_cuda_module_loading_for_http_workers()

    proc = ctx.Process(target=_report_cuda_module_loading, args=(queue,))
    proc.start()
    assert queue.get(timeout=60) == "LAZY"
    proc.join(timeout=60)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
