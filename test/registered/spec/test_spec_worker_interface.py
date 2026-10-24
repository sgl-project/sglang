"""Every spec worker must accept the keywords the scheduler passes it.

``Scheduler.run_batch`` drives all speculative decoding through
``BaseSpecWorker.forward_batch_generation``, and two of its keywords are
unconditional: ``on_publish`` on the overlap path, and ``pp_proxy_tensors`` on
the non-overlap one, which passes it whatever the pp size. A worker that falls
behind them raises ``TypeError`` on the first request down that path -- not at
import, not at construction -- which is how #39262 and #40155 were found.
``grammar_barrier`` is conditional on the algorithm, so it is left out here.

Signature inspection only: no model, no device, no forward.

    python -m pytest test/registered/spec/test_spec_worker_interface.py -v
"""

import importlib
import inspect
import pkgutil
import unittest

import sglang.srt.speculative as speculative
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# What Scheduler.run_batch hands every worker, regardless of algorithm.
SCHEDULER_KEYWORDS = ("on_publish", "pp_proxy_tensors")


def _workers_defining_the_call():
    """Every BaseSpecWorker subclass that writes its own forward_batch_generation.

    The package is walked rather than listed so a new worker is covered the
    day it lands. Subclasses that inherit the call are covered through the
    class that defines it.
    """
    for module in pkgutil.walk_packages(
        speculative.__path__, speculative.__name__ + "."
    ):
        importlib.import_module(module.name)

    workers = {}
    pending = list(BaseSpecWorker.__subclasses__())
    while pending:
        worker = pending.pop()
        pending.extend(worker.__subclasses__())
        if "forward_batch_generation" in worker.__dict__:
            workers[worker.__name__] = worker
    return workers


def _accepts_keyword(func, name: str) -> bool:
    parameters = inspect.signature(func).parameters
    if any(p.kind is p.VAR_KEYWORD for p in parameters.values()):
        return True
    parameter = parameters.get(name)
    return parameter is not None and parameter.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


class TestSpecWorkerForwardInterface(unittest.TestCase):
    def test_the_base_class_declares_the_call(self):
        self.assertIn(
            "forward_batch_generation",
            BaseSpecWorker.__abstractmethods__,
            "a worker missing the call the scheduler drives it through should "
            "fail at construction, not mid-request",
        )

    def test_the_walk_finds_workers(self):
        self.assertTrue(_workers_defining_the_call(), "no spec worker found to check")

    def test_workers_accept_the_scheduler_keywords(self):
        for name, worker in sorted(_workers_defining_the_call().items()):
            for keyword in SCHEDULER_KEYWORDS:
                with self.subTest(worker=name, keyword=keyword):
                    self.assertTrue(
                        _accepts_keyword(worker.forward_batch_generation, keyword),
                        f"{name}.forward_batch_generation drops {keyword}, which "
                        "Scheduler.run_batch passes to every spec worker",
                    )


if __name__ == "__main__":
    unittest.main()
