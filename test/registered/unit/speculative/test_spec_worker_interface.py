"""The scheduler-facing spec-worker contract: ``forward_batch_generation``.

The non-overlap scheduler path always calls the spec worker's
``forward_batch_generation`` with the ``pp_proxy_tensors`` keyword (None
unless pipeline parallelism is active). The workers used to diverge on this
signature one by one, and each divergence only surfaced as

    TypeError: <Worker>.forward_batch_generation() got an unexpected
    keyword argument 'pp_proxy_tensors'

on the first request through the affected algorithm (issue #40155; the
FrozenKVMTP and UNO instances were still broken on main). The base class now
declares the method, and this test walks every concrete ``BaseSpecWorker``
subclass so the next worker that drops the keyword fails here instead of in
production.
"""

import inspect
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import sglang.srt.speculative.dflash_worker_v2  # noqa: E402,F401
import sglang.srt.speculative.eagle_worker_v2  # noqa: E402,F401
import sglang.srt.speculative.frozen_kv_mtp_worker_v2  # noqa: E402,F401
import sglang.srt.speculative.multi_layer_eagle_worker_v2  # noqa: E402,F401
import sglang.srt.speculative.ngram_worker  # noqa: E402,F401
import sglang.srt.speculative.uno_worker_v2  # noqa: E402,F401
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker  # noqa: E402
from sglang.srt.speculative.dspark_components import dspark_worker_v2  # noqa: E402,F401

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _iter_subclasses(cls):
    for sub in cls.__subclasses__():
        yield sub
        yield from _iter_subclasses(sub)


def _concrete_spec_workers():
    for cls in _iter_subclasses(BaseSpecWorker):
        if not inspect.isabstract(cls):
            yield cls


class TestSpecWorkerForwardBatchGenerationSignature(CustomTestCase):
    def test_base_class_declares_scheduler_facing_signature(self):
        """BaseSpecWorker must keep declaring the method abstractly so a new
        worker that forgets to implement it fails at instantiation."""
        method = BaseSpecWorker.forward_batch_generation
        self.assertTrue(
            getattr(method, "__isabstractmethod__", False),
            "BaseSpecWorker.forward_batch_generation must stay @abstractmethod",
        )
        params = inspect.signature(method).parameters
        self.assertIn("batch", params)
        self.assertIn("pp_proxy_tensors", params)

    def test_every_concrete_worker_accepts_pp_proxy_tensors(self):
        """The non-overlap scheduler calls every worker through
        ``forward_batch_generation(batch, pp_proxy_tensors=...)``, so any
        concrete worker missing the keyword dies with a TypeError on its
        first request (issue #40155)."""
        workers = list(_concrete_spec_workers())
        self.assertGreater(
            len(workers), 0, "expected the worker imports above to register classes"
        )
        for cls in workers:
            with self.subTest(worker=cls.__name__):
                method = cls.forward_batch_generation
                self.assertFalse(
                    getattr(method, "__isabstractmethod__", False),
                    "concrete spec worker must not stay abstract",
                )
                params = inspect.signature(method).parameters
                self.assertIn(
                    "pp_proxy_tensors",
                    params,
                    "scheduler always passes pp_proxy_tensors (None without PP)",
                )
                self.assertIn(
                    params["pp_proxy_tensors"].kind,
                    (
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.KEYWORD_ONLY,
                    ),
                    "pp_proxy_tensors must be passable as a keyword",
                )

    def test_known_workers_are_covered_by_the_walk(self):
        """Guard against the walk silently missing modules: every scheduler-
        facing worker family must appear among the enumerated subclasses."""
        names = {cls.__name__ for cls in _concrete_spec_workers()}
        for known in (
            "EAGLEWorkerV2",
            "DSparkWorkerV2",
            "FrozenKVMTPWorkerV2",
            "NGRAMWorker",
            "UnoWorkerV2",
            "MultiLayerEagleWorkerV2",
            "DFlashWorkerV2",
        ):
            self.assertIn(known, names)


if __name__ == "__main__":
    unittest.main()
