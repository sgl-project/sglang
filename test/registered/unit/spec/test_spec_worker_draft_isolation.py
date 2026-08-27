"""The scheduler hands every draft worker a ServerArgs copy.

Regression: the v2 spec workers wrote the draft's context_length onto the
instance they share with the target worker, and the scheduler wrote the draft's
load_format onto that same object, so the target's config carried draft values
for the rest of the process. The copy is made once, before the worker factory,
so plugin algorithms registered through SpeculativeAlgorithm.register get it too.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _StopConstruction(Exception):
    """Cuts the draft worker off once its ServerArgs is captured."""


def _scheduler(server_args):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.server_args = server_args
    model_config = SimpleNamespace(context_len=4096)
    scheduler.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(model_config=model_config)
    )
    scheduler.ps = SimpleNamespace(gpu_id=0)
    scheduler.nccl_port = 0
    return scheduler


class TestSchedulerDraftServerArgs(CustomTestCase):
    def _seed(self, **fields):
        override = get_context().override_server_args(
            speculative_algorithm="EAGLE", **fields
        )
        server_args = override.install()
        self.addCleanup(override.restore)
        return server_args

    def _captured_draft_args(self, server_args):
        seen = {}

        def worker_class(**kwargs):
            seen["server_args"] = kwargs["server_args"]
            raise _StopConstruction

        scheduler = _scheduler(server_args)
        scheduler.spec_algorithm = SimpleNamespace(
            is_none=lambda: False,
            is_ngram=lambda: False,
            create_worker=lambda _sa: worker_class,
        )
        with self.assertRaises(_StopConstruction):
            scheduler.maybe_init_draft_worker()
        return seen["server_args"]

    def test_the_draft_gets_a_copy_carrying_the_target_context_length(self):
        server_args = self._seed(context_length=None)
        draft = self._captured_draft_args(server_args)
        self.assertIsNot(draft, server_args)
        self.assertEqual(draft.context_length, 4096)
        self.assertIsNone(server_args.context_length)

    def test_the_draft_config_is_published_while_the_draft_is_built(self):
        from sglang.srt.runtime_context import get_model

        server_args = self._seed(
            load_format="auto", speculative_draft_load_format="dummy"
        )
        seen = {}

        def worker_class(**kwargs):
            seen["published"] = get_model().load_format
            raise _StopConstruction

        scheduler = _scheduler(server_args)
        scheduler.spec_algorithm = SimpleNamespace(
            is_none=lambda: False,
            is_ngram=lambda: False,
            create_worker=lambda _sa: worker_class,
        )
        with self.assertRaises(_StopConstruction):
            scheduler.maybe_init_draft_worker()

        # Model-level weight loading reads the bags, not the instance it was
        # handed, so the draft's config has to be the published one while it
        # builds — and the target's has to be back afterwards.
        self.assertEqual(seen["published"], "dummy")
        self.assertEqual(get_model().load_format, "auto")

    def test_the_worker_factory_sees_the_draft_config(self):
        server_args = self._seed(
            load_format="auto", speculative_draft_load_format="dummy"
        )
        seen = {}

        def create_worker(factory_server_args):
            seen["load_format"] = factory_server_args.load_format
            raise _StopConstruction

        scheduler = _scheduler(server_args)
        scheduler.spec_algorithm = SimpleNamespace(
            is_none=lambda: False,
            is_ngram=lambda: False,
            create_worker=create_worker,
        )
        with self.assertRaises(_StopConstruction):
            scheduler.maybe_init_draft_worker()

        # A registered algorithm may pick its worker class from the config it
        # is handed, so the factory and the worker must see the same one.
        self.assertEqual(seen["load_format"], "dummy")

    def test_a_configured_draft_load_format_never_reaches_the_target(self):
        server_args = self._seed(
            load_format="auto", speculative_draft_load_format="dummy"
        )
        draft = self._captured_draft_args(server_args)
        self.assertEqual(draft.load_format, "dummy")
        self.assertEqual(server_args.load_format, "auto")


if __name__ == "__main__":
    unittest.main()
