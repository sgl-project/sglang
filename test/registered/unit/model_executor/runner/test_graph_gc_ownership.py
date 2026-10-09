"""Graph capture must not unfreeze the permanent generation owned by serving."""

import subprocess
import sys
import textwrap
import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=90, suite="base-a-test-cpu")


class TestGraphGCOwnership(unittest.TestCase):
    def probe(self, body):
        # GC's permanent generation is process-wide; keep each case isolated.
        setup = """
import gc
import weakref
from sglang.srt.model_executor.runner.base_cuda_graph_runner import freeze_gc
from sglang.srt.utils.common import freeze_gc as freeze_serving_gc

class Cycle:
    def __init__(self):
        self.cycle = self

def new_cycle():
    return weakref.ref(Cycle())

gc.unfreeze()
gc.collect()
"""
        result = subprocess.run(
            [sys.executable, "-c", setup + textwrap.dedent(body)],
            capture_output=True,
            check=False,
            text=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_temporary_freeze_is_removed_and_new_cycles_are_collected(self):
        self.probe("""
            old = Cycle()
            reference = weakref.ref(old)
            with freeze_gc(False):
                del old
                created = new_cycle()
            gc.collect()
            assert reference() is None, 'temporary freeze was not released'
            assert created() is None
            """)

    def test_existing_permanent_generation_survives_both_capture_modes(self):
        self.probe("""
            for enabled in (False, True):
                old = new_cycle()
                freeze_serving_gc('test')
                with freeze_gc(enabled):
                    created = new_cycle()
                gc.collect()
                assert old() is not None, 'preexisting frozen cycle was collected'
                assert created() is None, 'new request cycles became permanent'
                gc.unfreeze()
                gc.collect()
                assert old() is None
            """)

    def test_nested_capture_does_not_revoke_outer_freeze(self):
        self.probe("""
            old = Cycle()
            reference = weakref.ref(old)
            with freeze_gc(False):
                del old
                with freeze_gc(False):
                    pass
                assert reference() is not None, 'inner context revoked outer freeze'
            gc.collect()
            assert reference() is None
            """)

    def test_exception_preserves_caller_freeze_and_releases_owned_freeze(self):
        self.probe("""
            for permanent in (False, True):
                old = Cycle()
                reference = weakref.ref(old)
                if permanent:
                    freeze_serving_gc('test')
                try:
                    with freeze_gc(False):
                        del old
                        created = new_cycle()
                        raise RuntimeError('capture failed')
                except RuntimeError:
                    pass
                gc.collect()
                assert (reference() is not None) == permanent
                assert created() is None
                gc.unfreeze()
                gc.collect()
                assert reference() is None
            """)

    def test_api_can_make_an_active_graph_freeze_permanent(self):
        self.probe("""
            old = Cycle()
            reference = weakref.ref(old)
            with freeze_gc(False):
                del old
                freeze_serving_gc('test')
                created = new_cycle()
            gc.collect()
            assert reference() is not None
            assert created() is None
            gc.unfreeze()
            gc.collect()
            assert reference() is None
            """)

    def test_overlapping_threads_release_only_the_last_graph_scope(self):
        self.probe("""
            import threading
            inside, release, exited = (threading.Event() for _ in range(3))
            old = Cycle()
            reference = weakref.ref(old)
            def worker():
                try:
                    with freeze_gc(False):
                        inside.set()
                        assert release.wait(30)
                finally:
                    exited.set()
            thread = threading.Thread(target=worker)
            thread.start()
            try:
                assert inside.wait(30)
                del old
                with freeze_gc(False):
                    release.set()
                    assert exited.wait(30)
                    assert reference() is not None
            finally:
                release.set()
                thread.join(30)
            assert not thread.is_alive()
            gc.collect()
            assert reference() is None
            """)


if __name__ == "__main__":
    unittest.main()
