"""Unit tests for `entrypoints/prespawn.py` -- eligibility and the hand-over
protocol between `maybe_prespawn()` and `Engine._launch_subprocesses`.

Spawning itself needs a model and GPUs; what is tested here is the bookkeeping
around it: which configurations stay on the normal launch path, that
pre-spawned workers are adopted exactly once and only by the record they were
spawned from, and that unwanted workers are stopped.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import os
import unittest
from unittest import mock

from sglang.srt.entrypoints import prespawn
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import CustomTestCase


class _FakeProc:
    def __init__(self):
        self.terminated = False

    def terminate(self):
        self.terminated = True


def _prespawned(server_args, procs):
    return prespawn.Prespawned(
        server_args=server_args, port_args=object(), result=object(), procs=procs
    )


class TestEligibility(CustomTestCase):
    def test_default_record_is_eligible(self):
        self.assertTrue(prespawn.eligible(ServerArgs(model_path="dummy")))

    def test_weight_cache_daemons_take_the_normal_path(self):
        args = ServerArgs(model_path="dummy", weight_cache_mode="daemon")
        self.assertFalse(prespawn.eligible(args))

    def test_bootstrap_server_takes_the_normal_path(self):
        args = ServerArgs(
            model_path="dummy",
            remote_instance_weight_loader_start_seed_via_transfer_engine=True,
        )
        self.assertFalse(prespawn.eligible(args))

    def test_ray_takes_the_normal_path(self):
        self.assertFalse(
            prespawn.eligible(ServerArgs(model_path="dummy", use_ray=True))
        )


class TestHandOver(CustomTestCase):
    def setUp(self):
        self._saved = prespawn._PRESPAWNED
        prespawn._PRESPAWNED = None
        self.addCleanup(self._restore)

    def _restore(self):
        prespawn._PRESPAWNED = self._saved

    def test_nothing_prespawned(self):
        self.assertIsNone(prespawn.take(ServerArgs(model_path="dummy")))

    def test_only_the_spawning_record_may_adopt(self):
        mine = ServerArgs(model_path="dummy")
        other = ServerArgs(model_path="dummy")
        prespawn._PRESPAWNED = _prespawned(mine, procs=[_FakeProc()])
        pre = prespawn._PRESPAWNED
        self.assertIsNone(prespawn.take(other))
        self.assertIs(prespawn._PRESPAWNED, pre, "a refused take() keeps the workers")
        self.assertIs(prespawn.take(mine), pre)

    def test_adopted_once(self):
        mine = ServerArgs(model_path="dummy")
        pre = _prespawned(mine, procs=[_FakeProc()])
        prespawn._PRESPAWNED = pre
        self.assertIs(prespawn.take(mine), pre)
        self.assertIsNone(prespawn.take(mine))

    def test_abandon_stops_the_workers(self):
        procs = [_FakeProc(), _FakeProc()]
        prespawn.abandon(_prespawned(ServerArgs(model_path="dummy"), procs=procs))
        self.assertTrue(all(p.terminated for p in procs))

    def test_disabled_is_a_noop(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SGLANG_PRESPAWN_WORKERS", None)
            prespawn.maybe_prespawn(ServerArgs(model_path="dummy"))
        self.assertIsNone(prespawn._PRESPAWNED)


if __name__ == "__main__":
    unittest.main()
