import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestMooncakeStateTransferStatus(unittest.TestCase):
    """A failed extra-state transfer must fail the whole request.

    `_send_last_chunk_extras` sends the trailing state (DSA/SWA/Mamba/MiniMax)
    and then the aux metadata. Overall success must be the conjunction of both,
    so a successful aux transfer cannot mask a failed state transfer and let
    decode commit stale state pages.
    """

    def _manager(self, state_ret, aux_ret):
        mgr = MooncakeKVManager.__new__(MooncakeKVManager)
        mgr.maybe_send_extra = Mock(return_value=state_ret)
        mgr.send_aux = Mock(return_value=aux_ret)
        return mgr

    def _call(self, mgr, skip_state=False, state_indices=(0,)):
        req = object()
        executor = object()
        kv_chunk = SimpleNamespace(
            state_indices=list(state_indices),
            room=7,
            prefill_aux_index=0,
        )
        target = SimpleNamespace(dst_aux_ptrs=[1])
        return mgr._send_last_chunk_extras(req, kv_chunk, skip_state, executor, target)

    def test_state_success_aux_success_is_success(self):
        mgr = self._manager(state_ret=0, aux_ret=0)
        self.assertEqual(self._call(mgr), 0)
        mgr.maybe_send_extra.assert_called_once()
        mgr.send_aux.assert_called_once()

    def test_state_failure_aux_success_is_failure(self):
        mgr = self._manager(state_ret=1, aux_ret=0)
        self.assertNotEqual(self._call(mgr), 0)
        mgr.maybe_send_extra.assert_called_once()
        # Aux transfer is skipped once state has failed.
        mgr.send_aux.assert_not_called()

    def test_state_success_aux_failure_is_failure(self):
        mgr = self._manager(state_ret=0, aux_ret=1)
        self.assertNotEqual(self._call(mgr), 0)
        mgr.send_aux.assert_called_once()

    def test_no_state_indices_uses_aux_only(self):
        mgr = self._manager(state_ret=1, aux_ret=0)
        self.assertEqual(self._call(mgr, state_indices=()), 0)
        mgr.maybe_send_extra.assert_not_called()
        mgr.send_aux.assert_called_once()

    def test_skip_state_uses_aux_only(self):
        mgr = self._manager(state_ret=1, aux_ret=0)
        self.assertEqual(self._call(mgr, skip_state=True), 0)
        mgr.maybe_send_extra.assert_not_called()
        mgr.send_aux.assert_called_once()


if __name__ == "__main__":
    unittest.main()
