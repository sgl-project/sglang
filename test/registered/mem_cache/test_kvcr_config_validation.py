"""The KVCR backend must refuse configs whose failure mode is silent.

A worker with ``control_port = 0`` or a wildcard advertise host comes up,
offloads, gets indexed by the router and receives hints -- every fetch just
fails to reach it, which presents as "P2P does not work on this branch".

``get_timeout_s <= operation_timeout_ms`` is worse: it corrupts. See
:class:`TimeoutOrderingValidationTest`.

Needs no ``kvcr`` wheel: ``KVCRBackendConfig`` is a plain msgspec struct.

    python -m pytest test/registered/mem_cache/test_kvcr_config_validation.py -v
"""

from __future__ import annotations

import unittest

from sglang.srt.mem_cache.storage.kvcr.kvcr_config import KVCRBackendConfig
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# What the README's two-worker launch passes, minus whichever field is under
# test. Kept here so a test that expects a *rejection* cannot pass because some
# unrelated field was also invalid.
_DIALABLE = {
    "enable_remote_hint": True,
    "control_host": "127.0.0.1",
    "control_port": 25000,
    "control_advertise_host": "127.0.0.1",
}


class RemoteHintEndpointValidationTest(unittest.TestCase):
    def test_ephemeral_port_with_remote_hint_is_refused(self):
        """Port 0 is known only inside the scheduler, so nothing can register it."""
        with self.assertRaises(ValueError) as caught:
            KVCRBackendConfig(**{**_DIALABLE, "control_port": 0})
        self.assertIn("control_port", str(caught.exception))

    def test_wildcard_advertise_host_with_remote_hint_is_refused(self):
        """A bind wildcard is not an address; a peer handed it dials itself."""
        for host in ("0.0.0.0", "::"):
            with self.subTest(host=host):
                with self.assertRaises(ValueError) as caught:
                    KVCRBackendConfig(**{**_DIALABLE, "control_advertise_host": host})
                self.assertIn("control_advertise_host", str(caught.exception))


class TimeoutOrderingValidationTest(unittest.TestCase):
    """``get_timeout_s`` must outlast ``operation_timeout_ms``.

    ``_drain_until`` abandons an op at ``get_timeout_s`` and cannot cancel it:
    ``kvcr.abort()`` is a no-op stub and NIXL's cancellation releases the transfer
    handle without fencing an in-flight DMA. HiCache then frees that op's host pages
    and hands them to the next prefetch. Only the core giving up first keeps that
    safe. Invert the order and an abandoned transfer writes into pages another
    request owns -- KVCR block keys are token hashes with no content check, so it
    surfaces as wrong generated text, not an error.
    """

    def test_get_timeout_below_operation_timeout_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            KVCRBackendConfig(operation_timeout_ms=30000, get_timeout_s=10.0)
        self.assertIn("get_timeout_s", str(caught.exception))

    def test_the_shipped_defaults_satisfy_the_rule(self):
        """The negative branch: a stricter rule would fail every launch."""
        config = KVCRBackendConfig()
        self.assertGreater(config.get_timeout_s * 1000.0, config.operation_timeout_ms)


if __name__ == "__main__":
    unittest.main()
