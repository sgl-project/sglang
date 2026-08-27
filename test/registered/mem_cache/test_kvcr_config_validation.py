"""The KVCR backend must refuse configs whose failure mode is silent.

Two independent rules, both about values that do not break startup.

A worker with ``control_port = 0`` or a wildcard advertise host comes up,
offloads, gets indexed by the router and receives hints -- every fetch just
fails to reach it. That presents as "P2P does not work on this branch", which is
expensive to chase and has nothing to do with the transfer path.

``get_timeout_s <= operation_timeout_ms`` is worse: it does not fail, it
corrupts. See :class:`TimeoutOrderingValidationTest`.

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

    def test_missing_advertise_host_is_refused_even_when_the_bind_host_is_routable(
        self,
    ):
        """``control_host`` is where to bind, not what to tell peers.

        Falling back to it happens to work when an operator binds one routable
        address, and silently advertises the wrong interface as soon as they
        bind more than one. Requiring the advertise host makes the two
        independent, and keeps this in step with the dynamo bridge, which
        advertises the same field.
        """
        for host in (None, ""):
            with self.subTest(control_advertise_host=host):
                with self.assertRaises(ValueError) as caught:
                    KVCRBackendConfig(
                        **{
                            **_DIALABLE,
                            "control_host": "10.0.0.7",
                            "control_advertise_host": host,
                        }
                    )
                self.assertIn("control_advertise_host", str(caught.exception))

    def test_wildcard_bind_with_explicit_advertise_host_is_allowed(self):
        """Binding every interface is normal -- only advertising one is not.

        This is the documented escape hatch, so a check that rejected the
        wildcard *bind* would break a legitimate multi-NIC deployment while
        still passing the two tests above.
        """
        config = KVCRBackendConfig(
            **{
                **_DIALABLE,
                "control_host": "0.0.0.0",
                "control_advertise_host": "10.0.0.7",
            }
        )
        self.assertEqual(config.control_advertise_host, "10.0.0.7")

    def test_local_only_keeps_the_ephemeral_default(self):
        """Nothing dials a local-only worker, and an OS port cannot collide.

        The negative branch: without this, tightening the rule into an
        unconditional one would fail every local-only launch, including the
        default construction that ``from_extra_config`` returns for an empty
        blob.
        """
        self.assertEqual(KVCRBackendConfig().control_port, 0)
        self.assertEqual(KVCRBackendConfig.from_extra_config(None).control_port, 0)
        self.assertEqual(KVCRBackendConfig.from_extra_config({}).control_port, 0)

    def test_validation_runs_on_the_extra_config_path(self):
        """``from_extra_config`` is the only way this struct is built in prod.

        It goes through ``msgspec.convert``, not ``__init__``. If that stopped
        invoking ``__post_init__``, every test above would still pass while the
        check was dead on the path that matters.
        """
        with self.assertRaises(ValueError):
            KVCRBackendConfig.from_extra_config(
                {**_DIALABLE, "control_port": 0, "unknown_key": "ignored"}
            )


class TimeoutOrderingValidationTest(unittest.TestCase):
    """``get_timeout_s`` must outlast ``operation_timeout_ms``.

    ``_drain_until`` abandons an op at ``get_timeout_s`` and cannot cancel it:
    ``kvcr.abort()`` is a no-op stub and NIXL's cancellation releases the
    transfer handle without fencing an in-flight DMA. HiCache then frees that
    op's host pages and hands them to the next prefetch. Only the core giving up
    first -- both ends anchor to ``operation_timeout_ms`` -- keeps that safe.

    Invert the order and an abandoned transfer writes into pages another request
    owns. Nothing downstream can catch it: KVCR block keys are token hashes with
    no content check, so it surfaces as wrong generated text, not an error.
    """

    def test_get_timeout_below_operation_timeout_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            KVCRBackendConfig(operation_timeout_ms=30000, get_timeout_s=10.0)
        self.assertIn("get_timeout_s", str(caught.exception))

    def test_equal_timeouts_are_refused(self):
        """A tie is not safe: the core's deadline and ours would race."""
        with self.assertRaises(ValueError):
            KVCRBackendConfig(operation_timeout_ms=20000, get_timeout_s=20.0)

    def test_the_rule_applies_to_local_only_configs(self):
        """The pages are reused the same way whether or not a peer is involved.

        A deposit hands the core host pages as transfer *sources*, so abandoning
        one early lets the core read out of pages HiCache has already reused.
        Scoping this check to ``enable_remote_hint`` -- as the endpoint checks
        above are -- would leave that open.
        """
        with self.assertRaises(ValueError):
            KVCRBackendConfig(
                enable_remote_hint=False,
                operation_timeout_ms=30000,
                get_timeout_s=10.0,
            )

    def test_the_shipped_defaults_satisfy_the_rule(self):
        """The negative branch: a stricter rule would fail every launch."""
        config = KVCRBackendConfig()
        self.assertGreater(config.get_timeout_s * 1000.0, config.operation_timeout_ms)


if __name__ == "__main__":
    unittest.main()
