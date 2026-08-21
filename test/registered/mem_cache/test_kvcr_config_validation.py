"""The KVCR backend must refuse a remote-hint config no peer can dial.

Both offending values are *defaults*, and neither breaks startup: a worker with
``control_port = 0`` or a wildcard advertise host comes up, offloads, gets
indexed by the router and receives hints -- every fetch just fails to reach it.
That presents as "P2P does not work on this branch", which is expensive to chase
and has nothing to do with the transfer path. These tests pin the refusal.

Needs no ``kvcr`` wheel: ``KVCRBackendConfig`` is a plain msgspec struct.

    python -m pytest test/registered/mem_cache/test_kvcr_config_validation.py -v
"""

from __future__ import annotations

import unittest

from sglang.srt.mem_cache.storage.kvcr.kvcr_config import KVCRBackendConfig

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


if __name__ == "__main__":
    unittest.main()
