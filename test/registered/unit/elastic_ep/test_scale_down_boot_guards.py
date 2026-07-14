"""Unit tests: boot-time guards for the Mooncake-native scale-down path.

Validates that ``ServerArgs._handle_elastic_ep`` refuses to start when the
raw-NCCL fast-paths that bypass Mooncake's active_ranks mask are enabled,
and that the shm-broadcaster is auto-disabled with a warning.
"""

from __future__ import annotations

import unittest
from unittest import mock

from sglang.srt import environ as env_mod


class TestElasticMooncakeBootGuards(unittest.TestCase):
    """Guard rails enforced by ``ServerArgs._handle_elastic_ep`` under
    ``--elastic-ep-backend mooncake``. These live in the boot path so a
    user can't silently deploy a config that would deadlock on shrink."""

    def _make_args(self, **overrides):
        """Instantiate a minimal ServerArgs with elastic-mooncake enabled."""
        from sglang.srt.server_args import ServerArgs

        # ServerArgs has ~200 fields; use the class default and only override
        # the elastic-relevant ones. We bypass __post_init__ to avoid pulling
        # in the full server bring-up path -- we just need _handle_elastic_ep.
        args = ServerArgs.__new__(ServerArgs)
        for field_name, field in ServerArgs.__dataclass_fields__.items():
            default = field.default
            # dataclasses.MISSING is the sentinel; dataclass fields without a
            # default sit in the class dict already, so skip them.
            if default is not None and default.__class__.__name__ == "_MISSING_TYPE":
                continue
            setattr(args, field_name, default)
        args.elastic_ep_backend = "mooncake"
        args.elastic_ep_rejoin = False
        args.enable_eplb = False
        args.pp_size = 1
        args.mooncake_ib_device = None
        args.ep_join_mode = None
        args.ep_join_rank_offset = 0
        args.max_ep_size = None
        args.elastic_ep_initial_size = None
        args.enable_symm_mem = False
        args.enable_elastic_expert_backup = False
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def test_symm_mem_rejected_under_mooncake(self):
        from sglang.srt.server_args import ServerArgs

        args = self._make_args(enable_symm_mem=True)
        with mock.patch.object(
            ServerArgs,
            "_validate_ib_devices",
            side_effect=lambda x: x,
        ):
            with self.assertRaises(ValueError) as ctx:
                ServerArgs._handle_elastic_ep(args)
        self.assertIn("--enable-symm-mem is incompatible", str(ctx.exception))

    def test_sync_token_ids_rejected_under_mooncake(self):
        from sglang.srt.server_args import ServerArgs

        args = self._make_args()
        with mock.patch.object(
            ServerArgs,
            "_validate_ib_devices",
            side_effect=lambda x: x,
        ), mock.patch.object(
            env_mod.envs.SGLANG_SYNC_TOKEN_IDS_ACROSS_TP,
            "get",
            return_value=True,
        ):
            with self.assertRaises(ValueError) as ctx:
                ServerArgs._handle_elastic_ep(args)
        self.assertIn("SGLANG_SYNC_TOKEN_IDS_ACROSS_TP", str(ctx.exception))

    def test_shm_broadcaster_force_disabled(self):
        from sglang.srt.server_args import ServerArgs

        args = self._make_args()
        called = []

        original_get = env_mod.envs.SGLANG_USE_MESSAGE_QUEUE_BROADCASTER.get
        original_set = env_mod.envs.SGLANG_USE_MESSAGE_QUEUE_BROADCASTER.set

        def _fake_get():
            return True

        def _fake_set(value):
            called.append(value)

        with mock.patch.object(
            ServerArgs,
            "_validate_ib_devices",
            side_effect=lambda x: x,
        ), mock.patch.object(
            env_mod.envs.SGLANG_USE_MESSAGE_QUEUE_BROADCASTER,
            "get",
            side_effect=_fake_get,
        ), mock.patch.object(
            env_mod.envs.SGLANG_USE_MESSAGE_QUEUE_BROADCASTER,
            "set",
            side_effect=_fake_set,
        ):
            ServerArgs._handle_elastic_ep(args)

        self.assertEqual(called, [False])

    def test_nixl_backend_not_gated(self):
        """The guards only apply to ``mooncake`` -- NIXL / null backends must
        not trigger them (they use the append-only scale-up path where
        raw NCCL fast-paths are safe)."""
        from sglang.srt.server_args import ServerArgs

        args = self._make_args(elastic_ep_backend="nixl", enable_symm_mem=True)
        # nixl backend still needs an IB device validation pass; stub it.
        with mock.patch.object(
            ServerArgs, "_validate_ib_devices", side_effect=lambda x: x
        ):
            try:
                ServerArgs._handle_elastic_ep(args)
            except ValueError as exc:
                self.fail(f"NIXL backend should not raise boot guards: {exc}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
