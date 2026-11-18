"""When the shared Mooncake transfer engine is required vs. only reused by HiCache storage."""

import unittest
from types import SimpleNamespace
from typing import Optional
from unittest.mock import patch

from sglang.srt.distributed.device_communicators import mooncake_transfer_engine as mte
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _start(
    *,
    pd_backend: Optional[str] = None,
    hicache_backend: Optional[str] = None,
    init_error: Optional[Exception] = None,
) -> int:
    """Run maybe_init_shared_mooncake_transfer_engine; return how often it tried to start the engine."""
    disagg = SimpleNamespace(
        disaggregation_mode="prefill" if pd_backend else "null",
        disaggregation_transfer_backend=pd_backend or "mooncake",
        disaggregation_ib_device=None,
        encoder_only=False,
        language_only=False,
        encoder_transfer_backend="mooncake",
    )
    memory = SimpleNamespace(
        enable_hierarchical_cache=hicache_backend is not None,
        hicache_storage_backend=hicache_backend,
    )
    exec_ = SimpleNamespace(
        moe=SimpleNamespace(
            enable_elastic_expert_backup=False,
            elastic_ep_backend=None,
            mooncake_ib_device=None,
        )
    )
    attempts = []

    def fake_init(**kwargs):
        attempts.append(kwargs)
        if init_error is not None:
            raise init_error

    with (
        envs.SGLANG_HICACHE_MOONCAKE_REUSE_TE.override(True),
        patch.object(mte, "get_disagg", return_value=disagg),
        patch.object(mte, "get_memory", return_value=memory),
        patch.object(mte, "get_exec", return_value=exec_),
        patch.object(mte, "get_local_ip_auto", return_value="10.0.0.1"),
        patch.object(mte, "init_mooncake_transfer_engine", side_effect=fake_init),
    ):
        mte.maybe_init_shared_mooncake_transfer_engine(gpu_id=0)
    return len(attempts)


class TestSharedMooncakeEngineInit(CustomTestCase):
    def test_hicache_storage_survives_a_shared_engine_that_cannot_start(self):
        """On DGX Spark the engine cannot open its RoCE NICs (no GPUDirect), even for TCP;
        MooncakeStore starts its own engine, so startup must not abort."""
        with self.assertLogs(mte.logger, level="WARNING") as logs:
            attempts = _start(
                hicache_backend="mooncake",
                init_error=RuntimeError(
                    "Mooncake Transfer Engine initialization failed."
                ),
            )
        self.assertEqual(attempts, 1)
        self.assertIn("will start its own", "\n".join(logs.output))

    def test_pd_disaggregation_still_fails_when_the_engine_cannot_start(self):
        with self.assertRaises(RuntimeError):
            _start(
                pd_backend="mooncake",
                hicache_backend="mooncake",
                init_error=RuntimeError(
                    "Mooncake Transfer Engine initialization failed."
                ),
            )

    def test_nothing_starts_without_a_mooncake_user(self):
        self.assertEqual(_start(hicache_backend="file"), 0)


if __name__ == "__main__":
    unittest.main()
