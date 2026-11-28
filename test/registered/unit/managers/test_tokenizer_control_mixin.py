"""Unit tests for TokenizerControlMixin request normalization."""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic import TypeAdapter, ValidationError

from sglang.srt.managers import tokenizer_control_mixin
from sglang.srt.managers.io_struct import (
    FlushCacheReqOutput,
    ProfileReq,
    ProfileReqOutput,
    ProfileReqType,
)
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin
from sglang.srt.platforms.interface import SRTPlatform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestFlushCache(CustomTestCase):
    def test_busy_dp_replica_prevents_success(self):
        manager = MagicMock()
        busy = FlushCacheReqOutput(success=False, message="busy")
        manager.flush_cache_communicator = AsyncMock(
            return_value=[FlushCacheReqOutput(success=True), busy]
        )

        self.assertIs(asyncio.run(TokenizerControlMixin.flush_cache(manager)), busy)
        manager.mm_processor.clear_preprocess_cache.assert_not_called()


class _QuietProfilePlatform(SRTPlatform):
    def apply_profile_request_defaults(self, req: ProfileReq) -> None:
        if req.with_stack is None:
            req.with_stack = False
        if req.record_shapes is None:
            req.record_shapes = False
        if req.num_steps is None:
            req.num_steps = 2


class TestStartProfilePlatformDefaults(CustomTestCase):
    def _start_profile(self, req):
        manager = MagicMock()
        manager._execute_profile = AsyncMock(
            return_value=ProfileReqOutput(success=True, message="")
        )
        with (
            patch.object(
                tokenizer_control_mixin, "current_platform", _QuietProfilePlatform()
            ),
            patch.dict(
                "os.environ",
                {
                    "SGLANG_PROFILE_WITH_STACK": "true",
                    "SGLANG_PROFILE_RECORD_SHAPES": "true",
                },
            ),
        ):
            asyncio.run(TokenizerControlMixin.start_profile(manager, req))
        return manager._execute_profile.call_args.args[0]

    def test_platform_sees_unset_fields_before_env_defaults(self):
        """Platform defaults apply only to fields the caller left unset;
        the env defaults must not fill them first."""
        for req in (None, ProfileReq()):
            with self.subTest(req=req):
                sent = self._start_profile(req)
                self.assertEqual(sent.req_type, ProfileReqType.START_PROFILE)
                self.assertFalse(sent.with_stack)
                self.assertFalse(sent.record_shapes)
                self.assertEqual(sent.num_steps, 2)


class TestProfileReqFilenameComponents(CustomTestCase):
    def test_path_separators_are_rejected(self):
        """profile_id/profile_prefix are joined into the trace filename under
        output_dir; '../x' or an absolute value wrote the trace outside it
        (#44189). The HTTP body goes through the pydantic schema, which must
        surface the same error as a validation failure (400), not a 500."""
        adapter = TypeAdapter(ProfileReq)
        for field in ("profile_id", "profile_prefix"):
            for value in ("../escaped", "/tmp/elsewhere", "a\\..\\b"):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        ProfileReq(**{field: value})
                    with self.assertRaises(ValidationError):
                        adapter.validate_python({field: value})
        req = ProfileReq(profile_id="1763.5", profile_prefix="..run-1")
        self.assertEqual(req.profile_prefix, "..run-1")


if __name__ == "__main__":
    unittest.main()
