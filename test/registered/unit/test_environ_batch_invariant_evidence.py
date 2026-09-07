import sys

import pytest

from sglang.srt.environ import envs, exportable_env_vars
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestBatchInvariantEnvironmentEvidence:
    @pytest.mark.parametrize(
        "raw, expected", [("false", False), ("0", False), ("true", True), ("1", True)]
    )
    def test_the_fallback_setting_is_exported_with_its_actual_value(
        self, monkeypatch: pytest.MonkeyPatch, raw: str, expected: bool
    ) -> None:
        """The worker report must expose whether batch-variant fallback was enabled."""
        name = "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT"
        monkeypatch.setenv(name, raw)
        assert (
            envs.SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT.get() is expected
        )
        assert exportable_env_vars()[name] == raw

    def test_an_unset_fallback_keeps_the_disabled_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Registering the environment flag must preserve its disabled startup default."""
        name = "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT"
        monkeypatch.delenv(name, raising=False)
        assert envs.SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT.get() is False
        assert name not in exportable_env_vars()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
