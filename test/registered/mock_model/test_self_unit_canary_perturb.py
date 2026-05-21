import logging
from unittest.mock import Mock

import pytest
from _pytest.logging import LogCaptureFixture

from sglang.srt.kv_canary.perturb import real_kv_used
from sglang.srt.kv_canary.perturb.config import PerturbConfig
from sglang.srt.kv_canary.perturb.slot_picker import ActiveSlotTarget
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_real_kv_used_logs_when_target_group_has_no_real_kv_sources(
    monkeypatch: pytest.MonkeyPatch, caplog: LogCaptureFixture
) -> None:
    config = PerturbConfig(
        req_to_token_prob=0.0,
        real_kv_used_prob=1.0,
        real_kv_unused_cache_prob=0.0,
        target_group_kind="full",
        warmup_steps=0,
    )
    warmup_gate = Mock()
    warmup_gate.is_in_warmup.return_value = False

    monkeypatch.setattr(
        real_kv_used,
        "pick_active_slot",
        lambda **_: ActiveSlotTarget(req_pool_idx=0, position=0, slot=7),
    )

    with caplog.at_level(logging.INFO, logger=real_kv_used.logger.name):
        real_kv_used.run(
            forward_batch=Mock(),
            config=config,
            req_to_token_pool=Mock(),
            buffer_groups=(),
            warmup_gate=warmup_gate,
        )

    assert (
        "kv_canary perturb real_kv_used: skipped because no target group with "
        "real_kv_sources_k matched target_group_kind=full slot=7"
    ) in caplog.text
