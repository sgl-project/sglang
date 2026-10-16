from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.models import utils
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _exec_context(rl_on_policy_target):
    return SimpleNamespace(
        deterministic=SimpleNamespace(
            rl_on_policy_target=rl_on_policy_target,
        )
    )


def test_rl_on_policy_target_falls_back_before_kv_pool_lookup(monkeypatch):
    monkeypatch.setattr(utils, "get_exec", lambda: _exec_context("fsdp"))
    pool_lookup = Mock()
    monkeypatch.setattr(utils, "get_token_to_kv_pool", pool_lookup)

    assert not utils.enable_fused_set_kv_buffer(SimpleNamespace())
    pool_lookup.assert_not_called()


def test_standard_cuda_bf16_path_stays_fused(monkeypatch):
    monkeypatch.setattr(utils, "get_exec", lambda: _exec_context(None))
    monkeypatch.setattr(utils, "_is_cuda", True)
    monkeypatch.setattr(utils, "_is_hip", False)
    monkeypatch.setattr(utils, "is_cp_active", lambda forward_batch: False)
    monkeypatch.setattr(
        utils,
        "get_token_to_kv_pool",
        lambda: SimpleNamespace(dtype=torch.bfloat16),
    )

    forward_batch = SimpleNamespace(dcp_kv_mask=None)
    assert utils.enable_fused_set_kv_buffer(forward_batch)
