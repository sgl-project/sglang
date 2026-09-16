"""Verify-only replay buffers are unnecessary on a PD prefill target."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang.srt.mem_cache import kv_cache_configurator as mod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize("role", ["prefill", "decode", "null"])
@pytest.mark.parametrize("enable_replay", [False, True])
def test_prefill_pool_does_not_allocate_speculative_replay(
    monkeypatch, role, enable_replay
):
    monkeypatch.setattr(
        mod, "get_disagg", lambda: SimpleNamespace(disaggregation_mode=role)
    )
    monkeypatch.setattr(
        mod,
        "get_spec",
        lambda: SimpleNamespace(
            speculative_algorithm="DFLASH", speculative_eagle_topk=1
        ),
    )
    monkeypatch.setattr(mod, "max_speculative_num_draft_tokens", lambda: 8)
    monkeypatch.setattr(
        mod,
        "get_schedule",
        lambda: SimpleNamespace(
            max_mamba_cache_size=256, disable_overlap_schedule=False
        ),
    )
    monkeypatch.setattr(
        mod, "get_memory", lambda: SimpleNamespace(enable_page_major_kv_layout=False)
    )
    monkeypatch.setattr(
        mod,
        "get_exec",
        lambda: SimpleNamespace(
            features=SimpleNamespace(enable_memory_saver=False),
            mamba=SimpleNamespace(
                enable_linear_replayssm_spec=enable_replay,
                enable_linear_replayssm=False,
                linear_replayssm_cache_len=16,
                enable_mamba_extra_buffer=True,
                enable_mamba_extra_buffer_lazy=False,
            ),
        ),
    )
    monkeypatch.setattr(mod, "kimi_linear_config", lambda model: object())
    factory = Mock()
    monkeypatch.setattr(mod, "HybridReqToTokenPool", factory)
    kvc = SimpleNamespace(
        model_config=SimpleNamespace(context_len=4096),
        device="cpu",
        mambaish_config=SimpleNamespace(mamba2_cache_params=object()),
        layer_info=SimpleNamespace(start_layer=0),
        hybrid_gdn_config=None,
        _get_mamba_layer_ids_for_req_pool=lambda: [0],
        _get_ple_req_pool_kwargs=lambda: {},
    )
    mod.KVCacheConfigurator._build_hybrid_req_pool(
        kvc, max_num_reqs=64, extra_max_context_len=4
    )
    kwargs = factory.call_args.kwargs
    assert kwargs["enable_linear_replayssm_spec"] == (
        enable_replay and role != "prefill"
    )
    assert kwargs["speculative_num_draft_tokens"] == (None if role == "prefill" else 8)
    # Persistent state remains available for chunked prefill and PD handoff.
    assert kwargs["mamba_size"] == 256
    assert kwargs["mamba_layer_ids"] == [0]
