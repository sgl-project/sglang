"""MegaMoE small- and large-input switches must pick one path on every EP rank.

Exercises the real host dispatch in ``mega_moe_flydsl`` with CPU tensors and a
fake AITER operator that records which path each call takes. GPU collectives
and the kernels themselves are out of scope.
"""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import sglang.srt.layers.moe.mega_moe_flydsl as mega
from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

MTPR = 8192
SMALL_M_MTPR = 512
EXPERTS = 384
TOPK = 6
HIDDEN = 8
LAYERS = 4


def _layer():
    return SimpleNamespace(
        num_experts=EXPERTS,
        should_fuse_routed_scaling_factor_in_topk=True,
        uses_expert_location=False,
        **{
            name: torch.zeros((48, 4), dtype=torch.uint8)
            for name in ("_mega_w1", "_mega_w1_scale", "_mega_w2", "_mega_w2_scale")
        },
    )


@pytest.fixture
def runtime(monkeypatch):
    state = SimpleNamespace(
        calls=[], large_m_on=None, built={}, small_m_mtpr=SMALL_M_MTPR
    )
    monkeypatch.setenv("SGLANG_AMD_FLYDSL_MEGA_MOE_MTPR", str(MTPR))
    monkeypatch.setenv("SGLANG_AMD_FLYDSL_MEGA_QUANT", "a8w4")
    monkeypatch.setattr(mega, "_MEGA_MOE_INSTANCE", {})
    monkeypatch.setattr(mega, "_LAYER_SLOTS", {})
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(mega, "get_is_capture_mode", lambda: False)
    monkeypatch.setattr(mega, "get_is_extend_in_batch", lambda: False)
    monkeypatch.setattr(mega, "get_dp_global_num_tokens", lambda: state.global_tokens)
    monkeypatch.setattr(mega, "_ensure_mori_shmem", lambda: None)
    monkeypatch.setattr(mega, "_ep_rank_world", lambda: (0, 8))
    utils = ModuleType("sglang.srt.models.deepseek_common.utils")
    utils._use_aiter = True
    monkeypatch.setitem(sys.modules, utils.__name__, utils)

    class FakeMegaMoEV2:
        def __init__(self, **kwargs):
            state.built = kwargs
            self.small_m_mtpr = state.small_m_mtpr
            self.large_m_available = "large_m_state_slots" in kwargs

        def set_large_m_enabled(self, enabled):
            state.large_m_on = enabled

        def bind_layer(self, layer_slot, w1, w1_scale, w2, w2_scale):
            state.calls.append(("bind", layer_slot))

        def forward(self, x, weights, ids, slice_output=False):
            path = "large" if state.large_m_on else "main"
            state.calls.append((path, x.shape[0]))
            return torch.zeros_like(x)

        def forward_small_m(self, x, weights, ids, slice_output=False, max_tokens=None):
            state.calls.append(("small", max_tokens))
            return torch.zeros_like(x)

    monkeypatch.setattr(
        mega, "_import_aiter_megamoe", lambda: SimpleNamespace(op=FakeMegaMoEV2)
    )

    def run(local_rows, global_tokens, *, extend=False, layer=None):
        state.global_tokens = global_tokens
        topk = Mock(
            return_value=SimpleNamespace(
                topk_ids=torch.zeros(local_rows, TOPK, dtype=torch.int64),
                topk_weights=torch.ones(local_rows, TOPK),
            )
        )
        topk.routing_inputs.return_value = (None, None)
        moe = SimpleNamespace(
            config=SimpleNamespace(
                hidden_size=HIDDEN,
                num_experts_per_tok=TOPK,
                moe_intermediate_size=HIDDEN,
                swiglu_limit=0.0,
                num_hidden_layers=LAYERS,
            ),
            num_fused_shared_experts=0,
            experts=state.layers[0] if layer is None else layer,
            gate=lambda h, forward_batch=None: torch.zeros(h.shape[0], EXPERTS),
            topk=topk,
            is_hash=False,
            layer_id=0,
            routed_scaling_factor=1.0,
            routed_scaling_factor_fused_in_topk=True,
        )
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND if extend else ForwardMode.DECODE,
            num_token_non_padded=None,
            is_extend_in_batch=extend,
            original_global_num_tokens_cpu=global_tokens,
        )
        hidden = torch.zeros(local_rows, HIDDEN, dtype=torch.float32)
        mega._run_mega_routed(moe, hidden, forward_batch, None)
        return state.calls[-1]

    state.layers = [_layer() for _ in range(2)]
    state.run = run
    return state


def test_switches_off_keep_the_main_path(runtime):
    assert runtime.run(7, [7, 300]) == ("main", 7)
    assert runtime.run(4096, [4096], extend=True) == ("main", 4096)
    assert "large_m_state_slots" not in runtime.built
    assert not [call for call in runtime.calls if call[0] == "bind"]


@envs.SGLANG_OPT_USE_AITER_MEGA_SMALL_M.override(True)
def test_small_m_uses_the_dp_global_max_on_every_rank(runtime):
    # Ranks with different local rows see the same gathered counts, so all of
    # them call the small-input kernel with the same max_tokens.
    counts = [7, 300, 0, 64, 1, 512, 2, 9]
    assert [runtime.run(rows, counts) for rows in counts] == [("small", 512)] * 8
    # One rank above the small-input capacity moves the whole step to main.
    # (An idle rank runs one sentinel row.)
    counts = [7, 513, 0, 0, 0, 0, 0, 0]
    assert [runtime.run(rows, counts) for rows in counts] == [
        ("main", max(r, 1)) for r in counts
    ]
    # Extend steps never take the small-input path.
    assert runtime.run(7, [7], extend=True) == ("main", 7)


@envs.SGLANG_OPT_USE_AITER_MEGA_LARGE_M.override(True)
def test_large_m_follows_the_dp_global_max_of_extend_steps(runtime):
    # The extend threshold (default 1024) is on the DP-global max, not the local rows.
    assert runtime.run(1, [1, 1024], extend=True) == ("large", 1)
    assert runtime.run(1023, [1023, 5], extend=True) == ("main", 1023)
    assert runtime.run(7, [7]) == ("main", 7)
    assert runtime.built["large_m_state_slots"] == LAYERS + mega._LAYER_SLOT_MARGIN


@envs.SGLANG_OPT_USE_AITER_MEGA_LARGE_M.override(True)
def test_each_moe_layer_keeps_its_own_protocol_slot(runtime):
    for layer in runtime.layers * 2:
        runtime.run(8, [8], layer=layer)
    binds = [slot for call, slot in runtime.calls if call == "bind"]
    assert binds == [0, 1, 0, 1]


@envs.SGLANG_OPT_USE_AITER_MEGA_SMALL_M.override(True)
def test_switch_without_the_aiter_path_fails_at_first_use(runtime):
    runtime.small_m_mtpr = 0
    with pytest.raises(RuntimeError, match="AITER_MEGA_SMALL_M=1"):
        runtime.run(7, [7])
