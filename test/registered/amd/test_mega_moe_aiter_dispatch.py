"""Unit tests for ROCm MegaMoE (AITER) dispatch wiring."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture(autouse=True)
def reset_mega_moe_runtime_state():
    import sglang.srt.layers.moe.mega_moe_aiter as mega

    mega._MORI_SHMEM_READY = False
    mega._MEGA_RUNTIME_READY = False
    mega._MEGA_CACHE.clear()
    yield
    mega._MORI_SHMEM_READY = False
    mega._MEGA_RUNTIME_READY = False
    mega._MEGA_CACHE.clear()


def test_mega_moe_build_routes_to_aiter_on_rocm(monkeypatch):
    monkeypatch.setattr("sglang.srt.layers.moe.mega_moe._is_hip", True)
    monkeypatch.setattr(
        "sglang.srt.layers.moe.mega_moe._use_aiter_mega_moe",
        lambda: True,
    )

    called = {}

    def fake_build(experts):
        called["experts"] = experts
        experts._mega_moe_weights_built = True

    class Experts:
        _mega_moe_weights_built = False

    monkeypatch.setattr(
        "sglang.srt.layers.moe.mega_moe_aiter.build_mega_moe_aiter_weights",
        fake_build,
    )

    from sglang.srt.layers.moe.mega_moe import build_mega_moe_experts_weights

    experts = Experts()
    build_mega_moe_experts_weights(experts)
    assert called["experts"] is experts


def test_validate_mtpr_rejects_non_power_of_two():
    from sglang.srt.layers.moe.mega_moe_aiter import _validate_mtpr

    with pytest.raises(ValueError, match="power of two"):
        _validate_mtpr(1000)


def test_validate_mtpr_accepts_power_of_two():
    from sglang.srt.layers.moe.mega_moe_aiter import _validate_mtpr

    _validate_mtpr(1024)


def test_is_mega_moe_aiter_enabled_requires_megamoe_backend(monkeypatch):
    monkeypatch.setattr("sglang.srt.layers.moe.mega_moe_aiter._is_hip", True)
    monkeypatch.setattr("sglang.srt.layers.moe.mega_moe_aiter._use_aiter", True)

    class FakeBackend:
        @staticmethod
        def is_megamoe():
            return False

    monkeypatch.setattr(
        "sglang.srt.layers.moe.mega_moe_aiter.get_moe_a2a_backend",
        lambda: FakeBackend(),
    )

    from sglang.srt.layers.moe.mega_moe_aiter import is_mega_moe_aiter_enabled

    assert is_mega_moe_aiter_enabled() is False


def test_mori_shmem_init_uses_default_name_and_ep_cpu_group(monkeypatch):
    import sglang.srt.layers.moe.mega_moe_aiter as mega

    mega._MORI_SHMEM_READY = False
    calls = []
    cpu_group = object()
    ep_group = SimpleNamespace(cpu_group=cpu_group)

    fake_shmem = SimpleNamespace(
        shmem_torch_process_group_init=lambda name: calls.append(("init", name)),
        shmem_barrier_all=lambda: calls.append(("shmem_barrier",)),
    )
    monkeypatch.setitem(sys.modules, "mori", SimpleNamespace(shmem=fake_shmem))
    monkeypatch.setattr(
        mega,
        "get_parallel",
        lambda: SimpleNamespace(moe_ep_rank=3, moe_ep_size=8),
    )
    monkeypatch.setattr(
        torch._C._distributed_c10d,
        "_register_process_group",
        lambda name, group: calls.append(("register", name, group)),
    )
    monkeypatch.setattr(
        torch.distributed,
        "barrier",
        lambda *, group: calls.append(("dist_barrier", group)),
    )

    assert mega._ensure_mori_shmem(ep_group) == (3, 8)
    assert calls == [
        ("register", "default", cpu_group),
        ("dist_barrier", cpu_group),
        ("init", "default"),
        ("shmem_barrier",),
        ("dist_barrier", cpu_group),
    ]

    calls.clear()
    assert mega._ensure_mori_shmem(ep_group) == (3, 8)
    assert calls == []


def test_initialize_runtime_builds_operator_eagerly(monkeypatch):
    import sglang.srt.layers.moe.mega_moe_aiter as mega

    mega._MEGA_RUNTIME_READY = False
    built = {}
    cpu_group = object()
    ep_group = SimpleNamespace(cpu_group=cpu_group)

    experts = SimpleNamespace(
        _mega_moe_weights_built=True,
        _mega_moe_mtpr=4096,
        _mega_moe_model_dim=7168,
        _mega_moe_inter_dim=3072,
        _mega_moe_swiglu_limit=10.0,
        _mega_w1=object(),
        _mega_w1_scale=object(),
        _mega_w2=object(),
        _mega_w2_scale=object(),
        hidden_size=7168,
        num_experts=392,
        w13_weight=SimpleNamespace(shape=(49, 6144, 3584)),
    )
    moe = SimpleNamespace(
        experts=experts,
        config=SimpleNamespace(num_experts_per_tok=6),
        num_fused_shared_experts=1,
    )
    model = SimpleNamespace(modules=lambda: iter([SimpleNamespace(), moe]))

    monkeypatch.setattr(mega, "is_mega_moe_aiter_enabled", lambda: True)
    monkeypatch.setattr(mega, "get_moe_ep_group", lambda: ep_group)
    monkeypatch.setattr(mega, "_ensure_mori_shmem", lambda group: (2, 8))
    monkeypatch.setattr(
        mega, "_get_or_build_mega_moe", lambda **kwargs: built.update(kwargs)
    )
    monkeypatch.setitem(
        sys.modules,
        "mori",
        SimpleNamespace(
            shmem=SimpleNamespace(shmem_barrier_all=lambda: built.setdefault("barrier", True))
        ),
    )
    monkeypatch.setattr(torch.distributed, "barrier", lambda *, group: None)

    mega.initialize_mega_moe_aiter_runtime(model)

    assert mega._MEGA_RUNTIME_READY is True
    assert built["rank"] == 2
    assert built["world_size"] == 8
    assert built["experts"] == 392
    assert built["topk"] == 7
    assert built["mtpr"] == 4096
    assert built["barrier"] is True


def test_runtime_init_requires_prepared_weights(monkeypatch):
    import sglang.srt.layers.moe.mega_moe_aiter as mega

    mega._MEGA_RUNTIME_READY = False
    monkeypatch.setattr(mega, "is_mega_moe_aiter_enabled", lambda: True)
    monkeypatch.setattr(
        mega, "get_moe_ep_group", lambda: SimpleNamespace(cpu_group=object())
    )
    monkeypatch.setattr(mega, "_ensure_mori_shmem", lambda group: (0, 8))

    with pytest.raises(RuntimeError, match="no prepared MoE layer"):
        mega.initialize_mega_moe_aiter_runtime(
            SimpleNamespace(modules=lambda: iter([SimpleNamespace()]))
        )


def test_rocm_megamoe_rejects_mori_isolation_mode(monkeypatch):
    from sglang.srt.arg_groups.mega_moe_hook import handle_rocm_megamoe
    from sglang.srt.environ import envs

    monkeypatch.setattr("sglang.srt.utils.is_hip", lambda: True)
    monkeypatch.setattr(
        envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK,
        "get",
        lambda: 4096,
    )
    monkeypatch.setattr(envs.SGLANG_USE_AITER, "get", lambda: True)
    monkeypatch.setenv("MORI_SHMEM_MODE", "ISOLATION")

    with pytest.raises(ValueError, match="does not support MORI_SHMEM_MODE=ISOLATION"):
        handle_rocm_megamoe(
            SimpleNamespace(moe_a2a_backend="megamoe", ep_size=8)
        )


def test_empty_dp_attention_batch_participates_with_dummy_token(monkeypatch):
    import sglang.srt.layers.moe.mega_moe_aiter as mega

    mega._MEGA_RUNTIME_READY = True
    forwarded = {}

    class FakeMega:
        def forward(self, hidden_states, topk_weights, topk_ids):
            forwarded["shapes"] = (
                hidden_states.shape,
                topk_weights.shape,
                topk_ids.shape,
            )
            assert torch.count_nonzero(topk_weights) == 0
            return torch.zeros_like(hidden_states)

    monkeypatch.setattr(
        mega,
        "_get_or_build_mega_moe",
        lambda **kwargs: FakeMega(),
    )
    monkeypatch.setattr(
        mega,
        "get_parallel",
        lambda: SimpleNamespace(moe_ep_rank=0, moe_ep_size=8),
    )

    hidden_states = torch.empty((0, 7168))
    experts = SimpleNamespace(
        _mega_w1=torch.empty(1),
        _mega_w1_scale=torch.empty(1),
        _mega_w2=torch.empty(1),
        _mega_w2_scale=torch.empty(1),
        _mega_moe_mtpr=4096,
        _mega_moe_model_dim=7168,
        _mega_moe_inter_dim=3072,
        _mega_moe_swiglu_limit=10.0,
        hidden_size=7168,
        num_experts=392,
        w13_weight=SimpleNamespace(shape=(49, 6144, 3584)),
    )
    moe = SimpleNamespace(
        experts=experts,
        config=SimpleNamespace(num_experts_per_tok=6),
        num_fused_shared_experts=1,
    )
    output = mega.run_mega_moe_aiter_routed(
        moe=moe,
        hidden_states=hidden_states,
        forward_batch=None,
        input_ids_global=None,
        num_tokens=0,
    )

    assert output.shape == hidden_states.shape
    assert forwarded["shapes"] == (
        torch.Size((1, 7168)),
        torch.Size((1, 7)),
        torch.Size((1, 7)),
    )
