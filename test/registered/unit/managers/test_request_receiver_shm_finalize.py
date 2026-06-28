from types import SimpleNamespace

from sglang.srt.managers.scheduler_components import request_receiver


def _receiver(
    *,
    enable_dp_attention: bool,
    tp_size: int = 4,
    attn_tp_size: int = 4,
    attn_cp_size: int = 1,
    is_multimodal: bool = True,
):
    return SimpleNamespace(
        server_args=SimpleNamespace(enable_dp_attention=enable_dp_attention),
        ps=SimpleNamespace(
            tp_size=tp_size,
            attn_tp_size=attn_tp_size,
            attn_cp_size=attn_cp_size,
        ),
        model_config=SimpleNamespace(is_multimodal=is_multimodal),
        tp_cpu_group="tp",
        attn_tp_cpu_group="attn_tp",
        attn_cp_cpu_group="attn_cp",
    )


def _run_finalize(monkeypatch, receiver, *, has_shm=True):
    calls = []

    monkeypatch.setattr(
        request_receiver,
        "has_shm_features",
        lambda recv_reqs: has_shm,
    )
    monkeypatch.setattr(
        request_receiver,
        "barrier",
        lambda *, group: calls.append(("barrier", group)),
    )
    monkeypatch.setattr(
        request_receiver,
        "unwrap_shm_features",
        lambda req: calls.append(("unwrap", req)),
    )

    reqs = ["req0", "req1"]
    request_receiver.SchedulerRequestReceiver._finalize_shm_features(receiver, reqs)
    return calls


def test_finalize_shm_features_syncs_dp_attention_work_groups(monkeypatch):
    receiver = _receiver(
        enable_dp_attention=True,
        attn_tp_size=4,
        attn_cp_size=2,
    )

    calls = _run_finalize(monkeypatch, receiver)

    assert calls == [
        ("barrier", "attn_tp"),
        ("barrier", "attn_cp"),
        ("unwrap", "req0"),
        ("unwrap", "req1"),
    ]


def test_finalize_shm_features_keeps_tp_barrier_for_non_dp_attention(monkeypatch):
    receiver = _receiver(enable_dp_attention=False, tp_size=4)

    calls = _run_finalize(monkeypatch, receiver)

    assert calls == [
        ("barrier", "tp"),
        ("unwrap", "req0"),
        ("unwrap", "req1"),
    ]


def test_finalize_shm_features_skips_barrier_without_shm_features(monkeypatch):
    receiver = _receiver(enable_dp_attention=True)

    calls = _run_finalize(monkeypatch, receiver, has_shm=False)

    assert calls == [
        ("unwrap", "req0"),
        ("unwrap", "req1"),
    ]
