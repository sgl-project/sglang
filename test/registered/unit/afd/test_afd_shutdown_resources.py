from __future__ import annotations

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

from types import SimpleNamespace

import pytest

from sglang.srt.managers import tokenizer_manager


@pytest.mark.parametrize(
    "role,child_exit_seconds,expected_wait",
    [
        ("off", 20, 15),
        ("attention", 20, 20),
        ("attention", 80, 80),
        ("attention", 120, 95),
    ],
)
def test_tokenizer_broadcasts_shutdown_and_preserves_afd_close_budget(
    role, child_exit_seconds, expected_wait, monkeypatch
):
    import asyncio

    clock = [0.0]
    calls = []

    def sleep(seconds):
        clock[0] += seconds

    def exit_process(code):
        raise SystemExit(code)

    namespace = dict(
        asyncio=asyncio,
        _SCHEDULER_EXIT_TIMEOUT_SECS=15,
        get_bool_env_var=lambda _: False,
        ServerStatus=SimpleNamespace(UnHealthy="unhealthy"),
        logger=SimpleNamespace(info=lambda *args: None, warning=lambda *args: None),
        ShutdownReq=type("ShutdownReq", (), {}),
        time=SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep),
        collect_scheduler_processes=lambda: (
            [SimpleNamespace(pid=1)] if clock[0] < child_exit_seconds else []
        ),
        kill_process_tree=lambda *args, **kwargs: calls.append(
            ("parent_cleanup", clock[0])
        ),
        os=SimpleNamespace(getpid=lambda: 1),
        sys=SimpleNamespace(exit=exit_process),
    )
    for name, value in namespace.items():
        monkeypatch.setattr(tokenizer_manager, name, value)
    manager = SimpleNamespace(
        gracefully_exit=True,
        rid_to_state={},
        server_status="healthy",
        server_args=SimpleNamespace(
            afd_execution_mode=role,
            afd_config=SimpleNamespace(close_timeout_seconds=30),
        ),
        _server_stop_hook=None,
        _subprocess_watchdog=SimpleNamespace(
            stop=lambda: calls.append(("watchdog_stop", clock[0]))
        ),
        _dispatch_to_scheduler=lambda request: calls.append(
            ("broadcast_shutdown", clock[0])
        ),
    )
    with pytest.raises(SystemExit) as stopped:
        asyncio.run(tokenizer_manager.TokenizerManager.sigterm_watchdog(manager))
    assert stopped.value.code == 0
    assert [name for name, _ in calls] == [
        "watchdog_stop",
        "broadcast_shutdown",
        "parent_cleanup",
    ]
    assert calls[-1][1] == pytest.approx(expected_wait, abs=0.11)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-x"]))
