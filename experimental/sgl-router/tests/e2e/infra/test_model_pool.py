import signal

from infra import model_pool


class _Process:
    pid = 1234

    def __init__(self):
        self.wait_timeouts = []

    def poll(self):
        return None

    def send_signal(self, sig):
        raise AssertionError(f"signaled only the parent process: {sig}")

    def wait(self, timeout=None):
        self.wait_timeouts.append(timeout)
        return 0


def test_shutdown_waits_for_the_worker_process_group(monkeypatch):
    process = _Process()
    signals = []
    probes = iter([True, True, False])

    def killpg(pgid, sig):
        signals.append((pgid, sig))
        if sig == 0 and not next(probes):
            raise ProcessLookupError

    monkeypatch.setattr(model_pool.os, "killpg", killpg)
    monkeypatch.setattr(model_pool.time, "sleep", lambda _: None)

    instance = model_pool.ModelInstance(
        url="http://127.0.0.1:30000",
        port=30000,
        process=process,
        model_id="qwen3-0.6b",
    )
    instance.shutdown()

    assert signals == [
        (process.pid, signal.SIGTERM),
        (process.pid, 0),
        (process.pid, 0),
        (process.pid, 0),
    ]
    assert process.wait_timeouts == [60]
