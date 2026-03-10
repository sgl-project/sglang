from __future__ import annotations

import sglang.srt.utils.common as common


class _DummyProcess:
    def __init__(self, allowed):
        self._allowed = list(allowed)
        self._set = None

    def cpu_affinity(self, cpus=None):
        if cpus is None:
            return list(self._allowed)
        self._set = list(cpus)
        return list(self._set)


def test_set_gpu_proc_affinity_respects_allowed_cpus(monkeypatch) -> None:
    dummy = _DummyProcess(allowed=[0, 1, 2, 3, 4, 5, 6, 7])

    monkeypatch.setattr(common.os, "getpid", lambda: 123)
    monkeypatch.setattr(common.psutil, "Process", lambda _pid: dummy)
    monkeypatch.setattr(common.psutil, "cpu_count", lambda logical=True: 64 if logical else 32)

    common.set_gpu_proc_affinity(pp_size=1, tp_size=4, nnodes=1, gpu_id=1)

    assert dummy._set == [2, 3]
