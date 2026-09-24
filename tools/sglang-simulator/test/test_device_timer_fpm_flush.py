import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

_SOURCE = (
    Path(__file__).parents[3]
    / "python"
    / "sglang"
    / "srt"
    / "utils"
    / "device_timer.py"
)
_INSERTED_TORCH_STUB = "torch" not in sys.modules
if _INSERTED_TORCH_STUB:
    _TORCH = ModuleType("torch")
    _TORCH.cuda = SimpleNamespace(Event=object)
    sys.modules["torch"] = _TORCH
_SPEC = importlib.util.spec_from_file_location("device_timer_under_test", _SOURCE)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
if _INSERTED_TORCH_STUB:
    del sys.modules["torch"]
DeviceTimer = _MODULE.DeviceTimer


class _Event:
    def __init__(self):
        self.synchronized = False

    def query(self):
        return self.synchronized

    def synchronize(self):
        self.synchronized = True


class _Interval:
    def __init__(self, milliseconds):
        self.end_event = _Event()
        self.metadata = {"category": "forward"}
        self._milliseconds = milliseconds

    def elapsed_time(self):
        return self._milliseconds


def test_fpm_flush_waits_for_current_interval_and_reports_it_once():
    reported = []
    timer = DeviceTimer(lambda **values: reported.append(values))
    interval = _Interval(12.5)
    timer._intervals.append(interval)

    timer._report()
    assert reported == []

    timer.synchronize_and_report()
    assert reported == [{"t": pytest.approx(0.0125), "category": "forward"}]
    assert not timer._intervals

    timer.synchronize_and_report()
    assert len(reported) == 1


def test_fpm_flush_rejects_an_active_or_unfinished_interval():
    timer = DeviceTimer(lambda **_values: None)
    timer._in_wrap = True
    with pytest.raises(RuntimeError, match="inside a timed region"):
        timer.synchronize_and_report()

    timer._in_wrap = False
    timer._intervals.append(
        SimpleNamespace(end_event=None, metadata={}, elapsed_time=lambda: 0.0)
    )
    with pytest.raises(RuntimeError, match="unfinished"):
        timer.synchronize_and_report()
