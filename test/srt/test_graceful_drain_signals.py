"""Graceful-drain signal topology.

Server shutdown is coordinated by the tokenizer manager: on a stop signal it
drains in-flight requests, then explicitly stops the worker subprocesses.
These tests pin the two properties that make the drain reachable:

1. worker subprocesses ignore group-delivered SIGINT/SIGTERM instead of dying
   mid-forward at signal time, and
2. the main-process handler maps the first stop signal to the drain flag and a
   repeated stop signal to force exit.
"""

import signal
from types import SimpleNamespace

from sglang.srt.utils.common import ignore_external_stop_signals


def test_ignore_external_stop_signals_sets_sig_ign():
    old_int = signal.getsignal(signal.SIGINT)
    old_term = signal.getsignal(signal.SIGTERM)
    try:
        ignore_external_stop_signals()
        assert signal.getsignal(signal.SIGINT) is signal.SIG_IGN
        assert signal.getsignal(signal.SIGTERM) is signal.SIG_IGN
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)


def test_stop_signal_sets_drain_flag_and_escalates():
    from sglang.srt.managers.tokenizer_manager import SignalHandler

    tokenizer_manager = SimpleNamespace(gracefully_exit=False, drain_force_exit=False)
    handler = SignalHandler(tokenizer_manager)

    handler.sigterm_handler(signal.SIGTERM, None)
    assert tokenizer_manager.gracefully_exit
    assert not tokenizer_manager.drain_force_exit

    handler.sigterm_handler(signal.SIGINT, None)
    assert tokenizer_manager.drain_force_exit
