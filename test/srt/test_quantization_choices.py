"""CPU/CI test for quantization CLI choice/registry consistency.

Regression guard for #32250: ``--quantization marlin`` was accepted by the CLI
(because ``marlin`` was in ``QUANTIZATION_CHOICES``) but had no entry in
``QUANTIZATION_METHODS`` (the bare W4A16 ``marlin`` method was superseded by
``gptq_marlin`` / ``awq_marlin``). The server therefore crashed at model load
instead of failing fast at argument-parse time.

Run: python3 test/srt/test_quantization_choices.py
"""

import pytest

from sglang.srt.layers.quantization import get_quantization_config
from sglang.srt.server_args import QUANTIZATION_CHOICES


def test_marlin_not_in_quantization_choices():
    # The CLI must not advertise a quantization value the registry cannot serve.
    assert "marlin" not in QUANTIZATION_CHOICES


def test_marlin_quantization_method_is_unresolvable():
    # Mirrors the load-time failure users hit before the choice was removed.
    with pytest.raises(ValueError):
        get_quantization_config("marlin")


def test_marlin_family_methods_still_resolve():
    # The real marlin-backed methods must remain available and unchanged.
    assert get_quantization_config("gptq_marlin") is not None
    assert get_quantization_config("awq_marlin") is not None
