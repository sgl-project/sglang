"""CPU tests for how the ROCm gates under srt/layers read their env switches."""

import re
import unittest
from pathlib import Path

from sglang.srt.environ import envs
from sglang.srt.layers import activation
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_LAYERS_DIR = Path(activation.__file__).parent
# Boolean switches declared in environ.py that gate a ROCm path at import time.
_SWITCHES = (envs.SGLANG_USE_AITER, envs.SGLANG_ROCM_DISABLE_LINEARQUANT)
# Top-level modules and subpackages of srt/layers that hold those gates.
_READER_ROOTS = (
    "activation.py",
    "layernorm.py",
    "linear.py",
    "sampler.py",
    "layer_boundary",
    "moe",
)
# The MoRI dispatcher is not covered: it parses its switches itself.
_UNCOVERED = ("moe/token_dispatcher/moriep.py",)
_RAW_READ = re.compile(
    r"""(get_bool_env_var|getenv|environ(\.get)?)\s*[\(\[]\s*["'](%s)["']"""
    % "|".join(switch.name for switch in _SWITCHES)
)


def _reader_sources():
    for root in _READER_ROOTS:
        path = _LAYERS_DIR / root
        for source in sorted(path.rglob("*.py")) if path.is_dir() else [path]:
            if source.relative_to(_LAYERS_DIR).as_posix() not in _UNCOVERED:
                yield source


class TestRocmGateEnvReaders(CustomTestCase):
    def test_accepted_spellings(self):
        for switch in _SWITCHES:
            for raw, expected in (
                ("1", True),
                ("true", True),
                ("yes", True),
                ("y", True),
                ("0", False),
                ("false", False),
                ("no", False),
                ("n", False),
            ):
                with self.subTest(switch=switch.name, raw=raw), switch.override(raw):
                    self.assertIs(switch.get(), expected)

    def test_gates_read_the_switches_through_envs(self):
        """Every gate must read its switch through envs: a raw
        get_bool_env_var read treats yes / y as false, so the gates of one
        server would disagree about whether the switch is on."""
        raw_readers = [
            str(path.relative_to(_LAYERS_DIR))
            for path in _reader_sources()
            if _RAW_READ.search(path.read_text(encoding="utf-8"))
        ]
        self.assertEqual(raw_readers, [])


if __name__ == "__main__":
    unittest.main()
