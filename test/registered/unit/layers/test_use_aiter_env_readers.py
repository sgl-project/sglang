"""CPU tests for how the AITER gates under srt/layers read SGLANG_USE_AITER."""

import re
import unittest
from pathlib import Path

from sglang.srt.environ import envs
from sglang.srt.layers import activation
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_LAYERS_DIR = Path(activation.__file__).parent
# Top-level modules and subpackages of srt/layers that gate an AITER path.
_READER_ROOTS = ("activation.py", "layernorm.py", "sampler.py", "layer_boundary", "moe")
_RAW_READ = re.compile(
    r"""(get_bool_env_var|getenv|environ(\.get)?)\s*[\(\[]\s*["']SGLANG_USE_AITER["']"""
)


def _reader_sources():
    for root in _READER_ROOTS:
        path = _LAYERS_DIR / root
        yield from sorted(path.rglob("*.py")) if path.is_dir() else [path]


class TestUseAiterEnvReaders(CustomTestCase):
    def test_accepted_spellings(self):
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
            with self.subTest(raw=raw), envs.SGLANG_USE_AITER.override(raw):
                self.assertIs(envs.SGLANG_USE_AITER.get(), expected)

    def test_gates_read_the_variable_through_envs(self):
        """Every AITER gate must read SGLANG_USE_AITER through envs: a raw
        get_bool_env_var read treats yes / y as false, so the gates of one
        server would disagree about whether AITER is on."""
        raw_readers = [
            str(path.relative_to(_LAYERS_DIR))
            for path in _reader_sources()
            if _RAW_READ.search(path.read_text(encoding="utf-8"))
        ]
        self.assertEqual(raw_readers, [])


if __name__ == "__main__":
    unittest.main()
