"""Which aiter the triton backend trusts with gfx1250 Gluon MLA decode.

Before ROCm/aiter#5935 the kernel writes its segment softmax state over the
output whenever a batch runs one segment per sequence (batch 97 and above on
96 CUs). That output is wrong without an error, so an aiter without the fix
must leave MLA decode on the Triton kernel.
"""

import sys
import types
import unittest
from unittest import mock

from sglang.srt.environ import envs
from sglang.srt.layers.attention import triton_backend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_GLUON_PKG = "aiter.ops.triton._gluon_kernels.gfx1250.attention"
_FIXED_SOURCE = "if NUM_SEGMENTS_PER_SEQ > 1:\n    store_partials()\n"
_UNFIXED_SOURCE = "store_partials()\n"


def _fake_aiter(mla_decode_fwd):
    """sys.modules entries for the two aiter imports the probe makes."""
    gluon_attention = types.ModuleType(_GLUON_PKG)
    gluon_attention.mla = types.ModuleType(f"{_GLUON_PKG}.mla")
    triton_mla = types.ModuleType("aiter.ops.triton.attention.mla")
    triton_mla.mla_decode_fwd = mla_decode_fwd
    modules = {
        name: types.ModuleType(name)
        for name in (
            "aiter",
            "aiter.ops",
            "aiter.ops.triton",
            "aiter.ops.triton._gluon_kernels",
            "aiter.ops.triton._gluon_kernels.gfx1250",
            "aiter.ops.triton.attention",
        )
    }
    modules[_GLUON_PKG] = gluon_attention
    modules[f"{_GLUON_PKG}.mla"] = gluon_attention.mla
    modules["aiter.ops.triton.attention.mla"] = triton_mla
    return modules


class TestGluonMlaDecodeProbe(CustomTestCase):
    def setUp(self):
        self.kernel = object()
        triton_backend._gluon_mla_decode_fwd.cache_clear()
        self.addCleanup(triton_backend._gluon_mla_decode_fwd.cache_clear)
        for patcher in (
            mock.patch.object(triton_backend, "_is_hip", True),
            mock.patch.object(triton_backend, "is_gfx1250_supported", lambda: True),
            mock.patch.dict(sys.modules, _fake_aiter(self.kernel)),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def _probe(self, source, use_aiter=True):
        with (
            envs.SGLANG_USE_AITER.override(use_aiter),
            mock.patch.object(triton_backend.inspect, "getsource", return_value=source),
        ):
            return triton_backend._gluon_mla_decode_fwd()

    def test_aiter_with_the_fix_is_used(self):
        self.assertIs(self._probe(_FIXED_SOURCE), self.kernel)

    def test_aiter_without_the_fix_keeps_triton_decode(self):
        self.assertIsNone(self._probe(_UNFIXED_SOURCE))

    def test_off_unless_aiter_is_requested(self):
        self.assertIsNone(self._probe(_FIXED_SOURCE, use_aiter=False))

    def test_aiter_without_the_kernel_keeps_triton_decode(self):
        with mock.patch.dict(sys.modules, {_GLUON_PKG: None}):
            self.assertIsNone(self._probe(_FIXED_SOURCE))


if __name__ == "__main__":
    unittest.main()
