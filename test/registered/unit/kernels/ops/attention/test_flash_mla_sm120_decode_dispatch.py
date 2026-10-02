"""SM120 DSV4 decode dispatch: both FlashInfer shapes and the padded-heads knob, no torch.

Loads ``sglang/kernels/ops/attention/flash_mla_sm120.py`` from its file path with
``torch`` / ``triton`` / the sglang env registry stubbed and a stub
``flashinfer.mla._sparse_mla_sm120`` module in either of the two shapes FlashInfer
has published its decode dispatch in:

* ``pairs``: FlashInfer <= 0.6.18, ``_DECODE_DSV4_DISPATCH`` is a table of
  ``(num_heads, topk)`` tuples (#36655 was written against it);
* ``envelope``: FlashInfer main since flashinfer-ai/flashinfer#4802 (453aa7c7296e, in
  every 0.7.0 build), ``_DECODE_DSV4_DISPATCH`` is a ``_DecodeDispatchEnvelope``
  predicate with ``__contains__`` only and the head bound in ``_DECODE_MAX_HEADS``.
  Iterating it raises ``TypeError: '_DecodeDispatchEnvelope' object is not iterable``
  at EagerRunner warm-up, which is how a DeepSeek-V4 TP launch on SM120 died against
  flashinfer_python 0.7.0.

The routing knob ``SGLANG_SM120_DSV4_DECODE_PADDED`` (default 0: FlashInfer at the
exact per-rank width when covered, #36655's routing) is checked through the routing
helper the model calls, ``flashinfer_dsv4_decode_native_heads``, together with the
log line.

    python test/registered/unit/kernels/ops/attention/test_flash_mla_sm120_decode_dispatch.py
"""

import importlib.util
import logging
import pathlib
import sys
import types
import unittest
from unittest.mock import MagicMock

try:
    from sglang.test.ci.ci_register import register_cpu_ci
except Exception:  # pragma: no cover - the CI marker is read statically

    def register_cpu_ci(**_kwargs):
        return None


register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_SHIM = (
    pathlib.Path(__file__).resolve().parents[6]
    / "python"
    / "sglang"
    / "kernels"
    / "ops"
    / "attention"
    / "flash_mla_sm120.py"
)
_KV_LAYOUT_PACKAGES = (
    "sglang.kernels",
    "sglang.kernels.ops",
    "sglang.kernels.ops.attention",
    "sglang.kernels.ops.attention.dsv4",
)
_KV_LAYOUT_MODULE = "sglang.kernels.ops.attention.dsv4.kv_layout"
_FLASHINFER_MODULES = (
    "flashinfer",
    "flashinfer.mla",
    "flashinfer.mla._sparse_mla_sm120",
)
_STUBBED = (
    (
        "torch",
        "triton",
        "triton.language",
        "sglang",
        "sglang.srt",
        "sglang.srt.environ",
        "sglang.srt.utils",
    )
    + _KV_LAYOUT_PACKAGES
    + (_KV_LAYOUT_MODULE,)
    + _FLASHINFER_MODULES
)


class _Env:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


class _Envelope:
    """FlashInfer main's ``_DecodeDispatchEnvelope``: membership only, not iterable."""

    __slots__ = ("min_topk", "max_heads")

    def __init__(self, min_topk=1, max_heads=128):
        self.min_topk = min_topk
        self.max_heads = max_heads

    def __contains__(self, pair):
        if not isinstance(pair, tuple) or len(pair) != 2:
            return False
        h, k = pair
        return 1 <= h <= self.max_heads and k >= self.min_topk


def _install_kv_layout():
    kv_layout_path = _SHIM.parent / "dsv4" / "kv_layout.py"
    if not kv_layout_path.is_file():
        return
    for name in _KV_LAYOUT_PACKAGES:
        pkg = types.ModuleType(name)
        pkg.__path__ = []
        sys.modules[name] = pkg
    spec = importlib.util.spec_from_file_location(_KV_LAYOUT_MODULE, kv_layout_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[_KV_LAYOUT_MODULE] = module
    spec.loader.exec_module(module)


def _install_flashinfer(dispatch, max_tokens=64, max_heads=128, present=True):
    """A stub ``flashinfer.mla._sparse_mla_sm120``; ``present=False`` removes the
    package so the import fails; ``max_heads=None`` leaves ``_DECODE_MAX_HEADS`` out
    (an envelope build without the bound must fail closed)."""
    for name in _FLASHINFER_MODULES:
        sys.modules.pop(name, None)
    if not present:
        return
    flashinfer = types.ModuleType("flashinfer")
    flashinfer.__path__ = []
    mla = types.ModuleType("flashinfer.mla")
    mla.__path__ = []
    sm120 = types.ModuleType("flashinfer.mla._sparse_mla_sm120")
    sm120._DECODE_DSV4_DISPATCH = dispatch
    sm120._DECODE_MAX_TOKENS = max_tokens
    if max_heads is not None:
        sm120._DECODE_MAX_HEADS = max_heads
    mla._sparse_mla_sm120 = sm120
    flashinfer.mla = mla
    sys.modules.update(
        {
            "flashinfer": flashinfer,
            "flashinfer.mla": mla,
            "flashinfer.mla._sparse_mla_sm120": sm120,
        }
    )


class _Shim:
    """The shim under stubs plus the knob handle; every FlashInfer shape is installed
    into the same ``sys.modules`` snapshot and the dispatch cache cleared."""

    def __init__(self):
        self.saved = {name: sys.modules.get(name) for name in _STUBBED}
        for name in ("torch", "triton", "triton.language"):
            sys.modules[name] = MagicMock(name=name)
        sglang = types.ModuleType("sglang")
        srt = types.ModuleType("sglang.srt")
        environ = types.ModuleType("sglang.srt.environ")
        self.knob = _Env(False)
        environ.envs = types.SimpleNamespace(
            SGLANG_SM120_FLASHMLA_BACKEND=_Env("flashinfer"),
            SGLANG_SM120_FLASHINFER_EXTRA_PAGE_BLOCK_SIZES=_Env(()),
            SGLANG_SM120_PREFILL_FALLBACK=_Env(""),
            SGLANG_SM120_DSV4_DECODE_PADDED=self.knob,
        )
        utils = types.ModuleType("sglang.srt.utils")
        utils.is_hip = lambda: False
        sys.modules.update(
            {
                "sglang": sglang,
                "sglang.srt": srt,
                "sglang.srt.environ": environ,
                "sglang.srt.utils": utils,
            }
        )
        _install_kv_layout()
        spec = importlib.util.spec_from_file_location(
            "_flash_mla_sm120_decode_dispatch_under_test", _SHIM
        )
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def flashinfer(self, dispatch, **kwargs):
        _install_flashinfer(dispatch, **kwargs)
        self.module._flashinfer_dsv4_decode_dispatch.cache_clear()
        self.module._DSV4_DECODE_PATHS.clear()
        return self.module

    def restore(self):
        for name, mod in self.saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


PAIRS_0618 = frozenset(
    {(8, 128), (8, 256), (16, 128), (16, 256), (32, 128), (64, 128), (128, 128)}
)


class _Case(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shim = _Shim()

    @classmethod
    def tearDownClass(cls):
        cls.shim.restore()

    def setUp(self):
        self.shim.knob.value = False


class TestCapabilitiesReadBothShapes(_Case):
    def test_pairs_table_yields_its_head_widths(self):
        m = self.shim.flashinfer(PAIRS_0618)
        self.assertEqual(
            m._flashinfer_dsv4_decode_dispatch(),
            (64, frozenset({8, 16, 32, 64, 128}), "pairs"),
        )
        self.assertEqual(
            m._flashinfer_dsv4_decode_capabilities(),
            (64, frozenset({8, 16, 32, 64, 128})),
        )
        self.assertTrue(m.flashinfer_dsv4_decode_supports_num_heads(16, 64))
        self.assertFalse(m.flashinfer_dsv4_decode_supports_num_heads(16, 65))
        self.assertFalse(m.flashinfer_dsv4_decode_supports_num_heads(12, 1))
        self.assertEqual(
            m._describe_dsv4_decode_dispatch(),
            "pairs(max_tokens=64, heads=8,16,32,64,128)",
        )

    def test_envelope_is_probed_against_max_heads_not_iterated(self):
        m = self.shim.flashinfer(_Envelope(min_topk=1), max_heads=128)
        max_tokens, heads, shape = m._flashinfer_dsv4_decode_dispatch()
        self.assertEqual((max_tokens, shape), (64, "envelope"))
        self.assertEqual(heads, frozenset(range(1, 129)))
        self.assertTrue(m.flashinfer_dsv4_decode_supports_num_heads(16, 64))
        self.assertTrue(m.flashinfer_dsv4_decode_supports_num_heads(12, 1))
        self.assertFalse(m.flashinfer_dsv4_decode_supports_num_heads(129, 1))
        self.assertFalse(m.flashinfer_dsv4_decode_supports_num_heads(16, 65))
        self.assertEqual(
            m._describe_dsv4_decode_dispatch(), "envelope(max_tokens=64, heads=1-128)"
        )

    def test_envelope_probe_uses_the_envelopes_min_topk(self):
        # A sliding-window family envelope (min_topk 513) still yields its heads:
        # the probe topk is the envelope's own bound, not a fixed guess.
        m = self.shim.flashinfer(_Envelope(min_topk=513, max_heads=64), max_heads=64)
        self.assertEqual(
            m._flashinfer_dsv4_decode_dispatch()[1], frozenset(range(1, 65))
        )

    def test_envelope_without_max_heads_fails_closed(self):
        m = self.shim.flashinfer(_Envelope(), max_heads=None)
        self.assertEqual(m._flashinfer_dsv4_decode_dispatch(), (0, frozenset(), "none"))
        self.assertFalse(m.flashinfer_dsv4_decode_supports_num_heads(16, 1))
        self.assertEqual(m._describe_dsv4_decode_dispatch(), "none")

    def test_neither_shape_and_missing_flashinfer_fail_closed(self):
        for dispatch, kwargs in (
            (None, {}),
            (object(), {}),
            (frozenset({("x",)}), {}),
            (42, {"max_heads": 128}),
        ):
            with self.subTest(dispatch=dispatch):
                m = self.shim.flashinfer(dispatch, **kwargs)
                self.assertEqual(
                    m._flashinfer_dsv4_decode_dispatch(), (0, frozenset(), "none")
                )
        m = self.shim.flashinfer(None, present=False)
        self.assertEqual(m._flashinfer_dsv4_decode_dispatch(), (0, frozenset(), "none"))
        self.assertFalse(m.flashinfer_dsv4_decode_supports_num_heads(16, 1))

    def test_dispatch_is_read_once(self):
        m = self.shim.flashinfer(_Envelope())
        first = m._flashinfer_dsv4_decode_dispatch()
        _install_flashinfer(PAIRS_0618)  # no cache_clear: the first read stands
        self.assertIs(m._flashinfer_dsv4_decode_dispatch(), first)


class TestRoutingKnob(_Case):
    """``flashinfer_dsv4_decode_native_heads`` is what ``_kernel_num_heads`` asks."""

    def test_default_routes_to_the_native_width_when_covered(self):
        for dispatch, text in ((PAIRS_0618, "pairs("), (_Envelope(), "envelope(")):
            with self.subTest(dispatch=text):
                m = self.shim.flashinfer(dispatch)
                with self.assertLogs(m.logger, level=logging.INFO) as logs:
                    self.assertTrue(m.flashinfer_dsv4_decode_native_heads(16, 1))
                    self.assertTrue(m.flashinfer_dsv4_decode_native_heads(16, 64))
                self.assertEqual(len(logs.records), 1, "one log line per head count")
                msg = logs.records[0].getMessage()
                self.assertIn("path=native local_heads=16 flashinfer_supports=yes", msg)
                self.assertIn("flashinfer_dispatch=" + text, msg)
                self.assertIn("SGLANG_SM120_DSV4_DECODE_PADDED=0", msg)

    def test_padded_knob_keeps_the_64_head_path_even_when_covered(self):
        for dispatch in (PAIRS_0618, _Envelope()):
            with self.subTest(dispatch=type(dispatch).__name__):
                m = self.shim.flashinfer(dispatch)
                self.shim.knob.value = True
                with self.assertLogs(m.logger, level=logging.INFO) as logs:
                    self.assertFalse(m.flashinfer_dsv4_decode_native_heads(16, 64))
                    self.assertFalse(m.flashinfer_dsv4_decode_native_heads(16, 1))
                self.assertTrue(m.flashinfer_dsv4_decode_supports_num_heads(16, 64))
                msg = logs.records[0].getMessage()
                self.assertIn("path=padded local_heads=16 flashinfer_supports=yes", msg)
                self.assertIn("SGLANG_SM120_DSV4_DECODE_PADDED=1", msg)

    def test_uncovered_width_fails_closed_to_the_padded_path(self):
        # 0.6.18's table has no 12-head kernel; an envelope build without the
        # bound reads as none. Both take the padded path, knob or not.
        for dispatch, kwargs, heads in (
            (PAIRS_0618, {}, 12),
            (_Envelope(), {"max_heads": None}, 16),
        ):
            with self.subTest(heads=heads):
                m = self.shim.flashinfer(dispatch, **kwargs)
                with self.assertLogs(m.logger, level=logging.INFO) as logs:
                    self.assertFalse(m.flashinfer_dsv4_decode_native_heads(heads, 1))
                msg = logs.records[0].getMessage()
                self.assertIn(
                    f"path=padded local_heads={heads} flashinfer_supports=no", msg
                )

    def test_beyond_decode_max_tokens_is_not_native(self):
        m = self.shim.flashinfer(_Envelope())
        self.assertFalse(m.flashinfer_dsv4_decode_native_heads(16, 65))

    def test_a_changed_resolution_is_logged_again(self):
        m = self.shim.flashinfer(_Envelope())
        with self.assertLogs(m.logger, level=logging.INFO) as logs:
            self.assertTrue(m.flashinfer_dsv4_decode_native_heads(16, 1))
            self.shim.knob.value = True
            self.assertFalse(m.flashinfer_dsv4_decode_native_heads(16, 1))
            self.assertFalse(m.flashinfer_dsv4_decode_native_heads(16, 1))
        self.assertEqual(
            [
                ("path=native" in r.getMessage(), "path=padded" in r.getMessage())
                for r in logs.records
            ],
            [(True, False), (False, True)],
        )


if __name__ == "__main__":
    unittest.main()
