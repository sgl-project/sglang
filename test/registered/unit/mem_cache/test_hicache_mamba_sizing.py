"""Unit tests for --hicache-mamba-size-gb: parser, split rule, auto fixed point.

The module under test is torch-free and is loaded by path so this file runs on
a machine without torch or the sglang extension modules.
"""

import ast
import importlib.util
import math
import pathlib
import sys
import unittest
from unittest.mock import MagicMock

try:
    from sglang.test.ci.ci_register import register_cpu_ci
except Exception:  # pragma: no cover - checkout without torch

    def register_cpu_ci(*args, **kwargs):
        return None


register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_HYBRID_CACHE_DIR = (
    pathlib.Path(__file__).resolve().parents[4]
    / "python"
    / "sglang"
    / "srt"
    / "mem_cache"
    / "hybrid_cache"
)
_MODULE_PATH = _HYBRID_CACHE_DIR / "hicache_mamba_sizing.py"
_ASSEMBLER_PATH = _HYBRID_CACHE_DIR / "hybrid_pool_assembler.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("hicache_mamba_sizing", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sizing = _load_module()

# GLM-5.3-Flash TP4 RTX PRO 6000 (KV-ECONOMY-PLAN 2.3): 12 x 660 B per token per
# rank on device and host; one KDA checkpoint is 35 x (524,288 + 36,864) B.
GLM_KV_BYTES_PER_TOKEN = 7_920
GLM_MAMBA_BYTES_PER_SLOT = 19_640_320
GLM_CHUNK = 4096


def _split_hicache_size_reference(hicache_size, device_pool_bytes):
    """Verbatim arithmetic of hybrid_pool_assembler._split_hicache_size."""
    total = sum(device_pool_bytes)
    return tuple(hicache_size * size_bytes / total for size_bytes in device_pool_bytes)


class TestParseMambaHostSize(unittest.TestCase):
    def test_none_stays_unset(self):
        self.assertIsNone(sizing.parse_mamba_host_size(None))

    def test_number_strings_and_numbers(self):
        self.assertEqual(sizing.parse_mamba_host_size("14"), 14.0)
        self.assertEqual(sizing.parse_mamba_host_size(" 14.5 "), 14.5)
        self.assertEqual(sizing.parse_mamba_host_size(14), 14.0)
        self.assertEqual(sizing.parse_mamba_host_size(0.5), 0.5)

    def test_auto_any_case(self):
        self.assertEqual(sizing.parse_mamba_host_size("auto"), sizing.AUTO)
        self.assertEqual(sizing.parse_mamba_host_size(" AUTO "), sizing.AUTO)

    def test_rejects_garbage_and_non_positive(self):
        for bad in ("x", "", "14GB", "0", "-1", "nan", "inf", True):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    sizing.parse_mamba_host_size(bad)


class TestValidateKnob(unittest.TestCase):
    def test_unset_passes_without_hicache_size(self):
        self.assertIsNone(sizing.validate_mamba_host_size_knob(None, 0))

    def test_requires_fixed_hicache_size(self):
        with self.assertRaises(ValueError):
            sizing.validate_mamba_host_size_knob("14", 0)
        with self.assertRaises(ValueError):
            sizing.validate_mamba_host_size_knob("auto", 0)

    def test_explicit_must_leave_kv_a_share(self):
        with self.assertRaises(ValueError):
            sizing.validate_mamba_host_size_knob("32", 32)
        with self.assertRaises(ValueError):
            sizing.validate_mamba_host_size_knob("40", 32)
        self.assertEqual(sizing.validate_mamba_host_size_knob("14", 32), 14.0)
        self.assertEqual(sizing.validate_mamba_host_size_knob("auto", 32), sizing.AUTO)


class TestProportionalDefault(unittest.TestCase):
    def test_none_reproduces_split_hicache_size(self):
        # The existing assembler test's pools: 75 GB and (15 + 10) GB of device bytes.
        kv_bytes, mamba_bytes = 75 * 10**9, 15 * 10**9 + 10 * 10**9
        split = sizing.resolve_mamba_host_split(
            hicache_size_gb=100,
            knob=None,
            kv_bytes_per_token=GLM_KV_BYTES_PER_TOKEN,
            mamba_bytes_per_slot=GLM_MAMBA_BYTES_PER_SLOT,
            device_kv_bytes=kv_bytes,
            device_mamba_bytes=mamba_bytes,
            chunked_prefill_size=GLM_CHUNK,
            max_running_requests=8,
        )
        self.assertEqual(split.mode, "proportional")
        self.assertEqual((split.kv_gb, split.mamba_gb), (75.0, 25.0))
        self.assertEqual(
            (split.kv_gb, split.mamba_gb),
            _split_hicache_size_reference(100, (kv_bytes, mamba_bytes)),
        )
        self.assertEqual(
            sizing.proportional_split(100, (kv_bytes, mamba_bytes)),
            _split_hicache_size_reference(100, (kv_bytes, mamba_bytes)),
        )

    def test_none_is_byte_identical_on_awkward_ratios(self):
        # Odd byte counts: the helper must use the same expression order as
        # _split_hicache_size so the default path stays bit-for-bit today's.
        pools = (7_530_000_000 + 680_000_000, 3_920_000_000)
        self.assertEqual(
            sizing.proportional_split(32, pools),
            _split_hicache_size_reference(32, pools),
        )

    def test_default_glm_split_reports_the_shortfall(self):
        # The campaign box: 32 GB split against ~8.2 GB KV + 3.9 GB mamba on
        # device gives the mamba tier a few hundred slots against thousands of
        # checkpoints of KV host demand; the report must say so.
        split = sizing.resolve_mamba_host_split(
            hicache_size_gb=32,
            knob=None,
            kv_bytes_per_token=GLM_KV_BYTES_PER_TOKEN,
            mamba_bytes_per_slot=GLM_MAMBA_BYTES_PER_SLOT,
            device_kv_bytes=8_210_000_000,
            device_mamba_bytes=3_920_000_000,
            chunked_prefill_size=GLM_CHUNK,
            max_running_requests=8,
        )
        self.assertLess(split.coverage, 1.0)
        self.assertTrue(split.needs_warning)
        self.assertIn("coverage", split.boot_line())


class TestExplicitGigabytes(unittest.TestCase):
    def test_fourteen_of_thirty_two(self):
        split = sizing.resolve_mamba_host_split(
            hicache_size_gb=32,
            knob=14.0,
            kv_bytes_per_token=GLM_KV_BYTES_PER_TOKEN,
            mamba_bytes_per_slot=GLM_MAMBA_BYTES_PER_SLOT,
            device_kv_bytes=8_210_000_000,
            device_mamba_bytes=3_920_000_000,
            chunked_prefill_size=GLM_CHUNK,
            max_running_requests=8,
        )
        self.assertEqual(split.mode, "explicit")
        self.assertEqual(split.kv_gb, 18.0)
        self.assertEqual(split.mamba_gb, 14.0)
        # MambaPoolHost: int(host_size * 1e9 // size_per_token)
        self.assertEqual(split.slots, int(14.0 * 1e9 // GLM_MAMBA_BYTES_PER_SLOT))
        self.assertTrue(700 <= split.slots <= 720, split.slots)  # ~713 at 19.6 MB
        self.assertEqual(split.kv_tokens, int(18.0 * 1e9 // GLM_KV_BYTES_PER_TOKEN))
        # 18 GB / 7,920 B = 2.27M tokens -> 555 chunks + 32 finish slots.
        self.assertEqual(
            split.demanded_slots, math.ceil(split.kv_tokens / GLM_CHUNK) + 4 * 8
        )
        self.assertGreaterEqual(split.coverage, 1.0)
        self.assertFalse(split.needs_warning)

    def test_knob_at_or_above_budget_is_rejected(self):
        for knob in (32.0, 40.0):
            with self.subTest(knob=knob):
                with self.assertRaises(ValueError):
                    sizing.resolve_mamba_host_split(
                        hicache_size_gb=32,
                        knob=knob,
                        kv_bytes_per_token=GLM_KV_BYTES_PER_TOKEN,
                        mamba_bytes_per_slot=GLM_MAMBA_BYTES_PER_SLOT,
                        device_kv_bytes=1,
                        device_mamba_bytes=1,
                        chunked_prefill_size=GLM_CHUNK,
                        max_running_requests=8,
                    )

    def test_forced_small_value_flags_partial_coverage(self):
        split = sizing.resolve_mamba_host_split(
            hicache_size_gb=32,
            knob=1.0,
            kv_bytes_per_token=GLM_KV_BYTES_PER_TOKEN,
            mamba_bytes_per_slot=GLM_MAMBA_BYTES_PER_SLOT,
            device_kv_bytes=1,
            device_mamba_bytes=1,
            chunked_prefill_size=GLM_CHUNK,
            max_running_requests=8,
        )
        self.assertEqual(split.slots, 50)
        self.assertLess(split.coverage, 1.0)
        self.assertTrue(split.needs_warning)
        line = split.boot_line()
        self.assertIn("host mamba slots 50 cover", line)
        self.assertIn("KV host tier", line)
        self.assertIn("coverage", line)


class TestAuto(unittest.TestCase):
    def _auto(self, hicache_size_gb=32, max_running_requests=4, chunk=GLM_CHUNK):
        return sizing.resolve_mamba_host_split(
            hicache_size_gb=hicache_size_gb,
            knob=sizing.AUTO,
            kv_bytes_per_token=GLM_KV_BYTES_PER_TOKEN,
            mamba_bytes_per_slot=GLM_MAMBA_BYTES_PER_SLOT,
            device_kv_bytes=8_210_000_000,
            device_mamba_bytes=3_920_000_000,
            chunked_prefill_size=chunk,
            max_running_requests=max_running_requests,
        )

    def test_auto_covers_the_kv_tier(self):
        split = self._auto()
        self.assertEqual(split.mode, "auto")
        self.assertGreaterEqual(split.coverage, 1.0)
        self.assertGreaterEqual(
            split.slots, math.ceil(split.kv_tokens / GLM_CHUNK) + 4 * 4
        )
        self.assertAlmostEqual(split.kv_gb + split.mamba_gb, 32.0, places=6)
        self.assertGreater(split.kv_gb, 0)
        # The plan's expectation: ~2.5M anchorable KV host tokens at 32 GB.
        self.assertTrue(2_400_000 <= split.kv_tokens <= 2_600_000, split.kv_tokens)
        self.assertFalse(split.needs_warning)

    def test_auto_is_a_fixed_point(self):
        split = self._auto()
        # Re-deriving the demand from the resolved KV tier changes nothing.
        demand = sizing.checkpoint_demand(split.kv_tokens, GLM_CHUNK, 4)
        self.assertLessEqual(abs(split.slots - demand), 1)
        self.assertGreaterEqual(split.slots, demand)
        # And the pool constructor's floor division keeps every slot.
        self.assertEqual(
            sizing.host_slots(split.mamba_gb, GLM_MAMBA_BYTES_PER_SLOT), split.slots
        )

    def test_auto_tracks_running_requests(self):
        few, many = (
            self._auto(max_running_requests=4),
            self._auto(max_running_requests=8),
        )
        self.assertGreater(many.slots, few.slots)
        self.assertGreaterEqual(many.coverage, 1.0)

    def test_auto_at_ninety_six_gb(self):
        split = self._auto(hicache_size_gb=96, max_running_requests=8)
        self.assertGreaterEqual(split.coverage, 1.0)
        # ~7.5M anchorable tokens against 6.34M at the proportional split.
        self.assertGreater(split.kv_tokens, 7_000_000)

    def test_auto_needs_chunked_prefill(self):
        for chunk in (None, -1, 0):
            with self.subTest(chunk=chunk):
                with self.assertRaises(ValueError):
                    self._auto(chunk=chunk)

    def test_auto_rejects_a_budget_below_the_finish_slots(self):
        with self.assertRaises(ValueError):
            self._auto(hicache_size_gb=0.1, max_running_requests=8)


class TestCoverageReport(unittest.TestCase):
    def test_full_coverage_line(self):
        report = sizing.mamba_host_coverage(
            kv_tokens=2_270_000,
            slots=713,
            chunked_prefill_size=4096,
            max_running_requests=8,
        )
        self.assertEqual(report.demanded_slots, math.ceil(2_270_000 / 4096) + 32)
        self.assertTrue(report.is_full)
        self.assertEqual(report.covered_tokens, (713 - 32) * 4096)
        self.assertEqual(
            report.boot_line(),
            "host mamba slots 713 cover 2789376 tokens at one checkpoint per "
            "4096-token chunk; KV host tier 2270000 tokens (coverage 121%)",
        )

    def test_partial_coverage_line(self):
        report = sizing.mamba_host_coverage(
            kv_tokens=2_120_000,
            slots=205,
            chunked_prefill_size=4096,
            max_running_requests=8,
        )
        self.assertFalse(report.is_full)
        self.assertLess(report.coverage, 0.4)
        self.assertIn("(coverage 37%)", report.boot_line())

    def test_chunked_prefill_off(self):
        report = sizing.mamba_host_coverage(
            kv_tokens=1_000_000,
            slots=40,
            chunked_prefill_size=-1,
            max_running_requests=8,
        )
        self.assertEqual(report.demanded_slots, 32)
        self.assertEqual(report.covered_tokens, 0)
        self.assertTrue(report.is_full)
        self.assertIn("chunked prefill off", report.boot_line())

    def test_zero_demand_is_full(self):
        report = sizing.mamba_host_coverage(
            kv_tokens=0, slots=0, chunked_prefill_size=4096, max_running_requests=0
        )
        self.assertEqual(report.coverage, 1.0)


class TestCoverageInputGuard(unittest.TestCase):
    """The coverage boot line is skipped unless every input is a real size.

    Pool-assembly tests build the hybrid Mamba stack with MagicMock params and
    a patched MambaPoolHost; an unconditional coverage line raises TypeError
    inside checkpoint_demand there.
    """

    @staticmethod
    def _mock_stack_inputs():
        """What _log_mamba_host_coverage reads under that test: all mocks."""
        params = MagicMock()
        kv_host_pool = MagicMock()
        mamba_host_pool = MagicMock()
        return (
            kv_host_pool.size,
            mamba_host_pool.size,
            params.chunked_prefill_size,
            params.req_to_token_pool.size,
        )

    def test_checkpoint_demand_rejects_a_mock(self):
        # The failure the guard prevents: `chunked_prefill_size > 0` on a mock.
        with self.assertRaises(TypeError):
            sizing.checkpoint_demand(1_000, MagicMock(), 8)

    def test_mock_params_are_not_sizes(self):
        self.assertFalse(sizing.all_sizes_or_none(self._mock_stack_inputs()))

    def test_real_sizes_and_none_pass(self):
        self.assertTrue(sizing.all_sizes_or_none((2_680_896, 565, 4096, 8)))
        self.assertTrue(sizing.all_sizes_or_none((2_680_896, 565, None, None)))
        self.assertTrue(sizing.all_sizes_or_none(()))

    def test_other_types_fail(self):
        self.assertFalse(sizing.all_sizes_or_none((2_680_896, 565, 4096.0, 8)))
        self.assertFalse(sizing.all_sizes_or_none((2_680_896, 565, "4096", 8)))
        self.assertFalse(sizing.all_sizes_or_none((None, object(), None, None)))

    def test_assembler_guards_the_line_before_the_report(self):
        """AST pin: _log_mamba_host_coverage returns early on all_sizes_or_none."""
        tree = ast.parse(_ASSEMBLER_PATH.read_text(encoding="utf-8"))
        func = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_log_mamba_host_coverage"
        )
        guard_index = report_index = None
        for index, stmt in enumerate(func.body):
            if (
                isinstance(stmt, ast.If)
                and isinstance(stmt.test, ast.UnaryOp)
                and isinstance(stmt.test.op, ast.Not)
                and isinstance(stmt.test.operand, ast.Call)
                and getattr(stmt.test.operand.func, "id", None) == "all_sizes_or_none"
                and len(stmt.body) == 1
                and isinstance(stmt.body[0], ast.Return)
            ):
                guard_index = index
            if (
                isinstance(stmt, ast.Assign)
                and isinstance(stmt.value, ast.Call)
                and getattr(stmt.value.func, "id", None) == "mamba_host_coverage"
            ):
                report_index = index
        self.assertIsNotNone(guard_index, "no `if not all_sizes_or_none(...): return`")
        self.assertIsNotNone(report_index, "no `report = mamba_host_coverage(...)`")
        self.assertLess(guard_index, report_index)
        imported = {
            alias.name
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            and node.module == "sglang.srt.mem_cache.hybrid_cache.hicache_mamba_sizing"
            for alias in node.names
        }
        self.assertIn("all_sizes_or_none", imported)


if __name__ == "__main__":
    unittest.main()
