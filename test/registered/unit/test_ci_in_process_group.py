"""Unit tests for in_process_group CI bundling.

Guards derived properties of the partition-level test bundling path:
suite-uniformity, LPT atomicity, first-run est math (import amortization
+ per-file live_est), module-path conversion, and compute_partitions
unit counting. Each case fails on a concrete silent regression that
would otherwise only show up as CI wall-time or blame skew.
"""

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Optional

from sglang.test.ci.ci_register import (
    _BUNDLE_IMPORT_COST_SEC,
    CIBundle,
    CIRegistry,
    HWBackend,
    auto_partition,
    bundle_in_process_groups,
    register_cpu_ci,
    validate_in_process_groups,
)
from sglang.test.ci.ci_utils import (
    _assert_bundle_members_unittest_loadable,
    _file_has_unittest_testcase,
    _filename_to_module,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _reg(
    filename: str,
    est_time: float = 60.0,
    *,
    suite: str = "base-b-test-1-gpu-small",
    in_process_group: Optional[str] = None,
    nightly: bool = False,
    backend: HWBackend = HWBackend.CUDA,
) -> CIRegistry:
    return CIRegistry(
        backend=backend,
        filename=filename,
        est_time=est_time,
        suite=suite,
        nightly=nightly,
        in_process_group=in_process_group,
    )


class TestValidateInProcessGroups(CustomTestCase):
    """The authoritative uniformity check, run over the unfiltered registry.

    `bundle_in_process_groups` only ever receives a list already narrowed to
    one (backend, suite, nightly) triple, so its own check can't fire —
    a non-uniform group silently splits into partial bundles instead.
    """

    def test_multi_suite_group_raises(self):
        validate_in_process_groups([_reg("a.py", in_process_group="g")])  # uniform: ok
        files = [
            _reg("a.py", suite="suite-a", in_process_group="g"),
            _reg("b.py", suite="suite-b", in_process_group="g"),
        ]
        with self.assertRaises(ValueError) as ctx:
            validate_in_process_groups(files)
        self.assertIn("g", str(ctx.exception))

    def test_nightly_split_group_raises(self):
        """Same suite, mixed nightly -> one bundle per run type. Must not pass."""
        files = [
            _reg("a.py", in_process_group="g"),
            _reg("b.py", in_process_group="g", nightly=True),
        ]
        with self.assertRaises(ValueError) as ctx:
            validate_in_process_groups(files)
        self.assertIn("nightly", str(ctx.exception))

    def test_backend_split_group_raises(self):
        files = [
            _reg("a.py", in_process_group="g"),
            _reg("b.py", in_process_group="g", backend=HWBackend.AMD),
        ]
        with self.assertRaises(ValueError) as ctx:
            validate_in_process_groups(files)
        self.assertIn("g", str(ctx.exception))

    def test_ungrouped_files_ignored(self):
        validate_in_process_groups(
            [_reg("a.py", suite="suite-a"), _reg("b.py", suite="suite-b")]
        )


class TestBundleInProcessGroups(CustomTestCase):
    def test_multi_suite_group_raises(self):
        """Members of one group on different suites must not silently merge."""
        files = [
            _reg("a.py", suite="suite-a", in_process_group="g"),
            _reg("b.py", suite="suite-b", in_process_group="g"),
        ]
        with self.assertRaises(ValueError) as ctx:
            bundle_in_process_groups(files)
        self.assertIn("multiple suites", str(ctx.exception))
        self.assertIn("g", str(ctx.exception))

    def test_fallback_est_amortizes_import(self):
        """Without group live_est, est = sum(member) - (N-1)*import."""
        members = [
            _reg("a.py", est_time=30.0, in_process_group="attn"),
            _reg("b.py", est_time=30.0, in_process_group="attn"),
            _reg("c.py", est_time=30.0, in_process_group="attn"),
        ]
        units = bundle_in_process_groups(members)
        self.assertEqual(len(units), 1)
        bundle = units[0]
        self.assertIsInstance(bundle, CIBundle)
        expected = 3 * 30.0 - 2 * _BUNDLE_IMPORT_COST_SEC
        self.assertAlmostEqual(bundle.est_time, expected)

    def test_fallback_est_never_below_hard_lower_bounds(self):
        """A bundle can't beat one cold import, nor its slowest member.

        Amortizing (N-1)*import off a sum of sub-import members underflows
        (20 x 2s -> -150s). A bundle bin-packed as the cheapest unit in the
        suite makes its shard overrun the stage timeout, and the est can't
        self-correct because `live_est["group:<key>"]` only lands on a run
        that passed.
        """
        many_cheap = [
            _reg(f"c{i}.py", est_time=2.0, in_process_group="cheap") for i in range(20)
        ]
        bundle = bundle_in_process_groups(many_cheap)[0]
        self.assertGreaterEqual(bundle.est_time, _BUNDLE_IMPORT_COST_SEC)

        # Slowest member dominates when amortization overshoots.
        skewed = [_reg("slow.py", est_time=45.0, in_process_group="skew")] + [
            _reg(f"f{i}.py", est_time=1.0, in_process_group="skew") for i in range(8)
        ]
        skewed_bundle = bundle_in_process_groups(skewed)[0]
        self.assertGreaterEqual(skewed_bundle.est_time, 45.0)

    def test_fallback_prefers_per_file_live_est(self):
        """First-run (no group:<key>) must use per-file live_est when present."""
        a = _reg("/repo/test/a.py", est_time=100.0, in_process_group="g")
        b = _reg("/repo/test/b.py", est_time=100.0, in_process_group="g")
        live_est = {
            a.filename: 40.0,
            b.filename: 50.0,
        }
        bundle = bundle_in_process_groups([a, b], live_est=live_est)[0]
        expected = 40.0 + 50.0 - 1 * _BUNDLE_IMPORT_COST_SEC
        self.assertAlmostEqual(bundle.est_time, expected)

    def test_group_live_est_short_circuits(self):
        a = _reg("a.py", est_time=100.0, in_process_group="g")
        b = _reg("b.py", est_time=100.0, in_process_group="g")
        live_est = {"group:g": 55.0}
        bundle = bundle_in_process_groups([a, b], live_est=live_est)[0]
        self.assertAlmostEqual(bundle.est_time, 55.0)

    def test_ungrouped_files_pass_through(self):
        files = [_reg("solo.py"), _reg("g.py", in_process_group="g")]
        units = bundle_in_process_groups(files)
        kinds = {type(u).__name__ for u in units}
        self.assertEqual(kinds, {"CIRegistry", "CIBundle"})
        self.assertEqual(sum(1 for u in units if isinstance(u, CIRegistry)), 1)
        self.assertEqual(sum(1 for u in units if isinstance(u, CIBundle)), 1)


class TestAutoPartitionBundleAtomicity(CustomTestCase):
    def test_group_lands_in_exactly_one_partition(self):
        """LPT must not split an in_process_group across partitions."""
        # One heavy singleton + a 5-member group. With size=3, a naive
        # per-file pack would scatter the group; bundling keeps it atomic.
        files = [
            _reg(f"g{i}.py", est_time=20.0, in_process_group="attn") for i in range(5)
        ]
        files.append(_reg("solo_heavy.py", est_time=90.0))
        files.append(_reg("solo_light.py", est_time=10.0))

        membership = []
        for rank in range(3):
            part = auto_partition(files, rank=rank, size=3)
            group_count = sum(
                1 for u in part if isinstance(u, CIBundle) and u.group_key == "attn"
            )
            membership.append(group_count)
            # Bundle never partially appears as loose CIRegistry members.
            for u in part:
                if isinstance(u, CIRegistry):
                    self.assertIsNone(u.in_process_group)

        self.assertEqual(sum(membership), 1, membership)
        self.assertIn(1, membership)


class TestFilenameToModule(CustomTestCase):
    def test_relative_and_absolute_under_cwd(self):
        cwd = os.getcwd()
        rel = os.path.join("registered", "foo", "test_x.py")
        abs_path = os.path.join(cwd, rel)
        self.assertEqual(_filename_to_module(rel), "registered.foo.test_x")
        self.assertEqual(_filename_to_module(abs_path), "registered.foo.test_x")

    def test_non_identifier_segment_raises(self):
        """Paths like 4-gpu-models cannot be unittest module paths."""
        with self.assertRaises(ValueError) as ctx:
            _filename_to_module("registered/4-gpu-models/test_x.py")
        self.assertIn("cannot map", str(ctx.exception))
        self.assertIn("4-gpu-models", str(ctx.exception))

    def test_jit_path_maps_under_python_package_root(self):
        """JIT files live under repo/python/; cwd is test/ in CI.

        Naive relpath-to-cwd produces `../python/sglang/...` which is not
        importable. Mapping must yield `sglang.jit_kernel...`.
        """
        jit_abs = os.path.join(
            str(_REPO_ROOT),
            "python",
            "sglang",
            "jit_kernel",
            "tests",
            "test_add_constant.py",
        )
        # Unit tests may run from repo root or test/; normalize like CI.
        prev = os.getcwd()
        try:
            os.chdir(str(_REPO_ROOT / "test"))
            self.assertEqual(
                _filename_to_module(jit_abs),
                "sglang.jit_kernel.tests.test_add_constant",
            )
        finally:
            os.chdir(prev)


class TestBundleUnittestLoadable(CustomTestCase):
    def test_this_file_is_unittest_loadable(self):
        self.assertTrue(_file_has_unittest_testcase(__file__))

    def test_pytest_style_file_rejected(self):
        """Pure pytest modules would exit 0 with zero tests under unittest."""
        jit = (
            _REPO_ROOT
            / "python"
            / "sglang"
            / "jit_kernel"
            / "tests"
            / "test_add_constant.py"
        )
        if not jit.is_file():
            self.skipTest("jit test fixture not present on this checkout")
        self.assertFalse(_file_has_unittest_testcase(str(jit)))
        members = [
            CIRegistry(
                backend=HWBackend.CUDA,
                filename=str(jit),
                est_time=5.0,
                suite="base-b-kernel-unit-1-gpu-large",
                in_process_group="jit",
            )
        ]
        with self.assertRaises(ValueError) as ctx:
            _assert_bundle_members_unittest_loadable(members)
        self.assertIn("silently skipped", str(ctx.exception))
        self.assertIn(str(jit), str(ctx.exception))


def _load_run_suite_module():
    path = _REPO_ROOT / "test" / "run_suite.py"
    spec = importlib.util.spec_from_file_location("run_suite_ut", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class TestLoadLiveEstGroupKeys(CustomTestCase):
    def test_group_keys_preserved_verbatim(self):
        run_suite = _load_run_suite_module()

        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.json")
            with open(model_path, "w") as f:
                json.dump(
                    {
                        "est": {
                            "base-b-test-1-gpu-small": {
                                "test/registered/a.py": 12.0,
                                "group:attn": 45.0,
                            }
                        }
                    },
                    f,
                )
            live = run_suite.load_live_est(
                model_path, "base-b-test-1-gpu-small", str(_REPO_ROOT)
            )
            self.assertIsNotNone(live)
            self.assertIn("group:attn", live)
            self.assertEqual(live["group:attn"], 45.0)
            abs_a = os.path.join(str(_REPO_ROOT), "test/registered/a.py")
            self.assertEqual(live[abs_a], 12.0)


def _load_compute_partitions_module():
    path = _REPO_ROOT / "scripts" / "ci" / "utils" / "compute_partitions.py"
    spec = importlib.util.spec_from_file_location("compute_partitions_ut", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class TestComputePartitionsBundleAware(CustomTestCase):
    def _regs_via_cp_module(
        self, cp, n: int, *, est_time: float, suite: str, group: str
    ):
        # Must use the importlib-loaded CIRegistry/HWBackend: compute_partitions
        # compares backends by identity against its own enum objects.
        return [
            cp._ci_register.CIRegistry(
                backend=cp.HWBackend.CUDA,
                filename=os.path.join(str(_REPO_ROOT), f"test/registered/h{i}.py"),
                est_time=est_time,
                suite=suite,
                in_process_group=group,
            )
            for i in range(n)
        ]

    def test_group_live_est_yields_one_shard_not_file_count_error(self):
        """group:<key> live_est + bundling: 5 huge files → 1 cheap unit → size 1.

        Pre-fix path summed per-file est (5 * 10000) and capped shards by
        file count (5). With a tight stage timeout that would raise
        "needs N shards but has only 5 test file(s)" (or over-shard).
        Bundle-aware path uses group live_est so total=100 and units=1.
        """
        cp = _load_compute_partitions_module()
        huge = self._regs_via_cp_module(
            cp, 5, est_time=10_000.0, suite="tight-suite", group="huge"
        )
        live_model = {
            "est": {"tight-suite": {"group:huge": 100.0}},
            "fit": {},
        }
        # 0.75 * 5min * 60 = 225s target; total 100 => ideal_size 1.
        run_timeouts = {"tight-suite": 5}
        result = cp.compute_partitions(
            huge,
            repo_root=str(_REPO_ROOT),
            run_timeouts=run_timeouts,
            partition_model=live_model,
        )
        self.assertEqual(result["tight-suite"]["size"], 1)
        self.assertEqual(len(result["tight-suite"]["arr"]), 1)

    def test_unit_cap_message_uses_bin_pack_units(self):
        """When ideal_size exceeds units, error reports unit count not only files."""
        cp = _load_compute_partitions_module()
        # One atomic bundle of 5 files, each est 10_000, no group live_est.
        # Amortized total still >> tight target → ideal_size > 1 unit.
        huge = self._regs_via_cp_module(
            cp, 5, est_time=10_000.0, suite="tight-suite", group="huge"
        )
        run_timeouts = {"tight-suite": 5}
        with self.assertRaises(RuntimeError) as ctx:
            cp.compute_partitions(
                huge,
                repo_root=str(_REPO_ROOT),
                run_timeouts=run_timeouts,
                partition_model=None,
            )
        msg = str(ctx.exception)
        self.assertIn("bin-pack unit", msg)
        self.assertIn("1 bin-pack unit", msg)
        self.assertIn("5 file", msg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
