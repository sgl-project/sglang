"""Die-local routed MoE for K3 target verify (SGLANG_K3_ROUTED_MOE_DEEPGEMM): the in-tree per-width table and its
override, per-width dispatch with and without SGLANG_K3_AR_FUSION, the fallback to the default path, page homes, and
the assumptions about the code around it (no GPU, no deep_gemm call)."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import contextlib
import inspect
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.communication import k3_ar_fusion
from sglang.srt.layers.moe import k3_deepgemm_routed as routed
from sglang.srt.layers.moe import k3_die_local_moe as policy

BASE = [10, 20, 40, 80, 160, 235]
AR_FUSION = [80, 160, 235]
PREFIX = "SGLANG_K3_ROUTED_MOE_DEEPGEMM"
CONTROLS = (
    PREFIX,
    PREFIX + "_WEIGHT_PLACEMENT",
    PREFIX + "_DISPATCH",
    PREFIX + "_VERIFY_WIDTHS",
    "SGLANG_K3_AR_FUSION",
    "SGLANG_ENABLE_NV_MEGA_RUNTIME_K31_GB_OPTIMIZATION",
)
# Largest row count whose finalize the push all-reduce folds under SGLANG_K3_AR_FUSION
FOLD_MAX_ROWS = 73


@contextlib.contextmanager
def _env(table=None, **values):
    """The switches cleared, then set as given; `table` (a dict) is written to a file and named as the override."""
    with contextlib.ExitStack() as stack:
        for name in CONTROLS:
            field = getattr(envs, name)
            if name in values:
                stack.enter_context(field.override(values[name]))
            elif field.is_set():
                stack.enter_context(field.override(""))
                field.clear()
        if table is not None:
            path = Path(
                stack.enter_context(tempfile.TemporaryDirectory()), "widths.json"
            )
            path.write_text(table if isinstance(table, str) else json.dumps(table))
            stack.enter_context(
                envs.SGLANG_K3_ROUTED_MOE_DEEPGEMM_VERIFY_WIDTHS.override(str(path))
            )
        policy.reset_caches()
        stack.callback(policy.reset_caches)
        yield


def _in_verify(rows, verify=True):
    token = policy._verify.set((verify, False))
    try:
        return policy.takes(rows)
    finally:
        policy._verify.reset(token)


def _taken(verify=True):
    return [r for r in range(1, 300) if _in_verify(r, verify)]


ON = {PREFIX: True}
TABLE = dict(widths=BASE, widths_ar_fusion=AR_FUSION)


class TestInTreeTable(unittest.TestCase):
    def setUp(self):
        self.table = json.loads(policy._DEFAULT_TABLE.read_text())

    def test_default_path_is_next_to_the_module(self):
        self.assertEqual(policy._DEFAULT_TABLE.parent, Path(policy.__file__).parent)
        with _env(**ON):
            self.assertEqual(policy.table_path(), policy._DEFAULT_TABLE)

    def test_lists(self):
        widths, fused = self.table["widths"], self.table["widths_ar_fusion"]
        for rows in (widths, fused):
            self.assertEqual(rows, sorted(set(rows)))
            self.assertTrue(
                all(type(r) is int and 1 <= r <= policy.MAX_VERIFY_ROWS for r in rows)
            )
        # under AR fusion: the base widths above the fold limit, none at or below it
        self.assertEqual(self.table["ar_fusion_fold_max_rows"], FOLD_MAX_ROWS)
        self.assertEqual(fused, [r for r in widths if r > FOLD_MAX_ROWS])

    def test_is_what_the_module_dispatches(self):
        with _env(**ON, SGLANG_K3_AR_FUSION=False):
            self.assertEqual(_taken(), self.table["widths"])
        with _env(**ON, SGLANG_K3_AR_FUSION=True):
            self.assertEqual(_taken(), self.table["widths_ar_fusion"])

    def test_text_names_the_switches_and_how_to_regenerate(self):
        for key in ("description", "scope", "rule", "regenerate"):
            self.assertIsInstance(self.table[key], str)
        self.assertIn(
            "SGLANG_K3_ROUTED_MOE_DEEPGEMM_VERIFY_WIDTHS", self.table["regenerate"]
        )


class TestDispatch(unittest.TestCase):
    def test_off_reads_nothing(self):
        # the override names a file that does not exist: with the switch off nothing may read it
        for extra in (
            {},
            {"SGLANG_K3_AR_FUSION": True},
            {"SGLANG_ENABLE_NV_MEGA_RUNTIME_K31_GB_OPTIMIZATION": True},
        ):
            with _env(
                **extra, **{PREFIX + "_VERIFY_WIDTHS": "/nonexistent/widths.json"}
            ):
                self.assertFalse(policy.enabled())
                self.assertEqual(policy.verify_widths(), frozenset())
                self.assertEqual(_taken(), [])

    def test_off_leaves_the_model_forward_alone(self):
        def forward(self, input_ids, positions, forward_batch):
            return "out"

        with _env():
            self.assertIs(policy.target_forward(forward), forward)
        with _env(**ON, table=TABLE):
            scoped = policy.target_forward(forward)
            self.assertIsNot(scoped, forward)
            seen = []
            batch = types.SimpleNamespace(
                forward_mode=types.SimpleNamespace(is_target_verify=lambda: True)
            )
            with mock.patch.object(
                torch.cuda, "is_current_stream_capturing", lambda: False
            ):
                scoped_out = policy.target_forward(
                    lambda s, i, p, b: seen.append(policy.takes(80)) or "out"
                )(None, None, None, batch)
            self.assertEqual((scoped_out, seen), ("out", [True]))
            self.assertFalse(policy.takes(80))  # reset after the forward

    def test_without_ar_fusion_uses_widths(self):
        with _env(**ON, table=TABLE, SGLANG_K3_AR_FUSION=False):
            self.assertFalse(policy.ar_fusion())
            self.assertEqual(_taken(), BASE)
            # prefill / extend / plain decode never dispatch
            self.assertEqual(_taken(verify=False), [])

    def test_ar_fusion_uses_its_list(self):
        for extra in (
            {"SGLANG_K3_AR_FUSION": True},
            {"SGLANG_ENABLE_NV_MEGA_RUNTIME_K31_GB_OPTIMIZATION": True},
        ):
            with _env(**ON, table=TABLE, **extra):
                self.assertTrue(policy.ar_fusion())
                self.assertEqual(_taken(), AR_FUSION)
                self.assertEqual(_taken(verify=False), [])

    def test_ar_fusion_switch_wins_over_the_runtime_bundle(self):
        with _env(
            **ON,
            table=TABLE,
            SGLANG_K3_AR_FUSION=False,
            SGLANG_ENABLE_NV_MEGA_RUNTIME_K31_GB_OPTIMIZATION=True,
        ):
            self.assertEqual(_taken(), BASE)

    def test_dispatch_off_never_takes_a_call(self):
        with _env(**ON, table=TABLE, **{PREFIX + "_DISPATCH": False}):
            self.assertEqual(_taken(), [])

    def test_bad_tables_fail_when_read(self):
        with _env(**ON, **{PREFIX + "_VERIFY_WIDTHS": "/nonexistent/widths.json"}):
            with self.assertRaises(FileNotFoundError):
                policy.verify_widths()
        for bad in (
            dict(widths=BASE),
            dict(widths=BASE, widths_ar_fusion=[]),
            dict(widths=BASE, widths_ar_fusion=[80, 257]),
            dict(widths=BASE, widths_ar_fusion=[80.0]),
        ):
            with _env(**ON, table=bad, SGLANG_K3_AR_FUSION=True):
                with self.assertRaises(ValueError):
                    policy.verify_widths()

    def test_bad_placement_value(self):
        with _env(**ON, **{PREFIX + "_WEIGHT_PLACEMENT": "majority"}):
            with self.assertRaises(ValueError):
                policy.weight_placement()

    def test_handles(self):
        stock = types.SimpleNamespace()
        prepared = types.SimpleNamespace(mega_l1_weights=(object(), object()))
        with _env(**ON, table=TABLE, SGLANG_K3_AR_FUSION=True):
            token = policy._verify.set((True, False))
            try:
                for rows in (5, 20, 70, 80, 235):
                    self.assertFalse(
                        routed.handles(stock, rows)
                    )  # trtllm-gen: may defer its finalize
                    self.assertEqual(routed.handles(prepared, rows), rows in AR_FUSION)
            finally:
                policy._verify.reset(token)
            # outside target verify a prepared layer leaves every call to trtllm-gen
            self.assertFalse(
                any(routed.handles(prepared, rows) for rows in (5, 20, 80, 235))
            )


def _k31_layer(**overrides):
    cfg = types.SimpleNamespace(
        activation="situ",
        gemm1_alpha=4.0,
        gemm1_clamp_limit=25.0,
        gate_up_interleaved=False,
    )
    weight = lambda *shape: types.SimpleNamespace(
        shape=shape, device=types.SimpleNamespace(index=0), element_size=lambda: 1
    )
    layer = dict(
        moe_runner_config=cfg,
        num_experts=896,
        num_local_experts=112,
        w13_weight=weight(112, 6144, 1792),
        w2_weight=weight(112, 3584, 1536),
    )
    layer.update(overrides)
    return types.SimpleNamespace(**layer)


class TestFallback(unittest.TestCase):
    """With the switch on, anything that does not qualify keeps the default path and says so once."""

    def _wanted(self, layers, *, capability=None, sm=(10, 7), placement=None, env=None):
        warnings = []
        with (
            _env(**ON, table=TABLE, **(env or {})),
            mock.patch.object(routed, "missing_capability", lambda: capability),
            mock.patch.object(torch.distributed, "is_initialized", lambda: True),
            mock.patch.object(
                routed,
                "_deep_gemm",
                lambda: types.SimpleNamespace(get_num_locality_domains=lambda: 2),
            ),
            mock.patch.object(
                torch.cuda, "get_device_capability", lambda device=None: sm
            ),
            mock.patch.object(
                policy,
                "_placement_unavailable",
                lambda shape, itemsize, device: placement,
            ),
            mock.patch.object(policy, "print_warning_once", warnings.append),
        ):
            return [
                policy.wanted(layer, False, "default") for layer in layers
            ], warnings

    def test_switch_off_asks_nothing(self):
        with _env(), mock.patch.object(routed, "mismatch") as mismatch:
            self.assertFalse(policy.wanted(_k31_layer(), False, "default"))
            mismatch.assert_not_called()

    def test_qualifying_layer(self):
        got, warnings = self._wanted([_k31_layer()])
        self.assertEqual((got, warnings), ([True], []))

    def test_deep_gemm_without_the_api(self):
        reason = (
            "deep_gemm 0.2.9 lacks the one-rank MegaMoE API (numerics, topk_format)"
        )
        got, warnings = self._wanted(
            [_k31_layer() for _ in range(92)], capability=reason
        )
        self.assertEqual(got, [False] * 92)
        # the same text for every layer: print_warning_once logs it once per process
        self.assertEqual(len(set(warnings)), 1)
        self.assertIn(reason, warnings[0])
        self.assertIn("SGLANG_K3_ROUTED_MOE_DEEPGEMM is set but not used", warnings[0])
        self.assertIn("default path", warnings[0])

    def test_other_hardware_and_models(self):
        for kwargs, layer, text in (
            (dict(sm=(10, 0)), _k31_layer(), "SM 10.0"),
            (dict(sm=(9, 0)), _k31_layer(), "SM 9.0"),
            (dict(), _k31_layer(num_local_experts=448), "expert shape"),
            (dict(), _k31_layer(num_experts=256), "expert shape"),
            (
                dict(
                    placement="die-local placement is not available: no locality-domain API"
                ),
                _k31_layer(),
                "placement",
            ),
        ):
            got, warnings = self._wanted([layer], **kwargs)
            self.assertEqual(got, [False])
            self.assertEqual(len(warnings), 1)
            self.assertIn(text, warnings[0])

    def test_default_placement_needs_no_placement_api(self):
        got, warnings = self._wanted(
            [_k31_layer()],
            placement="unavailable",
            env={PREFIX + "_WEIGHT_PLACEMENT": "default"},
        )
        self.assertEqual((got, warnings), ([True], []))

    def test_layer_rule_comes_first(self):
        got, warnings = self._wanted([_k31_layer()], capability="no api")
        self.assertIn("no api", warnings[0])
        with (
            _env(**ON, table=TABLE),
            mock.patch.object(policy, "print_warning_once") as warn,
        ):
            self.assertFalse(policy.wanted(_k31_layer(), True, "default"))
            self.assertIn("expert bias", warn.call_args[0][0])

    def test_placement_without_driver_bindings(self):
        # the real probe: no cuda.bindings locality-domain API (or no CUDA at all) reads as "not available"
        policy._page_bytes.cache_clear()
        self.addCleanup(policy._page_bytes.cache_clear)
        with mock.patch.object(
            policy,
            "_page_bytes",
            side_effect=RuntimeError("cuda.bindings has no locality-domain API"),
        ):
            reason = policy._placement_unavailable(
                (112, 6144, 1792), 1, types.SimpleNamespace(index=0)
            )
        self.assertIn("not available", reason)
        with mock.patch.object(policy, "_page_bytes", return_value=2 << 20):
            self.assertIsNone(
                policy._placement_unavailable(
                    (112, 6144, 1792), 1, types.SimpleNamespace(index=0)
                )
            )
            self.assertIn(
                "does not split",
                policy._placement_unavailable(
                    (112, 6144, 1793), 1, types.SimpleNamespace(index=0)
                ),
            )


class TestPageHomes(unittest.TestCase):
    PAGE = 2 << 20

    def test_k31_shapes(self):
        for shape, pages in (((112, 6144, 1792), 588), ((112, 3584, 1536), 294)):
            homes = policy.page_homes(shape, 1, self.PAGE)
            self.assertEqual(len(homes), pages)
            self.assertEqual(homes.count(0), homes.count(1))
            # bytes read from the other domain's pages: about a quarter, against a half for alternating pages
            half = shape[1] * shape[2] // 2
            misplaced = (
                sum(
                    1
                    for page, home in enumerate(homes)
                    for byte in range(page * self.PAGE, (page + 1) * self.PAGE, 4096)
                    if byte // half % 2 != home
                )
                * 4096
            )
            self.assertLess(misplaced / (len(homes) * self.PAGE), 0.26)

    def test_page_aligned_halves_are_exact(self):
        self.assertEqual(
            policy.page_homes((3, 4, self.PAGE), 1, self.PAGE), [0, 0, 1, 1] * 3
        )

    def test_even_split_alternates(self):
        # every page holds half of each domain's rows: homes alternate by page index
        self.assertEqual(
            policy.page_homes((4, 2, self.PAGE // 2), 1, self.PAGE), [0, 1, 0, 1]
        )


class TestAssumptions(unittest.TestCase):
    """Each fails when the surrounding code moves, so the table and the load order get re-derived."""

    def test_fold_threshold(self):
        for slot, last in (
            (1 << 30, FOLD_MAX_ROWS),
            (16 << 20, FOLD_MAX_ROWS),
            (256 << 10, 36),
        ):
            state = types.SimpleNamespace(push_slot_bytes=slot)
            with mock.patch.object(k3_ar_fusion, "_get_state", return_value=state):
                self.assertTrue(k3_ar_fusion.finalize_push_fits(last))
                self.assertFalse(k3_ar_fusion.finalize_push_fits(last + 1))

    def test_deferral_gate(self):
        from sglang.srt.models import kimi_k3

        source = inspect.getsource(kimi_k3.KimiK3MoE._forward_fused)
        start = source.index("defer_finalize = (")
        gate = source[start : source.index("\n            )\n", start)]
        # A sequence-parallel pass never defers, so only the push-window rule and the
        # MegaMoE check decide for a decode or verify batch.
        self.assertIn("not scatters", gate)
        self.assertIn("k3_ar_fusion.finalize_push_fits(num_tokens)", gate)
        self.assertIn("not k3_deepgemm_routed.handles(self.experts, num_tokens)", gate)
        # The deferred branch is the only caller of the deferred runner entry point.
        model = inspect.getsource(kimi_k3.KimiK3MoE)
        self.assertEqual(model.count("self._forward_routed_deferred("), 1)
        self.assertEqual(model.count(".forward_deferred_finalize("), 1)

    def test_load_keeps_the_trtllm_workspace(self):
        from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod

        source = inspect.getsource(Mxfp4MoEMethod.process_weights_after_loading)
        reserve = source.index("reserve_trtllm_mxfp4_moe_workspace(\n")
        finish = source.index("k3_die_local_moe.finish(")
        self.assertLess(reserve, finish)
        self.assertNotIn("return", source[reserve:finish])


if __name__ == "__main__":
    unittest.main()
