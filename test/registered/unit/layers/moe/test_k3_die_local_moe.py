"""Die-local routed MoE for K3 target verify (SGLANG_K3_ROUTED_MOE_DEEPGEMM): the layer
rules and the DeepGEMM capability check, the per-width table and its dispatch, the
fallback to the default path, page homes, and the assumptions about the code around it
(no GPU, no deep_gemm call)."""

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
from sglang.test.test_utils import CustomTestCase

BASE = [10, 20, 40, 80, 160, 235]
AR_FUSION = [80, 160, 235]
TABLE = dict(widths=BASE, widths_ar_fusion=AR_FUSION)
PREFIX = "SGLANG_K3_ROUTED_MOE_DEEPGEMM"
FUSION = "SGLANG_K3_AR_FUSION"
BUNDLE = "SGLANG_ENABLE_NV_MEGA_RUNTIME_K31_GB_OPTIMIZATION"
SUFFIXES = ("", "_WEIGHT_PLACEMENT", "_DISPATCH", "_VERIFY_WIDTHS")
CONTROLS = (*(PREFIX + suffix for suffix in SUFFIXES), FUSION, BUNDLE)
ON = {PREFIX: True}
# Largest row count whose finalize the push all-reduce folds under SGLANG_K3_AR_FUSION
FOLD_MAX_ROWS = 73
# sgl-deep-gemm 0.2.9: has weight_layout (NVFP4 only) but none of the one-rank keywords
RELEASED_CALL = ("weight_layout", "global_num_tokens", "activation", "recipe")
RELEASED_BUFFER = ("mma_type", "activation", "base")


@contextlib.contextmanager
def _env(table=None, **values):
    """The switches cleared, then set as given; `table` is written as the override."""
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
            path.write_text(json.dumps(table))
            stack.enter_context(
                envs.SGLANG_K3_ROUTED_MOE_DEEPGEMM_VERIFY_WIDTHS.override(str(path))
            )
        policy.reset_caches()
        stack.callback(policy.reset_caches)
        yield


@contextlib.contextmanager
def _forward(verify=True):
    token = policy._verify.set((verify, False))
    try:
        yield
    finally:
        policy._verify.reset(token)


def _taken(verify=True):
    with _forward(verify):
        return [rows for rows in range(1, 300) if policy.takes(rows)]


def _layer(cfg=None, **overrides):
    """A K3.1 routed-expert layer as the rules read it; `cfg` overrides its config."""
    config = dict(
        activation="situ",
        gemm1_alpha=4.0,
        gemm1_clamp_limit=25.0,
        gate_up_interleaved=False,
    )
    config.update(cfg or {})
    weight = lambda *shape: types.SimpleNamespace(
        shape=shape, device=types.SimpleNamespace(index=0), element_size=lambda: 1
    )
    layer = dict(
        moe_runner_config=types.SimpleNamespace(**config),
        num_experts=896,
        num_local_experts=112,
        w13_weight=weight(112, 6144, 1792),
        w2_weight=weight(112, 3584, 1536),
    )
    layer.update(overrides)
    return types.SimpleNamespace(**layer)


def _fake_deep_gemm(call_keywords, buffer_keywords, version="0.0.0"):
    """A module with the signatures of a deep_gemm build, nothing else."""
    scope = {}
    exec(
        f"def fp8_fp4_mega_moe(y, l1_weights, l2_weights, sym_buffer, {', '.join(k + '=None' for k in call_keywords)}): pass\n"
        f"class SymmBuffer:\n"
        f"    def __init__(self, group, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden,"
        f" {', '.join(k + '=None' for k in buffer_keywords)}): pass\n",
        scope,
    )
    return types.SimpleNamespace(
        fp8_fp4_mega_moe=scope["fp8_fp4_mega_moe"],
        mega=types.SimpleNamespace(SymmBuffer=scope["SymmBuffer"]),
        get_num_locality_domains=lambda: 2,
        __version__=version,
    )


@contextlib.contextmanager
def _installed(module):
    """`module` as the deep_gemm the capability check finds (None: not importable)."""
    routed.missing_capability.cache_clear()
    with mock.patch.object(routed, "_deep_gemm", return_value=module):
        try:
            yield
        finally:
            routed.missing_capability.cache_clear()


class TestLayerRulesAndCall(CustomTestCase):
    def test_mismatch_reasons(self):
        for text, layer, with_bias, precision in (
            ("activation", _layer(dict(activation="silu")), False, "default"),
            ("SiTU", _layer(dict(gemm1_alpha=1.0)), False, "default"),
            ("interleaved", _layer(dict(gate_up_interleaved=True)), False, "default"),
            ("expert bias", _layer(), True, "default"),
            ("bf16", _layer(), False, "bf16"),
        ):
            self.assertIn(text, routed.mismatch(layer, with_bias, precision))
        # AR fusion is not a mismatch: the next rule in line answers
        with envs.SGLANG_K3_AR_FUSION.override(True), _installed(None):
            self.assertIn("deep_gemm", routed.mismatch(_layer(), False, "default"))

    def test_select_routing(self):
        ids, words, std = object(), object(), (object(), object())
        metadata = types.SimpleNamespace(expert_weights=object())
        # kimi_k3_fused_routing: plain ids + metadata weights, never the ids alone
        self.assertEqual(
            routed.select_routing(ids, (metadata, object()), lambda: std),
            (ids, metadata.expert_weights),
        )
        # route_quant_fused: packed words
        self.assertIs(routed.select_routing(words, None, lambda: std), words)
        # nothing prepared: the standard top-k
        self.assertIs(routed.select_routing(None, None, lambda: std), std)

    def test_call_keywords(self):
        with (
            mock.patch.object(routed._group, "first_expert", 336),
            mock.patch.object(routed._group, "stats", "stats"),
        ):
            small, large = routed._call_keywords(84), routed._call_keywords(85)
        # trtllm-gen: four accumulators up to 84 tokens, one chain above
        self.assertEqual(small["split4_combine_max_tokens"], 84)
        self.assertEqual(large["split4_combine_max_tokens"], 0)
        want = dict(
            weight_layout="trtllm",
            numerics="trtllm",
            topk_format="packed",
            topk_first_expert=336,
            activation="situ",
        )
        for keywords in (small, large):
            self.assertEqual({k: keywords[k] for k in want}, want)
            # every keyword the capability check asks for is one the call passes
            self.assertLessEqual(set(routed._CALL_KEYWORDS), set(keywords))


class TestCapability(CustomTestCase):
    def test_released_wheel_lacks_the_api(self):
        with _installed(_fake_deep_gemm(RELEASED_CALL, RELEASED_BUFFER, "0.2.9")):
            reason = routed.missing_capability()
            self.assertIn("deep_gemm 0.2.9 lacks", reason)
            for name in ("numerics", "topk_format", "topk_first_expert", "max_block_m"):
                self.assertIn(name, reason)
            # present in the release, for NVFP4 only
            self.assertNotIn("weight_layout", reason)
            self.assertEqual(routed.mismatch(_layer(), False, "default"), reason)
        # FP8 x FP4 `weight_layout='trtllm'` arrived with the buffer keywords: a build
        # without them does not qualify
        with _installed(_fake_deep_gemm(routed._CALL_KEYWORDS, RELEASED_BUFFER)):
            reason = routed.missing_capability()
            self.assertIn("exact_num_max_tokens", reason)
            self.assertNotIn("numerics", reason)

    def test_unusable_build_gives_a_reason_not_an_exception(self):
        # e.g. entry points behind a proxy whose lookup fails with another error type
        class Proxy:
            def __getattr__(self, name):
                raise RuntimeError(f"no op {name}")

        no_version = _fake_deep_gemm(RELEASED_CALL, RELEASED_BUFFER)
        del no_version.__version__
        for module, text in (
            (None, "deep_gemm cannot be imported"),
            (Proxy(), "no inspectable fp8_fp4_mega_moe"),
            (no_version, "deep_gemm unknown lacks"),
        ):
            with _installed(module):
                self.assertIn(text, routed.missing_capability())

    def test_capable_build_needs_two_locality_domains(self):
        module = _fake_deep_gemm(
            routed._CALL_KEYWORDS, routed._BUFFER_KEYWORDS + RELEASED_BUFFER
        )
        with (
            _installed(module),
            mock.patch("torch.distributed.is_initialized", return_value=True),
        ):
            self.assertIsNone(routed.missing_capability())
            self.assertIsNone(routed.mismatch(_layer(), False, "default"))
            for domains, text in (
                (lambda: 1, "locality domain"),
                (lambda: 1 / 0, "cannot report locality domains"),
            ):
                module.get_num_locality_domains = domains
                self.assertIn(text, routed.mismatch(_layer(), False, "default"))


class TestDispatch(CustomTestCase):
    def test_in_tree_table_is_what_the_module_dispatches(self):
        table = json.loads(policy._DEFAULT_TABLE.read_text())
        widths, fused = table["widths"], table["widths_ar_fusion"]
        # under AR fusion: the base widths above the fold limit, none at or below it
        self.assertEqual(table["ar_fusion_fold_max_rows"], FOLD_MAX_ROWS)
        self.assertEqual(fused, [rows for rows in widths if rows > FOLD_MAX_ROWS])
        for fusion, want in ((False, widths), (True, fused)):
            with _env(**ON, **{FUSION: fusion}):
                self.assertEqual(_taken(), want)

    def test_off_reads_nothing(self):
        # the override names a missing file: with the switch off nothing may read it
        missing = {PREFIX + "_VERIFY_WIDTHS": "/nonexistent/widths.json"}
        for extra in ({}, {BUNDLE: True}):
            with _env(**extra, **missing):
                self.assertFalse(policy.enabled())
                self.assertEqual(policy.verify_widths(), frozenset())
                self.assertEqual(_taken(), [])

    def test_target_forward_marks_verify_forwards_only_when_on(self):
        def forward(self, input_ids, positions, forward_batch):
            return policy.takes(80)

        batch = types.SimpleNamespace(
            forward_mode=types.SimpleNamespace(is_target_verify=lambda: True)
        )
        with _env():
            self.assertIs(policy.target_forward(forward), forward)
        with (
            _env(**ON, table=TABLE),
            mock.patch.object(torch.cuda, "is_current_stream_capturing", lambda: False),
        ):
            self.assertTrue(policy.target_forward(forward)(None, None, None, batch))
            self.assertFalse(policy.takes(80))  # reset after the forward

    def test_width_list_follows_the_switches(self):
        for switches, want in (
            ({FUSION: False}, BASE),
            ({FUSION: True}, AR_FUSION),
            ({BUNDLE: True}, AR_FUSION),
            ({FUSION: False, BUNDLE: True}, BASE),  # the AR fusion switch wins
            ({PREFIX + "_DISPATCH": False}, []),
        ):
            with self.subTest(switches=switches), _env(**ON, table=TABLE, **switches):
                self.assertEqual(_taken(), want)
                # prefill / extend / plain decode never dispatch
                self.assertEqual(_taken(verify=False), [])

    def test_bad_table_or_placement_fails_when_read(self):
        missing = {PREFIX + "_VERIFY_WIDTHS": "/nonexistent/widths.json"}
        with _env(**ON, **missing), self.assertRaises(FileNotFoundError):
            policy.verify_widths()
        for fused in ([], [80, 257], [80.0]):  # empty, above MAX_VERIFY_ROWS, not int
            bad = dict(widths=BASE, widths_ar_fusion=fused)
            with _env(**ON, table=bad, **{FUSION: True}), self.assertRaises(ValueError):
                policy.verify_widths()
        with _env(**ON, table=dict(widths=BASE), **{FUSION: True}):  # no fused list
            self.assertRaises(ValueError, policy.verify_widths)
        placement = {PREFIX + "_WEIGHT_PLACEMENT": "majority"}
        with _env(**ON, **placement), self.assertRaises(ValueError):
            policy.weight_placement()

    def test_handles(self):
        prepared = types.SimpleNamespace(mega_l1_weights=(object(), object()))
        with _env(**ON, table=TABLE, **{FUSION: True}):
            with _forward():
                for rows in (5, 20, 70, 80, 235):
                    # a layer left to trtllm-gen, which may defer its finalize
                    self.assertFalse(routed.handles(types.SimpleNamespace(), rows))
                    self.assertEqual(routed.handles(prepared, rows), rows in AR_FUSION)
            # outside target verify a prepared layer leaves every call to trtllm-gen
            self.assertFalse(any(routed.handles(prepared, rows) for rows in (80, 235)))


class TestFallback(CustomTestCase):
    """Switch on: what does not qualify keeps the default path and says so once."""

    def _wanted(self, layers, *, capability=None, sm=(10, 7), placement=None, env=None):
        warnings = []
        deep_gemm = types.SimpleNamespace(get_num_locality_domains=lambda: 2)
        with (
            _env(**ON, table=TABLE, **(env or {})),
            mock.patch.object(routed, "missing_capability", lambda: capability),
            mock.patch.object(torch.distributed, "is_initialized", lambda: True),
            mock.patch.object(routed, "_deep_gemm", lambda: deep_gemm),
            mock.patch.object(torch.cuda, "get_device_capability", lambda device: sm),
            mock.patch.object(policy, "_placement_unavailable", lambda *a: placement),
            mock.patch.object(policy, "print_warning_once", warnings.append),
        ):
            got = [policy.wanted(layer, False, "default") for layer in layers]
        return got, warnings

    def test_switch_off_asks_nothing(self):
        with _env(), mock.patch.object(routed, "mismatch") as mismatch:
            self.assertFalse(policy.wanted(_layer(), False, "default"))
            mismatch.assert_not_called()

    def test_qualifying_layer(self):
        self.assertEqual(self._wanted([_layer()]), ([True], []))
        # the default placement needs no placement API
        default = {PREFIX + "_WEIGHT_PLACEMENT": "default"}
        got = self._wanted([_layer()], placement="unavailable", env=default)
        self.assertEqual(got, ([True], []))

    def test_deep_gemm_without_the_api(self):
        reason = "deep_gemm 0.2.9 lacks the one-rank MegaMoE API (numerics)"
        got, warnings = self._wanted([_layer() for _ in range(3)], capability=reason)
        self.assertEqual(got, [False] * 3)
        # the same text for every layer: print_warning_once logs it once per process
        self.assertEqual(len(set(warnings)), 1)
        self.assertIn(reason, warnings[0])
        self.assertIn("SGLANG_K3_ROUTED_MOE_DEEPGEMM is set but not used", warnings[0])
        self.assertIn("default path", warnings[0])

    def test_other_hardware_and_models(self):
        no_api = "die-local placement is not available: no locality-domain API"
        for kwargs, layer, text in (
            (dict(sm=(10, 0)), _layer(), "SM 10.0"),
            (dict(), _layer(num_local_experts=448), "expert shape"),
            (dict(), _layer(num_experts=256), "expert shape"),
            (dict(placement=no_api), _layer(), "placement"),
        ):
            got, warnings = self._wanted([layer], **kwargs)
            self.assertEqual(got, [False])
            self.assertEqual(len(warnings), 1)
            self.assertIn(text, warnings[0])

    def test_placement_without_driver_bindings(self):
        # the real probe: no cuda.bindings locality-domain API reads as "not available"
        device = types.SimpleNamespace(index=0)
        error = RuntimeError("cuda.bindings has no locality-domain API")
        with mock.patch.object(policy, "_page_bytes", side_effect=error):
            reason = policy._placement_unavailable((112, 6144, 1792), 1, device)
        self.assertIn("not available", reason)
        with mock.patch.object(policy, "_page_bytes", return_value=2 << 20):
            self.assertIsNone(
                policy._placement_unavailable((112, 6144, 1792), 1, device)
            )
            reason = policy._placement_unavailable((112, 6144, 1793), 1, device)
            self.assertIn("does not split", reason)


class TestPageHomes(CustomTestCase):
    PAGE = 2 << 20

    def test_k31_shapes(self):
        for shape, pages in (((112, 6144, 1792), 588), ((112, 3584, 1536), 294)):
            homes = policy.page_homes(shape, 1, self.PAGE)
            self.assertEqual(len(homes), pages)
            self.assertEqual(homes.count(0), homes.count(1))
            # bytes read from the other domain's pages: a quarter, not the half of
            # alternating pages
            half = shape[1] * shape[2] // 2
            misplaced = 4096 * sum(
                1
                for page, home in enumerate(homes)
                for byte in range(page * self.PAGE, (page + 1) * self.PAGE, 4096)
                if byte // half % 2 != home
            )
            self.assertLess(misplaced / (len(homes) * self.PAGE), 0.26)

    def test_page_aligned_shapes(self):
        # page-aligned halves are exact
        self.assertEqual(
            policy.page_homes((3, 4, self.PAGE), 1, self.PAGE), [0, 0, 1, 1] * 3
        )
        # every page holds half of each domain's rows: homes alternate by page index
        self.assertEqual(
            policy.page_homes((4, 2, self.PAGE // 2), 1, self.PAGE), [0, 1, 0, 1]
        )


class TestAssumptions(CustomTestCase):
    """Each fails when the surrounding code moves: re-derive the table or load order."""

    def test_fold_threshold(self):
        state = types.SimpleNamespace(push_slot_bytes=1 << 30)
        with mock.patch.object(k3_ar_fusion, "_get_state", return_value=state):
            self.assertTrue(k3_ar_fusion.finalize_push_fits(FOLD_MAX_ROWS))
            self.assertFalse(k3_ar_fusion.finalize_push_fits(FOLD_MAX_ROWS + 1))

    def test_deferral_gate(self):
        from sglang.srt.models import kimi_k3

        source = inspect.getsource(kimi_k3.KimiK3MoE._forward_fused)
        start = source.index("defer_finalize = (")
        gate = source[start : source.index("\n            )\n", start)]
        # only the push-window rule and the MegaMoE check decide for a verify batch
        self.assertIn("k3_ar_fusion.finalize_push_fits(num_tokens)", gate)
        self.assertIn("not k3_deepgemm_routed.handles(self.experts, num_tokens)", gate)

    def test_load_keeps_the_trtllm_workspace(self):
        from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod

        source = inspect.getsource(Mxfp4MoEMethod.process_weights_after_loading)
        reserve = source.index("reserve_trtllm_mxfp4_moe_workspace(\n")
        finish = source.index("k3_die_local_moe.finish(")
        self.assertLess(reserve, finish)
        self.assertNotIn("return", source[reserve:finish])


if __name__ == "__main__":
    unittest.main()
