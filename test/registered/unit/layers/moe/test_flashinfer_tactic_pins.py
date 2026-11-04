"""CPU tests for pinned FlashInfer MoE tactics and the Qwen MoE prefill autotune hook."""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

from sglang.srt.environ import envs
from sglang.srt.layers.moe import flashinfer_tactic_pins as pins_module
from sglang.srt.layers.moe import prefill_autotune
from sglang.srt.layers.moe.flashinfer_tactic_pins import (
    MOE_OP,
    MoeTacticPins,
    parse_moe_tactic_pins,
    settle_decode_tactics,
    settle_prefill_tactics,
    valid_tactics_by_bucket,
)
from sglang.srt.model_executor.runner import flashinfer_autotune
from sglang.test.test_utils import CustomTestCase

SHAPE = (2560, 640, 512, 10)  # hidden, intermediate, experts, top_k


def _extras(hidden=2560, intermediate=640, experts=512, top_k=10, fused_shared=0):
    # MoERunner.get_cache_key_extras layout of FlashInfer 0.7.2.
    extras = [0] * 16
    extras[1], extras[2], extras[4] = top_k, experts, fused_shared
    extras[8], extras[10] = hidden, intermediate
    return tuple(extras)


def _key(bucket, *, op=MOE_OP, **shape):
    profile = [(bucket,)] * 5
    return SimpleNamespace(custom_op=op, nearest_profile=profile, extras=_extras(**shape))


METADATA = {
    "flashinfer_version": "0.7.2",
    "cuda_version": "13.0",
    "cublas_version": "13.1.0",
    "gpu": "NVIDIA B200",
    "sm": "10.0",
}
MOE_SECTION = dict(hidden_size=2560, intermediate_size=640, num_experts=512, top_k=10)
# Synthetic tactics: none of these is a tuned choice.
A, B, C, SPLIT = (64, 7), (64, 9), (256, 3), (4, 11)


def _pins(prefill=None, split_k=(), decode=None):
    return MoeTacticPins(
        flashinfer_version="0.7.2",
        shape=SHAPE,
        split_k=frozenset(split_k),
        prefill=prefill,
        decode=decode,
    )


class _HashableKey(SimpleNamespace):
    __hash__ = object.__hash__


def _winners(*entries):
    """{key: (tactic, profile)} the way FlashInfer's AutoTuner stores winners."""
    out = {}
    for bucket, tactic in entries:
        out[_HashableKey(**vars(_key(bucket)))] = (tactic, "profile")
    return out


class _TuningFileCase(CustomTestCase):
    def setUp(self):
        for cached in (pins_module.read_tuning_file, pins_module._moe_pins_of):
            cached.cache_clear()
            self.addCleanup(cached.cache_clear)

    def _file(self, document):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(document, f)
        self.addCleanup(os.unlink, f.name)
        return f.name


class TestPinFileParsing(_TuningFileCase):
    def test_parses_tables_and_rejects_malformed_entries(self):
        section = dict(MOE_SECTION, split_k_tactics=[list(SPLIT)])
        section["prefill"] = {"1024": list(A), "8192": -1}
        pins = parse_moe_tactic_pins({"_metadata": METADATA, "trtllm_fp4_block_scale_moe": section})
        self.assertEqual(pins.shape, SHAPE)
        self.assertEqual(pins.split_k, frozenset({SPLIT}))
        self.assertEqual(pins.prefill, {1024: A, 8192: -1})

        for bad_section in (
            {"prefill": {"1024": [64]}},
            {"prefill": {"1024": [64, -1]}},
            {"prefill": {"0": -1}},
            {"prefill": {"x": -1}},
            {"prefill": [[1024, 64, 7]]},
            {"split_k_tactics": [-1]},
            {"hidden_size": None},
        ):
            with self.subTest(bad_section=bad_section):
                document = {
                    "_metadata": METADATA,
                    "trtllm_fp4_block_scale_moe": dict(MOE_SECTION, **bad_section),
                }
                with self.assertRaises(ValueError):
                    parse_moe_tactic_pins(document)

    def test_every_metadata_field_is_required(self):
        for field in METADATA:
            metadata = {k: v for k, v in METADATA.items() if k != field}
            with self.subTest(missing=field), self.assertRaises(ValueError):
                parse_moe_tactic_pins(
                    {"_metadata": metadata, "trtllm_fp4_block_scale_moe": MOE_SECTION}
                )

    def test_pins_apply_only_on_the_exact_stack_and_device(self):
        # Tactic indices move between FlashInfer builds (post releases included),
        # cuBLAS and CUDA versions and SMs, so every field must match exactly.
        path = self._file({"_metadata": METADATA, "trtllm_fp4_block_scale_moe": MOE_SECTION})
        cases = [({}, True)] + [
            ({field: value}, False)
            for field, value in (
                ("flashinfer_version", "0.7.2.post1"),
                ("cuda_version", "13.1"),
                ("cublas_version", "13.1.1"),
                ("gpu", "NVIDIA B300"),
                ("sm", "10.3"),
            )
        ]
        with envs.SGLANG_FLASHINFER_TACTIC_PIN_FILE.override(path):
            for change, applies in cases:
                with self.subTest(change=change), mock.patch.object(
                    pins_module, "_current_metadata", return_value={**METADATA, **change}
                ):
                    self.assertEqual(pins_module.active_moe_tactic_pins() is not None, applies)
        self.assertIsNone(pins_module.active_moe_tactic_pins())

    def test_indeterminate_metadata_never_matches(self):
        # FlashInfer records "unknown" or "None" when it cannot probe a field, so
        # equal values there prove nothing about the stack.
        for value in ("unknown", "None"):
            unknown = {**METADATA, "cublas_version": value}
            path = self._file({"_metadata": unknown, "trtllm_fp4_block_scale_moe": MOE_SECTION})
            for cached in (pins_module.read_tuning_file, pins_module._moe_pins_of):
                cached.cache_clear()
            with self.subTest(value=value), envs.SGLANG_FLASHINFER_TACTIC_PIN_FILE.override(
                path
            ), mock.patch.object(pins_module, "_current_metadata", return_value=unknown):
                self.assertIsNone(pins_module.active_moe_tactic_pins())


class TestAutotuneCacheKey(_TuningFileCase):
    def _cache_path(self, cache_dir):
        import sys

        runner = SimpleNamespace(
            device="cuda",
            dtype=torch.bfloat16,
            is_draft_worker=False,
            model_config=SimpleNamespace(hf_config=SimpleNamespace()),
        )
        parallel = SimpleNamespace(
            tp_size=1, pp_size=1, attn_dp_size=1, moe_ep_size=1, tp_rank=0, pp_rank=0, dp_rank=0
        )
        with mock.patch.dict(sys.modules, {"flashinfer": SimpleNamespace(__version__="0.7.2")}), \
            mock.patch.object(torch.cuda, "get_device_capability", return_value=(10, 0)), \
            mock.patch.multiple(
                flashinfer_autotune,
                get_model=lambda: SimpleNamespace(model_path="m", quantization="q"),
                get_exec=lambda: SimpleNamespace(moe=SimpleNamespace(moe_runner_backend="flashinfer_trtllm")),
                get_parallel=lambda: parallel,
                get_flashinfer_autotune_skip_ops=lambda mr: set(),
            ), envs.SGLANG_CACHE_DIR.override(cache_dir):
            return flashinfer_autotune.flashinfer_autotune_cache_path(runner)

    def test_hook_and_tuning_file_get_their_own_cache(self):
        """With cache reuse on, a boot loads the saved tactics before any tuning, so
        settled tactics must not be found by a boot without the hook or the same file."""
        cache_dir = tempfile.mkdtemp()
        stock = self._cache_path(cache_dir)
        with envs.SGLANG_ENABLE_QWEN_MOE_PREFILL_AUTOTUNE.override(True):
            hook = self._cache_path(cache_dir)
        document = {"_metadata": METADATA, "trtllm_fp4_block_scale_moe": dict(
            MOE_SECTION, prefill={"1024": list(A)})}
        path = self._file(document)
        with envs.SGLANG_FLASHINFER_TACTIC_PIN_FILE.override(path):
            pinned = self._cache_path(cache_dir)
            # A cache saved under the first table, holding its pinned tactic.
            pinned.write_text(json.dumps({"_metadata": METADATA, "moe-key": ["MoERunner", list(A)]}))
            document["trtllm_fp4_block_scale_moe"]["prefill"] = {"1024": list(B)}
            with open(path, "w") as f:
                json.dump(document, f)
            changed = self._cache_path(cache_dir)
        self.assertEqual(len({stock, hook, pinned, changed}), 4)
        self.assertFalse(changed.exists(), "a changed table must not reuse the old cache")
        with envs.SGLANG_ENABLE_QWEN_MOE_PREFILL_AUTOTUNE.override(None):
            envs.SGLANG_ENABLE_QWEN_MOE_PREFILL_AUTOTUNE.clear()
            self.assertEqual(self._cache_path(cache_dir), stock)


class TestSettlePrefillTactics(CustomTestCase):
    def _settle(self, winners, pins, valid):
        before = dict(winners)
        rows = settle_prefill_tactics(winners, list(winners), pins=pins, valid=valid)
        changed = {k for k in winners if winners[k] != before[k]}
        return {row[0]: row[1:] for row in rows}, changed

    def test_without_pins_the_tuner_picks_stand(self):
        winners = _winners((768, list(SPLIT)), (1024, list(B)))
        rows, changed = self._settle(winners, None, {})
        self.assertEqual(changed, set())
        self.assertEqual(rows[768], (SPLIT, SPLIT, "tuner"))

    def test_a_table_falls_back_to_the_default_tactic_not_the_tuner(self):
        # An untuned prefill bucket runs the default tactic, so every bucket a
        # table cannot vouch for goes back to it rather than to the tuner's pick.
        winners = _winners(
            (768, list(SPLIT)), (1024, list(C)), (2048, list(C)), (4096, list(C)), (8192, list(C))
        )
        pins = _pins(
            prefill={768: A, 1024: -1, 2048: (64, 999), 4096: SPLIT},
            split_k={SPLIT},
        )
        valid = {768: {A}, 2048: {B}, 4096: {SPLIT}}
        rows, _ = self._settle(winners, pins, valid)
        self.assertEqual(rows[768][0::2], (A, "pin"))
        self.assertEqual(rows[1024][0::2], (-1, "pin_default"))
        self.assertEqual(rows[2048][0::2], (-1, "pin_invalid"))
        self.assertEqual(rows[4096][0::2], (-1, "pin_split_k"))
        self.assertEqual(rows[8192][0::2], (-1, "pin_absent"))
        stored = {k.nearest_profile[4][0]: v[0] for k, v in winners.items()}
        self.assertEqual(stored, {768: list(A), 1024: -1, 2048: -1, 4096: -1, 8192: -1})

    def test_split_k_is_excluded_from_tuner_picks_too(self):
        winners = _winners((768, list(SPLIT)), (1024, list(B)))
        rows, _ = self._settle(winners, _pins(split_k={SPLIT}), {})
        self.assertEqual(rows[768][0::2], (-1, "tuner_split_k"))
        self.assertEqual(rows[1024][0::2], (B, "tuner"))

    def test_another_shape_keeps_the_tuner_pick(self):
        key = _HashableKey(**vars(_key(768, experts=256)))
        winners = {key: (list(SPLIT), "profile")}
        rows, changed = self._settle(winners, _pins(prefill={}, split_k={SPLIT}), {})
        self.assertEqual(changed, set())
        self.assertEqual(rows[768][0::2], (SPLIT, "tuner"))

    def test_valid_tactics_are_read_per_bucket_for_the_pinned_shape(self):
        def query(num_tokens, top_k=10, hidden=2560, inter=640, experts=512):
            return (0, 0, 0, top_k, hidden, hidden, inter, experts, 0, 0, 0, 0, num_tokens, 0)

        valid = valid_tactics_by_bucket(
            {
                ("runner", query(768)): [list(A), [-1, -1]],
                ("runner", query(768, experts=256)): [list(B)],
                ("runner", query(1024)): [C],
            },
            SHAPE,
        )
        self.assertEqual(valid, {768: {A, -1}, 1024: {C}})


class TestDecodeTactics(CustomTestCase):
    def test_decode_buckets_must_be_powers_of_two_up_to_512(self):
        section = dict(MOE_SECTION, decode={"4": list(A), "512": -1})
        document = {"_metadata": METADATA, "trtllm_fp4_block_scale_moe": section}
        self.assertEqual(parse_moe_tactic_pins(document).decode, {4: A, 512: -1})
        for bucket in ("3", "1024"):
            section["decode"] = {bucket: -1}
            with self.subTest(bucket=bucket), self.assertRaises(ValueError):
                parse_moe_tactic_pins(document)

    def test_refused_pins_keep_the_tuner_pick(self):
        # Decode buckets are tuned at startup, so unlike prefill the stock choice a
        # refused pin falls back to is the tuner's pick, not the default tactic.
        winners = _winners(
            (4, list(C)), (8, list(B)), (16, list(B)), (32, list(C)), (64, list(B))
        )
        other_shape = _HashableKey(**vars(_key(4, intermediate=1024)))
        winners[other_shape] = (list(C), "profile")
        pins = _pins(
            decode={4: A, 8: SPLIT, 16: (64, 999), 32: -1, 128: A},
            split_k={SPLIT},
        )
        valid = {4: {A}, 8: {SPLIT}, 16: {B}}
        rows = settle_decode_tactics(winners, pins=pins, valid=valid)
        sources = sorted((row[0], row[3]) for row in rows)
        self.assertEqual(
            sources,
            [(4, "pin"), (4, "pin_shape_skip"), (8, "pin_split_k_refused"),
             (16, "pin_invalid"), (32, "pin_default")],
        )
        stored = {
            (k.nearest_profile[4][0], k is other_shape): v[0] for k, v in winners.items()
        }
        self.assertEqual(
            stored,
            {(4, False): list(A), (4, True): list(C), (8, False): list(B),
             (16, False): list(B), (32, False): -1, (64, False): list(B)},
        )

    def test_decode_pins_are_settled_before_the_cache_is_saved(self):
        # With cache reuse the next boot serves the saved tactics without tuning,
        # so the pins must be in place when the autotune context saves on exit.
        events = []

        class _Context:
            def __enter__(self):
                events.append("enter")

            def __exit__(self, *exc):
                events.append("save")

        with mock.patch.object(
            flashinfer_autotune, "flashinfer_autotune_context", lambda *a, **k: _Context()
        ), mock.patch.object(
            pins_module, "pin_decode_moe_tactics", side_effect=lambda: events.append("pin")
        ):
            with envs.SGLANG_FLASHINFER_TACTIC_PIN_FILE.override(None):
                flashinfer_autotune.run_flashinfer_autotune_forward(
                    None, lambda: events.append("tune"), run_lm_head=False
                )
            self.assertEqual(events, ["enter", "tune", "save"])
            events.clear()
            with envs.SGLANG_FLASHINFER_TACTIC_PIN_FILE.override("/tuning.json"):
                flashinfer_autotune.run_flashinfer_autotune_forward(
                    None, lambda: events.append("tune"), run_lm_head=False
                )
            self.assertEqual(events, ["enter", "tune", "pin", "save"])


class TestPrefillAutotuneHook(CustomTestCase):
    def _runner(self, model):
        return SimpleNamespace(
            model_runner=SimpleNamespace(
                model=model, is_generation=True, is_draft_worker=False, dtype=torch.bfloat16
            )
        )

    def _model(self):
        from sglang.srt.models.qwen4_exp import Qwen4ExpForConditionalGeneration

        return Qwen4ExpForConditionalGeneration.__new__(Qwen4ExpForConditionalGeneration)

    def test_off_by_default_the_extend_autotune_stays_stock(self):
        model = self._model()
        context = mock.MagicMock(side_effect=AssertionError("autotune context entered"))
        with envs.SGLANG_ENABLE_QWEN_MOE_PREFILL_AUTOTUNE.override(False), mock.patch.multiple(
            flashinfer_autotune,
            flashinfer_autotune_context=context,
            max_prefill_buffer_tokens=lambda: 16384,
        ), mock.patch.object(
            prefill_autotune, "_eligible_block", side_effect=AssertionError("model scanned")
        ):
            flashinfer_autotune.maybe_flashinfer_autotune_extend(
                self._runner(model), decode_num_tokens=512
            )
        context.assert_not_called()

    def test_on_the_runner_calls_the_hook_at_the_prefill_ceiling(self):
        model = self._model()
        calls = []
        backend = SimpleNamespace(is_flashinfer_trtllm=lambda: True, is_none=lambda: True)
        with envs.SGLANG_ENABLE_QWEN_MOE_PREFILL_AUTOTUNE.override(True), mock.patch.multiple(
            flashinfer_autotune,
            flashinfer_autotune_context=mock.MagicMock(),
            max_prefill_buffer_tokens=lambda: 16384,
        ), mock.patch.object(
            prefill_autotune, "_eligible_block", return_value=object()
        ), mock.patch.object(
            prefill_autotune,
            "autotune_qwen_moe_prefill",
            side_effect=lambda m, n, dtype: calls.append((n, dtype)) or 1,
        ), mock.patch(
            "sglang.srt.layers.moe.utils.get_moe_runner_backend", return_value=backend
        ), mock.patch(
            "sglang.srt.layers.moe.utils.get_moe_a2a_backend", return_value=backend
        ), mock.patch(
            "sglang.srt.layers.dp_attention.is_dp_attention_enabled", return_value=False
        ):
            flashinfer_autotune.maybe_flashinfer_autotune_extend(
                self._runner(model), decode_num_tokens=512
            )
        self.assertEqual(calls, [(16384, torch.bfloat16)])


if __name__ == "__main__":
    unittest.main()
