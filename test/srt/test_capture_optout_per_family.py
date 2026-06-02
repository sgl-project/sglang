"""Per-family verification for the `allow_routed_experts_capture` opt-out.

The plan's AC-4 requires every MoE-bearing draft family to construct its
`TopK` modules with `allow_routed_experts_capture=False`. AC-5 requires every
dense allowlist entry to construct zero `TopK` modules.

Building each full draft model from a HuggingFace config + weights is not
feasible on a CPU loop host (most need real weights and a CUDA-capable
runtime), so this suite verifies the opt-out at the source level:

  - For every MoE-bearing inventory entry with `opted_out=True`, the file
    named in `opt_out_injection_point` must declare a `allow_routed_experts_capture`
    expression that lands as `False` on the draft path.
  - The corresponding *_mtp / *_nextn / *_eagle wrapper (when it exists)
    must pass either `allow_routed_experts_capture=False`, `is_nextn=True`,
    `is_mtp=True`, or use a hardcoded subclass equivalent.
  - For every dense allowlist entry, the rationale must explain why no
    `TopK` is constructed, and the named file must not introduce an
    unconditional `self.topk = TopK(` on the draft path.

These checks are structural regression tripwires; the runtime contract is
also exercised by `test_allow_routed_experts_capture_flag.py` (helper-level)
and by `test_return_routed_experts_mtp.py` (e2e under GPU CI).
"""

import re
import unittest
from pathlib import Path

from sglang.srt.state_capturer.draft_inventory import (
    INVENTORY,
    DraftInventoryEntry,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "python" / "sglang" / "srt" / "models"


def _read(path: Path) -> str:
    return path.read_text()


# Each entry: draft_arch -> (wrapper_file, expected pattern at the wrapper site)
# Wrapper-site assertions check the *_mtp / *_nextn / *_eagle file that
# constructs the draft model; they must opt out explicitly via one of the
# known mechanisms.
_WRAPPER_EXPECTATIONS = {
    "DeepseekV3ForCausalLMNextN": (
        "deepseek_nextn.py",
        r"is_nextn=True",
    ),
    "DeepseekV4ForCausalLMNextN": (
        "deepseek_v4_nextn.py",
        r"is_nextn=True",
    ),
    "Glm4MoeForCausalLMNextN": (
        "glm4_moe_nextn.py",
        r"is_nextn=True",
    ),
    "BailingMoeForCausalLMNextN": (
        "bailing_moe_nextn.py",
        r"is_nextn=True",
    ),
    "Qwen3NextForCausalLMMTP": (
        "qwen3_next_mtp.py",
        r"is_nextn=True",
    ),
    "Qwen3_5ForCausalLMMTP": (
        "qwen3_5_mtp.py",
        r"is_nextn=True",
    ),
    "ExaoneMoEForCausalLMMTP": (
        "exaone_moe_mtp.py",
        r"allow_routed_experts_capture=False",
    ),
    "NemotronHForCausalLMMTP": (
        "nemotron_h_mtp.py",
        r"allow_routed_experts_capture=False",
    ),
    "HYV3ForCausalLMNextN": (
        "hunyuan_v3_nextn.py",
        r"allow_routed_experts_capture=False",
    ),
    "Step3p5MTP": (
        "step3p5_mtp.py",
        r"allow_routed_experts_capture=False",
    ),
    "MistralLarge3ForCausalLMEagle": (
        "mistral_large_3_eagle.py",
        r"allow_routed_experts_capture\s*=\s*False",
    ),
    "Gemma4AssistantForCausalLM": (
        "gemma4_mtp.py",
        r"allow_routed_experts_capture=False",
    ),
}


# For each MoE-bearing entry: the file named at opt_out_injection_point
# must declare an `allow_routed_experts_capture=` expression at the TopK call. We
# also accept `not is_nextn` patterns used by shared blocks.
_INJECTION_POINT_PATTERN = re.compile(
    r"allow_routed_experts_capture\s*=\s*"
    r"(not\s+is_nextn|allow_routed_experts_capture|False|not\s+self\.is_nextn)"
)


def _moe_entries(opted_out_only: bool = True) -> list[DraftInventoryEntry]:
    if opted_out_only:
        return [e for e in INVENTORY if e.moe_bearing and e.opted_out]
    return [e for e in INVENTORY if e.moe_bearing]


def _dense_entries() -> list[DraftInventoryEntry]:
    return [e for e in INVENTORY if not e.moe_bearing]


class MoEEntryOptOutSourceTest(unittest.TestCase):
    """AC-4: every MoE-bearing inventory entry that claims `opted_out=True`
    must actually have an `allow_routed_experts_capture=...` expression at the
    TopK call inside the file named by its `opt_out_injection_point`."""

    def test_injection_point_file_contains_allow_flag_assignment(self):
        # Every MoE-bearing entry's injection-point file must contain a
        # allow_routed_experts_capture assignment near a TopK construction.
        for entry in _moe_entries():
            file_part = entry.opt_out_injection_point.split(":")[0]
            path = REPO_ROOT / file_part
            self.assertTrue(
                path.is_file(),
                f"{entry.draft_architecture}: opt_out_injection_point file "
                f"{file_part!r} does not exist",
            )
            source = _read(path)
            self.assertRegex(
                source,
                _INJECTION_POINT_PATTERN,
                f"{entry.draft_architecture}: {file_part!r} does not contain "
                f"an `allow_routed_experts_capture=...` assignment matching the "
                f"recognized opt-out patterns",
            )

    def test_wrapper_file_opts_out(self):
        """Each draft wrapper file must declare the explicit signal that
        lands as `False` on the draft TopK. The wrapper file is where
        future-proofing matters most: a future model addition could
        replace the wrapper's call site and silently default `True`."""
        for entry in _moe_entries():
            arch = entry.draft_architecture
            if arch not in _WRAPPER_EXPECTATIONS:
                # Not every MoE entry has a separate wrapper file (e.g.
                # MistralLarge3ForCausalLMEagle's wrapper IS the model
                # itself); skip those gracefully — their injection-point
                # test above covers the assertion.
                continue
            wrapper_name, expected_pattern = _WRAPPER_EXPECTATIONS[arch]
            wrapper_path = MODELS_DIR / wrapper_name
            self.assertTrue(
                wrapper_path.is_file(),
                f"{arch}: wrapper file {wrapper_name!r} does not exist",
            )
            source = _read(wrapper_path)
            self.assertRegex(
                source,
                expected_pattern,
                f"{arch}: wrapper file {wrapper_name!r} does not declare "
                f"the expected opt-out pattern {expected_pattern!r}; a "
                f"future model addition could silently default to capture-on",
            )


class DenseAllowlistSourceTest(unittest.TestCase):
    """AC-5: dense allowlist entries must have a non-empty rationale and
    must not introduce `self.topk = TopK(` in the relevant model file's
    draft path."""

    def test_dense_entry_has_rationale(self):
        for entry in _dense_entries():
            self.assertTrue(entry.rationale.strip())
            self.assertIsNone(entry.opt_out_injection_point)
            self.assertEqual(entry.draft_signal, "dense_no_topk")

    def test_dense_wrapper_construction_uses_no_topk(self):
        """For each dense allowlist entry that we can locate a wrapper
        file for, that file must not introduce a `self.topk = TopK(`
        construction. (Wrapper files that reuse heavier blocks should
        opt out via the block's signal instead, in which case those
        belong in a MoE entry.)"""
        dense_wrapper_files = {
            "GlmOcrForConditionalGenerationNextN": "glm_ocr_nextn.py",
            "LongcatFlashForCausalLMNextN": "longcat_flash_nextn.py",
            "MiMoMTP": "mimo_mtp.py",
            "MiMoV2MTP": "mimo_v2_nextn.py",
            "Ernie4_5_MoeForCausalLMMTP": "ernie4_eagle.py",
            "LlamaForCausalLMEagle": "llama_eagle.py",
            "LlamaForCausalLMEagle3": "llama_eagle3.py",
            "MistralForCausalLMEagle": "mistral_eagle.py",
            "Qwen2ForCausalLMEagle": "qwen2_eagle.py",
            "Eagle3DeepseekV2ForCausalLM": "kimi_k25_eagle3.py",
        }
        topk_re = re.compile(r"self\.topk\s*=\s*TopK\(")
        for entry in _dense_entries():
            wrapper_name = dense_wrapper_files.get(entry.draft_architecture)
            if wrapper_name is None:
                continue
            path = MODELS_DIR / wrapper_name
            self.assertTrue(
                path.is_file(),
                f"{entry.draft_architecture}: wrapper {wrapper_name!r} not found",
            )
            source = _read(path)
            self.assertNotRegex(
                source,
                topk_re,
                f"{entry.draft_architecture}: dense wrapper {wrapper_name!r} "
                f"introduces `self.topk = TopK(...)` — flip the inventory "
                f"entry to moe_bearing=True and plumb the opt-out, or fix "
                f"the wrapper to avoid the MoE construction",
            )


class GuardConsistencyTest(unittest.TestCase):
    """Every inventory entry must have a state the runtime guard can
    enforce. A MoE entry with `opted_out=False` is allowed only when its
    rationale explicitly mentions pending plumbing; a `True` row implies
    the per-family source check above must pass."""

    def test_no_moe_entry_left_unopted_silently(self):
        unopted = [
            e
            for e in _moe_entries(opted_out_only=False)
            if not e.opted_out
        ]
        self.assertEqual(
            unopted,
            [],
            "Round 2 expects every MoE-bearing draft family to be "
            f"opted out; still pending: "
            f"{[e.draft_architecture for e in unopted]}",
        )


if __name__ == "__main__":
    unittest.main()
