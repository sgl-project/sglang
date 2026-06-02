"""AST-level verification for per-family `allow_routed_experts_capture` opt-out.

Tighter than the regex-based `test_capture_optout_per_family.py` checks:
this suite parses each wrapper file's AST and confirms that the wrapper
class's `__init__` declares the opt-out signal explicitly, either as a
constructor keyword or as a wrapper-boundary assignment. A dropped signal
in the middle of a multi-line constructor invocation would slip past
`grep -n` but is caught here because the AST walker reads structured
nodes rather than file prose.

Construction-level runtime tests (instantiating each MoE block and
walking `model.modules()` for `TopK`) require substantial distributed-
state fixtures (CUDA streams, model-parallel groups, MoE backend
selection) that are not feasible on a CPU loop host. The AST checks plus
the runtime mock-capturer tests in `test_allow_routed_experts_capture_flag.py`
plus the GPU CI tests in `test_return_routed_experts_mtp.py` together
cover the contract on every layer that can be exercised here.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

from sglang.srt.state_capturer.draft_inventory import (
    INVENTORY,
    DraftInventoryEntry,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "python" / "sglang" / "srt" / "models"


# Per draft architecture: (wrapper_file, expected kwarg, expected literal value).
# `expected_value` is either the boolean True/False (matched against an
# `ast.Constant`) or a string identifier (matched against `ast.Name`).
_WRAPPER_KWARG_EXPECTATIONS = {
    # is_nextn=True / is_mtp=True pattern (shared block consumes the signal).
    "DeepseekV3ForCausalLMNextN": ("deepseek_nextn.py", "is_nextn", True),
    "DeepseekV4ForCausalLMNextN": ("deepseek_v4_nextn.py", "is_nextn", True),
    "Glm4MoeForCausalLMNextN": ("glm4_moe_nextn.py", "is_nextn", True),
    "BailingMoeForCausalLMNextN": ("bailing_moe_nextn.py", "is_nextn", True),
    "Qwen3NextForCausalLMMTP": ("qwen3_next_mtp.py", "is_nextn", True),
    "Qwen3_5ForCausalLMMTP": ("qwen3_5_mtp.py", "is_nextn", True),
    # allow_routed_experts_capture=False pattern (explicit opt-out kwarg).
    "ExaoneMoEForCausalLMMTP": ("exaone_moe_mtp.py", "allow_routed_experts_capture", False),
    "NemotronHForCausalLMMTP": ("nemotron_h_mtp.py", "allow_routed_experts_capture", False),
    "HYV3ForCausalLMNextN": ("hunyuan_v3_nextn.py", "allow_routed_experts_capture", False),
    "Step3p5MTP": ("step3p5_mtp.py", "allow_routed_experts_capture", False),
    "Gemma4AssistantForCausalLM": ("gemma4_mtp.py", "allow_routed_experts_capture", False),
}

_WRAPPER_ASSIGN_EXPECTATIONS = {
    "MistralLarge3ForCausalLMEagle": (
        "mistral_large_3_eagle.py",
        "allow_routed_experts_capture",
        False,
    ),
}


def _find_kwarg_in_module(
    module_ast: ast.Module, kwarg_name: str
) -> list[ast.keyword]:
    """Walk the AST and collect every `keyword` argument whose `arg`
    equals `kwarg_name`. Returns all matches across all call sites so
    downstream assertions can filter to the value they expect."""
    matches: list[ast.keyword] = []
    for node in ast.walk(module_ast):
        if isinstance(node, ast.keyword) and node.arg == kwarg_name:
            matches.append(node)
    return matches


def _keyword_value_matches(kw: ast.keyword, expected) -> bool:
    """`expected` is either a Python literal (bool / int / str / None) or
    a string identifier. We check against `ast.Constant` literals and
    `ast.Name` references."""
    val = kw.value
    if isinstance(val, ast.Constant):
        return val.value == expected
    if isinstance(val, ast.Name):
        return val.id == expected
    return False


def _find_attr_assignment_values(module_ast: ast.Module, attr_name: str) -> list[ast.AST]:
    """Collect values assigned to an attribute named `attr_name`."""
    values: list[ast.AST] = []
    for node in ast.walk(module_ast):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr == attr_name:
                    values.append(node.value)
        elif isinstance(node, ast.AnnAssign):
            if (
                isinstance(node.target, ast.Attribute)
                and node.target.attr == attr_name
                and node.value is not None
            ):
                values.append(node.value)
    return values


def _ast_value_matches(value: ast.AST, expected) -> bool:
    if isinstance(value, ast.Constant):
        return value.value == expected
    if isinstance(value, ast.Name):
        return value.id == expected
    return False


class WrapperSignalASTTest(unittest.TestCase):
    """AC-4: every MoE-bearing draft wrapper must declare an explicit
    opt-out signal where it constructs or adjusts the draft model component.
    The AST walk confirms the specific keyword or assignment is present,
    defending against a dropped signal that file-wide regex would still match
    because the literal appears elsewhere."""

    def test_each_wrapper_declares_expected_signal(self):
        moe_entries = [e for e in INVENTORY if e.moe_bearing]
        for entry in moe_entries:
            arch = entry.draft_architecture
            self.assertIn(
                arch,
                _WRAPPER_KWARG_EXPECTATIONS | _WRAPPER_ASSIGN_EXPECTATIONS,
                f"AST expectation table missing entry for MoE-bearing arch "
                f"{arch!r}; add it to the wrapper expectation tables so the "
                f"wrapper's opt-out signal can be verified at AST level",
            )

        for arch, (
            wrapper_name,
            expected_kwarg,
            expected_value,
        ) in _WRAPPER_KWARG_EXPECTATIONS.items():
            wrapper_path = MODELS_DIR / wrapper_name
            self.assertTrue(
                wrapper_path.is_file(),
                f"{arch}: wrapper file {wrapper_name!r} not found",
            )
            tree = ast.parse(wrapper_path.read_text())
            matches = _find_kwarg_in_module(tree, expected_kwarg)
            self.assertTrue(
                matches,
                f"{arch}: wrapper {wrapper_name!r} has no `{expected_kwarg}=` "
                f"call-site kwarg anywhere; the draft signal is missing",
            )
            matching_values = [
                kw for kw in matches if _keyword_value_matches(kw, expected_value)
            ]
            self.assertTrue(
                matching_values,
                f"{arch}: wrapper {wrapper_name!r} has `{expected_kwarg}=` "
                f"call-site kwargs but none with value {expected_value!r}; "
                f"the draft opt-out signal is wrong",
            )

        for arch, (
            wrapper_name,
            expected_attr,
            expected_value,
        ) in _WRAPPER_ASSIGN_EXPECTATIONS.items():
            wrapper_path = MODELS_DIR / wrapper_name
            self.assertTrue(
                wrapper_path.is_file(),
                f"{arch}: wrapper file {wrapper_name!r} not found",
            )
            tree = ast.parse(wrapper_path.read_text())
            values = _find_attr_assignment_values(tree, expected_attr)
            self.assertTrue(
                values,
                f"{arch}: wrapper {wrapper_name!r} has no assignment to "
                f"`*.{expected_attr}`; the draft opt-out signal is missing",
            )
            self.assertTrue(
                any(_ast_value_matches(value, expected_value) for value in values),
                f"{arch}: wrapper {wrapper_name!r} assigns `*.{expected_attr}`, "
                f"but none has expected value {expected_value!r}",
            )


class InventoryConsistencyASTTest(unittest.TestCase):
    """AC-3/AC-4 consistency: every MoE-bearing inventory entry must have
    a corresponding AST expectation, and `opted_out=True` everywhere."""

    def test_all_moe_entries_have_ast_expectation(self):
        moe_entries = [e for e in INVENTORY if e.moe_bearing]
        missing = [
            e.draft_architecture
            for e in moe_entries
            if e.draft_architecture
            not in (_WRAPPER_KWARG_EXPECTATIONS | _WRAPPER_ASSIGN_EXPECTATIONS)
        ]
        self.assertEqual(
            missing,
            [],
            f"AST expectations missing for inventory entries: {missing}",
        )

    def test_no_pending_moe_optout(self):
        pending = [
            e.draft_architecture
            for e in INVENTORY
            if e.moe_bearing and not e.opted_out
        ]
        self.assertEqual(
            pending,
            [],
            f"MoE-bearing inventory entries still have opted_out=False: "
            f"{pending}",
        )


class DenseWrapperASTTest(unittest.TestCase):
    """AC-5: dense allowlist wrapper files must not introduce TopK at the
    AST level. A `self.topk = TopK(...)` assignment in the wrapper file
    is a structural signal that the dense classification is wrong."""

    _DENSE_WRAPPERS = {
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

    def test_dense_wrappers_have_no_topk_call(self):
        """Walk each dense wrapper's AST for a `Call` node whose function
        is named `TopK` (either bare name or attribute access). Reject
        any match — dense wrappers should construct no MoE TopK."""
        for arch, wrapper_name in self._DENSE_WRAPPERS.items():
            wrapper_path = MODELS_DIR / wrapper_name
            if not wrapper_path.is_file():
                self.skipTest(
                    f"{arch}: wrapper {wrapper_name!r} missing - skipping"
                )
                continue
            tree = ast.parse(wrapper_path.read_text())
            topk_calls = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    fn = node.func
                    if isinstance(fn, ast.Name) and fn.id == "TopK":
                        topk_calls.append(node)
                    elif isinstance(fn, ast.Attribute) and fn.attr == "TopK":
                        topk_calls.append(node)
            self.assertEqual(
                topk_calls,
                [],
                f"{arch}: dense wrapper {wrapper_name!r} contains "
                f"{len(topk_calls)} TopK() call(s); flip inventory entry "
                f"to moe_bearing=True and plumb the opt-out, or fix the "
                f"wrapper to drop the MoE construction",
            )


if __name__ == "__main__":
    unittest.main()
