"""Tests for `python/sglang/srt/state_capturer/draft_inventory.py`.

The inventory is the single source of truth the fail-closed runtime guard
consumes. These tests ensure:
  - every `_config_draft_model()` output is enumerated;
  - every on-disk `*Eagle*`, `*MTP*`, `*NextN*` `ForCausalLM` class is
    enumerated (or explicitly classified as non-architecture);
  - every entry's fields are well-formed (rationale non-empty, dense
    entries have `None` injection point, MoE entries have non-empty
    injection point).
"""

import re
import unittest
from pathlib import Path

from sglang.srt.state_capturer.draft_inventory import (
    INVENTORY,
    DraftInventoryEntry,
    lookup_draft_arch,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "python" / "sglang" / "srt" / "models"
MODEL_CONFIG_PATH = (
    REPO_ROOT / "python" / "sglang" / "srt" / "configs" / "model_config.py"
)


def _extract_draft_archs_from_config_source() -> set[str]:
    """Mine the `_config_draft_model()` source for every assignment of the
    form `self.hf_config.architectures[0] = "X"`. This is the authoritative
    set the inventory must cover; if the source list grows, the test fails
    until inventory grows too.
    """
    text = MODEL_CONFIG_PATH.read_text()
    pattern = re.compile(
        r'self\.hf_config\.architectures\[0\]\s*=\s*"([A-Za-z0-9_]+)"'
    )
    return set(pattern.findall(text))


def _discover_eagle_mtp_nextn_classes() -> set[str]:
    """Scan all model files for `class FooForCausalLM*` whose name carries
    `Eagle` / `MTP` / `NextN`. These are draft-architecture class names that
    might be loaded directly (EAGLE) or be produced by `_config_draft_model()`.
    Returning a set lets the test do `inventory >= discovered` set algebra.
    """
    found: set[str] = set()
    class_re = re.compile(r"^class\s+([A-Za-z0-9_]+)\(")
    for py in MODELS_DIR.glob("*.py"):
        try:
            text = py.read_text()
        except OSError:
            continue
        for match in class_re.finditer(text):
            name = match.group(1)
            if not name.endswith("CausalLM"):
                # Limit to top-level architecture class names. Inner blocks
                # like *Layer / *MoE / *Block are not architectures.
                if not (
                    name.endswith("CausalLMMTP")
                    or name.endswith("CausalLMNextN")
                    or name.endswith("CausalLMEagle")
                    or name.endswith("CausalLMEagle3")
                    or name == "MiMoMTP"
                    or name == "MiMoV2MTP"
                    or name == "Step3p5MTP"
                    or name.startswith("Eagle3")
                ):
                    continue
            if re.search(r"(Eagle|MTP|NextN)", name):
                found.add(name)
    return found


class InventoryShapeTest(unittest.TestCase):
    """AC-3: every entry well-formed."""

    def test_no_duplicate_draft_architectures(self):
        names = [e.draft_architecture for e in INVENTORY]
        self.assertEqual(len(names), len(set(names)))

    def test_lookup_returns_entry_for_each_listed(self):
        for entry in INVENTORY:
            looked_up = lookup_draft_arch(entry.draft_architecture)
            self.assertIs(looked_up, entry)

    def test_lookup_unknown_returns_none(self):
        self.assertIsNone(lookup_draft_arch("DefinitelyNotInThisInventory"))

    def test_each_entry_has_rationale(self):
        for entry in INVENTORY:
            self.assertTrue(
                entry.rationale.strip(),
                f"{entry.draft_architecture} has empty rationale",
            )

    def test_draft_signal_is_known_value(self):
        allowed = {"is_nextn", "is_mtp", "always_draft", "dense_no_topk"}
        for entry in INVENTORY:
            self.assertIn(entry.draft_signal, allowed)

    def test_dense_entries_have_no_injection_point(self):
        for entry in INVENTORY:
            if not entry.moe_bearing:
                self.assertIsNone(
                    entry.opt_out_injection_point,
                    f"{entry.draft_architecture} is dense but lists an "
                    f"injection point — drop it or flip moe_bearing",
                )
                self.assertEqual(
                    entry.draft_signal,
                    "dense_no_topk",
                    f"{entry.draft_architecture} is dense; draft_signal must "
                    f"be 'dense_no_topk'",
                )

    def test_moe_entries_have_injection_point(self):
        for entry in INVENTORY:
            if entry.moe_bearing:
                self.assertTrue(
                    entry.opt_out_injection_point
                    and entry.opt_out_injection_point.strip(),
                    f"{entry.draft_architecture} is MoE-bearing but has no "
                    f"opt_out_injection_point",
                )


class InventoryCoverageTest(unittest.TestCase):
    """AC-3 coverage cross-check: inventory must cover every
    `_config_draft_model()` output, and every discovered standalone
    EAGLE / MTP / NextN architecture class."""

    def test_covers_config_draft_model_outputs(self):
        source_archs = _extract_draft_archs_from_config_source()
        inventory_archs = {e.draft_architecture for e in INVENTORY}
        missing = source_archs - inventory_archs
        self.assertEqual(
            missing,
            set(),
            f"draft architectures produced by _config_draft_model() are "
            f"missing from inventory: {sorted(missing)}",
        )

    def test_covers_discovered_eagle_mtp_nextn_classes(self):
        discovered = _discover_eagle_mtp_nextn_classes()
        inventory_archs = {e.draft_architecture for e in INVENTORY}
        missing = discovered - inventory_archs
        # Exception: subclass-helper classes used internally are fine to
        # exclude (e.g. layer/block classes). The discovery filter already
        # narrows to `*CausalLM*`-style top-level classes; remaining
        # mismatches are real inventory gaps.
        self.assertEqual(
            missing,
            set(),
            f"on-disk draft architecture classes are missing from "
            f"inventory: {sorted(missing)}",
        )


if __name__ == "__main__":
    unittest.main()
