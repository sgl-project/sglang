"""Every live ServerArgs flag needs a row in server_arguments.mdx and every flag
the doc names must still register. The allowlist may only shrink.
"""

import argparse
import re
import unittest
from collections.abc import Iterable
from pathlib import Path

from sglang.srt.arg_groups.argparse_actions import (
    DeprecatedAction,
    DeprecatedAliasStoreAction,
    DeprecatedStoreConstAction,
    DeprecatedStoreTrueAction,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_DOC_DIR = Path(__file__).resolve().parents[4] / "docs" / "docs" / "advanced_features"
_DOC_PATH = _DOC_DIR / "server_arguments.mdx"
# Live flags with no row. May only shrink. Kept beside the doc, outside the
# main_package CI paths, so a PR that backfills rows runs only the CPU stage.
_UNDOCUMENTED_PATH = _DOC_DIR / "server_arguments_undocumented.txt"

_DEPRECATED_ACTION_TYPES = (
    DeprecatedAction,
    DeprecatedStoreTrueAction,
    DeprecatedStoreConstAction,
    DeprecatedAliasStoreAction,
)

# Argument, Description, Defaults, Options.
_DOC_COLUMNS = 4

# Only delimited mentions count, so a prose wildcard like `--cuda-graph-*`
# never becomes a fabricated flag name.
_ROW_FLAG_RE = re.compile(r"`(--[a-zA-Z0-9][\w-]*)`|<code>(--[a-zA-Z0-9][\w-]*)</code>")


def _undocumented() -> frozenset[str]:
    lines = _UNDOCUMENTED_PATH.read_text(encoding="utf-8").splitlines()
    return frozenset(
        line.strip() for line in lines if line.strip() and not line.startswith("#")
    )


_UNDOCUMENTED = _undocumented()


def _parser_actions() -> tuple[list[argparse.Action], list[argparse.Action]]:
    parser = argparse.ArgumentParser(add_help=False)
    ServerArgs.add_cli_args(parser)
    live, deprecated = [], []
    for action in parser._actions:
        (deprecated if isinstance(action, _DEPRECATED_ACTION_TYPES) else live).append(
            action
        )
    return live, deprecated


def _doc_rows() -> list[list[str]]:
    text = _DOC_PATH.read_text(encoding="utf-8-sig")
    rows = [
        re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)
        for row in re.findall(r"<tr[^>]*>(.*?)</tr>", text, re.S)
    ]
    # Header rows hold only <th> cells.
    return [cells for cells in rows if cells]


def _flags_in(cells: Iterable[str]) -> set[str]:
    return {
        match.group(1) or match.group(2)
        for cell in cells
        for match in _ROW_FLAG_RE.finditer(cell)
    }


class TestServerArgsDocsCoverage(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.live_actions, deprecated_actions = _parser_actions()
        cls.rows = _doc_rows()
        argument_column_flags = _flags_in(cells[0] for cells in cls.rows)
        cls.uncovered = {
            action.option_strings[0]
            for action in cls.live_actions
            # Any alias counts.
            if argument_column_flags.isdisjoint(action.option_strings)
        }
        registered = {
            o for a in cls.live_actions + deprecated_actions for o in a.option_strings
        }
        cls.stale = _flags_in(cell for cells in cls.rows for cell in cells) - registered

    def test_every_row_has_four_cells(self):
        """Coverage reads only the first cell. A row the cell regex splits wrongly
        can pass description text off as the Argument column and hide a gap."""
        misshapen = [cells[0][:80] for cells in self.rows if len(cells) != _DOC_COLUMNS]
        self.assertFalse(
            misshapen,
            f"these rows do not have {_DOC_COLUMNS} cells, so their first cell may "
            "not be the Argument column:\n  " + "\n  ".join(misshapen),
        )

    def test_no_new_undocumented_flags(self):
        added = sorted(self.uncovered - _UNDOCUMENTED)
        self.assertFalse(
            added,
            "these live flags have no row in server_arguments.mdx and are not "
            "on the allow list. Add a row (lift the help text from the "
            "field's Arg(help=...)) or if it is a deliberate omission, add "
            f"it to {_UNDOCUMENTED_PATH.name}:\n  " + "\n  ".join(added),
        )

    def test_undocumented_allowlist_is_current(self):
        live_options = {a.option_strings[0] for a in self.live_actions}
        outdated = []
        for flag in sorted(_UNDOCUMENTED - self.uncovered):
            reason = "now documented" if flag in live_options else "flag is gone"
            outdated.append(f"{flag} ({reason})")
        self.assertFalse(
            outdated,
            f"these entries in {_UNDOCUMENTED_PATH.name} are out of date. "
            "Remove them from the file:\n  " + "\n  ".join(outdated),
        )

    def test_no_stale_rows(self):
        stale = sorted(self.stale)
        self.assertFalse(
            stale,
            "server_arguments.mdx names these flags but they no longer register "
            "as a current or deprecated flag. The flag was renamed (possibly via "
            "cli_name=) or removed, or its field has no CLI surface (no A[] "
            "annotation, or Arg(no_cli=True)). Fix or delete the mention:\n  "
            + "\n  ".join(stale),
        )


if __name__ == "__main__":
    unittest.main()
