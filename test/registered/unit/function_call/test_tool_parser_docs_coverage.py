"""Every registered --tool-call-parser name needs a row in tool_parser.mdx, and
every name a row lists must still be registered.
"""

import re
import unittest
from pathlib import Path

from sglang.srt.function_call.parser_names import TOOL_CALL_PARSER_NAMES
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_DOC_PATH = (
    Path(__file__).resolve().parents[4]
    / "docs"
    / "docs"
    / "advanced_features"
    / "tool_parser.mdx"
)
_TABLE_HEADING = "## Currently supported parsers"

# Parser, Supported Models, Notes.
_DOC_COLUMNS = 3


def _doc_rows() -> list[list[str]]:
    text = _DOC_PATH.read_text(encoding="utf-8-sig")
    table = re.search(
        r"<tbody>(.*?)</tbody>", text[text.index(_TABLE_HEADING) :], re.S
    ).group(1)
    return [
        re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)
        for row in re.findall(r"<tr[^>]*>(.*?)</tr>", table, re.S)
    ]


class TestToolParserDocsCoverage(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = _doc_rows()
        # A row may list aliases in its first cell, e.g. `glm`, `glm45`.
        cls.documented = {
            name
            for cells in cls.rows
            if cells
            for name in re.findall(r"`([^`]+)`", cells[0])
        }

    def test_every_row_has_three_cells(self):
        """Coverage reads only the first cell. A row the cell regex splits wrongly
        can pass model or note text off as the Parser column and hide a gap."""
        misshapen = [
            (cells[0] if cells else "<no cells>")[:80]
            for cells in self.rows
            if len(cells) != _DOC_COLUMNS
        ]
        self.assertFalse(
            misshapen,
            f"these rows do not have {_DOC_COLUMNS} cells, so their first cell may "
            "not be the Parser column:\n  " + "\n  ".join(misshapen),
        )

    def test_every_registered_parser_is_documented(self):
        missing = sorted(set(TOOL_CALL_PARSER_NAMES) - self.documented)
        self.assertFalse(
            missing,
            "these --tool-call-parser names have no row in the supported parsers "
            f"table of {_DOC_PATH.name}. Add a row with the models the parser "
            "serves:\n  " + "\n  ".join(missing),
        )

    def test_no_stale_rows(self):
        stale = sorted(self.documented - set(TOOL_CALL_PARSER_NAMES))
        self.assertFalse(
            stale,
            f"{_DOC_PATH.name} lists these parsers but they are not in "
            "TOOL_CALL_PARSER_NAMES, so --tool-call-parser rejects them. Fix or "
            "delete the row:\n  " + "\n  ".join(stale),
        )


if __name__ == "__main__":
    unittest.main()
