"""Every live ServerArgs flag needs a row in server_arguments.mdx and every flag
the doc names must still register. The allowlist below may only shrink.
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

_DOC_PATH = (
    Path(__file__).resolve().parents[4]
    / "docs"
    / "docs"
    / "advanced_features"
    / "server_arguments.mdx"
)

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

# Live flags with no row. May only shrink.
_UNDOCUMENTED = frozenset(
    {
        "--c128-page-size",
        "--cuda-graph-prefill-max-context",
        "--decoupled-spec-bind-endpoint",
        "--decoupled-spec-connect-endpoints",
        "--decoupled-spec-rank",
        "--decoupled-spec-role",
        "--deepep-v2-mode",
        "--disaggregation-decode-extra-slots",
        "--disaggregation-decode-retraction-backup",
        "--disaggregation-enable-kv-checksum",
        "--dsa-paged-mqa-logits-backend",
        "--dsv4-attn-backend",
        "--dsv4-prefill-backend",
        "--dwdp-size",
        "--elastic-ep-initial-size",
        "--elastic-ep-join-mode",
        "--elastic-ep-join-rank-offset",
        "--elastic-ep-scale-timeout",
        "--enable-cp-decode-attn-tp",
        "--enable-dense-mlp-attn-tp",
        "--enable-dsa-cache-layer-split",
        "--enable-flexkv",
        "--enable-layernorm-sp",
        "--enable-lean-attention",
        "--enable-linear-replayssm",
        "--enable-linear-replayssm-spec",
        "--enable-response-store",
        "--enable-scattered-sconv",
        "--enable-session-radix-cache",
        "--enable-shared-experts-attn-tp",
        "--enable-tp-lm-head-all-to-all",
        "--enable-unified-cache-external-linker",
        "--enable-w4a4-mxfp4-megamoe",
        "--flexkv-config-file",
        "--fuseep-mode",
        "--gated-launch-port",
        "--grpc-port",
        "--hicache-host-memory-mode",
        "--hicache-storage-prefetch-retry-max-attempts",
        "--hicache-storage-prefetch-retry-poll-interval",
        "--http2-initial-connection-window-size",
        "--linear-attn-verify-backend",
        "--linear-replayssm-cache-len",
        "--mamba-max-states-per-path",
        "--max-ep-size",
        "--min-free-slots-delay",
        "--mm-feature-transport",
        "--mm-global-cache-backend",
        "--mm-io-worker-num",
        "--mm-preprocess-cache-size-mb",
        "--mm-processor-worker-num",
        "--otlp-service-name",
        "--prefill-decode-interval",
        "--radix-eviction-policy-config",
        "--return-input-ids",
        "--return-output-ids",
        "--sampling-mask-max-tokens",
        "--sidecar",
        "--sidecar-args",
        "--smg-grpc-mode",
        "--spec-trace-dir",
        "--speculative-domino-candidate-pool-size",
        "--speculative-draft-kv-cache-dtype",
        "--speculative-dspark-align-verify-tokens-to-graph-tier",
        "--speculative-dspark-block-size",
        "--speculative-dspark-confidence-sts-path",
        "--speculative-dspark-sps-table-path",
        "--speculative-use-rejection-sampling",
        "--startup-weight-load-mode",
        "--trust-mm-content-hashes",
        "--unified-cache-external-linker-backend",
        "--uno-lora-path",
        "--weight-cache-mode",
        "--weight-cache-socket",
        "--weight-cache-timeout",
    }
)


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
            f"it to _UNDOCUMENTED in {Path(__file__).name}:\n  " + "\n  ".join(added),
        )

    def test_undocumented_allowlist_is_current(self):
        live_options = {a.option_strings[0] for a in self.live_actions}
        outdated = []
        for flag in sorted(_UNDOCUMENTED - self.uncovered):
            reason = "now documented" if flag in live_options else "flag is gone"
            outdated.append(f"{flag} ({reason})")
        self.assertFalse(
            outdated,
            "these entries in _UNDOCUMENTED are out of date. Remove them from "
            f"the set in {Path(__file__).name}:\n  " + "\n  ".join(outdated),
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
