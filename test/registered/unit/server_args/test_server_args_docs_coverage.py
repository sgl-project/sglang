"""A safeguard that prevents the CLI and its documentation from getting out of sync.
Every implemented action must be documented and every documented action must exist
in the CLI.

Builds the real argparse surface (``ServerArgs.add_cli_args``) rather than
AST parsing the dataclass fields. This means ``cli_name=`` overrides, ``aliases=``
and argparse's generated ``--no-*`` forms all resolve for free. The check does
not care which module a field is declared in.

Two fixed sets of names (not counts). We track names rather than counts because a
count could stay the same even if one flag is documented and a different flag
becomes undocumented:

- ``_UNDOCUMENTED``: Actions with no row. May only shrink.
- ``_STALE_ROWS``: Documented flags that no longer register. May only shrink.

Each set is checked three ways (the ratchet idiom in
``test_global_config_read_ratchet.py``): a new member outside the set fails
("you broke coverage"), a member that no longer belongs fails ("lock in the
win by editing the set") and a member that is not a real member of the
current live/stale computation fails ("the set has drifted from reality or
you fixed it without updating the baseline").
"""

import argparse
import dataclasses
import re
import unittest
from pathlib import Path
from typing import Annotated, get_args, get_origin, get_type_hints

from sglang.srt.arg_groups.arg_utils import Arg
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

# Coverage and staleness are different questions, so they use different
# regexes on purpose. A symmetric matcher either misses a bare prose mention or
# promotes a prose wildcard like `--cuda-graph-*` into a fabricated flag name.
_ROW_FLAG_RE = re.compile(r"`(--[a-zA-Z0-9][\w-]*)`|<code>(--[a-zA-Z0-9][\w-]*)</code>")

# Actions allowed to have no docs row.
# This set may only shrink. See TestNoNewUndocumentedFlags below.
# Re-measure with the two functions in this file before editing it.
_UNDOCUMENTED = frozenset(
    {
        "--c128-page-size",
        "--decoupled-spec-bind-endpoint",
        "--decoupled-spec-connect-endpoints",
        "--decoupled-spec-rank",
        "--decoupled-spec-role",
        "--deepep-v2-mode",
        "--disaggregation-decode-extra-slots",
        "--disaggregation-decode-retraction-backup",
        "--dsa-paged-mqa-logits-backend",
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
        "--optimistic-prefill-attempts",
        "--prefill-decode-interval",
        "--radix-eviction-policy-config",
        "--return-input-ids",
        "--return-output-ids",
        "--sidecar",
        "--sidecar-args",
        "--smg-grpc-mode",
        "--spec-trace-dir",
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

# Rows naming a flag that no longer registers.
# This set may only shrink. See TestNoNewStaleRows below. Three of these ten
# (`--hybrid-kvcache-ratio`, `--debug-tensor-dump-inject`,
# `--optimistic-prefill-retries`) are claimed by open PRs #37556 / #37571 and
# the other seven by #38413. Drop each from this set as its fix merges.
_STALE_ROWS = frozenset(
    {
        "--custom-sigquit-handler",
        "--debug-tensor-dump-inject",
        "--dsa-prefill-cp-mode",
        "--enable-dsa-prefill-context-parallel",
        "--enable-nsa-prefill-context-parallel",
        "--enable-prefill-context-parallel",
        "--hybrid-kvcache-ratio",
        "--nsa-prefill-cp-mode",
        "--optimistic-prefill-retries",
        "--prefill-cp-mode",
    }
)


def _parser_actions():
    """(live, deprecated) argparse actions, ``-h``/``--help`` excluded."""
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    live, deprecated = [], []
    for action in parser._actions:
        options = [o for o in action.option_strings if o not in ("-h", "--help")]
        if not options:
            continue
        (deprecated if isinstance(action, _DEPRECATED_ACTION_TYPES) else live).append(
            action
        )
    return live, deprecated


def _doc_cells() -> list[str]:
    text = _DOC_PATH.read_text(encoding="utf-8-sig")
    return re.findall(r"<td[^>]*>(.*?)</td>", text, re.S)


def _uncovered(live_actions, cells_blob: str) -> set[str]:
    """Live actions with no spelling mentioned anywhere in a cell.
    The action is the unit and any of its option strings counts, with a
    trailing boundary so `--model` cannot match inside `--model-path`."""
    uncovered = set()
    for action in live_actions:
        options = [o for o in action.option_strings if o not in ("-h", "--help")]
        if not any(
            re.search(re.escape(opt) + r"(?![\w-])", cells_blob) for opt in options
        ):
            uncovered.add(action.option_strings[0])
    return uncovered


def _stale(cells: list[str], live_actions, deprecated_actions) -> set[str]:
    """Delimited flag mentions (backtick or ``<code>``) that name neither a
    live nor a deprecated registered action."""
    documented = set()
    for cell in cells:
        for match in _ROW_FLAG_RE.finditer(cell):
            documented.add(match.group(1) or match.group(2))
    registered = {
        o for a in live_actions + deprecated_actions for o in a.option_strings
    }
    return documented - registered


def _classify_stale(flag: str) -> str:
    """Why a documented flag no longer registers. Advisory only. This
    never gates pass/fail. A gap or a break in this classifier can only make
    the failure message less useful, never turn a real gap into a false
    green (see TestClassifierIsAdvisoryOnly)."""
    field_name = flag.lstrip("-").replace("-", "_")
    field = next(
        (f for f in dataclasses.fields(ServerArgs) if f.name == field_name), None
    )
    if field is None:
        return "no dataclass field at all -- renamed or removed"

    hint = get_type_hints(ServerArgs, include_extras=True).get(field_name)
    if get_origin(hint) is not Annotated:
        return "field exists with no A[] annotation -- never had a CLI surface"

    arg_meta = next((a for a in get_args(hint)[1:] if isinstance(a, Arg)), None)
    if arg_meta is not None and arg_meta.no_cli:
        return "field exists with Arg(no_cli=True) -- never had a CLI surface"

    return (
        "field exists with live-looking CLI metadata under this name -- the "
        "real spelling likely differs via cli_name=; check by hand"
    )


class TestServerArgsDocsCoverage(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.live_actions, cls.deprecated_actions = _parser_actions()
        cls.cells = _doc_cells()
        cls.cells_blob = "\n".join(cls.cells)

    def test_cell_parser_still_matches_the_doc(self):
        """An under matching regex fails loudly (everything reads
        missing) but an over matching one fails quietly, inflating the
        documented set and hiding real gaps, This is dangerous direction for a
        guardrail. Pin both a minimum cell count and one known row."""
        self.assertGreater(
            len(self.cells),
            1700,
            f"only {len(self.cells)} <td> cells recovered from {_DOC_PATH}. "
            "The cell regex no longer matches the table markup",
        )
        self.assertIn(
            "--model-path",
            self.cells_blob,
            "a flag known to be documented was not found in any cell. The "
            "cell regex is over or under matching",
        )

    def test_no_new_undocumented_flags(self):
        uncovered = _uncovered(self.live_actions, self.cells_blob)
        added = sorted(uncovered - _UNDOCUMENTED)
        self.assertFalse(
            added,
            "these live flags have no row in server_arguments.mdx and are not "
            "on the allow-list. Add a row (lift the help text from the "
            f"field's Arg(help=...)) or if it is a deliberate omission, add "
            f"it to _UNDOCUMENTED in {Path(__file__).name}:\n  " + "\n  ".join(added),
        )

    def test_undocumented_allowlist_only_shrinks(self):
        healed = sorted(_UNDOCUMENTED - _uncovered(self.live_actions, self.cells_blob))
        self.assertFalse(
            healed,
            "these flags are documented now but still listed in _UNDOCUMENTED "
            f"Remove them from the set in {Path(__file__).name} to lock in "
            "the win:\n  " + "\n  ".join(healed),
        )

    def test_undocumented_allowlist_has_no_ghosts(self):
        live_options = {a.option_strings[0] for a in self.live_actions}
        ghosts = sorted(_UNDOCUMENTED - live_options)
        self.assertFalse(
            ghosts,
            "these entries in _UNDOCUMENTED no longer name a live action. "
            f"Drop them from the set in {Path(__file__).name}, the flag is "
            "gone:\n  " + "\n  ".join(ghosts),
        )

    def test_no_new_stale_rows(self):
        stale = _stale(self.cells, self.live_actions, self.deprecated_actions)
        added = sorted(stale - _STALE_ROWS)
        if added:
            detail = "\n  ".join(f"{f} ({_classify_stale(f)})" for f in added)
            self.fail(
                "these rows in server_arguments.mdx name a flag that no "
                "longer registers, live or deprecated and are not on the "
                f"allow-list. Fix or delete the row or add it to "
                f"_STALE_ROWS in {Path(__file__).name}:\n  {detail}"
            )

    def test_stale_allowlist_only_shrinks(self):
        stale = _stale(self.cells, self.live_actions, self.deprecated_actions)
        healed = sorted(_STALE_ROWS - stale)
        self.assertFalse(
            healed,
            "these rows are no longer stale but still listed in _STALE_ROWS. "
            f"Remove them from the set in {Path(__file__).name} to lock in "
            "the win:\n  " + "\n  ".join(healed),
        )

    def test_stale_allowlist_has_no_ghosts(self):
        stale = _stale(self.cells, self.live_actions, self.deprecated_actions)
        ghosts = sorted(_STALE_ROWS - stale)
        self.assertFalse(
            ghosts,
            "these entries in _STALE_ROWS no longer correspond to an actual "
            f"stale row. Drop them from the set in {Path(__file__).name}:\n  "
            + "\n  ".join(ghosts),
        )


class TestClassifierIsAdvisoryOnly(CustomTestCase):
    """The four buckets steer the fix, never the verdict. A classifier
    that raised or returned something unexpected for every stale row would
    only degrade the failure message above. Pin that it stays total and
    string typed over the current baseline and not that its buckets are
    exhaustive which is exactly what the fourth catch-all bucket admits it
    is not."""

    def test_classifies_every_current_stale_row_without_raising(self):
        for flag in sorted(_STALE_ROWS):
            with self.subTest(flag=flag):
                verdict = _classify_stale(flag)
                self.assertIsInstance(verdict, str)
                self.assertTrue(verdict)

    def test_known_bucket_examples(self):
        # No field at all (renamed/removed): 9 of the 10 fall here.
        self.assertEqual(
            _classify_stale("--prefill-cp-mode"),
            "no dataclass field at all -- renamed or removed",
        )
        # Field exists, but was never annotated for CLI at all.
        self.assertEqual(
            _classify_stale("--custom-sigquit-handler"),
            "field exists with no A[] annotation -- never had a CLI surface",
        )


if __name__ == "__main__":
    unittest.main()
