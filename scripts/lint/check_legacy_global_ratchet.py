"""Ratchet guard: legacy global-accessor call-sites may only decrease.

The process-wide ``ServerArgs`` is owned by the runtime context; the legacy
``get_global_server_args`` / ``set_global_server_args_for_*`` names survive as
thin shims for the existing call-sites. New code should use the
``sglang.srt.runtime_context`` accessors (``get_server_args()`` /
``get_context().set_server_args()``), so the shim call-site counts below must
never grow. When your change removes call-sites, lower the matching baseline
to the new count.
"""

import re
from pathlib import Path

_SRT_ROOT = Path(__file__).resolve().parents[2] / "python" / "sglang" / "srt"

# Baselines counted over python/sglang/srt/**/*.py, including each function's
# own def line. Ratchet: decrease-only.
_RATCHETS = [
    # Down to the shim definition itself; every call-site now goes through
    # runtime_context.get_server_args().
    ("get_global_server_args", r"\bget_global_server_args\s*\(", 1),
    (
        "set_global_server_args_for_*",
        r"\bset_global_server_args_for_(?:scheduler|tokenizer)\s*\(",
        4,
    ),
]


def check_legacy_global_ratchet():
    sources = [
        path.read_text(encoding="utf-8", errors="replace")
        for path in sorted(_SRT_ROOT.rglob("*.py"))
    ]
    for name, pattern, baseline in _RATCHETS:
        count = sum(len(re.findall(pattern, source)) for source in sources)
        if count > baseline:
            raise AssertionError(
                f"{name} call-sites grew: {count} > baseline {baseline}. "
                "New code must use the sglang.srt.runtime_context accessors "
                "(get_server_args() / get_context().set_server_args())."
            )
        if count < baseline:
            raise AssertionError(
                f"{name} call-sites shrank: {count} < baseline {baseline}. "
                "Lower the baseline in this file to lock in the progress."
            )


if __name__ == "__main__":
    check_legacy_global_ratchet()
