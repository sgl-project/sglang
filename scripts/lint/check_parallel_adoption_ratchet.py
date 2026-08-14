"""Guard: no legacy parallel-getter calls in the swept directories.

``models/`` and ``layers/`` read parallel topology through
``get_parallel().<dim>`` (the read-through wrapper in ``runtime_context``),
which gives one import, one naming scheme, and the scoped ``override()`` test
primitive. Exemptions are pinned in ``_EXEMPT``, each with its reason; sweeping
one must remove it from there.
"""

import re
from pathlib import Path

_SRT_ROOT = Path(__file__).resolve().parents[2] / "python" / "sglang" / "srt"

_BANNED_CALLS = re.compile(
    r"\b(?:dcp_enabled|get_(?:"
    r"tensor_model_parallel_(?:world_size|rank)"
    r"|pipeline_model_parallel_(?:world_size|rank)"
    r"|moe_expert_parallel_(?:world_size|rank)"
    r"|moe_tensor_parallel_(?:world_size|rank)"
    r"|moe_data_parallel_(?:world_size|rank)"
    r"|attn_tensor_model_parallel_(?:world_size|rank)"
    r"|attn_context_model_parallel_(?:world_size|rank)"
    r"|dcp_(?:world_size|rank)"
    r"|dcp_group(?:_no_assert)?"
    r"|attention_dcp_(?:world_size|rank)"
    r"|attention_(?:tp|cp)_(?:group|rank|size)"
    r"))\(\)"
)

# The whole package is swept; the exemptions are the substrate itself.
_SWEPT_DIRS = ("",)

_EXEMPT = (
    "distributed/",  # parallel_state: defines the canonical getters
    "runtime_context.py",  # delegates DCP reads to canonical getters
    "layers/dp_attention.py",  # delegation substrate for the attn-DP dims
    "layers/dcp/comm.py",  # deprecated out-of-tree DCP compatibility shims
    # The dumper's megatron plugin calls third-party getters that share the
    # parallel_state names (self._mpu.get_tensor_model_parallel_rank()).
    "debug_utils/dumper.py",
)


def check_parallel_adoption_ratchet():
    offenders = []
    for top in _SWEPT_DIRS:
        for path in sorted((_SRT_ROOT / top).rglob("*.py")):
            rel = path.relative_to(_SRT_ROOT).as_posix()
            if rel.startswith(_EXEMPT):
                continue
            for line_number, line in enumerate(path.read_text().split("\n"), 1):
                if _BANNED_CALLS.search(line):
                    offenders.append(f"{rel}:{line_number}")
    if offenders:
        raise AssertionError(
            "legacy parallel-getter calls in swept directories (use "
            f"get_parallel().<dim> instead): {offenders}",
        )


if __name__ == "__main__":
    check_parallel_adoption_ratchet()
