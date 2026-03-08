from __future__ import annotations

import re
from typing import NamedTuple, Optional

from sglang.srt.debug_utils.comparator.dims_spec.types import (
    _AXIS_LOOKUP,
    ParallelAxis,
)

_DP_ALIAS_PATTERN = re.compile(r"^dp:=(\w+)$")
_REPLICATED_PATTERN = re.compile(r"^(\w+):replicated$")


class _CommentSuffix(NamedTuple):
    dp_group_alias: Optional[str] = None
    replicated_axes: frozenset[ParallelAxis] = frozenset()


def _parse_comment_suffix(declaration_part: str) -> _CommentSuffix:
    """Parse the ``#`` comment section for dp alias and replicated declarations."""
    dp_group_alias: Optional[str] = None
    replicated_axes: set[ParallelAxis] = set()

    for token in declaration_part.strip().split():
        dp_match = _DP_ALIAS_PATTERN.match(token)
        if dp_match is not None:
            if dp_group_alias is not None:
                raise ValueError(
                    f"Duplicate dp alias declaration: already have {dp_group_alias!r}, "
                    f"got {dp_match.group(1)!r}"
                )
            dp_group_alias = dp_match.group(1)
            continue

        repl_match = _REPLICATED_PATTERN.match(token)
        if repl_match is not None:
            axis_str: str = repl_match.group(1)
            axis: Optional[ParallelAxis] = _AXIS_LOOKUP.get(axis_str)
            if axis is None:
                raise ValueError(
                    f"Unknown axis {axis_str!r} in replicated declaration: {token!r}"
                )
            if axis in replicated_axes:
                raise ValueError(
                    f"Duplicate replicated declaration for axis {axis_str!r}"
                )
            replicated_axes.add(axis)
            continue

        raise ValueError(
            f"Unrecognized token {token!r} in # comment section. "
            f"Expected 'dp:=<group>' or '<axis>:replicated'."
        )

    return _CommentSuffix(
        dp_group_alias=dp_group_alias,
        replicated_axes=frozenset(replicated_axes),
    )
