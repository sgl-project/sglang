"""Resolution + prefix-length alignment for a FIXED system prompt.

The fixed system prompt is shared by every chat request. When `--system-prompt`
is set the value is either the literal prompt text or a path to a UTF-8 file
(an existing path is auto-detected). Everything in this module is inert when
`--system-prompt` is unset: callers must check ``resolve_system_prompt(...)``
for None before calling anything else, so the default code paths stay
byte-identical.
"""

from __future__ import annotations

import os
from typing import Optional


def resolve_system_prompt(server_args) -> Optional[str]:
    """Return the fixed system-prompt text, or None if the feature is disabled.

    The literal value is used as-is unless it names an existing file, in which
    case the file is read as UTF-8 (trailing whitespace stripped). Long content
    (large files) is fully supported because we only hold the text in memory.
    """
    value = getattr(server_args, "system_prompt", None)
    if value is None:
        return None
    if os.path.isfile(value):
        try:
            with open(value, "r", encoding="utf-8") as f:
                return f.read().rstrip()
        except OSError as e:
            raise ValueError(
                f"--system-prompt file '{value}' could not be read: {e}"
            ) from e
    return value


def aligned_prefix_len(prefix_len: int, chunk: int) -> int:
    """Round a raw prefix token count up to the next ``chunk`` boundary.

    Linear-attention (SSM / Mamba) state only checkpoints at multiples of
    ``mamba_cache_chunk_size``, so a reusable standalone state at the system
    boundary requires the fixed prefix length to be a multiple of the chunk.
    Returns the un-padded length unchanged when it already is a multiple.
    """
    if chunk <= 0:
        return prefix_len
    return ((prefix_len + chunk - 1) // chunk) * chunk