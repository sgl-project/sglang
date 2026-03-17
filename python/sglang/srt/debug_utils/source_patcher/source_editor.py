from sglang.srt.debug_utils.source_patcher.types import EditSpec, PatchApplicationError


def apply_edits(*, source: str, edits: list[EditSpec]) -> str:
    """Apply a sequence of match/replacement edits to source text.

    Each edit is applied sequentially so later edits see the result of earlier ones.
    """
    result: str = source
    for edit in edits:
        result = _apply_single_edit(source=result, edit=edit)
    return result


def _apply_single_edit(*, source: str, edit: EditSpec) -> str:
    """Apply a single match/replacement edit to the source text."""
    match_text: str = edit.match.strip()
    if not match_text:
        raise PatchApplicationError("empty match text")

    source_lines: list[str] = source.splitlines()
    match_lines: list[str] = match_text.splitlines()

    start_idx: int = _find_match(source_lines=source_lines, match_lines=match_lines)
    match_len: int = len(match_lines)

    original_indent: int = _leading_spaces(source_lines[start_idx])

    effective_replacement: str = _resolve_replacement(edit=edit, match_text=match_text)
    replacement_lines: list[str] = (
        effective_replacement.splitlines() if effective_replacement else []
    )
    aligned: list[str] = _realign_replacement(
        replacement_lines=replacement_lines, original_indent=original_indent
    )
    new_lines: list[str] = (
        source_lines[:start_idx] + aligned + source_lines[start_idx + match_len :]
    )

    trailing_newline: str = "\n" if source.endswith("\n") else ""
    return "\n".join(new_lines) + trailing_newline


def _resolve_replacement(*, edit: EditSpec, match_text: str) -> str:
    """Return the effective replacement text, handling replacement, prepend, and append modes."""
    if edit.prepend.strip():
        return edit.prepend.strip() + "\n" + match_text
    if edit.append.strip():
        return match_text + "\n" + edit.append.strip()
    return edit.replacement.strip()


def _find_match(*, source_lines: list[str], match_lines: list[str]) -> int:
    """Find the start index of match_lines in source_lines (strip-compared).

    Returns the index of the first matching line.
    Raises PatchApplicationError if not found or found multiple times.
    """
    stripped_source: list[str] = [line.strip() for line in source_lines]
    stripped_match: list[str] = [line.strip() for line in match_lines]
    match_len: int = len(stripped_match)

    found_indices: list[int] = [
        i
        for i in range(len(stripped_source) - match_len + 1)
        if stripped_source[i : i + match_len] == stripped_match
    ]

    if len(found_indices) == 0:
        preview: str = "\n".join(match_lines)
        raise PatchApplicationError(f"match text not found in source:\n{preview}")
    if len(found_indices) > 1:
        preview = "\n".join(match_lines)
        raise PatchApplicationError(
            f"match text found multiple times ({len(found_indices)} occurrences) in source:\n{preview}"
        )

    return found_indices[0]


def _realign_replacement(
    *, replacement_lines: list[str], original_indent: int
) -> list[str]:
    """Realign replacement lines to the original indentation level.

    Strategy:
    - Take the leading spaces of the first non-empty replacement line as base_indent
    - For each replacement line: remove base_indent, add original_indent
    """
    non_empty: list[str] = [line for line in replacement_lines if line.strip()]
    if not non_empty:
        return []

    base_indent: int = _leading_spaces(non_empty[0])
    result: list[str] = []

    for line in replacement_lines:
        if not line.strip():
            result.append("")
        else:
            stripped = line[min(base_indent, len(line) - len(line.lstrip())) :]
            result.append(" " * original_indent + stripped)

    return result


def _leading_spaces(line: str) -> int:
    return len(line) - len(line.lstrip(" "))
