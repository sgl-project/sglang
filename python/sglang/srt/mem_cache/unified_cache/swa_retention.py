"""Page-aligned SWA ranges needed by sparse external-cache checkpoints."""


def retained_swa_ranges(
    start: int,
    end: int,
    *,
    prompt_boundary: int,
    window: int,
    interval: int,
    page_size: int,
    include_prompt_boundary: bool = True,
) -> list[tuple[int, int]]:
    """Intersect available token positions with checkpoint/replay windows.

    Coordinates are absolute positions in the prefix key, not offsets within
    a radix node. The caller supplies the actual restorable prompt boundary
    (including any last-token/bigram adjustment). No GPU values are inspected.
    This selects storage only; it does not establish that every selected
    window is present across other nodes or ranks.
    """
    if page_size <= 0 or interval <= 0 or window <= 0:
        raise ValueError("page_size, interval and window must be positive")
    if not 0 <= start <= end <= prompt_boundary:
        raise ValueError("expected 0 <= start <= end <= prompt_boundary")
    if any(x % page_size for x in (start, end, prompt_boundary, interval)):
        raise ValueError("positions and interval must be page aligned")
    if start == end:
        return []
    window = (window + page_size - 1) // page_size * page_size
    first = (start // interval + 1) * interval
    last = min(prompt_boundary, end + window - page_size)
    boundaries = list(range(first, last + 1, interval))
    if include_prompt_boundary and (
        not boundaries or boundaries[-1] != prompt_boundary
    ):
        boundaries.append(prompt_boundary)
    ranges: list[tuple[int, int]] = []
    for boundary in boundaries:
        left, right = max(start, boundary - window), min(end, boundary)
        if left >= right:
            continue
        if ranges and left <= ranges[-1][1]:
            ranges[-1] = (ranges[-1][0], max(right, ranges[-1][1]))
        else:
            ranges.append((left, right))
    return ranges
