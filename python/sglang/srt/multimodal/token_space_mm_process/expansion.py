"""Expand media placeholders without decoding the caller's token IDs."""

from collections.abc import Mapping, Sequence


# TODO(Rockdu): port to rust/sglang-mm (the `_multimodal` PyO3 extension) so
# expansion runs without holding the GIL on the processor worker thread.
def expand_token_placeholders(
    input_ids: Sequence[int],
    mm_token_expansion_spec: Mapping[int, Sequence[Sequence[int]]],
    mm_token_expansion_start_len: int = 0,
) -> list[int]:
    """Replace each placeholder token after the prefix with its media's tokens.

    The first `mm_token_expansion_start_len` IDs are already expanded, so the
    placeholders after them take the trailing expansions; inserted tokens are
    never scanned again.
    """
    start = mm_token_expansion_start_len
    if not isinstance(start, int) or not 0 <= start <= len(input_ids):
        raise ValueError("mm_token_expansion_start_len must be in [0, len(input_ids)].")
    input_ids = list(input_ids)
    placements = []
    for token_id, expansions in mm_token_expansion_spec.items():
        positions = _token_positions(input_ids, token_id, start)
        count = len(positions)
        # A full expansion must use every media; a partial one only the last few.
        if count > len(expansions) or (start == 0 and count != len(expansions)):
            raise ValueError(
                f"Found {count} placeholders for token {token_id}, "
                f"but received {len(expansions)} media expansions"
            )
        placements.extend(zip(positions, expansions[len(expansions) - count :]))
    placements.sort(key=lambda placement: placement[0])
    # Copy the runs between placeholders as slices; Python loops only per media.
    expanded_input_ids = input_ids[:start]
    end = start
    for position, expansion in placements:
        expanded_input_ids += input_ids[end:position]
        expanded_input_ids += expansion
        end = position + 1
    expanded_input_ids += input_ids[end:]
    return expanded_input_ids


def _token_positions(input_ids: list[int], token_id: int, start: int) -> list[int]:
    positions = []
    position = start - 1
    try:
        while True:
            position = input_ids.index(token_id, position + 1)
            positions.append(position)
    except ValueError:
        return positions
