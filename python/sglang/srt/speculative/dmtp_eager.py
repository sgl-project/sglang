"""Pack only the accepted DMTP continuation for the private eager MTP runner."""

import torch


def pack_eager_commit(candidates, bonus, target_hidden, cache_locations, lengths):
    if len(lengths) != candidates.shape[0] or any(
        not 1 <= count <= candidates.shape[1] for count in lengths
    ):
        raise ValueError(
            "DMTP commit lengths must include the anchor and fit the block"
        )
    tokens, hidden, slots, last_rows = [], [], [], []
    total = 0
    for i, count in enumerate(lengths):
        # The old anchor's private KV is pending already. Pair each newly
        # appended token (including the next anchor) with its preceding target
        # hidden row and the slot of that preceding token.
        tokens.append(torch.cat((candidates[i, 1:count], bonus[i : i + 1])))
        hidden.append(target_hidden[i, :count])
        slots.append(cache_locations[i, :count])
        total += count
        last_rows.append(total - 1)
    return torch.cat(tokens), torch.cat(hidden), torch.cat(slots), last_rows
