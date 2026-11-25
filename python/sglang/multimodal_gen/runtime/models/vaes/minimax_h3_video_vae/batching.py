# SPDX-License-Identifier: Apache-2.0
"""Order-preserving batches of complete H3 tiles and temporal windows."""

from collections.abc import Callable, Iterator, Sequence

import torch


def partitions(
    values: Sequence[torch.Tensor], *, capacity: int
) -> Iterator[tuple[int, int]]:
    """Group adjacent equal-shaped tensors without reordering samples."""
    if capacity < 1:
        raise ValueError("Batch capacity must be positive")
    start = 0
    while start < len(values):
        first = values[start]
        if first.shape[0] < 1:
            raise ValueError("Cannot batch empty sample dimensions")
        width = max(1, capacity // int(first.shape[0]))
        stop = start + 1
        key = (first.shape, first.dtype, first.device)
        while stop < len(values) and stop - start < width:
            value = values[stop]
            if (value.shape, value.dtype, value.device) != key:
                break
            stop += 1
        yield start, stop
        start = stop


def forward_many(values, *, forward: Callable, capacity: int, collector=None):
    """Run complete samples; collect class tokens for single-tile encoder calls."""
    outputs = []
    for start, stop in partitions(values, capacity=capacity):
        group = values[start:stop]
        batch = int(group[0].shape[0])
        x = group[0] if len(group) == 1 else torch.cat(group, dim=0)
        y = forward(x)
        if y.shape[0] != len(group) * batch:
            raise RuntimeError("Batched VAE changed the leading sample dimension")
        if collector is not None:
            collector.collect()
        outputs.extend(
            [y] if len(group) == 1 else y.unflatten(0, (len(group), batch)).unbind(0)
        )
    return outputs


def decode_windows(
    vae, z, *, head, tail, count: int, window_limit: int
) -> Iterator[tuple[int, torch.Tensor]]:
    """Only equal-shaped adjacent windows share a forward; time stays an axis."""
    if window_limit < 0:
        raise ValueError("Window batch limit cannot be negative")
    windows = []
    for index in range(count):
        start = index * vae.tokens_chunk_size
        value = z[:, :, start : start + vae.tokens_chunk_size + vae.token_overlap]
        if index == 0 and head is not None:
            value = torch.cat([head, value], dim=2)
        if index == count - 1 and tail is not None:
            value = torch.cat([value, tail], dim=2)
        windows.append(value)
    limit = window_limit or max(1, len(windows))
    for start, stop in partitions(windows, capacity=limit * int(z.shape[0])):
        group = windows[start:stop]
        value = group[0] if len(group) == 1 else torch.cat(group, dim=0)
        output = vae._adaptive_decode(value)
        batch = int(group[0].shape[0])
        if output.shape[0] != len(group) * batch:
            raise RuntimeError("Window decode changed the sample count")
        decoded = (
            [output]
            if len(group) == 1
            else output.unflatten(0, (len(group), batch)).unbind(0)
        )
        for offset, result in enumerate(decoded):
            yield start + offset, result
        # Release the previous output batch before decoding the next group.
        del decoded, output, result
