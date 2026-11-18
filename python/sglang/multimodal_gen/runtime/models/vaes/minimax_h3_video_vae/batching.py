# SPDX-License-Identifier: Apache-2.0
"""Order-preserving batches of complete H3 tiles and temporal windows."""

import functools
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
    """Only equal-shaped adjacent windows share a forward; time stays an axis.

    Each group's tiles are decoded, and their all-gather started, before the
    previous group is assembled, so the gather runs beside the next group's
    decode instead of between the two.
    """
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
    batch = int(z.shape[0])

    def started() -> Iterator[tuple[int, Callable[[], list[torch.Tensor]]]]:
        for start, stop in partitions(windows, capacity=limit * batch):
            group_windows = windows[start:stop]
            value = (
                group_windows[0]
                if len(group_windows) == 1
                else torch.cat(group_windows, dim=0)
            )
            yield (
                start,
                functools.partial(
                    _split_windows,
                    vae._start_adaptive_decode(value),
                    len(group_windows),
                    batch,
                ),
            )

    def finished(start, finish):
        for offset, result in enumerate(finish()):
            yield start + offset, result

    pending = None
    for item in started():
        if pending is not None:
            yield from finished(*pending)
        pending = item
    if pending is not None:
        yield from finished(*pending)


def _split_windows(finish: Callable[[], torch.Tensor], windows: int, batch: int):
    output = finish()
    if output.shape[0] != windows * batch:
        raise RuntimeError("Window decode changed the sample count")
    if windows == 1:
        return [output]
    return list(output.unflatten(0, (windows, batch)).unbind(0))
