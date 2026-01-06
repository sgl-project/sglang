# -*- coding: utf-8 -*-

# Copyright (c) 2017-2021 NVIDIA CORPORATION. All rights reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).


"""Miscellaneous utility functions."""

import importlib
import itertools as itt
import os
import re
import sys
from typing import Any, Callable, Iterator, Union
import torch
import numpy as np


def make_seed(*args):
    seed = 0
    for arg in args:
        seed = (seed * 31 + hash(arg)) & 0x7FFFFFFF
    return seed


class PipelineStage:
    def invoke(self, *args, **kw):
        raise NotImplementedError


def identity(x: Any) -> Any:
    """Return the argument as is."""
    return x


def safe_eval(s: str, expr: str = "{}"):
    """Evaluate the given expression more safely."""
    if re.sub("[^A-Za-z0-9_]", "", s) != s:
        raise ValueError(f"safe_eval: illegal characters in: '{s}'")
    return eval(expr.format(s))


def lookup_sym(sym: str, modules: list):
    """Look up a symbol in a list of modules."""
    for mname in modules:
        module = importlib.import_module(mname, package="webdataset")
        result = getattr(module, sym, None)
        if result is not None:
            return result
    return None


def repeatedly0(
    loader: Iterator, nepochs: int = sys.maxsize, nbatches: int = sys.maxsize
):
    """Repeatedly returns batches from a DataLoader."""
    for _ in range(nepochs):
        yield from itt.islice(loader, nbatches)


def guess_batchsize(batch: Union[tuple, list]):
    """Guess the batch size by looking at the length of the first element in a tuple."""
    return len(batch[0])


def repeatedly(
    source: Iterator,
    nepochs: int = None,
    nbatches: int = None,
    nsamples: int = None,
    batchsize: Callable[..., int] = guess_batchsize,
):
    """Repeatedly yield samples from an iterator."""
    epoch = 0
    batch = 0
    total = 0
    while True:
        for sample in source:
            yield sample
            batch += 1
            if nbatches is not None and batch >= nbatches:
                return
            if nsamples is not None:
                total += guess_batchsize(sample)
                if total >= nsamples:
                    return
        epoch += 1
        if nepochs is not None and epoch >= nepochs:
            return


def pytorch_worker_info(group=None):  # sourcery skip: use-contextlib-suppress
    """Return node and worker info for PyTorch and some distributed environments."""
    rank = 0
    world_size = 1
    worker = 0
    num_workers = 1
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        try:
            import torch.distributed

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                group = group or torch.distributed.group.WORLD
                rank = torch.distributed.get_rank(group=group)
                world_size = torch.distributed.get_world_size(group=group)
        except ModuleNotFoundError:
            pass
    if "WORKER" in os.environ and "NUM_WORKERS" in os.environ:
        worker = int(os.environ["WORKER"])
        num_workers = int(os.environ["NUM_WORKERS"])
    else:
        try:
            import torch.utils.data

            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                worker = worker_info.id
                num_workers = worker_info.num_workers
        except ModuleNotFoundError:
            pass

    return rank, world_size, worker, num_workers


def pytorch_worker_seed(group=None):
    """Compute a distinct, deterministic RNG seed for each worker and node."""
    rank, world_size, worker, num_workers = pytorch_worker_info(group=group)
    return rank * 1000 + worker

def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()
    worker_id = worker_info.id

    # dataset = worker_info.dataset
    # split_size = dataset.num_records // worker_info.num_workers
    # # reset num_records to the true number to retain reliable length information
    # dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
    # current_id = np.random.choice(len(np.random.get_state()[1]), 1)
    # return np.random.seed(np.random.get_state()[1][current_id] + worker_id)

    return np.random.seed(np.random.get_state()[1][0] + worker_id)


def collation_fn(samples, combine_tensors=True, combine_scalars=True):
    """

    Args:
        samples (list[dict]):
        combine_tensors:
        combine_scalars:

    Returns:

    """

    result = {}

    keys = samples[0].keys()

    for key in keys:
        result[key] = []

    for sample in samples:
        for key in keys:
            val = sample[key]
            result[key].append(val)

    for key in keys:
        val_list = result[key]
        if isinstance(val_list[0], (int, float)):
            if combine_scalars:
                result[key] = np.array(result[key])

        elif isinstance(val_list[0], torch.Tensor):
            if combine_tensors:
                result[key] = torch.stack(val_list)

        elif isinstance(val_list[0], np.ndarray):
            if combine_tensors:
                result[key] = np.stack(val_list)

    return result
