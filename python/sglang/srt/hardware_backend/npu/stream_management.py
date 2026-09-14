# SPDX-License-Identifier: Apache-2.0
"""Centralized NPU stream management for SGLang hardware backend."""

from __future__ import annotations

import torch

cmo_stream = None
share_stream = None
routed_stream = None
indexer_weight_stream = None


def _device_module():
    return torch.get_device_module()


# --- CMO (cache management / weight prefetch) streams ---


def get_cmo_stream():
    """Return the CMO prefetch stream, or None if not initialized."""
    return cmo_stream


def set_cmo_stream(stream):
    global cmo_stream
    cmo_stream = stream


def prepare_weight_cache(handle, cache, PREFETCH_MAX_SIZE=1000000000):
    """Prefetch weight tensors on a dedicated stream for overlap with compute."""
    import torch_npu

    stream = get_cmo_stream()
    if stream is None:
        stream = torch.npu.Stream()
        set_cmo_stream(stream)
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        if isinstance(cache, list):
            for weight in cache:
                torch_npu.npu_prefetch(weight, handle, PREFETCH_MAX_SIZE)
        else:
            torch_npu.npu_prefetch(cache, handle, PREFETCH_MAX_SIZE)


def wait_cmo_stream():
    stream = get_cmo_stream()
    if stream is not None:
        torch.npu.current_stream().wait_stream(stream)


# --- Shared / routed expert streams ---


def get_share_stream():
    return share_stream


def set_share_stream(stream):
    global share_stream
    share_stream = stream


def get_routed_stream():
    return routed_stream


def set_routed_stream(stream):
    global routed_stream
    routed_stream = stream


def wait_share_stream():
    stream = get_share_stream()
    if stream is not None:
        _device_module().current_stream().wait_stream(stream)


def wait_routed_stream():
    stream = get_routed_stream()
    if stream is not None:
        _device_module().current_stream().wait_stream(stream)


def process_shared_expert(hidden_states, forward_func):
    stream = get_share_stream()
    dev = _device_module()
    if stream is None:
        stream = dev.Stream()
        set_share_stream(stream)
    stream.wait_stream(dev.current_stream())
    with dev.stream(stream):
        return forward_func(hidden_states)


def process_routed_expert(hidden_states, topk_output, forward_func):
    stream = get_routed_stream()
    dev = _device_module()
    if stream is None:
        stream = dev.Stream()
        set_routed_stream(stream)
    stream.wait_stream(dev.current_stream())
    with dev.stream(stream):
        return forward_func(hidden_states, topk_output)


# Backward-compatible alias used by qwen2_moe via cmo.py
shared_expert_on_independent_stream = process_shared_expert


# --- Indexer weight stream (DSA) ---


def get_indexer_weight_stream():
    global indexer_weight_stream
    if indexer_weight_stream is None:
        indexer_weight_stream = torch.npu.Stream()
    return indexer_weight_stream
