import sys

import pytest
import torch

from sglang.srt.model_executor.cuda_graph_buffer_registry import (
    build_prefill_registry,
)
from sglang.srt.model_executor.runner_utils.buffers import PrefillInputBuffers
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _create_buffers(is_multimodal, enable_input_embeds=None):
    return PrefillInputBuffers.create(
        device=torch.device("cpu"),
        max_bs=2,
        max_num_tokens=8,
        cache_loc_dtype=torch.int64,
        is_multimodal=is_multimodal,
        hidden_size=4,
        dtype=torch.float32,
        enable_mamba_track=False,
        enable_input_embeds=enable_input_embeds,
    )


def test_buffers_default_follows_multimodal():
    assert _create_buffers(False).input_embeds is None
    assert _create_buffers(False).mrope_positions is None
    assert _create_buffers(True).input_embeds is not None
    assert _create_buffers(True).mrope_positions is not None


def test_buffers_text_only_draft_opts_into_input_embeds():
    """An EAGLE draft serving a multimodal target is text-only but consumes
    externally composed embeddings, so it must allocate the slot."""
    buffers = _create_buffers(False, enable_input_embeds=True)
    assert buffers.input_embeds is not None
    assert buffers.input_embeds.shape == (8, 4)
    assert buffers.mrope_positions is None


def _build_registry(is_multimodal, register_input_embeds=None):
    return build_prefill_registry(
        device=torch.device("cpu"),
        max_bs=2,
        max_num_token=8,
        cache_loc_dtype=torch.int64,
        is_multimodal=is_multimodal,
        hidden_size=4,
        embed_dtype=torch.float32,
        enable_mamba_track=False,
        enable_num_token_non_padded=False,
        require_gathered_buffer=False,
        enable_prefill_cp=False,
        register_input_embeds=register_input_embeds,
        source=None,
    )


def test_registry_defaults_to_multimodal():
    assert _build_registry(True).has_slot("input_embeds")
    assert not _build_registry(False).has_slot("input_embeds")


def test_registry_text_only_draft_registers_input_embeds():
    registry = _build_registry(False, register_input_embeds=True)
    assert registry.has_slot("input_embeds")
    assert not registry.has_slot("mrope_positions")


def test_registry_mm_can_skip_input_embeds_slot():
    """The eager extend path passes False so embeddings are carried from the
    batch rather than written in-graph."""
    registry = _build_registry(True, register_input_embeds=False)
    assert not registry.has_slot("input_embeds")
    assert registry.has_slot("mrope_positions")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
