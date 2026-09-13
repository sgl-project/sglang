from types import SimpleNamespace

import pytest
import torch

import sglang.srt.utils as srt_utils
from sglang.srt.layers.attention.graph_variants import (
    DsaGraphVariants,
    create_attention_graph_variants,
)
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "hip, architecture, index_kpool, expected_dual_graph",
    [
        (True, "Glm5NextForConditionalGeneration", 4, False),
        (True, "Glm5NextForConditionalGeneration", 1, True),
        (False, "Glm5NextForConditionalGeneration", 4, False),
        (False, "Glm5NextForConditionalGeneration", 1, False),
        (True, "LlamaForCausalLM", 4, False),
    ],
)
def test_dsa_graph_variants_match_the_indexer_path(
    monkeypatch, hip, architecture, index_kpool, expected_dual_graph
):
    monkeypatch.setattr(srt_utils, "is_hip", lambda: hip)
    config = SimpleNamespace(
        architectures=[architecture], index_topk=2048, index_kpool=index_kpool
    )

    variants = create_attention_graph_variants(config)

    if expected_dual_graph:
        assert isinstance(variants, DsaGraphVariants)
        assert variants.capture_labels == ("dense", "sparse")
    else:
        assert variants is None


@pytest.mark.parametrize(
    "seq_lens, expected",
    [
        ([1, 2048], "dense"),
        ([1, 2049], "sparse"),
    ],
)
def test_dsa_select_uses_available_cpu_mirror(seq_lens, expected):
    batch = SimpleNamespace(
        seq_lens_cpu=torch.tensor(seq_lens),
        seq_lens=torch.tensor(seq_lens),
    )

    assert DsaGraphVariants(index_topk=2048).select(batch) == expected


def test_dsa_select_does_not_read_device_scalar_without_cpu_mirror():
    class DeviceSeqLens:
        def numel(self):
            return 8

        def max(self):
            raise AssertionError("device seq_lens must not be synchronized")

    batch = SimpleNamespace(seq_lens_cpu=None, seq_lens=DeviceSeqLens())

    assert DsaGraphVariants(index_topk=2048).select(batch) == "sparse"


def test_resolve_attention_variant_uses_runner_variants():
    runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
    runner.attention_graph_variants = DsaGraphVariants(index_topk=2048)
    batch = SimpleNamespace(
        seq_lens_cpu=torch.tensor([1, 2049]),
        seq_lens=torch.tensor([1, 2049]),
    )

    assert runner._resolve_attention_variant(batch) == "sparse"


def test_resolve_attention_variant_returns_none_without_variants():
    runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
    runner.attention_graph_variants = None

    assert runner._resolve_attention_variant(SimpleNamespace()) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
