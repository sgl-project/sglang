from types import SimpleNamespace

import pytest

import sglang.srt.utils as srt_utils
from sglang.srt.layers.attention.graph_variants import (
    DsaGraphVariants,
    create_attention_graph_variants,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "hip, architecture, index_kpool, expected_dual_graph",
    [
        (True, "Glm5NextForConditionalGeneration", 4, False),
        (True, "Glm5NextForConditionalGeneration", 1, True),
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


def test_dsa_select_does_not_read_device_scalar_without_cpu_mirror():
    class DeviceSeqLens:
        def numel(self):
            return 8

        def max(self):
            raise AssertionError("device seq_lens must not be synchronized")

    batch = SimpleNamespace(seq_lens_cpu=None, seq_lens=DeviceSeqLens())

    assert DsaGraphVariants(index_topk=2048).select(batch) == "sparse"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
