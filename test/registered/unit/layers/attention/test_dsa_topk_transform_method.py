import pytest

from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod
from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_backend(prefill_impl: str, *, fp8_kv_cache: bool = True):
    backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
    backend.dsa_kv_cache_store_fp8 = fp8_kv_cache
    backend.dsa_prefill_impl = prefill_impl
    return backend


@pytest.mark.parametrize("prefill_impl", ["flashmla_sparse", "flashmla_sparse_q8"])
@pytest.mark.parametrize("forward_mode", [ForwardMode.EXTEND, ForwardMode.MIXED])
def test_fp8_sparse_prefill_uses_ragged_topk_transform(
    prefill_impl: str, forward_mode: ForwardMode
):
    backend = _make_backend(prefill_impl)

    assert backend.get_topk_transform_method(forward_mode) == TopkTransformMethod.RAGGED


@pytest.mark.parametrize(
    "forward_mode",
    [ForwardMode.DECODE, ForwardMode.TARGET_VERIFY, ForwardMode.DRAFT_EXTEND_V2],
)
def test_speculative_and_decode_modes_keep_paged_topk_transform(
    forward_mode: ForwardMode,
):
    backend = _make_backend("flashmla_sparse_q8")

    assert backend.get_topk_transform_method(forward_mode) == TopkTransformMethod.PAGED


def test_non_fp8_or_flashmla_kv_prefill_keeps_paged_topk_transform():
    assert (
        _make_backend(
            "flashmla_sparse_q8", fp8_kv_cache=False
        ).get_topk_transform_method(ForwardMode.MIXED)
        == TopkTransformMethod.PAGED
    )
    assert (
        _make_backend("flashmla_kv").get_topk_transform_method(ForwardMode.MIXED)
        == TopkTransformMethod.PAGED
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
