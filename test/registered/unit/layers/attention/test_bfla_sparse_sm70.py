import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="TileLang BFLA kernels require an NVIDIA V100",
)


def test_bfla_rejects_implicit_approximate_attention(monkeypatch):
    pytest.importorskip("tilelang")
    from sglang.srt.layers.attention.tilelang_fa_v100._paged_adapter import (
        _build_bfla_mask,
    )

    q = torch.randn(256, 6, 256, device="cuda", dtype=torch.float16)
    k = torch.randn(4096, 1, 256, device="cuda", dtype=torch.float16)
    monkeypatch.setenv("SGLANG_V100_BFLA_KEEP_RATIO", "0.1")
    monkeypatch.delenv("SGLANG_V100_BFLA_ALLOW_APPROXIMATE", raising=False)

    with pytest.raises(ValueError, match="changes attention semantics"):
        _build_bfla_mask(q, k, 3840, 1)


def test_bfla_approximate_mask_keeps_anchor_and_local_blocks(monkeypatch):
    pytest.importorskip("tilelang")
    from sglang.srt.layers.attention.tilelang_fa_v100._paged_adapter import (
        _build_bfla_mask,
    )

    torch.manual_seed(31)
    q = torch.randn(256, 6, 256, device="cuda", dtype=torch.float16)
    k = torch.randn(4096, 1, 256, device="cuda", dtype=torch.float16)
    monkeypatch.setenv("SGLANG_V100_BFLA_KEEP_RATIO", "0.1")
    monkeypatch.setenv("SGLANG_V100_BFLA_ALLOW_APPROXIMATE", "1")
    monkeypatch.setenv("SGLANG_V100_BFLA_LOCAL_BLOCKS", "1")

    mask = _build_bfla_mask(q, k, 3840, 1).bool()

    assert mask.shape == (6, 1, 16)
    assert mask[:, :, 0].all()
    assert mask[:, :, 14:16].all()
    assert mask.float().mean().item() < 0.5


def test_bfla_all_keep_matches_dense_d256_attention(monkeypatch):
    pytest.importorskip("tilelang")
    from sglang.srt.layers.attention.tilelang_fa_v100._kernels_dense_d256 import (
        get_dense_prefix_d256_kernel,
    )
    from sglang.srt.layers.attention.tilelang_fa_v100._kernels_dense_d256_sparse import (
        get_dense_prefix_d256_sparse_kernel,
    )
    from sglang.srt.layers.attention.tilelang_fa_v100._paged_adapter import (
        _build_bfla_mask,
    )

    torch.manual_seed(37)
    query_tokens, kv_tokens, prefix_tokens = 256, 768, 512
    q = torch.randn(query_tokens, 6, 256, device="cuda", dtype=torch.float16) * 0.1
    k = torch.randn(kv_tokens, 1, 256, device="cuda", dtype=torch.float16) * 0.1
    v = torch.randn_like(k) * 0.1
    monkeypatch.setenv("SGLANG_V100_BFLA_KEEP_RATIO", "1.0")
    mask = _build_bfla_mask(q, k, prefix_tokens, 1)

    dense = get_dense_prefix_d256_kernel(6, 1)(q, k, v, prefix_tokens, 256**-0.5)
    sparse = get_dense_prefix_d256_sparse_kernel(
        6, 1, mask.shape[1], mask.shape[2], 256
    )(q, k, v, prefix_tokens, 256**-0.5, mask)

    torch.testing.assert_close(sparse, dense, rtol=2e-4, atol=2e-5)
