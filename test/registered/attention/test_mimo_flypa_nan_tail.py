"""Unused page tails must not contribute to full-attention output."""

import math
import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=25, suite="stage-b-test-1-gpu-small-amd")


@pytest.fixture(autouse=True)
def require_flypa_backend():
    if not torch.cuda.is_available() or not torch.version.hip:
        pytest.skip("FlyPA tail tests require ROCm")
    if "gfx942" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("This suite qualifies the gfx942 FP8/BF16 path")
    pytest.importorskip("flydsl")


CASES = [
    (32, (1,), (1,), 1, False, "bf16"),
    (32, (65,), (141,), 1, True, "bf16"),
    (64, (32,), (70,), 1, False, "bf16"),
    (64, (64,), (64,), 1, True, "bf16"),
    (64, (129,), (257,), 2, True, "bf16"),
    (128, (65,), (141,), 1, True, "bf16"),
    (128, (257,), (513,), 2, True, "bf16"),
    (64, (1, 32, 129), (1, 70, 257), 2, True, "bf16"),
    (64, (32683,), (65451,), 1, True, "bf16"),
    # The penultimate Q tile can also reach the partial KV page.
    (64, (257,), (519,), 1, True, "bf16"),
    # Mix fast and masked tiles across batch items and local KV heads.
    (64, (256, 257), (512, 519), 2, True, "bf16"),
    (64, (256, 257), (512, 519), 2, True, "fp8"),
    (32, (1,), (1,), 1, True, "fp8"),
    (64, (32,), (70,), 1, True, "fp8"),
    (128, (65,), (141,), 2, True, "fp8"),
]


@pytest.mark.parametrize("page,qs,ks,kv_heads,side_stream,kind", CASES)
def test_unused_nan_v_tail(
    page, qs, ks, kv_heads, side_stream, kind, causal=True, v_dim=128
):
    from sglang.srt.layers.attention.flydsl_pa import PagedAttention

    torch.manual_seed(1297)
    device = torch.device("cuda")
    dtype = torch.bfloat16 if kind == "bf16" else torch.float8_e4m3fnuz
    stream = torch.cuda.Stream() if side_stream else torch.cuda.current_stream()
    heads, qk_dim, pack = 16, 192, 8 if kind == "bf16" else 16
    with torch.cuda.stream(stream):
        q = (torch.randn(sum(qs), heads, qk_dim, device=device) * 0.2).to(dtype)
        k = (torch.randn(sum(ks), kv_heads, qk_dim, device=device) * 0.2).to(dtype)
        v_float = torch.randn(sum(ks), kv_heads, v_dim, device=device) * 0.2
        offset = 0
        for length in ks:
            # Make dropping the last valid V token visibly incorrect.
            v_float[offset + length - 1] = length * 0.1
            offset += length
        v = v_float.to(dtype)
        page_counts = [math.ceil(n / page) for n in ks]
        n_pages = sum(page_counts)
        page_ids = torch.randperm(n_pages, device=device) + 1
        kt = torch.full(
            (n_pages + 1, page, kv_heads, qk_dim),
            float("nan"),
            dtype=dtype,
            device=device,
        )
        vt = torch.full(
            (n_pages + 1, page, kv_heads, v_dim),
            float("nan"),
            dtype=dtype,
            device=device,
        )
        offset = page_offset = 0
        for length, count in zip(ks, page_counts):
            for j in range(count):
                valid = min(page, length - j * page)
                index = page_ids[page_offset + j]
                kt[index, :valid] = k[offset + j * page : offset + j * page + valid]
                vt[index, :valid] = v[offset + j * page : offset + j * page + valid]
            offset += length
            page_offset += count
        kc = (
            kt.view(-1, page, kv_heads, qk_dim // pack, pack)
            .permute(0, 2, 3, 1, 4)
            .contiguous()
        )
        vc = (
            vt.view(-1, page // pack, pack, kv_heads, v_dim)
            .permute(0, 3, 1, 4, 2)
            .contiguous()
        )
        cu = torch.tensor([0] + list(qs), device=device, dtype=torch.int32).cumsum(
            0, dtype=torch.int32
        )
        indptr = torch.tensor(
            [0] + page_counts, device=device, dtype=torch.int32
        ).cumsum(0, dtype=torch.int32)
        indices = torch.nn.functional.pad(page_ids.to(torch.int32), (0, 256))
        tails = torch.tensor(
            [(n - 1) % page + 1 for n in ks], device=device, dtype=torch.int32
        )
        scale = torch.ones(1, device=device)
        fn = PagedAttention(
            num_qo_heads=heads,
            num_kv_heads=kv_heads,
            head_dim_qk=qk_dim,
            head_dim_v=v_dim,
            page_size=page,
            is_causal=causal,
            quant_query_mode="per-tensor",
        )
        out = fn(
            q,
            kc,
            vc,
            cu,
            None,
            indptr,
            indices,
            max(qs),
            max(ks),
            causal,
            scale,
            scale,
            scale,
            tails,
            stream=stream,
        )
        assert torch.isfinite(out).all()
        q_offset = k_offset = 0
        for nq, nk in zip(qs, ks):
            samples = (
                torch.arange(nq, device=device)
                if nq <= 257
                else torch.tensor(
                    [0, 1, 31, 127, nq // 2, nq - 129, nq - 32, nq - 2, nq - 1],
                    device=device,
                )
            )
            k_ref = (
                k[k_offset : k_offset + nk]
                .float()
                .repeat_interleave(heads // kv_heads, dim=1)
            )
            v_ref = (
                v[k_offset : k_offset + nk]
                .float()
                .repeat_interleave(heads // kv_heads, dim=1)
            )
            scores = torch.einsum(
                "qhd,khd->hqk", q[q_offset + samples].float(), k_ref
            ) / math.sqrt(qk_dim)
            mask = torch.arange(nk, device=device)[None] <= (nk - nq + samples)[:, None]
            if causal:
                scores.masked_fill_(~mask[None], -float("inf"))
            ref = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v_ref).bfloat16()
            tolerance = 0.02 if kind == "bf16" else 0.055
            torch.testing.assert_close(
                out[q_offset + samples], ref, atol=tolerance, rtol=tolerance
            )
            q_offset += nq
            k_offset += nk
    stream.synchronize()


def test_noncausal_nan_tail():
    test_unused_nan_v_tail(128, (65,), (141,), 2, True, "bf16", causal=False)


def test_v192_nan_tail():
    test_unused_nan_v_tail(64, (32,), (70,), 1, True, "fp8", v_dim=192)


def test_valid_nan_values_are_not_sanitized():
    from sglang.srt.layers.attention.flydsl_pa import PagedAttention

    device = "cuda"
    q = torch.zeros((1, 16, 192), device=device, dtype=torch.bfloat16)
    k = torch.zeros((1, 1, 24, 64, 8), device=device, dtype=torch.bfloat16)
    v = torch.full((1, 1, 8, 128, 8), float("nan"), device=device, dtype=torch.bfloat16)
    cu = torch.tensor([0, 1], device=device, dtype=torch.int32)
    ids = torch.zeros(257, device=device, dtype=torch.int32)
    tails = torch.ones(1, device=device, dtype=torch.int32)
    one = torch.ones(1, device=device)
    fn = PagedAttention(
        num_qo_heads=16,
        num_kv_heads=1,
        head_dim_qk=192,
        head_dim_v=128,
        page_size=64,
        is_causal=True,
        quant_query_mode="per-tensor",
    )
    out = fn(q, k, v, cu, None, cu, ids, 1, 1, True, one, one, one, tails)
    assert torch.isnan(out).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
