"""MPS fresh-prefill batching must preserve request isolation and Torch KV state."""

import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch.nn.functional import scaled_dot_product_attention

from sglang.srt.layers.attention import torch_native_backend as native
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.mem_cache.memory_pool import KVWriteLoc, unwrap_write_loc
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.test.ci.ci_register import register_mps_ci

register_mps_ci(est_time=5, suite="stage-a-unit-test-mps")


class KVPool:
    def __init__(self, size, kv_heads, head_dim, dtype):
        self.dtype = dtype
        self.is_quantized_kv_cache = False
        self.k = torch.zeros(size + 1, kv_heads, head_dim, dtype=dtype, device="mps")
        self.v = torch.zeros_like(self.k)

    def set_kv_buffer(self, layer, loc_info, k, v):
        loc, _, _ = unwrap_write_loc(loc_info)
        self.k[loc] = k.to(self.dtype)
        self.v[loc] = v.to(self.dtype)

    def get_key_buffer(self, layer_id):
        return self.k

    def get_value_buffer(self, layer_id):
        return self.v


def make_case(
    lengths=(7, 7, 7),
    prefixes=(0, 0, 0),
    *,
    heads=4,
    kv_heads=2,
    dtype=torch.bfloat16,
    strided=False,
):
    generator = torch.Generator().manual_seed(42)
    total = sum(lengths)
    head_dim = 16
    width = (heads + 2 * kv_heads) * head_dim
    packed = torch.randn(total, width, generator=generator).to(dtype).to("mps")
    q, k, v = packed.split(
        [heads * head_dim, kv_heads * head_dim, kv_heads * head_dim], dim=-1
    )
    if not strided:
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    k = k.view(total, kv_heads, head_dim)
    v = v.view(total, kv_heads, head_dim)
    full_lengths = [length + prefix for length, prefix in zip(lengths, prefixes)]
    pool = KVPool(sum(full_lengths), kv_heads, head_dim, dtype)
    # Non-monotonic physical slots catch accidental dependence on cache layout.
    slots = (torch.randperm(sum(full_lengths), generator=generator) + 1).to("mps")
    req_to_token = torch.zeros(
        len(lengths) + 1, max(full_lengths), dtype=torch.int32, device="mps"
    )
    writes = []
    slot_offset = 0
    for index, (length, prefix) in enumerate(zip(lengths, prefixes), start=1):
        req_slots = slots[slot_offset : slot_offset + length + prefix]
        req_to_token[index, : length + prefix] = req_slots.to(torch.int32)
        if prefix:
            pool.k[req_slots[:prefix]] = (
                torch.randn(prefix, kv_heads, head_dim, generator=generator)
                .to(dtype)
                .to("mps")
            )
            pool.v[req_slots[:prefix]] = (
                torch.randn(prefix, kv_heads, head_dim, generator=generator)
                .to(dtype)
                .to("mps")
            )
        writes.append(req_slots[prefix:])
        slot_offset += length + prefix
    batch = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=len(lengths),
        input_ids=torch.zeros(total, dtype=torch.int64, device="mps"),
        req_pool_indices=torch.arange(
            1, len(lengths) + 1, device="mps", dtype=torch.int32
        ),
        seq_lens=torch.tensor(full_lengths, dtype=torch.int32, device="mps"),
        out_cache_loc=torch.cat(writes),
        seq_lens_sum=sum(full_lengths),
        extend_seq_lens=torch.tensor(lengths, dtype=torch.int32, device="mps"),
        extend_prefix_lens=torch.tensor(prefixes, dtype=torch.int32, device="mps"),
        extend_seq_lens_cpu=list(lengths),
        extend_prefix_lens_cpu=list(prefixes),
    )
    runner = SimpleNamespace(
        device="mps",
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        token_to_kv_pool=pool,
    )
    backend = native.TorchNativeAttnBackend(runner)
    backend.init_forward_metadata(batch)
    layer = SimpleNamespace(
        layer_id=0,
        tp_q_head_num=heads,
        tp_k_head_num=kv_heads,
        tp_v_head_num=kv_heads,
        qk_head_dim=head_dim,
        v_head_dim=head_dim,
        scaling=0.17,
        is_cross_attention=False,
        attn_type=AttentionType.DECODER,
        sliding_window_size=-1,
    )
    return backend, layer, batch, q, k, v


def cpu_reference(backend, layer, batch, q):
    # Independent CPU float32 reference includes the prefix and uses an
    # explicit causal mask, rather than the fallback's padded-query algorithm.
    output = []
    offset = 0
    for index, (length, prefix) in enumerate(
        zip(batch.extend_seq_lens_cpu, batch.extend_prefix_lens_cpu), start=1
    ):
        slots = (
            backend.req_to_token_pool.req_to_token[index, : length + prefix]
            .cpu()
            .long()
        )
        queries = (
            q[offset : offset + length]
            .cpu()
            .float()
            .reshape(length, layer.tp_q_head_num, layer.qk_head_dim)
            .transpose(0, 1)[None]
        )
        keys = (
            backend.token_to_kv_pool.k.cpu()[slots]
            .to(q.dtype)
            .float()
            .transpose(0, 1)[None]
        )
        values = (
            backend.token_to_kv_pool.v.cpu()[slots]
            .to(q.dtype)
            .float()
            .transpose(0, 1)[None]
        )
        mask = (
            torch.arange(length + prefix)[None, :]
            <= torch.arange(prefix, prefix + length)[:, None]
        )
        if layer.attn_type == AttentionType.ENCODER_ONLY:
            mask = None
        elif layer.sliding_window_size > -1:
            mask &= (
                torch.arange(length + prefix)[None, :]
                >= torch.arange(prefix, prefix + length)[:, None]
                - layer.sliding_window_size
            )
        result = scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=mask,
            enable_gqa=layer.tp_q_head_num != layer.tp_k_head_num,
            scale=layer.scaling,
        )
        output.append(result[0].transpose(0, 1).flatten(1))
        offset += length
    return torch.cat(output)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS")
@pytest.mark.parametrize(
    "heads,kv_heads,dtype,strided,length",
    [
        (4, 2, torch.bfloat16, False, 7),
        (4, 4, torch.float32, False, 7),
        (4, 2, torch.float32, True, 7),
        (4, 2, torch.bfloat16, True, 64),
        (4, 2, torch.bfloat16, False, 1),
    ],
)
def test_fresh_batched_prefill_matches_reference_and_writes_kv(
    heads, kv_heads, dtype, strided, length
):
    backend, layer, batch, q, k, v = make_case(
        lengths=(length,) * 3,
        heads=heads,
        kv_heads=kv_heads,
        dtype=dtype,
        strided=strided,
    )
    with mock.patch.object(
        native, "scaled_dot_product_attention", wraps=scaled_dot_product_attention
    ) as sdpa:
        actual = backend.forward_extend(q, k, v, layer, batch)
    assert sdpa.call_count == 1
    tolerance = 0.02 if dtype == torch.bfloat16 else 2e-6
    torch.testing.assert_close(
        actual.cpu().float(),
        cpu_reference(backend, layer, batch, q),
        atol=tolerance,
        rtol=tolerance,
    )
    torch.testing.assert_close(
        backend.token_to_kv_pool.k[batch.out_cache_loc], k, atol=0, rtol=0
    )
    torch.testing.assert_close(
        backend.token_to_kv_pool.v[batch.out_cache_loc], v, atol=0, rtol=0
    )
    # Changing one request must not leak into either neighboring request.
    changed_k, changed_v = k.clone(), v.clone()
    changed_k[length : 2 * length] *= -3
    changed_v[length : 2 * length] += 10
    changed = backend.forward_extend(q, changed_k, changed_v, layer, batch)
    torch.testing.assert_close(changed[:length], actual[:length], atol=0, rtol=0)
    torch.testing.assert_close(
        changed[2 * length :], actual[2 * length :], atol=0, rtol=0
    )


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS")
@pytest.mark.parametrize(
    "lengths,prefixes", [((3, 7), (0, 0)), ((7, 7), (2, 0)), ((7, 7), (2, 3))]
)
def test_mixed_lengths_and_cached_prefixes_use_fallback(lengths, prefixes):
    backend, layer, batch, q, k, v = make_case(lengths, prefixes)
    assert backend._fresh_prefill_shape is None
    with mock.patch.object(
        native, "scaled_dot_product_attention", wraps=scaled_dot_product_attention
    ) as sdpa:
        actual = backend.forward_extend(q, k, v, layer, batch)
    assert sdpa.call_count == len(lengths)
    torch.testing.assert_close(
        actual.cpu().float(),
        cpu_reference(backend, layer, batch, q),
        atol=0.02,
        rtol=0.02,
    )


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS")
@pytest.mark.parametrize(
    "case",
    [
        "sliding_window",
        "encoder",
        "read_only",
        "missing_k",
        "missing_v",
        "kv_cast",
        "quantized_kv",
    ],
)
def test_unsupported_layers_and_cache_contracts_keep_the_fallback(case):
    backend, layer, batch, q, k, v = make_case()
    backend.token_to_kv_pool.set_kv_buffer(layer, KVWriteLoc(batch.out_cache_loc), k, v)
    save = True
    if case == "sliding_window":
        layer.sliding_window_size = 2
    elif case == "encoder":
        layer.attn_type = AttentionType.ENCODER_ONLY
    elif case == "read_only":
        save = False
        k, v = k + 5, v - 3
    elif case == "missing_k":
        k = None
    elif case == "missing_v":
        v = None
    elif case == "kv_cast":
        backend.token_to_kv_pool.dtype = torch.float16
        backend.token_to_kv_pool.k = backend.token_to_kv_pool.k.to(torch.float16)
        backend.token_to_kv_pool.v = backend.token_to_kv_pool.v.to(torch.float16)
    elif case == "quantized_kv":
        backend.token_to_kv_pool.is_quantized_kv_cache = True
    with mock.patch.object(
        backend, "_run_sdpa_forward_extend", wraps=backend._run_sdpa_forward_extend
    ) as fallback:
        actual = backend.forward_extend(q, k, v, layer, batch, save_kv_cache=save)
    fallback.assert_called_once()
    torch.testing.assert_close(
        actual.cpu().float(),
        cpu_reference(backend, layer, batch, q),
        atol=0.02,
        rtol=0.02,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_non_mps_metadata_does_not_select_the_fast_path(device):
    backend = native.TorchNativeAttnBackend(
        SimpleNamespace(device=device, req_to_token_pool=None, token_to_kv_pool=None)
    )
    # No host metadata is needed on the unchanged non-MPS path.
    backend.init_forward_metadata(SimpleNamespace())
    assert backend._fresh_prefill_shape is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"forward_mode": ForwardMode.DECODE},
        {"forward_mode": ForwardMode.MIXED},
        {"batch_size": 1},
        {"batch_size": 0},
        {"encoder_lens": object()},
        {"spec_info": object()},
        {"extend_seq_lens_cpu": None},
        {"extend_prefix_lens_cpu": None},
        {"extend_seq_lens_cpu": [7]},
        {"extend_prefix_lens_cpu": [0]},
        {"extend_seq_lens_cpu": [0, 0]},
        {"extend_seq_lens_cpu": [3, 11]},
        {"extend_prefix_lens_cpu": [0, 1]},
        {"input_ids": torch.zeros(15)},
    ],
)
def test_metadata_rejects_unsupported_batches_and_clears_previous_shape(overrides):
    backend = native.TorchNativeAttnBackend(
        SimpleNamespace(device="mps", req_to_token_pool=None, token_to_kv_pool=None)
    )
    fields = dict(
        forward_mode=ForwardMode.EXTEND,
        batch_size=2,
        encoder_lens=None,
        spec_info=None,
        extend_seq_lens_cpu=[7, 7],
        extend_prefix_lens_cpu=[0, 0],
        input_ids=torch.zeros(14),
    )
    backend.init_forward_metadata(SimpleNamespace(**fields))
    assert backend._fresh_prefill_shape == (2, 7)
    backend.init_forward_metadata(SimpleNamespace(**dict(fields, **overrides)))
    assert backend._fresh_prefill_shape is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
