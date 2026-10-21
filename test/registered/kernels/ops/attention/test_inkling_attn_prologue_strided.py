# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""The fused Inkling attention prologue on the unified pool's views.

Under --enable-unified-memory every per-layer conv cache and KV buffer is a
strided view whose slot stride is one whole token-major entry. Each prologue
(decode, target verify, extend) must produce what it produces on contiguous
copies of the same values, and write only its own rows of the shared buffers.

    python -m pytest test/registered/kernels/ops/attention/test_inkling_attn_prologue_strided.py -v
"""

import unittest

import torch

from sglang.kernels.ops.attention.inkling_attn_prologue import (
    inkling_attn_prologue_decode,
    inkling_attn_prologue_extend,
    inkling_attn_prologue_verify,
)
from sglang.srt.mem_cache.layout.token_major import (
    build_dense_views,
    build_mamba_entry_views,
    mamba_entry_bytes,
)
from sglang.srt.mem_cache.unified_memory_pool import MHASubPoolSpec
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="1-gpu-large")

HEAD = 128
W = 4
DQ, DKV = 2048, 256
ROW = DQ + 2 * DKV + 64  # packed qkvr row with an r tail
EPS = 1e-6
LAYERS = 3
LAYER = 1  # a middle layer, so every entry holds rows on both sides of it
CONV_SLOTS = 16
KV_SLOTS = 64
# Inkling's six conv streams (k/v full, k/v local, attn, mlp), channel-last.
CONV_SHAPES = [(W - 1, DKV)] * 2 + [(W - 1, HEAD)] * 2 + [(W - 1, 512)] * 2
KV_SPEC = MHASubPoolSpec(
    name="full",
    layer_num=LAYERS,
    grow_direction="up",
    head_num=DKV // HEAD,
    head_dim=HEAD,
    store_dtype=torch.bfloat16,
)


def _conv_kwargs():
    return dict(
        layer_num=LAYERS,
        conv_state_shapes=CONV_SHAPES,
        conv_dtype=torch.bfloat16,
        temporal_state_shape=(0, 0, 0),
        temporal_dtype=torch.bfloat16,
    )


def _raw(nbytes):
    raw = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    raw.view(torch.bfloat16).normal_()
    return raw


def _conv_views(raw):
    views, _ = build_mamba_entry_views(raw, max_slots=CONV_SLOTS, **_conv_kwargs())
    return views[0][LAYER], views[1][LAYER]


def _kv_views(raw):
    return tuple(
        build_dense_views(
            raw, layout=KV_SPEC.layout(), part_name=part, num_slots=KV_SLOTS
        )[LAYER]
        for part in ("k", "v")
    )


def _assert_matches_contiguous(test, call):
    """Run ``call(k_cache, v_cache, k_buf, v_buf)`` on the unified views and on
    contiguous copies: equal outputs, and the shared buffers changed exactly
    where the contiguous run changed its copies."""
    conv_raw = _raw(CONV_SLOTS * mamba_entry_bytes(**_conv_kwargs()))
    kv_raw = _raw(KV_SLOTS * KV_SPEC.entry_bytes())
    want_conv, want_kv = conv_raw.clone(), kv_raw.clone()
    views = _conv_views(conv_raw) + _kv_views(kv_raw)
    test.assertGreater(views[0].stride(0), (W - 1) * DKV)
    test.assertGreater(views[2].stride(0), DKV)
    copies = [v.clone(memory_format=torch.contiguous_format) for v in views]

    got = call(*views)
    want = call(*copies)
    for a, b in zip(got, want):
        test.assertTrue(torch.equal(a, b))
    for view, copy in zip(_conv_views(want_conv) + _kv_views(want_kv), copies):
        view.copy_(copy)
    test.assertTrue(torch.equal(conv_raw, want_conv))
    test.assertTrue(torch.equal(kv_raw, want_kv))


def _weights():
    kw = torch.randn(DKV, W, device="cuda", dtype=torch.bfloat16) * 0.3
    vw = torch.randn(DKV, W, device="cuda", dtype=torch.bfloat16) * 0.3
    qg = torch.randn(HEAD, device="cuda", dtype=torch.bfloat16)
    kg = torch.randn(HEAD, device="cuda", dtype=torch.bfloat16)
    return dict(k_weight=kw, v_weight=vw, q_gamma=qg, k_gamma=kg, eps=EPS)


def _cuda(values, dtype):
    return torch.tensor(values, dtype=dtype, device="cuda")


def _qkv_offsets():
    return dict(q_off=0, k_off=DQ, v_off=DQ + DKV, dq=DQ, dkv=DKV)


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestInklingAttnPrologueStrided(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_decode(self):
        qkvr = torch.randn(6, ROW, device="cuda", dtype=torch.bfloat16)
        args = dict(
            qkvr=qkvr,
            cache_indices=_cuda([3, 7, -1, 11, 2, 9], torch.int32),
            cache_mask=_cuda([1, 0, 1, 1, 0, 1], torch.bool),
            track_mask=_cuda([1, 0, 0, 1, 0, 0], torch.bool),
            track_indices=_cuda([13, 0, 0, 14, 0, 0], torch.int64),
            loc=_cuda([5, 40, 17, 63, 1, 22], torch.int64),
            **_weights(),
            **_qkv_offsets(),
        )

        def call(k_cache, v_cache, k_buf, v_buf):
            return inkling_attn_prologue_decode(
                k_cache=k_cache, v_cache=v_cache, k_buf=k_buf, v_buf=v_buf, **args
            )[:3]

        _assert_matches_contiguous(self, call)

    def test_verify(self):
        bs, draft_token_num = 3, 4
        num_tokens = bs * draft_token_num
        loc = torch.randperm(KV_SLOTS, device="cuda")[:num_tokens]
        loc[5] = -1  # a full->swa translation sentinel: no store
        args = dict(
            qkvr=torch.randn(num_tokens, ROW, device="cuda", dtype=torch.bfloat16),
            cache_indices=_cuda([4, -1, 10], torch.int32),
            cache_mask=_cuda([1, 1, 0], torch.bool),
            loc=loc,
            draft_token_num=draft_token_num,
            **_weights(),
            **_qkv_offsets(),
        )

        def call(k_cache, v_cache, k_buf, v_buf):
            inter = [
                torch.zeros(
                    bs, draft_token_num, W - 1, DKV, device="cuda", dtype=torch.bfloat16
                )
                for _ in range(2)
            ]
            out = inkling_attn_prologue_verify(
                k_cache=k_cache,
                v_cache=v_cache,
                k_inter=inter[0],
                v_inter=inter[1],
                k_buf=k_buf,
                v_buf=v_buf,
                **args,
            )
            return list(out[:3]) + inter

        _assert_matches_contiguous(self, call)

    def test_extend(self):
        # Lengths 1 and 2 are shorter than the conv window: their cache update
        # shifts in old rows (with history) or zeros (without).
        lens = _cuda([5, 1, 9, 2], torch.int64)
        cu = torch.nn.functional.pad(torch.cumsum(lens, 0), (1, 0))
        num_tokens = int(cu[-1])
        has_initial_state = _cuda([1, 1, 0, 0], torch.bool)
        cache_indices = _cuda([6, 12, -1, 1], torch.int32)
        track_rows = (
            cu[1:, None] - (W - 1) + torch.arange(W - 1, device="cuda")
        ).clamp(min=0)
        args = dict(
            qkvr=torch.randn(num_tokens, ROW, device="cuda", dtype=torch.bfloat16),
            cache_indices=cache_indices,
            cache_mask=has_initial_state & (cache_indices != -1),
            has_initial_state=has_initial_state,
            cu=cu,
            si=torch.repeat_interleave(
                torch.arange(4, device="cuda", dtype=torch.int32), lens
            ),
            track_rows=track_rows,
            track_mask=_cuda([1, 0, 0, 1], torch.bool),
            track_dst=_cuda([13, 0, 0, 14], torch.int64),
            loc=torch.randperm(KV_SLOTS, device="cuda")[:num_tokens],
            **_weights(),
            **_qkv_offsets(),
        )

        def call(k_cache, v_cache, k_buf, v_buf):
            return inkling_attn_prologue_extend(
                k_cache=k_cache, v_cache=v_cache, k_buf=k_buf, v_buf=v_buf, **args
            )[:3]

        _assert_matches_contiguous(self, call)


if __name__ == "__main__":
    unittest.main()
