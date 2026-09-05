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
"""GPU parity of the per-layer-view `UnifiedMLATokenToKVPool` against the stock
`MLATokenToKVPool` on real K3 MLA geometry (L=24, D=512+64).

The unified pool receives kernel-facing locs (kernel_id(t) = (t//ps)*(ps*L) +
t%ps) where the reference pool receives raw token ids; every (layer, token)
cell must hold identical bytes afterwards. The TMA JIT fast path (n_loc >= 768)
flattens the buffer via `.view(shape[0], -1)`, which is legal only because the
per-layer views are contiguous.
"""

import types
import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")

_HAS_CUDA = torch.cuda.is_available()
_DEV = "cuda"

_L = 24
_LORA = 512
_ROPE = 64
_D = _LORA + _ROPE
_DTYPE = torch.bfloat16


def _kernel_id(t: torch.Tensor, ps: int) -> torch.Tensor:
    return (t // ps) * (ps * _L) + t % ps


def _make_pools(ps: int, n_tokens: int = 4096):
    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
    from sglang.srt.mem_cache.unified_memory_pool import (
        MambaSubPoolSpec,
        MLASubPoolSpec,
        UnifiedKVPool,
        UnifiedMLATokenToKVPool,
    )

    full = MLASubPoolSpec(
        name="full",
        layer_num=_L,
        kv_lora_rank=_LORA,
        qk_rope_head_dim=_ROPE,
        store_dtype=_DTYPE,
        grow_direction="down",
    )
    mamba = MambaSubPoolSpec(
        name="mamba",
        layer_num=2,
        conv_state_shapes=((8, 16),),
        conv_dtype=torch.bfloat16,
        temporal_state_shape=(4, 8, 8),
        temporal_dtype=torch.float32,
        grow_direction="up",
    )
    total = full.entry_bytes() * n_tokens + mamba.entry_bytes() * 16
    pool = UnifiedKVPool(
        total_bytes=total,
        sub_pool_specs=[full, mamba],
        device=_DEV,
        enable_memory_saver=False,
        page_size=ps,
    )
    unified = UnifiedMLATokenToKVPool(
        unified_buffer=pool,
        sub_pool_name="full",
        kv_cache_dtype=_DTYPE,
        page_size=ps,
    )
    max_tokens = pool.max_slots("full")
    ref = MLATokenToKVPool(
        size=max_tokens - ps,
        page_size=ps,
        dtype=_DTYPE,
        kv_lora_rank=_LORA,
        qk_rope_head_dim=_ROPE,
        layer_num=_L,
        device=_DEV,
        enable_memory_saver=False,
    )
    return unified, ref, max_tokens


def _rand_locs(max_tokens: int, ps: int, n: int) -> torch.Tensor:
    # distinct physical token ids clear of the reserved page 0
    g = torch.Generator(device="cpu").manual_seed(1234 + n + ps)
    perm = torch.randperm(max_tokens - ps, generator=g)[:n] + ps
    return perm.to(_DEV)


@unittest.skipUnless(_HAS_CUDA, "requires CUDA")
class TestUnifiedMLAPoolGPUParity(unittest.TestCase):
    def _assert_parity(self, unified, ref, locs, ps, layers=range(_L)):
        for l in layers:
            got = unified.get_key_buffer(l)[_kernel_id(locs, ps)]
            want = ref.get_key_buffer(l)[locs]
            torch.testing.assert_close(got, want, rtol=0, atol=0)

    def _run_set_mla(self, ps: int, n_loc: int):
        unified, ref, max_tokens = _make_pools(ps)
        locs = _rand_locs(max_tokens, ps, n_loc)
        torch.manual_seed(7)
        for l in range(_L):
            layer = types.SimpleNamespace(layer_id=l)
            nope = torch.randn(n_loc, 1, _LORA, dtype=_DTYPE, device=_DEV)
            rope = torch.randn(n_loc, 1, _ROPE, dtype=_DTYPE, device=_DEV)
            unified.set_mla_kv_buffer(layer, _kernel_id(locs, ps), nope, rope)
            ref.set_mla_kv_buffer(layer, locs, nope, rope)
        torch.cuda.synchronize()
        self._assert_parity(unified, ref, locs, ps)

    def test_set_mla_kv_buffer_matches_stock_pool(self):
        """Both kernel paths at both page sizes: n_loc < 768 takes the Triton
        fallback, n_loc >= 768 the TMA JIT fast path."""
        for ps, n_loc in ((1, 256), (1, 1024), (64, 256), (64, 1024)):
            with self.subTest(page_size=ps, n_loc=n_loc):
                self._run_set_mla(ps=ps, n_loc=n_loc)

    def test_set_kv_buffer_combined_write(self):
        for ps in (1, 64):
            unified, ref, max_tokens = _make_pools(ps)
            n_loc = 512
            locs = _rand_locs(max_tokens, ps, n_loc)
            torch.manual_seed(11)
            for l in (0, _L // 2, _L - 1):
                layer = types.SimpleNamespace(layer_id=l)
                k = torch.randn(n_loc, 1, _D, dtype=_DTYPE, device=_DEV)
                unified.set_kv_buffer(layer, _kernel_id(locs, ps), k, None)
                ref.set_kv_buffer(layer, locs, k, None)
            torch.cuda.synchronize()
            self._assert_parity(unified, ref, locs, ps, layers=(0, _L // 2, _L - 1))

    def test_get_mla_kv_buffer_roundtrip(self):
        for ps in (1, 64):
            unified, ref, max_tokens = _make_pools(ps)
            n_loc = 1024  # exercise both get paths against the same bytes
            locs = _rand_locs(max_tokens, ps, n_loc)
            torch.manual_seed(13)
            layer = types.SimpleNamespace(layer_id=3)
            nope = torch.randn(n_loc, 1, _LORA, dtype=_DTYPE, device=_DEV)
            rope = torch.randn(n_loc, 1, _ROPE, dtype=_DTYPE, device=_DEV)
            unified.set_mla_kv_buffer(layer, _kernel_id(locs, ps), nope, rope)
            got_nope, got_rope = unified.get_mla_kv_buffer(layer, _kernel_id(locs, ps))
            torch.cuda.synchronize()
            torch.testing.assert_close(got_nope, nope, rtol=0, atol=0)
            torch.testing.assert_close(got_rope, rope, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
