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
"""MXFP8 KV scales on the unified memory pool (needs FA4 / SM100).

A scale that lands on the wrong page is silent: the payload still decodes,
just against another slot's exponents.
"""

import unittest

import pytest
import torch

from sglang.srt.mem_cache.pool_host.mha_mxfp8 import MHATokenToKVPoolMXFP8Host
from sglang.srt.mem_cache.unified_memory_pool import (
    MHASubPoolSpec,
    UnifiedKVPool,
    build_unified_mha_pool,
)
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

requires_sm100 = pytest.mark.skipif(
    not torch.cuda.is_available() or get_device_sm() < 100,
    reason="MXFP8 KV cache requires the FA4 backend (SM100+).",
)

_DEV = "cuda"
_SBS = 32
_PS = 128
_H = 2
_D = 128  # the interleave kernel requires sf_dim == 4
_L = 2
_SF_DIM = _D // _SBS
_NUM_PAGES = 8


def _spec(name, grow):
    return MHASubPoolSpec(
        name=name,
        layer_num=_L,
        head_num=_H,
        head_dim=_D,
        store_dtype=torch.uint8,
        kv_cache_dtype=torch.float8_e4m3fn,
        scale_block_size=_SBS,
        grow_direction=grow,
    )


def _build_pool():
    full = _spec("full", "down")
    buffer = UnifiedKVPool(
        total_bytes=_NUM_PAGES * _PS * full.entry_bytes(),
        sub_pool_specs=[full, _spec("swa", "up")],
        device=_DEV,
        enable_memory_saver=False,
        page_size=_PS,
    )
    return buffer, build_unified_mha_pool(
        unified_buffer=buffer, sub_pool_name="full", page_size=_PS
    )


def _random_kv(num_tokens):
    payload = torch.randint(
        1, 200, (num_tokens, _H, _D), dtype=torch.uint8, device=_DEV
    ).view(torch.float8_e4m3fn)
    scales = torch.randint(
        100, 140, (num_tokens, _H, _SF_DIM), dtype=torch.uint8, device=_DEV
    ).view(torch.float8_e8m0fnu)
    return payload, scales


@requires_sm100
class TestUnifiedMXFP8Pool(unittest.TestCase):
    def setUp(self):
        self.buffer, self.pool = _build_pool()

    def _write(self, page, slots, layer_id=0):
        loc = torch.tensor(
            [page * _PS + s for s in slots], dtype=torch.int64, device=_DEV
        )
        k, k_sf = _random_kv(len(slots))
        v, v_sf = _random_kv(len(slots))
        self.pool.set_kv_buffer(None, loc, k, v, k_sf, v_sf, layer_id_override=layer_id)
        return loc, (k, v, k_sf, v_sf)

    def test_payload_and_scales_round_trip(self):
        loc, (k, v, k_sf, v_sf) = self._write(page=3, slots=[0, 1, 17, 127])

        self.assertTrue(
            torch.equal(
                self.pool.k_buffer[0][loc].view(torch.uint8), k.view(torch.uint8)
            )
        )
        got_k_sf, got_v_sf = self.pool._read_scales(0, loc)
        self.assertTrue(torch.equal(got_k_sf.view(torch.uint8), k_sf.view(torch.uint8)))
        self.assertTrue(torch.equal(got_v_sf.view(torch.uint8), v_sf.view(torch.uint8)))

    def test_scales_land_on_this_slots_page(self):
        self._write(page=3, slots=[5], layer_id=0)

        self.assertGreater(
            int(self.pool.k_scale_buffer[0][3].view(torch.uint8).max()), 0
        )
        for buf in (
            self.pool.k_scale_buffer[0][2],
            self.pool.v_scale_buffer[0][3],
            self.pool.k_scale_buffer[1][3],
        ):
            self.assertEqual(int(buf.view(torch.uint8).max()), 0)

    def test_move_kv_cache_carries_scales(self):
        slots = list(range(_PS))
        _, (k, _, k_sf, v_sf) = self._write(page=3, slots=slots)
        src = torch.tensor([3 * _PS + s for s in slots], dtype=torch.int64, device=_DEV)
        tgt = torch.tensor([5 * _PS + s for s in slots], dtype=torch.int64, device=_DEV)

        self.pool.move_kv_cache(tgt, src)

        moved = tgt
        self.assertTrue(
            torch.equal(
                self.pool.k_buffer[0][moved].view(torch.uint8), k.view(torch.uint8)
            )
        )
        got_k_sf, got_v_sf = self.pool._read_scales(0, moved)
        self.assertTrue(torch.equal(got_k_sf.view(torch.uint8), k_sf.view(torch.uint8)))
        self.assertTrue(torch.equal(got_v_sf.view(torch.uint8), v_sf.view(torch.uint8)))


@requires_sm100
class TestUnifiedMXFP8HostRoundTrip(unittest.TestCase):
    def setUp(self):
        self.buffer, self.pool = _build_pool()
        self.host = MHATokenToKVPoolMXFP8Host(
            self.pool,
            host_to_device_ratio=1.0,
            host_size=0,
            page_size=_PS,
            layout="page_first",
            pin_memory=False,
            device="cpu",
        )

    def tearDown(self):
        self.host.destroy()

    def test_scales_round_trip_through_the_host_tier(self):
        slots = list(range(_PS))
        loc = torch.tensor([3 * _PS + s for s in slots], dtype=torch.int64, device=_DEV)
        expected = []
        for layer in range(_L):
            k, k_sf = _random_kv(len(slots))
            v, v_sf = _random_kv(len(slots))
            self.pool.set_kv_buffer(
                None, loc, k, v, k_sf, v_sf, layer_id_override=layer
            )
            expected.append((k_sf, v_sf))

        # The staged write-back takes host indices on the host, H2D on the device.
        host_loc = torch.arange(_PS, dtype=torch.int64)
        self.host.backup_from_device_all_layer(
            self.pool, host_loc, loc, io_backend="kernel"
        )
        torch.cuda.synchronize()
        for layer in range(_L):
            self.pool.k_scale_buffer[layer].view(torch.uint8).fill_(0)
            self.pool.v_scale_buffer[layer].view(torch.uint8).fill_(0)
        for layer in range(_L):
            self.host.load_to_device_per_layer(
                self.pool, host_loc.to(_DEV), loc, layer, io_backend="kernel"
            )
        torch.cuda.synchronize()

        for layer, (k_sf, v_sf) in enumerate(expected):
            got_k, got_v = self.pool._read_scales(layer, loc)
            self.assertTrue(
                torch.equal(got_k.view(torch.uint8), k_sf.view(torch.uint8))
            )
            self.assertTrue(
                torch.equal(got_v.view(torch.uint8), v_sf.view(torch.uint8))
            )


if __name__ == "__main__":
    unittest.main()
