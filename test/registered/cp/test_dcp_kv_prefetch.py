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

"""Multi-GPU functional bitwise parity tests for Zero-SM DCP KV prefetch."""

from __future__ import annotations

import socket
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.kernels.ops.kvcache.mla_buffer import get_mla_kv_buffer_triton
from sglang.srt.layers.dcp.comm import (
    all_gather_kv_cache_for_dcp,
    all_gather_kv_cache_for_mha_extend,
)
from sglang.srt.layers.dcp.kv_prefetch import DcpKvPrefetcher
from sglang.srt.layers.dcp.layout import filter_dcp_local_chunk_kv_indices
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="2-gpu-large")


class _Batch:
    pass


class _KvPoolView:
    def __init__(self, buffers: list[torch.Tensor]) -> None:
        self._buffers = buffers
        self.layer_transfer_counter = None
        self.kv_scale_buffer = None
        self.use_dsa = False

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        return self._buffers[layer_id]


class _KvPool:
    def __init__(self, buffers: list[torch.Tensor]) -> None:
        view = _KvPoolView(buffers)
        self.layer_transfer_counter = None
        self.layer_router = SimpleNamespace(
            layer=lambda lid: SimpleNamespace(
                view=view, key=lambda lid=lid: view.get_key_buffer(lid)
            )
        )


class _ProcessGroupAdapter:
    def __init__(self, device_group, cpu_group, world_size: int, rank: int) -> None:
        self.device_group = device_group
        self.cpu_group = cpu_group
        self.world_size = world_size
        self.rank_in_group = rank

    def all_gather_into_tensor(self, out: torch.Tensor, inp: torch.Tensor) -> None:
        dist.all_gather_into_tensor(out, inp, group=self.device_group)


def _make_kv_buffers(
    num_layers: int,
    num_slots: int,
    kv_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    strided: bool,
) -> list[torch.Tensor]:
    torch.manual_seed(seed)
    if strided:
        # Match token-major KVArena slot stride (num_layers * kv_dim).
        arena = torch.empty(
            (num_slots, num_layers, 1, kv_dim), dtype=torch.bfloat16, device=device
        ).normal_(0, 1)
        if dtype != torch.bfloat16:
            arena = arena.to(dtype)
        return [arena[:, lid, :, :] for lid in range(num_layers)]
    bufs = []
    for _ in range(num_layers):
        t = torch.randn((num_slots, 1, kv_dim), dtype=torch.bfloat16, device=device)
        bufs.append(t.to(dtype) if dtype != torch.bfloat16 else t)
    return bufs


def _local_indices(
    seq_lens: list[int],
    starts: list[int],
    num_slots: int,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    global_ids = torch.randint(
        0, num_slots, (sum(seq_lens),), generator=gen, dtype=torch.int32
    ).to(device)
    return filter_dcp_local_chunk_kv_indices(
        global_ids,
        torch.tensor(starts, dtype=torch.int32),
        torch.tensor(seq_lens, dtype=torch.int32),
    )


def _reference_gather(
    kv_buf: torch.Tensor,
    local_ids: torch.Tensor,
    kv_lora_rank: int,
    qk_rope_dim: int,
    dst_dtype: torch.dtype,
    seq_lens: list[int],
    starts: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cache_nope = torch.empty(
        (local_ids.numel(), 1, kv_lora_rank), dtype=dst_dtype, device=kv_buf.device
    )
    cache_rope = torch.empty(
        (local_ids.numel(), 1, qk_rope_dim), dtype=dst_dtype, device=kv_buf.device
    )
    if local_ids.numel() > 0:
        get_mla_kv_buffer_triton(kv_buf, local_ids, cache_nope, cache_rope)
    gathered = all_gather_kv_cache_for_dcp(
        cache_nope,
        cache_rope,
        torch.tensor(seq_lens, dtype=torch.int32),
        torch.tensor(starts, dtype=torch.int32),
    )
    kv_a, k_pe = gathered.split([kv_lora_rank, qk_rope_dim], dim=-1)
    return kv_a.squeeze(1).contiguous(), k_pe.contiguous(), gathered.contiguous()


def _run_bitwise_worker(rank: int, world_size: int, port: int) -> None:
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        cpu_group = dist.new_group(backend="gloo")
        group = _ProcessGroupAdapter(dist.group.WORLD, cpu_group, world_size, rank)

        kv_lora_rank, qk_rope_dim = 512, 64
        kv_dim = kv_lora_rank + qk_rope_dim
        num_layers, num_slots, max_cap = 4, 4096, 2048

        parallel = SimpleNamespace(
            dcp_enabled=True, dcp_size=world_size, dcp_rank=rank, dcp_group=group
        )
        platform = SimpleNamespace(is_sm100=False)

        configs = [
            # (kv_cache_dtype, out_dtype, dst_dtype, strided)
            (torch.bfloat16, torch.bfloat16, torch.bfloat16, True),
            (torch.float16, torch.float16, torch.float16, False),
            (torch.float8_e4m3fn, torch.bfloat16, torch.bfloat16, True),
            (torch.float8_e4m3fn, torch.bfloat16, torch.float8_e4m3fn, True),
            (torch.float8_e5m2, torch.bfloat16, torch.bfloat16, False),
        ]

        with (
            patch("sglang.srt.layers.dcp.comm.get_parallel", return_value=parallel),
            patch("sglang.srt.layers.dcp.layout.get_parallel", return_value=parallel),
            patch("sglang.srt.layers.dcp.comm.get_platform", return_value=platform),
        ):
            for idx, (kv_dtype, out_dtype, dst_dtype, strided) in enumerate(configs):
                pf = DcpKvPrefetcher(
                    group,
                    device,
                    kv_lora_rank=kv_lora_rank,
                    qk_rope_head_dim=qk_rope_dim,
                    kv_cache_dtype=kv_dtype,
                    out_dtype=out_dtype,
                    max_chunk_capacity=max_cap,
                )
                assert pf.ready, f"Prefetcher unavailable for {kv_dtype} -> {dst_dtype}"
                assert pf.out_pool.shape == (max_cap * kv_dim,)

                kv_buffers = _make_kv_buffers(
                    num_layers,
                    num_slots,
                    kv_dim,
                    kv_dtype,
                    device,
                    1000 + rank * 37 + idx,
                    strided,
                )
                pool = _KvPool(kv_buffers)

                # Learn layer order on prefix-free extend step (chunk 0).
                b0 = _Batch()
                for lid in range(num_layers):
                    pf.observe_layer(lid, b0)

                if dst_dtype != torch.float8_e4m3fn:
                    # 1. MHA_CHUNKED_KV: 2 chunks, multi-request, zero-len req & unaligned starts.
                    c0_lens, c0_starts = [37, 64, 13], [0, 0, 0]
                    c1_lens, c1_starts = [0, 29, 2], [37, 64, 13]
                    c0_ids = _local_indices(
                        c0_lens, c0_starts, num_slots, device, 11 + idx
                    )
                    c1_ids = _local_indices(
                        c1_lens, c1_starts, num_slots, device, 22 + idx
                    )
                    b_chunk = _Batch()
                    for lid in range(num_layers):
                        pf.begin_layer(
                            lid,
                            b_chunk,
                            pool,
                            mode="mha_chunked",
                            dst_dtype=dst_dtype,
                            kv_indices_list=[c0_ids, c1_ids],
                            seq_lens_list=[c0_lens, c1_lens],
                            starts_list=[c0_starts, c1_starts],
                        )
                        pf.start_first()
                        for ci, (ids, lens, starts) in enumerate(
                            [(c0_ids, c0_lens, c0_starts), (c1_ids, c1_lens, c1_starts)]
                        ):
                            res = pf.join_split(lid, ci)
                            assert res is not None
                            got_a, got_pe = res[0].clone(), res[1].clone()
                            pf.start_next(lid, ci)
                            ref_a, ref_pe, _ = _reference_gather(
                                kv_buffers[lid],
                                ids,
                                kv_lora_rank,
                                qk_rope_dim,
                                dst_dtype,
                                lens,
                                starts,
                            )
                            assert torch.equal(got_a, ref_a)
                            assert torch.equal(got_pe, ref_pe)

                    # 2. MHA_ONE_SHOT: tiny unaligned req (empty local rows on some ranks) + TP-pad interleave.
                    os_lens, os_starts = [2], [0]
                    os_ids = _local_indices(
                        os_lens, os_starts, num_slots, device, 33 + idx
                    )
                    ext_a = torch.randn(
                        (5, kv_lora_rank), dtype=dst_dtype, device=device
                    )  # 3 real + 2 TP pad
                    ext_pe = torch.randn(
                        (5, 1, qk_rope_dim), dtype=dst_dtype, device=device
                    )
                    b_one = _Batch()
                    for lid in range(num_layers):
                        pf.begin_layer(
                            lid,
                            b_one,
                            pool,
                            mode="mha_one_shot",
                            dst_dtype=dst_dtype,
                            kv_indices_list=[os_ids],
                            seq_lens_list=[os_lens],
                        )
                        pf.start_first()
                        res = pf.join_split(lid, 0)
                        assert res is not None
                        got_cat_a, got_cat_pe = all_gather_kv_cache_for_mha_extend(
                            None,
                            None,
                            None,
                            os_lens,
                            [3],
                            ext_a,
                            ext_pe,
                            prefetched=res,
                        )
                        pf.start_next(lid, 0)
                        ref_a, ref_pe, _ = _reference_gather(
                            kv_buffers[lid],
                            os_ids,
                            kv_lora_rank,
                            qk_rope_dim,
                            dst_dtype,
                            os_lens,
                            os_starts,
                        )
                        assert torch.equal(
                            got_cat_a, torch.cat([ref_a, ext_a[:3]], dim=0)
                        )
                        assert torch.equal(
                            got_cat_pe, torch.cat([ref_pe, ext_pe[:3]], dim=0)
                        )
                else:
                    # 3. Absorbed MLA: FP8 -> FP8 combined output buffer.
                    mla_lens, mla_starts = [41, 19], [0, 0]
                    mla_ids = _local_indices(
                        mla_lens, mla_starts, num_slots, device, 44 + idx
                    )
                    b_mla = _Batch()
                    for lid in range(num_layers):
                        pf.begin_layer(
                            lid,
                            b_mla,
                            pool,
                            mode="mla_absorb",
                            dst_dtype=dst_dtype,
                            kv_indices_list=[mla_ids],
                            seq_lens_list=[mla_lens],
                        )
                        pf.start_first()
                        got = pf.join_combined(lid, 0)
                        assert got is not None
                        got_copy = got.clone()
                        pf.start_next(lid, 0)
                        _, _, ref_comb = _reference_gather(
                            kv_buffers[lid],
                            mla_ids,
                            kv_lora_rank,
                            qk_rope_dim,
                            dst_dtype,
                            mla_lens,
                            mla_starts,
                        )
                        assert torch.equal(
                            got_copy.view(torch.uint8), ref_comb.view(torch.uint8)
                        )

                torch.cuda.synchronize(device)
                dist.barrier(group=cpu_group)
    finally:
        dist.destroy_process_group()


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestDcpKvPrefetch(unittest.TestCase):
    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.device_count() >= 2,
        "requires at least 2 CUDA GPUs",
    )
    def test_multi_gpu_bitwise_parity(self):
        world_size = 4 if torch.cuda.device_count() >= 4 else 2
        mp.spawn(
            _run_bitwise_worker,
            args=(world_size, _find_free_port()),
            nprocs=world_size,
            join=True,
        )


if __name__ == "__main__":
    unittest.main()
