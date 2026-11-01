"""K-pool prefill row sharding: row arithmetic, then the real path on real TP ranks."""

import os
import tempfile
import unittest
from dataclasses import dataclass
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch
import torch.multiprocessing as mp

from sglang.srt.layers.attention.dsa import dsa_indexer_kpool as indexer_module
from sglang.srt.layers.attention.dsa.dsa_indexer_kpool import shard_row_chunks
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase, publish_build_topology

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
register_cuda_ci(est_time=60, stage="extra-b", runner_config="4-gpu-b200")


@dataclass
class _Metadata:
    topk_indices_offset: torch.Tensor
    kpool_extend_plan: object = None


def _run_ranks(rank, world, init_method):
    from sglang.srt.distributed.parallel_state import (
        GroupCoordinator,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.environ import envs
    from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod
    from sglang.srt.layers.attention.dsa.kpool_plan import init_kpool_extend_metadata
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.runtime_context import get_forward, get_parallel

    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world))
    torch.cuda.set_device(rank)
    init_distributed_environment(
        world_size=world,
        rank=rank,
        local_rank=rank,
        distributed_init_method=init_method,
        backend="nccl",
    )
    publish_build_topology(tp_size=world, world_rank=rank)
    initialize_model_parallel()
    device = torch.device(f"cuda:{rank}")

    # 12289 rows behind a cached prefix do not divide evenly across TP ranks,
    # and q_fp8 has padding rows past them.
    prefix, length, pad_rows = 20480, 32769, 5
    n_real, pages = length - prefix, length // 256
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        batch_size=1,
        req_pool_indices=torch.tensor([1], device=device),
        seq_lens=torch.tensor([length], device=device),
        seq_lens_cpu=torch.tensor([length]),
        extend_seq_lens_cpu=[n_real],
        attn_cp_metadata=None,
    )
    metadata = SimpleNamespace(
        attn_metadata=init_kpool_extend_metadata(
            _Metadata(torch.zeros(n_real, device=device, dtype=torch.int32)),
            batch,
            kpool=4,
            physical_page_size=64,
            index_page_size=64,
            topk_transform_method=TopkTransformMethod.RAGGED,
            full_real_page_table=torch.arange(
                pages * 4 + 4, device=device, dtype=torch.int32
            )[None],
            full_seqlens_expanded=torch.arange(
                prefix + 1, length + 1, device=device, dtype=torch.int32
            ),
        ),
        topk_transform_method=TopkTransformMethod.RAGGED,
    )
    # The same generator on every rank, because attention TP replicates inputs.
    generator = torch.Generator().manual_seed(7)
    keys = torch.randn((pages + 1, 64, 128), generator=generator)
    scales = torch.rand((pages + 1, 64), generator=generator) + 0.5
    buf = torch.cat(
        [
            keys.to(torch.float8_e4m3fn).view(torch.uint8).flatten(1),
            scales.view(torch.uint8).flatten(1),
        ],
        dim=1,
    ).to(device)
    q_fp8 = torch.randn((n_real + pad_rows, 32, 128), generator=generator).to(
        device, torch.float8_e4m3fn
    )
    weights = torch.rand((n_real + pad_rows, 32, 1), generator=generator).to(device)
    indexer = SimpleNamespace(
        index_kpool=4,
        index_topk=2048,
        _get_index_k_read_buffer=lambda pool, layer_id: buf,
        _fp8_mqa_logits=indexer_module.IndexerKPool._fp8_mqa_logits,
    )
    for name in ("_topk_from_kpool_logits", "_kpool_topk_by_row_chunks"):
        method = getattr(indexer_module.IndexerKPool, name)
        setattr(indexer, name, MethodType(method, indexer))

    def topk(enabled, q_rows):
        envs.SGLANG_OPT_USE_DSA_KPOOL_PREFILL_TP.set(enabled)
        return indexer_module.IndexerKPool._get_topk_ragged_kpool_plan(
            indexer, batch, 0, q_fp8[:q_rows], weights[:q_rows], metadata
        )

    # Eager prefill has no padding rows, so the gather needs more rows than the output.
    for q_rows in (n_real, n_real + pad_rows):
        with (
            patch.object(
                indexer_module,
                "get_token_to_kv_pool",
                return_value=SimpleNamespace(index_page_size=64),
            ),
            patch.object(
                GroupCoordinator,
                "all_gather_into_tensor",
                autospec=True,
                side_effect=GroupCoordinator.all_gather_into_tensor,
            ) as gather,
        ):
            replicated = topk(False, q_rows)
            gather.assert_not_called()
            sharded = topk(True, q_rows)
            gather.assert_called_once()
            with get_forward().scoped(attn_input_scattered=True):
                topk(True, q_rows)
            gather.assert_called_once()

        # Top-k order is not stable, and an equal score at the cutoff can swap a
        # selected group, so a few rows may differ. A misplaced shard moves thousands.
        same = (sharded.sort(dim=-1).values == replicated.sort(dim=-1).values).all(
            dim=-1
        )
        assert (~same).sum() < 5, f"{(~same).sum()} rows differ"
        assert sharded.shape == replicated.shape and (sharded[n_real:] == -1).all()
        gathered = [torch.empty_like(sharded) for _ in range(world)]
        torch.distributed.all_gather(
            gathered, sharded, group=get_parallel().attn_tp_group.device_group
        )
        assert all(torch.equal(other, sharded) for other in gathered)


class TestPrefillTPRows(CustomTestCase):
    def test_chunks_reassemble_without_missing_or_duplicate_rows(self):
        for num_rows in (8, 17, 4097, 16384):
            for world_size in (2, 4, 8):
                for chunk_size in (3, 1024, 16384):
                    chunks = tuple(
                        slice(i, min(i + chunk_size, num_rows))
                        for i in range(0, num_rows, chunk_size)
                    )
                    all_rows = []
                    for rank in range(world_size):
                        local, padded = shard_row_chunks(chunks, world_size, rank)
                        rows = [
                            i for chunk in local for i in range(chunk.start, chunk.stop)
                        ]
                        self.assertLessEqual(len(rows), padded // world_size)
                        self.assertEqual(padded % world_size, 0)
                        all_rows.extend(rows)
                    self.assertEqual(all_rows, list(range(num_rows)))

    def test_sharded_path_matches_replicated_on_tp_ranks(self):
        for world in (2, 4):
            if torch.cuda.device_count() < world:
                self.skipTest(f"requires {world} CUDA GPUs")
            with tempfile.TemporaryDirectory(prefix="kpool-tp-") as tmp:
                mp.spawn(_run_ranks, args=(world, f"file://{tmp}/init"), nprocs=world)


if __name__ == "__main__":
    unittest.main()
