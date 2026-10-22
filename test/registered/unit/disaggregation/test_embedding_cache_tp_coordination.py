"""Exercise real TP cache coordination with CPU substitutes for GPU/storage I/O."""

import asyncio
import tempfile
import time
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist

import sglang.srt.disaggregation.encoder.server as encoder_server
from sglang.srt.disaggregation.encoder.server import MMEncoder
from sglang.srt.managers.schedule_batch import Modality
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=25, suite="base-a-test-cpu")


class _CPUEmbeddingCache:
    """Implement cache boundaries without CUDA, RDMA, or a storage service."""

    def __init__(self):
        self.global_entries = {}
        self.local_entries = {}
        self.fail_lookup = False
        self.fail_get = False
        self.fail_prefetch = False
        self.fail_wait = False
        self.puts = []

    async def batch_is_exist(self, hashes):
        if self.fail_lookup:
            raise RuntimeError("injected lookup failure")
        return [h in self.local_entries or h in self.global_entries for h in hashes]

    def prefetch(self, req_id, hashes, token_counts, modality):
        if self.fail_prefetch:
            raise RuntimeError("injected prefetch failure")
        if not self.fail_get:
            for h in hashes:
                if h not in self.local_entries:
                    self.local_entries[h] = self.global_entries[h].clone()

    def check_prefetch_progress(self, req_id):
        if self.fail_wait:
            raise TimeoutError("injected prefetch completion failure")
        return True

    def has_local_embedding(self, h):
        return h in self.local_entries

    def store_to_pool_async(self, hashes, tensors, modality):
        for h, tensor in zip(hashes, tensors):
            self.local_entries[h] = tensor.clone()
        return []

    def wait_store_to_pool(self, handles):
        pass

    def insert_batch(self, hashes, modality):
        for h in hashes:
            self.global_entries[h] = self.local_entries[h].clone()
        self.puts.extend(hashes)

    def get_embedding_dim(self, modality):
        return 4

    def get_pool_views(self, hashes):
        return [[self.local_entries[h]] for h in hashes]

    def release_pool_views(self, hashes):
        pass


async def _run_cache_scenarios(encoder, rank):
    token_counts = [2, 3]
    expected_slices = [
        torch.full((count, 4), idx + 1, dtype=torch.float32)
        for idx, count in enumerate(token_counts)
    ]
    scenarios = (
        ("all_hit", [0, 1], {}, []),
        ("partial_hit", [0], {}, [[1]]),
        ("all_miss", [], {}, [[0, 1]]),
        ("lookup_failure", [0, 1], {"fail_lookup": True}, [[0, 1]]),
        ("get_failure", [0], {"fail_get": True}, [[1], [0]]),
        ("prefetch_failure", [0], {"fail_prefetch": True}, [[1], [0]]),
        ("completion_failure", [0], {"fail_wait": True}, [[1], [0]]),
    )
    for modality in (Modality.IMAGE, Modality.VIDEO, Modality.AUDIO):
        for name, hits, failures, expected_calls in scenarios:
            cache = encoder.mm_global_cache
            hashes = [f"{modality.name}-{name}-{i}" for i in range(2)]
            if rank == 0:
                cache.global_entries = {hashes[i]: expected_slices[i] for i in hits}
                cache.local_entries.clear()
                cache.puts.clear()
                for flag in ("fail_lookup", "fail_get", "fail_prefetch", "fail_wait"):
                    setattr(cache, flag, failures.get(flag, False))
            forward_calls = []

            def get_features(indices):
                forward_calls.append(list(indices))
                embedding = torch.cat([expected_slices[i] for i in indices])
                # A model collective makes mismatched forward decisions fail.
                dist.all_reduce(embedding)
                return embedding / dist.get_world_size()

            ctx = SimpleNamespace(
                req_id=f"{modality.name}-{name}",
                num_items=2,
                modality=modality,
                str_mm_hashes=hashes if rank == 0 else None,
                mm_feature=None,
                get_feature_fn=get_features,
                preprocess_result=SimpleNamespace(token_counts=token_counts),
            )
            result = await encoder._compute_global_cache_embedding(
                ctx, keep_on_gpu=False
            )
            if encoder.background_tasks:
                await asyncio.gather(*list(encoder.background_tasks))
            assert forward_calls == expected_calls, (rank, name, forward_calls)
            traces = [None, None]
            dist.all_gather_object(traces, forward_calls)
            assert traces[0] == traces[1], traces
            if rank == 0:
                torch.testing.assert_close(result, torch.cat(expected_slices))
                assert result.shape == (5, 4)
                assert result.dtype == torch.float32
                assert len(cache.puts) == sum(
                    len(indices) for indices in expected_calls
                )
                # Repeat without resetting the cache: local hits skip all forwards.
                for flag in ("fail_lookup", "fail_get", "fail_prefetch", "fail_wait"):
                    setattr(cache, flag, False)
            else:
                assert result is None
                assert encoder.mm_global_cache is None
            forward_calls.clear()
            result = await encoder._compute_global_cache_embedding(
                ctx, keep_on_gpu=False
            )
            assert not forward_calls, (rank, name, forward_calls)
            if rank == 0:
                torch.testing.assert_close(result, torch.cat(expected_slices))
            else:
                assert result is None


def _cache_coordination_worker(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    try:
        encoder = MMEncoder.__new__(MMEncoder)
        encoder.rank = rank
        encoder._embedding_dims = {Modality.IMAGE: 4}
        encoder._embedding_dtype = torch.float32
        encoder.device = "cpu"
        encoder.background_tasks = set()
        encoder._build_model_mm_items = lambda feature, result, indices, modality: (
            indices
        )
        parallel = SimpleNamespace(
            tp_size=2,
            tp_rank=rank,
            tp_group=SimpleNamespace(cpu_group=dist.group.WORLD),
        )
        real_empty = torch.empty

        def cpu_empty(*args, **kwargs):
            kwargs.pop("pin_memory", None)
            return real_empty(*args, **kwargs)

        with (
            patch.object(encoder_server, "get_parallel", return_value=parallel),
            patch.object(
                encoder_server,
                "get_mm",
                return_value=SimpleNamespace(
                    enable_mm_global_cache=True, mm_global_cache_backend="mooncake"
                ),
            ),
            patch(
                "sglang.srt.mem_cache.embedding_store.EmbeddingStoreFactory.create_backend"
            ) as backend,
            patch(
                "sglang.srt.mem_cache.embedding_cache_controller.EmbeddingCacheController",
                side_effect=lambda **kwargs: _CPUEmbeddingCache(),
            ) as controller,
            patch.object(encoder_server, "encoder_metrics_collector", None),
            patch.object(torch, "empty", side_effect=cpu_empty),
            patch.object(torch.cuda, "current_stream", return_value=Mock()),
        ):
            # Real initialization and mask groups: neither collective is mocked.
            encoder._init_global_cache()
            assert encoder.mm_global_cache_enabled
            assert encoder.global_cache_tp_group is not None
            assert controller.call_count == backend.call_count == int(rank == 0)
            asyncio.run(_run_cache_scenarios(encoder, rank))
    finally:
        dist.destroy_process_group()


class TestEmbeddingCacheTPCoordination(CustomTestCase):
    @unittest.skipUnless(dist.is_gloo_available(), "requires Gloo")
    def test_two_rank_cache_pipeline(self):
        with tempfile.TemporaryDirectory(prefix="embedding-cache-tp-") as directory:
            context = torch.multiprocessing.spawn(
                _cache_coordination_worker,
                args=((Path(directory) / "store").as_uri(),),
                nprocs=2,
                join=False,
            )
            deadline = time.monotonic() + 90
            try:
                while not context.join(timeout=1):
                    if time.monotonic() >= deadline:
                        self.fail("TP cache coordination exceeded 90 seconds")
            finally:
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
