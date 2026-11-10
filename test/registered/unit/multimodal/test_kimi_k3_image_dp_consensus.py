"""Two real Gloo ranks: cache drift heals and local failures stop before payloads."""

import datetime
import json
import socket
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.managers import mm_schedule
from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult, MultiModalStaticCache
from sglang.srt.multimodal.image_dp_consensus import ImageDPCoordinationError
from sglang.srt.multimodal.mm_utils import run_dp_sharded_mrope_vision_model
from sglang.srt.multimodal.modality import Modality
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

CASES = [
    "cold",
    "hot",
    "partial",
    "drift",
    "stale_union",
    "one_hit",
    "empty",
    "identity",
    "missing_input",
    "memory",
    "early",
    "projector",
    "grid",
    "padding",
    "split",
    "split_memory",
    "by_item",
    "full",
]
FAILURES = {
    "identity",
    "missing_input",
    "memory",
    "early",
    "projector",
    "grid",
    "padding",
    "split_memory",
}


def _key(item):
    return getattr(item, "cache_key", item.hash)


def _worker(rank, port, output):
    import sglang.srt.multimodal.mm_utils as helpers

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=10),
    )

    class Group:
        world_size = 2
        rank_in_group = rank
        cpu_group = dist.group.WORLD

        def __init__(self):
            self.payloads = 0

        def broadcast(self, tensor, src):
            self.payloads += 1
            dist.broadcast(tensor, src)

        def all_gather(self, tensor, dim):
            self.payloads += 1
            values = [torch.empty_like(tensor) for _ in range(2)]
            dist.all_gather(values, tensor)
            return torch.cat(values, dim)

    class Tower:
        device = torch.device("cpu")
        merge_kernel_size = (1, 1)
        config = SimpleNamespace(hidden_size=1)
        patch_embed = SimpleNamespace(proj=SimpleNamespace(weight=torch.empty(1)))

        def __call__(self, pixels, **_kwargs):
            if rank == 0 and (
                owner.mode == "memory"
                or owner.mode == "split_memory"
                and pixels[0, 0] == 2
            ):
                raise MemoryError("injected owner failure")
            return pixels.reshape(-1, 1, 1)

    class Owner:
        mode = "cold"

        def get_image_dp_group(self):
            return group

        def get_image_feature(self, misses):
            if self.mode == "early" and rank == 0:
                raise ValueError("injected preprocessing failure")

            def run(items):
                grids = [
                    [1, 2, 2] if self.mode == "grid" and rank == 1 else [1, 1, 4]
                ] * len(items)
                return run_dp_sharded_mrope_vision_model(
                    Tower(),
                    torch.cat([item.feature for item in items]),
                    grids,
                    rope_type="rope_2d",
                    parallel_group=group,
                )

            result = (
                torch.cat([run([item]) for item in misses])
                if self.mode in ("split", "split_memory")
                else run(misses)
            )
            if self.mode == "projector" and rank == 0:
                raise MemoryError("injected projector failure")
            return result

    group, owner = Group(), Owner()
    rows = []
    original_pad = helpers._pad_mrope_vision_embeddings_for_tp_gather
    try:
        with get_parallel().override(attn_tp_size=1, attn_tp_rank=0):
            for case in CASES:
                dist.barrier()
                owner.mode = case
                group.payloads = 0
                cache = MultiModalStaticCache(4096)
                mm_schedule.embedding_cache = cache
                items = [
                    mm_schedule.MultimodalDataItem(
                        modality=Modality.IMAGE,
                        hash=i + 1,
                        offsets=[(4 * i, 4 * i + 3)],
                        feature=torch.full((4, 1), float(i + 1)),
                    )
                    for i in range(2)
                ]
                seed = items if case == "hot" or case == "one_hit" and rank == 0 else []
                if case in ("drift", "by_item"):
                    seed = [items[rank]]
                elif case == "partial":
                    seed = [items[0]]
                if case == "stale_union" and rank == 0:
                    cache.set(
                        _key(items[0]),
                        EmbeddingResult(embedding=items[1].feature.clone()),
                    )
                for item in seed:
                    cache.set(
                        _key(item), EmbeddingResult(embedding=item.feature.clone())
                    )
                if case == "full" and rank == 0:
                    cache.set(
                        MultiModalStaticCache.combine_hashes([_key(i) for i in items]),
                        EmbeddingResult(
                            embedding=torch.cat([i.feature for i in items])
                        ),
                    )
                selected = (
                    list(reversed(items)) if case == "identity" and rank == 1 else items
                )
                if case == "missing_input" and rank == 1 or case == "empty":
                    selected = []
                request = mm_schedule.PerImageRequestInfo(
                    0, selected, [(0, 3), (4, 7)], 0, 8
                )
                if case == "padding" and rank == 0:

                    def fail_pad(*_args):
                        raise MemoryError("injected gather allocation failure")

                    helpers._pad_mrope_vision_embeddings_for_tp_gather = fail_pad
                error = None
                try:
                    if case == "by_item":
                        result = mm_schedule._get_chunked_embedding_by_item(
                            owner.get_image_feature, items, 0, 8, torch.device("cpu")
                        )
                        assert torch.equal(
                            torch.cat(result), torch.cat([i.feature for i in items])
                        )
                    elif case == "full":
                        result, _ = mm_schedule._get_chunked_embedding_full(
                            owner.get_image_feature,
                            items,
                            [(0, 3), (4, 7)],
                            0,
                            8,
                            torch.arange(8),
                            torch.device("cpu"),
                        )
                        assert torch.equal(
                            result, torch.cat([i.feature for i in items])
                        )
                    else:
                        result = mm_schedule._batch_encode_per_image_misses(
                            owner.get_image_feature, [request], torch.device("cpu")
                        )
                        for item in selected:
                            assert torch.equal(result[(_key(item), 4)], item.feature)
                except ImageDPCoordinationError as exc:
                    error = str(exc)
                finally:
                    helpers._pad_mrope_vision_embeddings_for_tp_gather = original_pad
                assert bool(error) == (case in FAILURES), (case, rank, error)
                if case in FAILURES - {"projector", "split_memory"}:
                    assert group.payloads == 0, (case, group.payloads)
                if case == "hot":
                    assert group.payloads == 0
                if error:
                    # The metadata communicator must remain usable after a coordinated stop.
                    owner.mode = "cold"
                    request = mm_schedule.PerImageRequestInfo(
                        0, items, [(0, 3), (4, 7)], 0, 8
                    )
                    recovered = mm_schedule._batch_encode_per_image_misses(
                        owner.get_image_feature, [request], torch.device("cpu")
                    )
                    for item in items:
                        assert torch.equal(recovered[(_key(item), 4)], item.feature)
                rows.append(
                    dict(case=case, coordinated_error=error, next_request_ok=True)
                )
        (Path(output) / f"rank{rank}.json").write_text(json.dumps(rows, indent=2))
    finally:
        dist.destroy_process_group()


class TestImageDPConsensus(CustomTestCase):
    def test_real_gloo_cache_drift_and_failures(self):
        with tempfile.TemporaryDirectory() as output:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            mp.spawn(_worker, args=(port, output), nprocs=2, join=True)
            rows = [
                json.loads((Path(output) / f"rank{r}.json").read_text()) for r in [0, 1]
            ]
            self.assertEqual([r["case"] for r in rows[0]], CASES)
            print(
                f"Validated {len(CASES)} scenarios on two real Gloo ranks", flush=True
            )
            self.assertEqual(
                [bool(r["coordinated_error"]) for r in rows[0]],
                [bool(r["coordinated_error"]) for r in rows[1]],
            )


if __name__ == "__main__":
    unittest.main()
