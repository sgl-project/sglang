"""PP source assembly with real Gloo collectives and rank-local KV slots."""

import json
import multiprocessing as mp
import tempfile
import time
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import torch
import torch.distributed as dist
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    UnquantizedKVCacheMethod,
)
from sglang.srt.speculative.dspark_components import (
    dspark_target_kv_inject as injection,
)
from sglang.srt.speculative.dspark_components.dspark_target_kv_encoder import (
    TargetKVContextEncoder,
)
from sglang.srt.training_capture.identity import LocalLayerContract, RankTargetContract
from sglang.srt.training_capture.protocol import DTYPES, canonical_bytes
from sglang.srt.training_capture.startup import CaptureStartupError
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dspark_target_kv_utils import (
    RecordingTargetKVDraft,
    make_target_kv_contract,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=45, suite="base-a-test-cpu")


class GlooGroup:
    broadcast = GroupCoordinator.broadcast

    def __init__(self, ranks, group, rank):
        self.ranks = ranks
        self.world_size = len(ranks)
        self.rank_in_group = ranks.index(rank)
        self.cpu_group = self.device_group = group

    def all_gather(self, value, *, dim):
        values = [torch.empty_like(value) for _ in self.ranks]
        dist.all_gather(values, value, group=self.device_group)
        return torch.cat(values, dim=dim)


def make_groups(rank, tp_size, pp_size):
    tp_group = pp_group = None
    for stage in range(pp_size):
        ranks = list(range(stage * tp_size, (stage + 1) * tp_size))
        group = dist.new_group(ranks, timeout=timedelta(seconds=20))
        if rank in ranks:
            tp_group = GlooGroup(ranks, group, rank)
    for tensor_rank in range(tp_size):
        ranks = list(range(tensor_rank, 4, tp_size))
        group = dist.new_group(ranks, timeout=timedelta(seconds=20))
        if rank in ranks:
            pp_group = GlooGroup(ranks, group, rank)
    return tp_group, pp_group


def make_fixture(rank, tp_group, pp_group, heads, dtype):
    contract = make_target_kv_contract()
    contract = msgspec.structs.replace(
        contract,
        kv=msgspec.structs.replace(
            contract.kv,
            dtype=dtype,
            codec=f"dense_{'bf16' if dtype == 'bfloat16' else 'fp16'}_post_rope_v1",
            layers=[
                msgspec.structs.replace(layer, num_kv_heads=heads)
                for layer in contract.kv.layers
            ],
        ),
    )
    contract = type(contract).decode(contract)
    start = pp_group.rank_in_group * (4 // pp_group.world_size)
    end = start + 4 // pp_group.world_size
    local_heads = max(1, heads // tp_group.world_size)
    replicas = max(1, tp_group.world_size // heads)
    first = (tp_group.rank_in_group // replicas) * local_heads
    local_contract = RankTargetContract(
        teacher=contract.teacher,
        tp_rank=tp_group.rank_in_group,
        tp_size=tp_group.world_size,
        pp_rank=pp_group.rank_in_group,
        pp_size=pp_group.world_size,
        dp_rank=0,
        pp_layer_range=(start, end),
        num_attention_layers=4,
        selected_layer_ids=contract.kv.selected_layer_ids,
        dtype=dtype,
        source_page_size=contract.kv.source_page_size,
        storage_chunk_tokens=contract.kv.storage_chunk_tokens,
        layers=[
            LocalLayerContract(
                geometry=layer,
                head_range=(first, first + local_heads),
                rope_config=contract.kv.rope_config,
                source_k_norm=contract.kv.source_k_norm,
            )
            for layer in contract.kv.layers
            if start <= layer.layer_id < end
        ],
    )
    slots = (torch.arange(9) + 2 * rank) % 9
    source, buffers = {}, {}
    generator = torch.Generator().manual_seed(172)
    for layer in contract.kv.layers:
        for component, dim in (("k", layer.key_head_dim), ("v", layer.value_head_dim)):
            name = f"target_{component}.{layer.layer_id}"
            source[name] = torch.randn(
                9, heads, dim, dtype=DTYPES[dtype], generator=generator
            )
            if start <= layer.layer_id < end:
                value = source[name][:, first : first + local_heads].clone()
                if tp_group.rank_in_group % replicas:
                    value.fill_(-999)
                buffers[name] = torch.empty_like(value)
                buffers[name][slots] = value
    target_pool = SimpleNamespace(
        get_key_buffer=lambda layer: buffers[f"target_k.{layer}"],
        get_value_buffer=lambda layer: buffers[f"target_v.{layer}"],
    )
    draft_pool = object.__new__(MHATokenToKVPool)
    draft_pool.quant_method = UnquantizedKVCacheMethod()
    draft_pool.use_hnd = False
    writer = RecordingTargetKVDraft()
    writer.target_kv_contract = contract
    injector = injection.TargetKVInjector(
        draft_model=writer,
        draft_model_runner=SimpleNamespace(
            token_to_kv_pool=draft_pool,
            model_config=SimpleNamespace(model_path="draft-fixture"),
        ),
        model_runner=SimpleNamespace(
            tp_group=tp_group,
            pp_group=pp_group,
            model=object(),
            model_config=SimpleNamespace(hf_text_config=SimpleNamespace(hidden_size=8)),
            token_to_kv_pool=target_pool,
        ),
    )
    return injector, local_contract, source, slots, buffers


def exercise_injector(injector, source, slots, buffers):
    assert set(injector.sources) == set(buffers)
    positions = torch.tensor([7, 1, 4])
    selected = injector._select_target_kv(slots[positions])
    assert list(selected) == list(source)
    for name, value in selected.items():
        torch.testing.assert_close(value, source[name][positions], rtol=0, atol=0)

    torch.manual_seed(17)
    encoder = TargetKVContextEncoder(injector.draft_model.target_kv_contract).to(
        dtype=next(iter(source.values())).dtype
    )
    torch.testing.assert_close(
        encoder(selected, positions),
        encoder({name: value[positions] for name, value in source.items()}, positions),
        rtol=0,
        atol=0,
    )

    req = SimpleNamespace(dspark_projected_context=None)
    for start, end in ((0, 3), (3, 5)):
        result = injector.inject_target_kv(
            req,
            committed_prefix_end=end,
            source_ranges=(
                injection.TargetKVSourceRange(
                    start=start,
                    positions=torch.arange(start, end),
                    cache_locs=slots[start:end],
                ),
            ),
            draft_weight_version=injector.weight_version,
        )
        assert result is None
        values, actual_positions, destination = injector.draft_model.writes[-1]
        torch.testing.assert_close(actual_positions, torch.arange(start, end))
        torch.testing.assert_close(destination, slots[start:end])
        for name, value in values.items():
            torch.testing.assert_close(value, source[name][start:end], rtol=0, atol=0)
        assert req.dspark_projected_context.end == end
    assert injector.projected_token_ct == 5
    assert len(injector.draft_model.writes) == 2


CASES = (
    (2, 2, 2, "bfloat16"),
    (2, 2, 1, "bfloat16"),
    (1, 4, 2, "bfloat16"),
    (4, 1, 2, "bfloat16"),
    (4, 1, 8, "bfloat16"),
    (2, 2, 2, "float16"),
)


def pipeline_worker(rank, root):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=(Path(root) / "rendezvous").as_uri(),
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=20),
    )
    results = []
    try:
        for tp_size, pp_size, heads, dtype in CASES:
            tp_group, pp_group = make_groups(rank, tp_size, pp_size)
            # Stage zero owns no selected layers in PP4, but must vote on errors.
            attempts = (
                ("binding", "pool", "weights", "ready") if pp_size == 4 else ("ready",)
            )
            for attempt in attempts:
                injector, local, source, slots, buffers = make_fixture(
                    rank, tp_group, pp_group, heads, dtype
                )
                if attempt == "pool" and rank == 0:
                    injector.draft_model_runner.token_to_kv_pool.use_hnd = True

                def bind_local(local=local, attempt=attempt, **kwargs):
                    assert kwargs["tp_rank"] == local.tp_rank
                    assert kwargs["pp_rank"] == local.pp_rank
                    assert kwargs["tp_size"] == local.tp_size
                    assert kwargs["pp_size"] == local.pp_size
                    if attempt == "binding" and rank == 0:
                        raise OSError("local target artifact unavailable")
                    return local

                digest = "b" * 64 if attempt == "weights" and rank == 0 else "a" * 64
                with (
                    patch.object(injection, "bind_rank_target_contract", bind_local),
                    patch.object(
                        injection, "local_safetensors_digest", return_value=digest
                    ),
                    patch.object(
                        injection,
                        "get_world_group",
                        return_value=SimpleNamespace(cpu_group=dist.group.WORLD),
                    ),
                ):
                    try:
                        injector.bind(
                            tokenizer_path="tokenizer-fixture",
                            prediction_count=3,
                            mask_token_id=255,
                        )
                    except CaptureStartupError as error:
                        expected = {
                            "binding": ("binding", (0,)),
                            "pool": ("policy", (0,)),
                            "weights": ("policy_agreement", (0, 1, 2, 3)),
                        }
                        assert (error.phase, error.failed_ranks) == expected[attempt]
                        assert injector.sources is None
                        results.append({"attempt": attempt, "phase": error.phase})
                    else:
                        assert attempt == "ready"
                        exercise_injector(injector, source, slots, buffers)
                        results.append(
                            {
                                "attempt": attempt,
                                "tp": tp_size,
                                "pp": pp_size,
                                "heads": heads,
                                "dtype": dtype,
                                "local_sources": len(buffers),
                            }
                        )
                if rank == 0:
                    print(json.dumps(results[-1], sort_keys=True), flush=True)
        (Path(root) / f"rank-{rank}.json").write_bytes(canonical_bytes(results))
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo required")
class TestTargetKVPipeline(CustomTestCase):
    def test_pipeline_sources_and_startup_failures(self):
        with tempfile.TemporaryDirectory() as root:
            context = mp.get_context("spawn")
            workers = [
                context.Process(target=pipeline_worker, args=(rank, root))
                for rank in range(4)
            ]
            try:
                for worker in workers:
                    worker.start()
                deadline = time.monotonic() + 120
                for worker in workers:
                    worker.join(timeout=max(0, deadline - time.monotonic()))
                self.assertEqual([worker.exitcode for worker in workers], [0] * 4)
                results = [
                    json.loads((Path(root) / f"rank-{rank}.json").read_bytes())
                    for rank in range(4)
                ]
                for rows in results:
                    self.assertEqual(len(rows), len(CASES) + 3)
                pp4 = [
                    next(row for row in rows if row.get("pp") == 4) for rows in results
                ]
                self.assertEqual([row["local_sources"] for row in pp4], [0, 2, 0, 2])
            finally:
                for worker in workers:
                    if worker.pid is not None and worker.is_alive():
                        worker.terminate()
                        worker.join(timeout=5)
                        if worker.is_alive():
                            worker.kill()
                            worker.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
