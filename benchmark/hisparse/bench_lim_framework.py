"""Measure the real sparse-KV manager/attention path with BF16 native LI/LIM.

This is a fixed-input framework subpath benchmark, without model projections,
weights, scheduler or sampling. New-token D2H is optional. The native and LI reference modes
share prefix-TopK + visible-tail semantics. Legacy uses full-sequence TopK.
Run on a dedicated Ascend A2/A3 device with the operator library installed.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from array import array
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch_npu

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.attention import (
    forward_sparsity_driven_kv_offload,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.manager import (
    SparseKVCacheManager,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool, ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams

K, PAGE, CKV, KPE = 2048, 128, 512, 64


def inputs(
    batch, length, misses, index_heads, query_heads, seed, hotset_layout="scattered"
):
    """Wide score margins keep TopK set parity independent of ties/order."""
    rng = random.Random(seed)
    generator = torch.Generator().manual_seed(seed)
    context = length + PAGE
    keys = torch.zeros(batch, context, 128)
    queries = torch.zeros(2, batch, index_heads, 128)
    weights = torch.ones(2, batch, index_heads)
    expected = [[], []]
    for row in range(batch):
        population = length if hotset_layout == "scattered" else K + misses
        tokens = rng.sample(range(population), K + misses)
        common = tokens[: K - misses]
        only_a, only_b = tokens[K - misses : K], tokens[K:]
        keys[row, :, :2] = -10
        keys[row, common, :2] = 10
        keys[row, only_a, :2] = torch.tensor([6.0, -10.0])
        keys[row, only_b, :2] = torch.tensor([-10.0, 6.0])
        queries[0, row, :, 0] = queries[1, row, :, 1] = 1
        keys[row, :, 2:] = torch.randn(context, 126, generator=generator) * 0.005
        queries[:, row, :, 2:] = (
            torch.randn(2, index_heads, 126, generator=generator) * 0.005
        )
        expected[0].append(sorted(common + only_a))
        expected[1].append(sorted(common + only_b))
    return SimpleNamespace(
        context=context,
        key=keys.to(device="npu", dtype=torch.bfloat16).reshape(-1, PAGE, 1, 128),
        queries=queries.to(device="npu", dtype=torch.bfloat16),
        weights=weights.to(device="npu", dtype=torch.bfloat16),
        table=torch.arange(
            batch * context // PAGE, device="npu", dtype=torch.int32
        ).view(batch, -1),
        expected=torch.tensor(expected, device="npu", dtype=torch.int32),
        host=torch.randn(batch + 1, context, 1, CKV + KPE, generator=generator).to(
            torch.bfloat16
        ),
        q=torch.randn(batch, query_heads, CKV, generator=generator).to(
            device="npu", dtype=torch.bfloat16
        ),
        rope=torch.randn(batch, query_heads, KPE, generator=generator).to(
            device="npu", dtype=torch.bfloat16
        ),
    )


class FixtureIndexPool(MLATokenToKVPool):
    """An index-cache boundary; manager and ReqToTokenPool run their real init."""

    def __init__(self, key, layers):
        self.key = key
        self.kv_lora_rank, self.qk_rope_head_dim = CKV, KPE
        self.store_dtype = torch.bfloat16
        self.start_layer, self.layer_num = 0, layers

    def get_index_k_buffer(self, layer_id):
        return self.key


class FrameworkCase:
    def __init__(
        self,
        mode,
        fixture,
        args,
        *,
        request_capacity=None,
        graph_batch_capacity=0,
        allocator_kv=None,
    ):
        self.mode, self.fixture, self.args = mode, fixture, args
        self.pool = ReqToTokenPool(
            request_capacity or args.batch, fixture.context, "npu", False
        )
        kv = FixtureIndexPool(fixture.key, args.layers)
        with (
            get_context().override_server_args(
                device="npu",
                attention_backend="ascend",
                tp_size=1,
                max_running_requests=request_capacity or args.batch,
                disable_cuda_graph=True,
            ),
            envs.SGLANG_NPU_SPARSE_KV_INDEXER.override(mode),
            envs.SGLANG_NPU_SPARSE_KV_CACHE_SIZE.override(args.capacity),
            envs.SGLANG_NPU_SPARSE_KV_HBM_BUDGET_MB.override(0),
        ):
            self.manager = SparseKVCacheManager(
                self.pool,
                SimpleNamespace(
                    get_kvcache=lambda: allocator_kv if allocator_kv is not None else kv
                ),
                K,
                copy_sfa_padded_query_heads=args.query_heads,
                graph_batch_capacity=graph_batch_capacity,
                paged_kv_cache=kv,
            )
        for layer in range(args.layers):
            self.manager.host_kv_buffer[layer].copy_(
                fixture.host[: self.manager.size] + layer
            )
        self.batch = SimpleNamespace(
            batch_size=args.batch,
            forward_mode=ForwardMode.DECODE,
            req_pool_indices=torch.arange(
                1, args.batch + 1, device="npu", dtype=torch.int32
            ),
            seq_lens=torch.full(
                (args.batch,), args.context + args.tail, device="npu", dtype=torch.int32
            ),
            out_cache_loc=fixture.table[:, -1] * PAGE + args.tail - 1,
            seq_lens_cpu=torch.full(
                (args.batch,), args.context + args.tail, dtype=torch.int32
            ),
        )
        self.new_kv = []
        if getattr(args, "include_new_token_d2h", False):
            token = args.context + args.tail - 1
            for layer in range(args.layers):
                kv = (fixture.host[1:, token] + layer).to("npu")
                self.new_kv.append(
                    (kv[..., :CKV].contiguous(), kv[..., CKV:].contiguous())
                )
        self.backend = SimpleNamespace(
            device="npu",
            kv_lora_rank=CKV,
            qk_rope_head_dim=KPE,
            sparse_kv_manager=self.manager,
            forward_metadata=SimpleNamespace(
                block_tables=fixture.table,
                actual_seq_lengths_q=None,
                actual_seq_lengths_kv=None,
                seq_lens_cpu_int=[args.context + args.tail] * args.batch,
                seq_lens=self.batch.seq_lens,
                sparse_kv_lim_plan=None,
            ),
        )
        self.layer = SimpleNamespace(
            layer_id=0,
            tp_k_head_num=1,
            tp_q_head_num=args.query_heads,
            scaling=(CKV + KPE) ** -0.5,
        )
        self.outputs = [
            torch.empty(
                args.batch, args.query_heads * CKV, device="npu", dtype=torch.bfloat16
            )
            for _ in range(2)
        ]
        self.latest_ids = None

    def select(self, route):
        f, a = self.fixture, self.manager.lim_adapter
        if a is not None:
            plan = a.select(
                self.manager,
                query=f.queries[route],
                weights=f.weights[route],
                key=f.key,
                block_table=f.table,
                forward_batch=self.batch,
                layer_id=0,
                reference_indexer=torch_npu.npu_lightning_indexer,
            )
            self.backend.forward_metadata.sparse_kv_lim_plan = plan
            self.latest_ids = plan.source_ids
        else:
            result = torch_npu.npu_lightning_indexer(
                query=f.queries[route],
                key=f.key,
                weights=f.weights[route],
                actual_seq_lengths_query=torch.arange(
                    1, self.args.batch + 1, dtype=torch.int32, device="npu"
                ),
                actual_seq_lengths_key=self.batch.seq_lens,
                block_table=f.table,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=K,
                sparse_mode=3,
            )
            self.latest_ids = (
                result[0] if isinstance(result, (tuple, list)) else result
            ).reshape(self.args.batch, K)
        return self.latest_ids

    def step(self, route, *, save_kv_cache=False):
        ids = self.select(route)
        for layer in range(self.args.layers):
            self.layer.layer_id = layer
            key, key_rope = (
                self.new_kv[layer]
                if save_kv_cache
                else (self.fixture.q[:, :1], self.fixture.rope[:, :1])
            )
            result = forward_sparsity_driven_kv_offload(
                self.backend,
                self.fixture.q,
                key,
                None,
                self.layer,
                self.batch,
                save_kv_cache=save_kv_cache,
                q_rope=self.fixture.rope,
                k_rope=key_rope,
                topk_indices=ids,
            )
        self.outputs[route].copy_(result)
        return result

    def verify_payload(self):
        a = self.manager.lim_adapter
        ids = (
            a.combined_ids[: self.args.batch]
            if a and not a.full_visible
            else self.latest_ids
        )
        got = (
            a.combined_kv[: self.args.batch]
            if a and not a.full_visible
            else self.manager.selected_kv_buffer[: self.args.batch]
        )
        ids_cpu = ids.cpu().long()
        for row in range(self.args.batch):
            valid = ids_cpu[row] >= 0
            if not bool(valid.any()):
                continue
            expected = (self.fixture.host[row + 1] + (self.args.layers - 1))[
                ids_cpu[row, valid]
            ]
            if a and a.full_visible and self.manager.last_decode_path == "copy_sfa":
                slots = a.device_slots[row, 0].cpu().long()[valid]
                actual = self.manager.device_kv_buffer[-1][row + 1, :, 0].cpu()[slots]
            else:
                actual = got[row].cpu()[valid]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def reset(self):
        self.manager.reset_requests(list(range(1, self.manager.size)))

    def verify_indexer_entry_and_decode_offload(self):
        """Exercise the real framework entry with fixed projection boundaries."""
        from sglang.srt.layers.attention.dsa import dsa_npu_indexer as module

        class Rotation:
            is_neox_style = False

            def __call__(self, positions, query, key):
                return query, key

        indexer = module.DSANPUIndexerMixin()
        indexer.n_heads = indexer.hidden_size = self.args.index_heads
        indexer.index_topk = K
        indexer.head_dim, indexer.rope_head_dim = 128, 64
        indexer.rotary_emb, indexer.dsa_enable_prefill_cp = Rotation(), False
        indexer.rotary_emb.cos_sin_cache = torch.empty(
            self.fixture.context, 64, device="npu", dtype=torch.bfloat16
        )
        indexer.wq_b = lambda x: (x,)
        indexer.weights_proj = lambda x: (x,)
        indexer.k_norm = lambda x: x
        locations = self.fixture.table[:, -1] * PAGE + self.args.tail - 1
        self.batch.out_cache_loc = locations
        flat_key = self.fixture.key.view(-1, 128)
        indexer.wk = lambda x: (flat_key.index_select(0, locations.long()),)
        pool = SimpleNamespace(
            start_layer=0,
            index_k_scale_buffer=None,
            get_index_k_buffer=lambda layer: self.fixture.key,
            set_index_k_buffer=lambda layer, loc, value: flat_key.index_copy_(
                0, loc.long(), value.reshape(-1, 128)
            ),
        )
        with (
            patch.object(module, "get_attn_backend", lambda: self.backend),
            patch.object(module, "get_token_to_kv_pool", lambda: pool),
            patch.object(
                module, "get_parallel", lambda: SimpleNamespace(dcp_enabled=False)
            ),
            envs.SGLANG_NPU_USE_MULTI_STREAM.override(False),
        ):
            ids = indexer.forward_npu(
                self.fixture.weights[0],
                self.fixture.queries[0].flatten(1),
                self.batch.seq_lens - 1,
                self.batch,
                0,
            )
        ids = ids.reshape(self.args.batch, K)
        torch.testing.assert_close(
            ids.sort(dim=1).values, self.fixture.expected[0], rtol=0, atol=0
        )
        # Write a new compact decode token to DRAM and immediately consume it
        # through SFA on the same forward. Source cache contains its CPU oracle.
        token = self.args.context + self.args.tail - 1
        for layer in range(self.args.layers):
            self.layer.layer_id = layer
            new_kv_cpu = self.fixture.host[1:, token] + layer
            new_kv = new_kv_cpu.to("npu")
            self.manager.host_kv_buffer[layer][1:, token].zero_()
            forward_sparsity_driven_kv_offload(
                self.backend,
                self.fixture.q,
                new_kv[..., :CKV],
                None,
                self.layer,
                self.batch,
                save_kv_cache=True,
                q_rope=self.fixture.rope,
                k_rope=new_kv[..., CKV:],
                topk_indices=ids,
            )
            torch.npu.synchronize()
            torch.testing.assert_close(
                self.manager.host_kv_buffer[layer][1:, token],
                new_kv_cpu,
                rtol=0,
                atol=0,
            )
        self.verify_payload()


def time_graph(graph, iterations, pairs_per_graph):
    start, end = (
        torch.npu.Event(enable_timing=True),
        torch.npu.Event(enable_timing=True),
    )
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (iterations * 2 * pairs_per_graph)


def run(args):
    torch.npu.set_device(0)
    # Some shared hosts cannot use the production shm allocator (ret=-4).
    # An explicit external ACL registered-host allocator keeps real host reads
    # in the measured copy kernels. Never replace host memory with device KV.
    registered_owners = []
    if args.registered_host_helper:
        sys.path.insert(0, str(args.registered_host_helper))
        from hisparse_registered_host_view import (
            allocate_registered_host,
            make_npu_view,
        )

        def allocate_host(*, shape, dtype, **kwargs):
            owner = allocate_registered_host(torch.zeros(shape, dtype=dtype))
            view, host_ptr, _, mapped_ptr = make_npu_view(owner)
            registered_owners.append((owner, view))
            return owner, int(host_ptr), int(mapped_ptr)

        allocator = patch(
            "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.manager.create_shm_tensor",
            allocate_host,
        )
        allocator.start()
    fixture = inputs(
        args.batch,
        args.context,
        args.misses,
        args.index_heads,
        args.query_heads,
        args.seed,
        args.hotset_layout,
    )
    cases = {mode: FrameworkCase(mode, fixture, args) for mode in args.modes.split(",")}
    correctness = {}
    for mode, case in cases.items():
        for route in (0, 1, 0, 1):
            case.step(route)
            torch.npu.synchronize()
            torch.testing.assert_close(
                case.latest_ids.sort(dim=1).values,
                fixture.expected[route],
                rtol=0,
                atol=0,
            )
            case.verify_payload()
        # A reset must force a cold plan even when this request slot is reused.
        case.reset()
        case.step(0)
        torch.npu.synchronize()
        case.verify_payload()
        if mode == "lim":
            torch.testing.assert_close(
                case.manager.lim_adapter.miss_counts[: args.batch],
                torch.full_like(
                    case.manager.lim_adapter.miss_counts[: args.batch], args.capacity
                ),
            )
        correctness[mode] = {
            "topk_set": "pass",
            "kv_payload_bitwise": "pass",
            "request_reset": "pass",
        }
        # Exercise transitions on device, without changing the captured shapes.
        case.batch.seq_lens.fill_(1024)
        case.step(0)
        torch.npu.synchronize()
        short_ids = case.latest_ids.sort(dim=1).values
        expected_short = torch.cat(
            (
                torch.full((args.batch, 1024), -1, dtype=torch.int32, device="npu"),
                torch.arange(1024, dtype=torch.int32, device="npu").expand(
                    args.batch, -1
                ),
            ),
            dim=1,
        )
        torch.testing.assert_close(short_ids, expected_short, rtol=0, atol=0)
        case.verify_payload()
        case.batch.seq_lens.zero_()
        case.batch.req_pool_indices.zero_()
        case.step(0)
        torch.npu.synchronize()
        if mode != "legacy" and not bool((case.latest_ids == -1).all()):
            raise AssertionError("padded row retained valid LIM source IDs")
        torch.testing.assert_close(
            case.outputs[0], torch.zeros_like(case.outputs[0]), rtol=0, atol=1e-6
        )
        case.batch.seq_lens.fill_(args.context + args.tail)
        case.batch.req_pool_indices.copy_(
            torch.arange(1, args.batch + 1, device="npu", dtype=torch.int32)
        )
        case.reset()
        correctness[mode]["short_and_padded_rows"] = "pass"
        requests = [
            Req(f"lim-test-{i}", "", array("i", [1]), SamplingParams(max_new_tokens=1))
            for i in range(args.batch)
        ]
        allocated = case.pool.alloc(requests)
        if allocated != list(range(1, args.batch + 1)):
            raise AssertionError("real request allocator returned unexpected rows")
        for request in requests:
            case.manager.init_req(request)
        case.step(0)
        torch.npu.synchronize()
        case.verify_payload()
        correctness[mode]["real_req_allocation_and_init"] = "pass"
        case.verify_indexer_entry_and_decode_offload()
        correctness[mode]["indexer_entry_and_new_token_d2h"] = (
            "pass (fixed projection boundary)"
        )
    if "lim" in cases and "li_reference" in cases:
        for route in (0, 1):
            cases["li_reference"].step(route)
            cases["lim"].step(route)
            torch.npu.synchronize()
            torch.testing.assert_close(
                cases["lim"].outputs[route],
                cases["li_reference"].outputs[route],
                rtol=0.01,
                atol=0.002,
            )
        correctness["lim_vs_li_reference"] = (
            "attention output passes rtol=0.01, atol=0.002"
        )
    graphs = {}
    for mode, case in cases.items():
        for _ in range(args.warmup):
            case.step(1, save_kv_cache=args.include_new_token_d2h)
            case.step(0, save_kv_cache=args.include_new_token_d2h)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            for _ in range(args.pairs_per_graph):
                case.step(1, save_kv_cache=args.include_new_token_d2h)
                case.step(0, save_kv_cache=args.include_new_token_d2h)
        graph.replay()
        torch.npu.synchronize()
        case.verify_payload()
        graphs[mode] = graph
    samples = {mode: [] for mode in cases}
    for round_id in range(args.rounds):
        modes = list(cases) if round_id % 2 == 0 else list(cases)[::-1]
        for mode in modes:
            for _ in range(args.warmup):
                graphs[mode].replay()
            samples[mode].append(
                time_graph(graphs[mode], args.iterations, args.pairs_per_graph)
            )
    medians = {mode: statistics.median(values) for mode, values in samples.items()}
    # Validate the state after all timed replays, and audit actual misses.
    # C>K does not imply that the native policy behaves like Python round-LRU.
    from sgl_kernel_npu.sparsity_driven_kv_offload import slot_map_lookup

    miss_audit = {}
    for mode, case in cases.items():
        case.verify_payload()
        miss_audit[mode] = {}
        for route in (1, 0):
            if mode != "lim":
                ids = case.select(route)
                hits, _ = slot_map_lookup(
                    case.manager.device_slot_map[0],
                    case.batch.req_pool_indices,
                    ids.contiguous(),
                )
                misses = ((~hits.bool()) & (ids >= 0)).sum(1).cpu().tolist()
            case.step(route)
            torch.npu.synchronize()
            case.verify_payload()
            torch.testing.assert_close(
                case.latest_ids.sort(dim=1).values,
                fixture.expected[route],
                rtol=0,
                atol=0,
            )
            if mode == "lim":
                misses = (
                    case.manager.lim_adapter.topk_misses[: args.batch].cpu().tolist()
                )
            miss_audit[mode]["A" if route == 0 else "B"] = misses
        case.reset()
        graphs[mode].replay()
        torch.npu.synchronize()
        case.verify_payload()
    result = {
        "scope": "real manager init + select + KV materialize/refill + SFA; fixed input; graph replay",
        "included_new_token_d2h": args.include_new_token_d2h,
        "excluded": [
            "model projections/weights",
            "scheduler",
            "sampling",
        ]
        + ([] if args.include_new_token_d2h else ["new-token D2H"]),
        "configuration": vars(args)
        | {
            "output": str(args.output),
            "registered_host_helper": str(args.registered_host_helper)
            if args.registered_host_helper
            else None,
        },
        "correctness": correctness,
        "graph_capture_replay": "pass",
        "post_timing_payload": "pass",
        "request_reset_during_graph_replay": "pass",
        "actual_steady_topk_misses": miss_audit,
        "samples_us_per_decode": samples,
        "median_us_per_decode": medians,
        "managed_hbm_bytes": {
            mode: case.manager.budget.managed_hbm_bytes for mode, case in cases.items()
        },
        "steady_native_misses": cases["lim"]
        .manager.lim_adapter.topk_misses[: args.batch]
        .cpu()
        .tolist()
        if "lim" in cases
        else None,
        "runtime": {
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        },
    }
    if "lim" in medians and "li_reference" in medians:
        result["lim_vs_reference_latency_change_percent"] = (
            medians["lim"] / medians["li_reference"] - 1
        ) * 100
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if args.registered_host_helper:
        allocator.stop()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--context", type=int, default=32768)
    p.add_argument("--capacity", type=int, default=2048)
    p.add_argument("--misses", type=int, default=128)
    p.add_argument(
        "--hotset-layout",
        choices=("scattered", "prefix"),
        default="scattered",
        help="prefix fits the alternating hot union into native cold-fill C>K for an all-hit control",
    )
    p.add_argument("--tail", type=int, default=1)
    p.add_argument("--index-heads", type=int, choices=(32, 64), default=32)
    p.add_argument("--query-heads", type=int, default=8)
    p.add_argument(
        "--layers",
        type=int,
        default=1,
        help="IndexShare layers consuming one anchor plan",
    )
    p.add_argument("--modes", default="li_reference,lim,legacy")
    p.add_argument(
        "--include-new-token-d2h",
        action="store_true",
        help="Include compact new-token KV writes to registered DRAM in each timed decode step",
    )
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument(
        "--pairs-per-graph",
        type=int,
        default=8,
        help="BA pairs in one graph to amortize host replay dispatch gaps",
    )
    p.add_argument("--seed", type=int, default=20261020)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--registered-host-helper",
        type=Path,
        help="Optional directory containing an ACL registered-host allocator extension",
    )
    args = p.parse_args()
    if (
        args.context % PAGE
        or not 1 <= args.tail <= PAGE
        or args.context < K + args.misses
    ):
        p.error("context must be page aligned and >= K+misses; tail must be 1..128")
    run(args)
