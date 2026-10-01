"""Four-rank collector/worker callbacks with synthetic forwards and real Store."""

import socket
import threading
import time
from array import array
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist

from sglang.srt.managers.io_struct import (
    TokenizedGenerateReqInput,
    msgpack_decode,
    msgpack_encode,
)
from sglang.srt.managers.schedule_batch import FINISH_ABORT, FINISH_LENGTH, Req
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.training_capture.catalog import HTTPCaptureCatalog
from sglang.srt.training_capture.cohort import CaptureCohortAllocator
from sglang.srt.training_capture.cohort_coordinator import CohortCaptureCoordinator
from sglang.srt.training_capture.config import CaptureConfig, StoreSetup
from sglang.srt.training_capture.kv_exporter import SelectedLayerKVExporter
from sglang.srt.training_capture.mooncake_store import MooncakeSnapshotStore
from sglang.srt.training_capture.protocol import canonical_bytes
from sglang.srt.training_capture.resources import CaptureResources
from sglang.srt.training_capture.topology import plan_capture_layout
from sglang.test.training_capture_utils import make_snapshot


def _connect(master):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        address = f"127.0.0.1:{sock.getsockname()[1]}"
    return MooncakeSnapshotStore.connect(
        {
            "local_hostname": address,
            "master_server_addr": master,
            "metadata_server": "P2PHANDSHAKE",
            "protocol": "tcp",
            "rdma_devices": "",
            "global_segment_size": 0,
            "local_buffer_size": 16 << 20,
        }
    )


def _incoming(rid):
    return TokenizedGenerateReqInput(
        rid=rid,
        input_text=None,
        input_ids=array("q", [3, 4]),
        input_embeds=None,
        mm_inputs=None,
        token_type_ids=None,
        sampling_params=SamplingParams(
            max_new_tokens=4, temperature=0.8, is_normalized=True
        ),
        return_logprob=False,
        logprob_start_len=-1,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=False,
    )


def _wait(coordinator, predicate):
    deadline = time.monotonic() + 30
    while not predicate():
        assert coordinator.error is None
        assert coordinator.service.error is None
        assert coordinator.writer_actor.error is None
        if time.monotonic() >= deadline:
            raise TimeoutError(f"collector did not progress: {coordinator.stats()}")
        time.sleep(0.01)


def _sources(base, tensors, partition):
    buffers = {}
    heads = {head.layer_id: head for head in partition.heads}
    for layer in partition.local_layers(base.kv):
        for component in ("k", "v"):
            buffers[f"target_{component}.{layer.layer_id}"] = torch.zeros(
                16, layer.num_kv_heads, 4, dtype=torch.bfloat16
            )
    for obj in base.objects:
        if obj.kind == "kv" and obj.layer_id in heads:
            head = heads[obj.layer_id]
            start, end = obj.token_range
            for offset in (0, 8):
                buffers[obj.name][offset + start : offset + end].copy_(
                    tensors[obj.key][:, head.start : head.end]
                )
    return buffers


def _forward(coordinator, req, *, end, raw_logits, pp_last, token):
    extend = end == 2
    batch = SimpleNamespace(
        reqs=[req],
        forward_mode=ForwardMode.EXTEND if extend else ForwardMode.DECODE,
        seq_lens_cpu=torch.tensor([end]),
        hicache_consumer_index=None,
    )
    forward = SimpleNamespace(
        extend_seq_lens_cpu=[2] if extend else None,
        positions=torch.arange(0 if extend else end - 1, end),
        is_prefill_only=False,
        return_logprob=False,
        apply_deprecated_skip_attn_backend_init=lambda _: None,
    )
    logits = SimpleNamespace(next_token_logits=raw_logits.clone()) if pp_last else None
    out = SimpleNamespace(
        logits_output=logits,
        can_run_graph=False,
        expert_distribution_metrics=None,
        routed_experts_output=None,
        indexer_topk_output=None,
    )

    def sample(logits_output, _):
        # The real worker's capture callback must run before these mutations.
        logits_output.next_token_logits.fill_(-999)
        return torch.tensor([token])

    worker = SimpleNamespace(
        training_capture=coordinator,
        set_hicache_consumer=lambda _: None,
        is_dllm=lambda: False,
        pp_group=SimpleNamespace(is_last_rank=pp_last),
        model_runner=SimpleNamespace(
            forward=lambda *args, **kwargs: out, sample=sample
        ),
        enable_overlap=False,
        enable_spec=False,
    )
    with patch(
        "sglang.srt.managers.tp_worker.ForwardBatch.init_new", return_value=forward
    ):
        result = TpModelWorker.forward_batch_generation(worker, batch)
    if pp_last:
        assert result.next_token_ids.tolist() == [token]
    return result.training_capture if pp_last else None


def _run_case(rank, root, master, endpoint, case):
    base, tensors = make_snapshot(response_length=4)
    replicated = case == "replicated"
    ranges = (
        [(0, 4)]
        if replicated
        else ([(0, 1), (1, 4)] if case == "inactive_ingress" else [(0, 2), (2, 4)])
    )
    tp_size = 4 if replicated else 2
    layout = plan_capture_layout(
        base.kv, tp_size=tp_size, pp_layer_ranges=ranges, aux_tp_rank=1
    )
    partition = layout.partitions[rank]
    config = CaptureConfig(
        dataset_id=f"coordinator-{case}",
        model_id="fixture",
        producer_revision="test",
        selected_layer_ids=base.kv.selected_layer_ids,
        catalog_endpoint=endpoint,
        journal_directory=str(Path(root) / f"journal-{case}-{rank}"),
        store=StoreSetup(local_hostname=f"rank-{rank}", master_server_addr=master),
        max_sample_tokens=8,
        max_inflight_samples=2,
        max_host_bytes=4 << 20,
        sample_ratio=1.0,
    )
    sources = _sources(base, tensors, partition)
    resources = CaptureResources()
    if partition.active:
        resources.store = _connect(master)
        resources.catalog = HTTPCaptureCatalog(endpoint)
        if partition.heads:
            resources.exporter = SelectedLayerKVExporter(
                base.kv, sources, partition=partition
            )
        resources._allocate(config, base.kv, partition, pin_memory=False)
    control = dist.new_group(backend="gloo", timeout=timedelta(seconds=20))
    coordinator = CohortCaptureCoordinator(
        allocator=CaptureCohortAllocator(
            group=control,
            layout=layout,
            config=config,
            teacher=base.teacher,
            kv=base.kv,
            resources=resources,
            timeout_seconds=15,
        ),
        req_to_token=SimpleNamespace(req_to_token=torch.arange(16).reshape(2, 8)),
        capture_mode="speculative_accepted_target_path"
        if case == "verify"
        else "autoregressive",
        autostart=False,
    )
    recovery_entered, release_recovery = threading.Event(), threading.Event()
    startup_patch = None
    if case == "inactive_ingress" and partition.include_aux:
        recover = coordinator.writer_actor.writer.recover

        def held_recovery():
            recovery_entered.set()
            assert release_recovery.wait(15)
            return recover()

        startup_patch = patch.object(
            coordinator.writer_actor.writer, "recover", held_recovery
        )
        startup_patch.start()
    try:
        coordinator.activate()
        if case == "inactive_ingress":
            if partition.include_aux:
                assert recovery_entered.wait(10)
            dist.barrier()
            if rank == 0:
                _wait(coordinator, lambda: coordinator.stats()["reservations"] == 2)
                raw = _incoming("not-ready")
                coordinator.request_router.prepare([raw])
                assert raw.training_capture_ticket is None
            dist.barrier()
            release_recovery.set()
        _wait(
            coordinator, lambda: coordinator.stats()["states"].get("available", 0) == 2
        )
        requests = []
        for index in range(2):
            raw = _incoming(f"{case}-{index}")
            if rank == 0:
                coordinator.request_router.prepare([raw])
                assert raw.training_capture_ticket is not None
            messages = [msgpack_encode(raw) if rank == 0 else None]
            dist.broadcast_object_list(messages, src=0)
            raw = msgpack_decode(messages[0])
            req = Req(
                raw.rid,
                raw.input_text,
                raw.input_ids,
                raw.sampling_params,
                vocab_size=256,
                training_capture_ticket=raw.training_capture_ticket,
            )
            req.req_pool_idx = index
            coordinator.request_router.attach(raw, req)
            requests.append(req)
        coordinator.before_forward(requests)
        records = [req.training_capture_context for req in requests]
        assert all(record is not None for record in records), coordinator.stats()
        dist.barrier()
        raw_rows = torch.randn(4, 256, generator=torch.Generator().manual_seed(42))
        pp_last = rank // tp_size == len(ranges) - 1
        for index in (0, 1) if rank < 2 else (1, 0):
            req = requests[index]
            first = _forward(
                coordinator,
                req,
                end=2,
                raw_logits=raw_rows[:1],
                pp_last=pp_last,
                token=5,
            )
            req.output_ids.append(5)
            coordinator.after_result(first, requests=[req])
            if case == "verify":
                batch = SimpleNamespace(reqs=[req], seq_lens_cpu=torch.tensor([2]))
                forward = SimpleNamespace(
                    input_ids=torch.tensor([5, 6, 7]),
                    positions=torch.arange(2, 5),
                    out_cache_loc=torch.arange(index * 8 + 2, index * 8 + 5),
                )
                logits = (
                    SimpleNamespace(next_token_logits=raw_rows[1:].clone())
                    if pp_last
                    else None
                )
                ticket = coordinator.after_verify_forward(
                    batch, forward, logits, width=3, can_run_cuda_graph=False
                )
                assert ticket is not None
                if logits is not None:
                    logits.next_token_logits.zero_()
                ticket = coordinator.after_verify_accept(
                    ticket,
                    commit_lens=torch.tensor([3]),
                    out_tokens=torch.tensor([[6, 7, 8]]),
                )
                req.output_ids.extend([6, 7, 8])
            else:
                for step, end in enumerate((3, 4, 5), start=1):
                    ticket = _forward(
                        coordinator,
                        req,
                        end=end,
                        raw_logits=raw_rows[step : step + 1],
                        pp_last=pp_last,
                        token=5 + step,
                    )
                    req.output_ids.append(5 + step)
                    if end < 5:
                        coordinator.after_result(ticket, requests=[req])
            req.finished_len = 4
            req.finished_reason = (
                FINISH_ABORT("test cancellation")
                if case == "abort" and index == 0 and rank == 0
                else FINISH_LENGTH(4)
            )
            if rank == 0 and req.training_capture_finalize is not None:
                # Match KV cache release invoking finalization before result bookkeeping.
                req.training_capture_finalize(req)
            coordinator.after_result(ticket if pp_last else None, requests=[req])
            for source in sources.values():
                source[index * 8 : index * 8 + 8].zero_()
        _wait(
            coordinator,
            lambda: (
                not coordinator.records
                and coordinator.writer_actor.stats()["pending"] == 0
            ),
        )
        rows = [
            {
                "capture_id": req.training_capture_route.handle.cohort.lease.capture_id,
                "sample_id": req.training_capture_route.handle.cohort.lease.sample_id,
                "expected": "FAILED" if case == "abort" and index == 0 else "AVAILABLE",
            }
            for index, req in enumerate(requests)
        ]
        if case != "abort":
            assert all(record.context.state == "SEALED" for record in records)
        dist.barrier()
        assert coordinator.close(), coordinator.stats()
        assert resources.closed
        return {"case": case, "samples": rows, "stats": coordinator.stats()}
    finally:
        release_recovery.set()
        if startup_patch is not None:
            startup_patch.stop()
        if not coordinator.close():
            raise RuntimeError("collector still owns live capture state")
        dist.destroy_process_group(control)


def cohort_runtime_worker(rank, root, master, endpoint):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=(Path(root) / "rendezvous").as_uri(),
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=45),
    )
    try:
        results = []
        for case in ("tp_pp", "replicated", "inactive_ingress", "verify", "abort"):
            with (
                get_context().override_server_args(enable_dp_attention=False),
                get_parallel().override(
                    tp_rank=rank % (4 if case == "replicated" else 2)
                ),
            ):
                results.append(_run_case(rank, root, master, endpoint, case))
            dist.barrier()
        (Path(root) / f"rank-{rank}.json").write_bytes(canonical_bytes(results))
    finally:
        dist.destroy_process_group()
