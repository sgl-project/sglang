"""Fixed-token, full-checkpoint decode timing; launch with two torchrun ranks.

Reuses the upstream one_batch loader and input preparation. Every decode uses
the same teacher-forced token sequence across configurations, keeping KV history
comparable even if the argmax changes. Sampling and HTTP are outside this test.
"""

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

from .followup_server import MODEL, REVISION, save, source_head


@dataclass(frozen=True)
class Workload:
    buckets: tuple = (8, 32, 64)
    warmups: int = 32
    samples: int = 64
    rounds: int = 2
    seed: int = 32774
    attention_split_tile: int = 256

    def validate(self):
        if (
            not self.buckets
            or len(set(self.buckets)) != len(self.buckets)
            or any(b < 2 or b > 64 or b % 2 for b in self.buckets)
            or self.warmups < 2
            or self.samples < 2
            or self.rounds < 1
            or self.attention_split_tile < self.warmups + self.samples + 16
            or self.attention_split_tile & (self.attention_split_tile - 1)
        ):
            raise ValueError(
                "Use unique even buckets in [2, 64], warmups/samples >=2, rounds >=1, "
                "and a power-of-two attention tile covering the configured context"
            )

    def tokens(self, rank, bucket, step):
        # Values below 10k are valid for the pinned checkpoint. Rank-specific
        # prompts prevent the two EP ranks from processing identical requests.
        return [
            100 + (self.seed + rank * 17 + row * 13 + step * 7) % 8192
            for row in range(bucket)
        ]

    def fingerprint(self):
        self.validate()
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True).encode()
        ).hexdigest()


def model_args(workload, configuration, nccl_port, *, resolve_tokenizer=False):
    """Construct CLI options, optionally resolving the cached pinned tokenizer."""
    workload.validate()
    if configuration not in ("serial", "sbo", "tbo", "sbo-tbo"):
        raise ValueError(configuration)
    args = [
        "--model-path",
        MODEL,
        "--revision",
        REVISION,
        "--trust-remote-code",
        "--tp-size",
        "2",
        "--dp-size",
        "2",
        "--ep-size",
        "2",
        "--enable-dp-attention",
        "--enable-dp-lm-head",
        "--moe-dense-tp-size",
        "1",
        "--moe-a2a-backend",
        "nccl_ep",
        "--moe-runner-backend",
        "triton",
        "--fp8-gemm-backend",
        "triton",
        "--attention-backend",
        "triton",
        # Adaptive KV splitting changes reduction partitions with TBO's batch
        # size, amplifying BF16 rounding through FP8 routing. One context-sized
        # tile keeps this non-EP computation comparable without relaxing logits.
        "--triton-attention-split-tile-size",
        str(workload.attention_split_tile),
        "--nccl-ep-mode",
        "low_latency",
        "--enable-nccl-ep-cuda-graph",
        "--nccl-ep-num-max-dispatch-tokens-per-rank",
        "64",
        "--disable-shared-experts-fusion",
        "--disable-overlap-schedule",
        "--cuda-graph-bs-decode",
        *map(str, workload.buckets),
        "--chunked-prefill-size",
        "64",
        "--page-size",
        "1",
        "--context-length",
        str(workload.warmups + workload.samples + 16),
        "--max-running-requests",
        str(2 * max(workload.buckets)),  # Global limit is divided by attention DP.
        "--max-total-tokens",
        str(max(workload.buckets) * (workload.warmups + workload.samples + 16) + 1024),
        "--mem-fraction-static",
        "0.6",
        "--random-seed",
        str(workload.seed),
        "--nccl-port",
        str(nccl_port),
    ]
    if resolve_tokenizer:
        from huggingface_hub import snapshot_download

        # one_batch does not forward revision to its tokenizer loader. Resolve
        # the cached snapshot before ServerArgs becomes read-only at setup.
        args += [
            "--tokenizer-path",
            snapshot_download(MODEL, revision=REVISION, local_files_only=True),
        ]
    if "sbo" in configuration:
        args.append("--enable-single-batch-overlap")
    if "tbo" in configuration:
        args.append("--enable-two-batch-overlap")
    return args


def decode_step(tokens, batch, runner):
    from sglang.benchmark.one_batch import _maybe_prepare_mlp_sync_batch
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    batch.input_ids = tokens
    batch.prepare_for_decode()
    _maybe_prepare_mlp_sync_batch(batch, runner)
    forward_batch = ForwardBatch.init_new(
        batch, runner, return_hidden_states_before_norm=False
    )
    tbo = forward_batch.can_run_tbo
    output = runner.forward(forward_batch)
    return output.logits_output.next_token_logits, output.can_run_graph, tbo


def load_benchmark_model(server, nccl_port, rank):
    from sglang.benchmark.one_batch import load_model

    # The loader uses its own TCP endpoint. torchrun's agent only serves
    # MASTER_PORT; inheriting its store policy leaves our port without a server.
    os.environ.pop("TORCHELASTIC_USE_AGENT_STORE", None)
    return load_model(server, SimpleNamespace(nccl_port=nccl_port), rank, rank)


def run(
    workload, configuration, reports, nccl_port, *, profile=False, profile_bucket=None
):
    import torch
    import torch.distributed as dist

    from sglang.benchmark.one_batch import prepare_synthetic_inputs_for_latency_test
    from sglang.srt.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.layers.moe import initialize_moe_config
    from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
    from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
    from sglang.srt.server_args import ServerArgs

    from .environment import Unavailable, binding_check, prepare_jit
    from .followup_server import environment

    rank = int(os.environ["RANK"])
    if int(os.environ["WORLD_SIZE"]) != 2 or rank not in (0, 1):
        raise ValueError("Launch exactly two local ranks")
    if torch.cuda.device_count() < 2:
        raise Unavailable("Two visible GPUs required")
    torch.cuda.set_device(rank)
    if torch.cuda.get_device_capability(rank)[0] < 9:
        raise Unavailable("Native NCCL EP requires SM90+")
    report = dict(
        passed=False,
        source_head=source_head(),
        rank=rank,
        implementation="nccl_ep_full_model_decode_v2",
        configuration=configuration,
        workload=asdict(workload),
        workload_fingerprint=workload.fingerprint(),
        profiled=profile,
        model=MODEL,
        revision=REVISION,
        records=[],
    )
    reports.mkdir(parents=True, exist_ok=True)
    target = reports / f"model-rank{rank}.json"
    save(target, report)
    loaded = False
    try:
        report["bindings"] = binding_check()
        report["jit"] = prepare_jit()
        report["environment"] = environment()
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        server = ServerArgs.from_cli_args(
            parser.parse_args(
                model_args(workload, configuration, nccl_port, resolve_tokenizer=True)
            )
        )
        if server.enable_eplb or server.json_model_override_args != "{}":
            raise ValueError(
                "Performance runs require unmodified full weights and no EPLB migration"
            )
        _set_envs_and_config(server)
        initialize_moe_config(server)
        initialize_fp8_gemm_config(server)
        initialize_fp4_gemm_config(server)
        # Each worker uses the same externally checked port. PortArgs.init_new
        # would race its availability probe against rank 0 binding the store.
        wrapper, _ = load_benchmark_model(server, nccl_port, rank)
        loaded = True
        runner = wrapper.torch_runner
        report["captured_buckets"] = list(runner.decode_cuda_graph_runner.capture_bs)
        if not set(workload.buckets) <= set(report["captured_buckets"]):
            raise ValueError("Not all requested per-rank Graph buckets were captured")
        attention = runner.decode_cuda_graph_runner.attn_backend
        attention_backends = (
            [attention.primary, *attention.children]
            if hasattr(attention, "children")
            else [attention]
        )
        report["attention_partition"] = dict(
            tile=workload.attention_split_tile,
            max_kv_splits=[b.max_kv_splits for b in attention_backends],
        )
        if any(b.max_kv_splits != 1 for b in attention_backends):
            raise ValueError("The bounded workload must use one attention KV partition")
        config = runner.model_config.hf_config
        if (
            config.num_hidden_layers,
            config.first_k_dense_replace,
            config.moe_layer_freq,
        ) != (27, 1, 1):
            raise ValueError("Expected the complete 27-layer / 26-MoE pinned model")
        report["model_shape"] = dict(
            layers=27,
            moe_layers=26,
            hidden=config.hidden_size,
            experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            intermediate=config.moe_intermediate_size,
        )
        report["resolved_args"] = {
            name: getattr(server, name)
            for name in (
                "tp_size",
                "dp_size",
                "ep_size",
                "enable_two_batch_overlap",
                "enable_single_batch_overlap",
                "enable_eplb",
                "nccl_ep_num_max_dispatch_tokens_per_rank",
                "triton_attention_split_tile_size",
            )
        }
        report["measurement_scope"] = (
            "Synchronized ModelRunner decode: batch preparation, DP synchronization, full forward and Graph replay. Excludes prefill, warmup, sampling, HTTP and correctness copies. Per-rank B; global tokens/step = 2B. KV length grows identically across configurations."
        )
        checkpoints = {}
        with torch.inference_mode():
            for bucket in workload.buckets:
                if profile and bucket != (profile_bucket or workload.buckets[0]):
                    continue
                # Preloaded device inputs; no RNG or H2D transfer in timed steps.
                tokens = [
                    torch.tensor(
                        workload.tokens(rank, bucket, step),
                        device="cuda",
                        dtype=torch.int64,
                    )
                    for step in range(workload.warmups + workload.samples + 1)
                ]
                for rnd in range(1 if profile else workload.rounds):
                    wrapper.clear()
                    prompts = [[token] for token in workload.tokens(rank, bucket, 0)]
                    reqs = prepare_synthetic_inputs_for_latency_test(bucket, 1, prompts)
                    # B*1 <=64: the unchunked one_batch prefill respects LL's budget.
                    _, _, batch = wrapper.extend(reqs)
                    for step in range(workload.warmups):
                        decode_step(tokens[step + 1], batch, runner)
                    torch.cuda.synchronize()
                    events = [
                        (
                            torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True),
                        )
                        for _ in range(
                            min(8, workload.samples) if profile else workload.samples
                        )
                    ]
                    for start, end in events:
                        start.record()
                        end.record()
                    torch.cuda.synchronize()
                    dist.barrier()
                    if profile:
                        if rank == 0:
                            torch.cuda.cudart().cudaProfilerStart()
                        dist.barrier()
                    measured = dict(cuda_step_ms=[], host_step_ms=[])
                    graph_passes = tbo_passes = 0
                    for sample, (start, end) in enumerate(events):
                        if profile:
                            torch.cuda.nvtx.range_push(
                                f"nccl_ep_model/rank={rank}/B={bucket}/step={sample}"
                            )
                        host_start = time.perf_counter()
                        start.record()
                        logits, graph, tbo = decode_step(
                            tokens[workload.warmups + sample + 1], batch, runner
                        )
                        end.record()
                        end.synchronize()
                        measured["host_step_ms"].append(
                            (time.perf_counter() - host_start) * 1000
                        )
                        measured["cuda_step_ms"].append(start.elapsed_time(end))
                        if profile:
                            torch.cuda.nvtx.range_pop()
                        if not graph or bool(tbo) != ("tbo" in configuration):
                            raise RuntimeError(
                                f"Unexpected measured path: graph={graph}, tbo={tbo}"
                            )
                        graph_passes += 1
                        tbo_passes += int(tbo)
                        # Two named logit checkpoints, outside the event/host interval.
                        if sample in (0, len(events) - 1):
                            cpu = logits.detach().float().cpu()
                            if not torch.isfinite(cpu).all():
                                raise RuntimeError("Nonfinite full-model logits")
                            checkpoints[f"B{bucket}/round{rnd}/sample{sample}"] = cpu
                    if profile:
                        dist.barrier()
                        if rank == 0:
                            torch.cuda.cudart().cudaProfilerStop()
                    report["records"].append(
                        dict(
                            bucket=bucket,
                            round=rnd,
                            samples=measured,
                            graph_passes=graph_passes,
                            tbo_passes=tbo_passes,
                            first_decode_context=workload.warmups + 2,
                            ep_rounds_per_rank_per_step=26
                            * (2 if "tbo" in configuration else 1),
                        )
                    )
                    save(target, report)
        torch.save(checkpoints, reports / f"logits-rank{rank}.pt")
        report["logit_checkpoints"] = list(checkpoints)
        report["native_ep_tested"] = True
        report["passed"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            if loaded and report["passed"]:
                torch.cuda.synchronize()
                destroy_model_parallel()
                destroy_distributed_environment()
                report["cleanup_completed"] = True
        except Exception as error:
            report["passed"] = False
            report["cleanup_error"] = str(error)
            raise
        finally:
            save(target, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configuration", choices=("serial", "sbo", "tbo", "sbo-tbo"), default="serial"
    )
    parser.add_argument("--buckets", nargs="+", type=int, default=[8, 32, 64])
    parser.add_argument("--warmups", type=int, default=32)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--attention-split-tile", type=int, default=256)
    parser.add_argument("--nccl-port", type=int, default=29619)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-bucket", type=int)
    parser.add_argument("--describe", action="store_true")
    args = parser.parse_args()
    workload = Workload(
        tuple(args.buckets),
        args.warmups,
        args.samples,
        args.rounds,
        attention_split_tile=args.attention_split_tile,
    )
    workload.validate()
    if args.profile_bucket is not None and (
        not args.profile or args.profile_bucket not in workload.buckets
    ):
        parser.error("--profile-bucket must select a configured bucket with --profile")
    if args.describe:
        print(
            json.dumps(
                dict(
                    workload=asdict(workload),
                    fingerprint=workload.fingerprint(),
                    server_args=model_args(
                        workload, args.configuration, args.nccl_port
                    ),
                ),
                indent=2,
            )
        )
        return
    if args.profile:
        from .performance_trace import annotate_replays

        annotate_replays()
    run(
        workload,
        args.configuration,
        args.reports,
        args.nccl_port,
        profile=args.profile,
        profile_bucket=args.profile_bucket,
    )


if __name__ == "__main__":
    main()
