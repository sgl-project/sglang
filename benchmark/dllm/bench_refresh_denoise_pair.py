"""Model-level comparison of stock SDAR and refresh/denoise pairing.

Both variants use SGLang weights, layers, FlashInfer, and captured model graphs.
The fixed-batch driver owns the decoding loop. HTTP and scheduler costs are
outside this experiment. This file does not submit jobs.
"""

import argparse
import importlib.metadata
import json
import os
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

MASK = 151669


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def make_args(checkpoint, block, directory, concurrency):
    from sglang.srt.server_args import ServerArgs

    kwargs = dict(
        model_path=checkpoint,
        dtype="bfloat16",
        kv_cache_dtype="auto",
        attention_backend="flashinfer",
        trust_remote_code=True,
        disable_radix_cache=True,
        tp_size=1,
        random_seed=0,
        page_size=1,
        max_running_requests=max(concurrency, 16),
        max_total_tokens=32768,
        context_length=4096,
        mem_fraction_static=0.65,
        cuda_graph_bs_decode=[1, 4, 16],
        cuda_graph_bs_prefill=[128],
    )
    if block:
        config = directory / "algorithm.yaml"
        config.write_text(f"block_size: {block}\nthreshold: 0.95\n")
        kwargs.update(dllm_algorithm="LowConfidence", dllm_algorithm_config=str(config))
    return ServerArgs(**kwargs)


def initialize(args):
    from sglang.benchmark.one_batch import load_model
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.layers.moe import initialize_moe_config
    from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
    from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
    from sglang.srt.runtime_context import SpawnRanks, publish, spawn_world_rank
    from sglang.srt.server_args import PortArgs

    args.resolve_once()
    _set_envs_and_config(args)
    ports = PortArgs.init_new(args)
    publish(
        args,
        role="scheduler",
        ranks=SpawnRanks(
            world_rank=spawn_world_rank(args, tp_rank=0, pp_rank=0), gpu_id=0
        ),
    )
    initialize_moe_config()
    initialize_fp8_gemm_config()
    initialize_fp4_gemm_config()
    wrapper, _ = load_model(args, ports, 0, 0)
    return wrapper.torch_runner


class GraphCall:
    """One fixed-shape model graph; plan attention outside every replay."""

    def __init__(self, runner, backend, slots, req_indices, extent, mode, prefix=128):
        import torch

        from sglang.srt.model_executor.forward_batch_info import ForwardBatch

        self.runner, self.backend = runner, backend
        self.slots, self.extent = slots, extent
        self.bs = len(req_indices)
        n = self.bs
        device = slots.device
        self.batch = ForwardBatch(
            forward_mode=mode,
            batch_size=n,
            input_ids=torch.full(
                (n * extent,),
                MASK if mode.is_dllm_extend() else 100,
                dtype=torch.int64,
                device=device,
            ),
            req_pool_indices=req_indices,
            seq_lens=torch.full(
                (n,), prefix + extent, dtype=torch.int32, device=device
            ),
            seq_lens_cpu=torch.full((n,), prefix + extent, dtype=torch.int32),
            out_cache_loc=slots[:, prefix : prefix + extent].contiguous().reshape(-1),
            seq_lens_sum=n * (prefix + extent),
            positions=torch.arange(prefix, prefix + extent, device=device).repeat(n),
            extend_num_tokens=n * extent,
            extend_seq_lens=torch.full((n,), extent, dtype=torch.int32, device=device),
            extend_prefix_lens=torch.full(
                (n,), prefix, dtype=torch.int32, device=device
            ),
            extend_start_loc=torch.arange(n, device=device, dtype=torch.int32) * extent,
            extend_seq_lens_cpu=[extent] * n,
            extend_prefix_lens_cpu=[prefix] * n,
            extend_logprob_start_lens_cpu=[0] * n,
        )
        runner.prepare_dummy_forward_batch(self.batch)
        backend.init_cuda_graph_state(n, n * extent)
        backend.init_forward_metadata_out_graph(self.batch, in_capture=True)
        # Capture independent shapes with independent wrapper state and graph memory.
        for _ in range(3):
            self.model()
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = self.model()
        self.replays = 0

    def model(self):
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )

        with forward_context(ForwardContext(attn_backend=self.backend)):
            return self.runner.model(
                self.batch.input_ids, self.batch.positions, self.batch
            )

    def update(self, ids, prefix):
        import torch

        require(ids.shape == (self.bs, self.extent), "Wrong graph input shape")
        require(
            0 <= prefix <= self.slots.shape[1] - self.extent,
            "KV range exceeds owned slots",
        )
        b = self.batch
        b.input_ids.copy_(ids.reshape(-1))
        b.positions.copy_(
            torch.arange(prefix, prefix + self.extent, device=ids.device).repeat(
                self.bs
            )
        )
        b.out_cache_loc.copy_(self.slots[:, prefix : prefix + self.extent].reshape(-1))
        b.seq_lens.fill_(prefix + self.extent)
        b.seq_lens_cpu.fill_(prefix + self.extent)
        b.seq_lens_sum = self.bs * (prefix + self.extent)
        b.extend_prefix_lens.fill_(prefix)
        b.extend_prefix_lens_cpu = [prefix] * self.bs

    def __call__(self, ids, prefix, *, eager=False):
        self.update(ids, prefix)
        self.backend.init_forward_metadata_out_graph(self.batch)
        if eager:
            output = self.model()
        else:
            self.graph.replay()
            self.replays += 1
            output = self.output
        logits = (
            output.full_logits
            if self.batch.forward_mode.is_dllm_extend()
            else output.next_token_logits
        )
        return logits


def select(ids, logits, block, threshold):
    from sglang.srt.dllm.algorithm.low_confidence import LowConfidence

    nblocks = ids.numel() // block
    algorithm = SimpleNamespace(block_size=block, mask_id=MASK, threshold=threshold)
    batch = SimpleNamespace(batch_size=nblocks, input_ids=ids.reshape(-1))
    LowConfidence.step(algorithm, batch, logits, None)
    # Both comparison arms pay this post-selection completion check.
    return not bool((ids == MASK).any().item())


def execute(
    solo,
    pair,
    prompt,
    block,
    threshold,
    variant,
    *,
    output_tokens=256,
    diagnostic=False,
    record_phases=False,
):
    import torch

    require(variant in ("separate", "paired", "qwen"), "Unknown variant")
    counts, events = Counter(), []
    n = prompt.shape[0]
    begin, prefilled, finish = [torch.cuda.Event(enable_timing=True) for _ in range(3)]

    def call(graph, ids, prefix, phase):
        if diagnostic:
            a, b = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            a.record()
        with torch.profiler.record_function(phase) if record_phases else nullcontext():
            result = graph(ids, prefix)
        if diagnostic:
            b.record()
            events.append((phase, a, b))
        counts[phase] += 1
        return result

    torch.cuda.synchronize()
    started = time.perf_counter()
    begin.record()
    if not block:
        logits = call(pair, prompt, 0, "prefill")
        current = logits.argmax(-1).reshape(n, 1)
        output = [current.clone()]
        prefilled.record()
        for index in range(output_tokens - 1):
            current = (
                call(solo, current, prompt.shape[1] + index, "ar_decode")
                .argmax(-1)
                .reshape(n, 1)
            )
            output.append(current.clone())
    else:
        for prefix in range(0, prompt.shape[1], block):
            call(solo, prompt[:, prefix : prefix + block], prefix, "prefill")
        prefilled.record()
        output = []
        current = torch.full((n, block), MASK, dtype=torch.long, device=prompt.device)
        complete = False
        for index in range(output_tokens // block):
            prefix = prompt.shape[1] + index * block
            for _ in range(block + 1):
                if complete:
                    break
                logits = call(solo, current, prefix, "denoising")
                complete = select(current, logits, block, threshold)
            require(complete, "Denoising exceeded the native progress bound")
            if variant == "paired" and index + 1 < output_tokens // block:
                next_mask = torch.full_like(current, MASK)
                combined = torch.cat([current, next_mask], dim=1)
                logits = call(pair, combined, prefix, "refresh_and_denoise")
                complete = select(combined, logits, block, threshold)
                output.append(combined[:, :block].clone())
                current = combined[:, block:].contiguous()
            else:
                logits = call(solo, current, prefix, "refresh")
                require(
                    select(current, logits, block, threshold),
                    "Refresh changed completion",
                )
                output.append(current.clone())
                current = torch.full_like(current, MASK)
                complete = False
    finish.record()
    finish.synchronize()
    wall = time.perf_counter() - started
    ids = torch.cat(output, dim=1)
    require(ids.shape == (n, output_tokens), "Output budget changed")
    phases = Counter()
    for phase, a, b in events:
        phases[phase] += a.elapsed_time(b)
    return dict(
        variant=variant,
        threshold=threshold if block else None,
        batch_size=n,
        block_size=block,
        output_tokens_per_request=output_tokens,
        output_ids=ids.tolist(),
        forward_counts=dict(counts),
        wall_request_s=wall,
        cuda_event_request_ms=begin.elapsed_time(finish),
        cuda_event_prefill_ms=begin.elapsed_time(prefilled),
        cuda_event_generation_ms=prefilled.elapsed_time(finish),
        diagnostic_phase_ms=dict(phases),
        diagnostic=diagnostic,
        terminal_refresh_retained=bool(block),
        scope="Fixed-batch model execution with graph replay, planning and selection; excludes HTTP and serving scheduler.",
    )


def kv_snapshot(runner, slots, length):
    # Copy only the retained request range, not the full pool.
    result = []
    selected = slots[:, :length].reshape(-1)
    for layer in range(runner.model_config.num_hidden_layers):
        k, v = runner.token_to_kv_pool.get_kv_buffer(layer)
        result.append((k[selected].clone(), v[selected].clone()))
    return result


def qualify(solo, pair, prompt, block, runner, slots):
    import torch

    prefix = prompt.shape[1]
    for offset in range(0, prefix, block):
        solo(prompt[:, offset : offset + block], offset)
    # Tokens from the checkpoint's own first denoise pass form the completed block.
    masked = torch.full(
        (prompt.shape[0], block), MASK, dtype=torch.long, device=prompt.device
    )
    logits = solo(masked, prefix)
    committed = logits.argmax(-1).reshape_as(masked)
    a = solo(committed, prefix).clone()
    b = solo(masked, prefix + block).clone()
    retained = kv_snapshot(runner, slots, prefix + 2 * block)
    combined = torch.cat([committed, masked], dim=1)
    fused = pair(combined, prefix).clone().view(prompt.shape[0], 2 * block, -1)
    replay_kv = kv_snapshot(runner, slots, prefix + 2 * block)
    checks = []
    for label, reference, candidate in (
        ("refresh", a, fused[:, :block].reshape_as(a)),
        ("denoising", b, fused[:, block:].reshape_as(b)),
    ):
        checks.append(
            dict(
                phase=label,
                max_abs=float((reference - candidate).abs().max()),
                logits_close=bool(
                    torch.allclose(reference, candidate, atol=0.25, rtol=0.02)
                ),
                argmax_equal=bool(
                    torch.equal(reference.argmax(-1), candidate.argmax(-1))
                ),
            )
        )
    max_kv = 0.0
    close = True
    for reference, candidate in zip(retained, replay_kv):
        for left, right in zip(reference, candidate):
            max_kv = max(max_kv, float((left - right).abs().max()))
            close = close and bool(torch.allclose(left, right, atol=0.125, rtol=0.02))
    eager = pair(combined, prefix, eager=True).clone()
    graph = pair(combined, prefix).clone()
    graph_close = bool(torch.allclose(eager, graph, atol=0.125, rtol=0.02))
    decisions = []
    ref = torch.stack([a.view_as(fused[:, :block]), b.view_as(fused[:, block:])], dim=1)
    ref = ref.reshape(prompt.shape[0] * 2 * block, -1)
    for threshold in (0.0, 0.5, 0.95):
        left, right = combined.clone(), combined.clone()
        select(left, ref, block, threshold)
        select(right, fused.reshape_as(ref), block, threshold)
        decisions.append(
            dict(threshold=threshold, equal=bool(torch.equal(left, right)))
        )
    prefix_exact = all(
        torch.equal(
            left.reshape(prompt.shape[0], prefix + 2 * block, *left.shape[1:])[
                :, :prefix
            ],
            right.reshape(prompt.shape[0], prefix + 2 * block, *right.shape[1:])[
                :, :prefix
            ],
        )
        for a_layer, b_layer in zip(retained, replay_kv)
        for left, right in zip(a_layer, b_layer)
    )
    return dict(
        logits=checks,
        kv_close=close,
        max_kv_abs=max_kv,
        cached_prefix_exact=prefix_exact,
        graph_matches_eager=graph_close,
        decisions=decisions,
        passed=close
        and graph_close
        and prefix_exact is not False
        and all(x["logits_close"] for x in checks)
        and all(x["equal"] for x in decisions),
        tolerances=dict(logits_atol=0.25, kv_atol=0.125, rtol=0.02),
        note="BF16 numeric tolerances are fixed before GPU execution. Selected tokens must match exactly.",
    )


def worker(args):
    import torch

    from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
    from sglang.srt.layers.attention.flashinfer_block_pair_backend import (
        FlashInferBlockPairBackend,
    )
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    directory = args.output
    directory.mkdir(parents=True, exist_ok=False)
    prompt_ids = json.loads(args.prompt_ids.read_text())
    require(
        isinstance(prompt_ids, list) and len(prompt_ids) == 128,
        "Supply 128 prompt token IDs",
    )
    require(
        all(type(token) is int and token >= 0 for token in prompt_ids),
        "Prompt IDs must be nonnegative integers",
    )
    block = args.block_size
    model = {"checkpoint": args.model_path, "block_size": block}
    args.model = f"sdar-b{block}" if block else "qwen"
    server_args = make_args(args.model_path, block, directory, args.concurrency)
    runner = initialize(server_args)
    save(
        directory / "runtime.json",
        dict(
            versions={
                k: importlib.metadata.version(k)
                for k in ("torch", "sglang", "flashinfer-python", "sglang-kernel")
            },
            dtype=str(runner.dtype),
            kv_dtype=str(runner.kv_cache_dtype),
            checkpoint=model,
            sglang_source=str(Path(__import__("sglang").__file__).resolve()),
            graph_policy="Separate captured full-model graphs for B and 2B; planning stays outside replay",
            gpu=torch.cuda.get_device_name(),
            job=os.environ.get("SLURM_JOB_ID"),
        ),
    )
    require(
        runner.dtype == torch.bfloat16 and runner.kv_cache_dtype == torch.bfloat16,
        "The experiment requires BF16 weights, activations, and KV",
    )
    n = args.concurrency
    req_rows = runner.req_to_token_pool.alloc_rows(n)
    require(req_rows is not None, "Request allocation failed")
    req = torch.tensor(req_rows, dtype=torch.int64, device="cuda")
    slots = runner.token_to_kv_pool_allocator.alloc(n * 512)
    require(slots is not None, "KV allocation failed")
    slots = slots.reshape(n, 512)
    runner.req_to_token_pool.req_to_token[req, :512] = slots.to(torch.int32)
    prompt = torch.tensor(prompt_ids, dtype=torch.long, device="cuda")[None].repeat(
        n, 1
    )
    require(prompt.shape[1] == 128, "Use the existing 128-token matched prompt")
    solo_backend = FlashInferAttnBackend(runner, init_new_workspace=True)
    pair_backend = (
        FlashInferBlockPairBackend(runner)
        if block
        else FlashInferAttnBackend(runner, init_new_workspace=True)
    )
    with torch.inference_mode():
        # Initialize the captured attention prefix before warmup reads it.
        for layer in range(runner.model_config.num_hidden_layers):
            for tensor in runner.token_to_kv_pool.get_kv_buffer(layer):
                tensor[slots.reshape(-1)] = 0
        solo = GraphCall(
            runner,
            solo_backend,
            slots,
            req,
            block or 1,
            ForwardMode.DLLM_EXTEND if block else ForwardMode.DECODE,
        )
        pair = GraphCall(
            runner,
            pair_backend,
            slots,
            req,
            2 * block if block else 128,
            ForwardMode.DLLM_EXTEND if block else ForwardMode.EXTEND,
        )
        if block:
            gate = qualify(solo, pair, prompt, block, runner, slots)
            save(directory / "equivalence.json", gate)
            require(
                gate["passed"],
                "Pair equivalence failed; preserve diagnostics and stop this cell",
            )
        for threshold in [0.0, 0.5, 0.95] if block else [0.0]:
            for variant in ["separate", "paired"] if block else ["qwen"]:
                warm = execute(solo, pair, prompt, block, threshold, variant)
                save(directory / f"warmup-{threshold}-{variant}.json", warm)
            for trial in range(args.trials):
                variants = (
                    (
                        ["separate", "paired"]
                        if trial % 2 == 0
                        else ["paired", "separate"]
                    )
                    if block
                    else ["qwen"]
                )
                results = {}
                for variant in variants:
                    result = execute(solo, pair, prompt, block, threshold, variant)
                    result.update(
                        trial=trial,
                        model=args.model,
                        graph_replays=solo.replays + pair.replays,
                    )
                    save(directory / f"t{threshold}-r{trial}-{variant}.json", result)
                    results[variant] = result
                if block:
                    exact = (
                        results["separate"]["output_ids"]
                        == results["paired"]["output_ids"]
                    )
                    save(
                        directory / f"t{threshold}-r{trial}-comparison.json",
                        dict(
                            outputs_exact=exact,
                            admitted=exact,
                            decode_speedup=results["separate"][
                                "cuda_event_generation_ms"
                            ]
                            / results["paired"]["cuda_event_generation_ms"]
                            if exact
                            else None,
                        ),
                    )
            for variant in ["separate", "paired"] if block else ["qwen"]:
                result = execute(
                    solo, pair, prompt, block, threshold, variant, diagnostic=True
                )
                save(directory / f"diagnostic-{threshold}-{variant}.json", result)
                if args.profile:
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ]
                    ) as profiler:
                        profiled = execute(
                            solo,
                            pair,
                            prompt,
                            block,
                            threshold,
                            variant,
                            diagnostic=True,
                            record_phases=True,
                        )
                    trace = directory / f"profile-{threshold}-{variant}.json"
                    profiler.export_chrome_trace(str(trace))
                    profiled["profile_trace"] = trace.name
                    save(
                        directory / f"profile-timing-{threshold}-{variant}.json",
                        profiled,
                    )
                    with (
                        directory / f"profile-operators-{threshold}-{variant}.txt"
                    ).open("x") as stream:
                        stream.write(
                            profiler.key_averages().table(
                                sort_by="self_cuda_time_total", row_limit=100
                            )
                        )
    save(
        directory / "complete.json",
        dict(state="complete", graphs_replayed=solo.replays + pair.replays),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument(
        "--block-size",
        type=int,
        choices=[0, 16, 32],
        required=True,
        help="Use zero for Qwen AR",
    )
    p.add_argument(
        "--prompt-ids",
        type=Path,
        required=True,
        help="JSON list of 128 matched prompt token IDs",
    )
    p.add_argument("--concurrency", type=int, choices=[1, 4, 16], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument(
        "--profile",
        action="store_true",
        help="Save separate CPU/CUDA traces after clean timings",
    )
    worker(p.parse_args())
