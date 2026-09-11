import numpy as np
from sglang_simulator.simulation.types import RequestStats, SchedulerConfig
from sglang_simulator.spec.accelerator import AcceleratorInfo
from sglang_simulator.spec.model import ModelInfo
from sglang_simulator.time_predictor.aiconfigurator import get_perf_model


def calc_kv_cache_cell_elems(model_info: ModelInfo, tp_size: int, pp_size: int) -> int:
    num_layers = model_info.num_hidden_layers // pp_size
    if model_info.is_mla():
        return (model_info.kv_lora_rank + model_info.qk_rope_head_dim) * num_layers
    else:
        num_kv_heads = max(model_info.num_key_value_heads // tp_size, 1)
        return num_kv_heads * model_info.head_dim * num_layers * 2


def calc_kv_cache_per_layer_elems(
    model_info: ModelInfo, tp_size: int, pp_size: int
) -> int:
    if model_info.is_mla():
        return model_info.kv_lora_rank + model_info.qk_rope_head_dim
    else:
        num_kv_heads = max(model_info.num_key_value_heads // tp_size, 1)
        return num_kv_heads * model_info.head_dim * 2


def profile_device_available_bytes(
    model: ModelInfo, device: AcceleratorInfo, scheduler_config: SchedulerConfig
) -> int:
    """Return the simulated per-GPU byte budget available to KV-cache pools."""
    # Simulation capacity must come from the declared target accelerator. Do
    # not fall back to the local CUDA device: doing so would make an identical
    # simulation config host-dependent and could silently simulate the wrong
    # hardware.
    if device.hbm_capacity_gb is None:
        raise ValueError(
            "Cannot estimate max_total_num_tokens: the simulated accelerator "
            f"{device.name!r} has no hbm_capacity_gb. Add the accelerator to "
            "the simulator hardware registry, provide hbm_capacity_gb in the "
            "simulation config, or set max_total_tokens explicitly. The "
            "simulator never falls back to the local GPU memory capacity."
        )

    perf_model = get_perf_model(scheduler_config, model)
    weights = 0
    for op in perf_model.context_ops:
        weights += op.get_weights()
    # Count weights on a single GPU
    weights /= perf_model.config.pp_size
    framework_reserved_mem_gb = 1.4
    rest_memory = (
        scheduler_config.mem_fraction_static * device.hbm_capacity_gb
        - framework_reserved_mem_gb
    ) * (1 << 30) - weights
    return int(rest_memory)


def calc_input_token_metrics(
    total_input: int,
    total_reused_tokens: int,
    total_dur_s: float,
) -> dict:
    """Compute model-independent new-input token count and throughput."""
    dur_s = max(total_dur_s, 1e-9)

    total_new_input_tokens = total_input - total_reused_tokens
    new_input_write_thr_tokens = total_new_input_tokens / dur_s

    return {
        "total_new_input": total_new_input_tokens,
        "new_input_write_throughput_tokens_per_s": new_input_write_thr_tokens,
    }


def calc_iteration_metrics(
    iteration_stats: list[dict], request_metrics: dict | None = None
) -> dict:
    """Aggregate per-iteration simulator latency into result metrics."""
    iterations = len(iteration_stats)
    forward_s = sum(
        float(item.get("forward_latency", 0) or 0) for item in iteration_stats
    )
    l2_load_s = sum(
        float(item.get("l2_load_latency", 0) or 0) for item in iteration_stats
    )
    cpu_s = sum(float(item.get("cpu_overhead", 0) or 0) for item in iteration_stats)
    total_s = forward_s + l2_load_s + cpu_s
    avg_iter_latency_ms = total_s / iterations * 1000 if iterations else 0
    metrics = {
        "iterations": iterations,
        "avg_iter_latency_ms": avg_iter_latency_ms,
    }

    if request_metrics:
        mean_ttft_ms = request_metrics.get("mean_ttft_ms")
        mean_queue_ms = request_metrics.get("mean_queue_ms")
        if mean_ttft_ms is not None and mean_queue_ms is not None:
            mean_exec_ms = mean_ttft_ms - mean_queue_ms
            metrics["mean_exec_ms"] = mean_exec_ms
            metrics["avg_iters_per_req"] = (
                mean_exec_ms / avg_iter_latency_ms if avg_iter_latency_ms else None
            )

    return metrics


def calc_metrics(requests: list[RequestStats]) -> dict:
    ttfts = []
    tpots = []
    itls = []
    e2e_latencies = []
    total_dur_s = 1e-9
    total_input = 0
    total_output = 0
    completed = 0
    total_reused_tokens = 0
    total_device_hit_tokens = 0
    total_host_hit_tokens = 0
    total_storage_hit_tokens = 0
    queue_durs = []
    dispatch_wait_durs = []
    arrival_to_prefill_durs = []
    output_token_timestamps = []
    concurrency_events = []
    for req in requests:
        if not req.is_complete():
            continue
        completed += 1
        ttfts.append(req.gen_token_latencies[0])
        # Queue latency is the time spent in SGLang's waiting queue before
        # the request's first prefill admission.
        queue_durs.append(req.queue_end - req.queue_start)
        dispatch_wait_durs.append(req.queue_start - req.created_time)
        arrival_to_prefill_durs.append(req.queue_end - req.created_time)
        if len(req.gen_token_latencies) > 1:
            # output length > 1
            tpots.append(np.mean(req.gen_token_latencies[1:]))
        itls.extend(req.gen_token_latencies[1:])
        e2e_latencies.append(sum(req.gen_token_latencies))
        token_timestamp = req.created_time
        for token_latency in req.gen_token_latencies:
            token_timestamp += token_latency
            output_token_timestamps.append(token_timestamp)
        concurrency_events.append((req.created_time, 1))
        concurrency_events.append((req.last_event_time, -1))
        total_dur_s = max(total_dur_s, req.last_event_time)
        total_input += req.input_length
        total_output += req.output_length
        total_reused_tokens += req.final_device_hit_len
        total_device_hit_tokens += req.final_device_hit_len - req.final_host_hit_len
        total_host_hit_tokens += req.final_host_hit_len - req.final_storage_hit_len
        total_storage_hit_tokens += req.final_storage_hit_len

    input_token_metrics = calc_input_token_metrics(
        total_input=total_input,
        total_reused_tokens=total_reused_tokens,
        total_dur_s=total_dur_s,
    )

    max_output_tokens_per_s = 0.0
    if output_token_timestamps:
        first_created_time = min(
            req.created_time for req in requests if req.is_complete()
        )
        num_buckets = int(max(output_token_timestamps) - first_created_time) + 1
        output_tokens_per_s = np.zeros(max(num_buckets, 1))
        for timestamp in output_token_timestamps:
            bucket = int(timestamp - first_created_time)
            output_tokens_per_s[bucket] += 1
        max_output_tokens_per_s = float(np.max(output_tokens_per_s))

    max_concurrent_requests = 0
    current_concurrent_requests = 0
    # Treat request intervals as [created_time, last_event_time): requests that
    # finish exactly when another arrives are not simultaneously active.
    for _, delta in sorted(concurrency_events, key=lambda event: (event[0], event[1])):
        current_concurrent_requests += delta
        max_concurrent_requests = max(
            max_concurrent_requests, current_concurrent_requests
        )

    return {
        "num_requests": len(requests),
        "completed": completed,
        "total_input": total_input,
        "total_output": total_output,
        "duration": total_dur_s,
        "request_throughput": completed / total_dur_s,
        "input_throughput": total_input / total_dur_s,
        "output_throughput": total_output / total_dur_s,
        "total_throughput": (total_input + total_output) / total_dur_s,
        "prefix_cache_reused_ratio": (
            0 if total_input == 0 else total_reused_tokens / total_input
        ),
        "kv_cache_storage_hit_ratio": (
            0 if total_input == 0 else total_storage_hit_tokens / total_input
        ),
        "kv_cache_host_hit_ratio": (
            0 if total_input == 0 else total_host_hit_tokens / total_input
        ),
        "kv_cache_device_hit_ratio": (
            0 if total_input == 0 else total_device_hit_tokens / total_input
        ),
        **input_token_metrics,
        "mean_ttft_ms": np.mean(ttfts or 0) * 1000,
        "median_ttft_ms": np.median(ttfts or 0) * 1000,
        "std_ttft_ms": np.std(ttfts or 0) * 1000,
        "p90_ttft_ms": np.percentile(ttfts or 0, 90) * 1000,
        "p95_ttft_ms": np.percentile(ttfts or 0, 95) * 1000,
        "p99_ttft_ms": np.percentile(ttfts or 0, 99) * 1000,
        "mean_queue_ms": max(np.mean(queue_durs or 0), 0.0) * 1000,
        "mean_dispatch_wait_ms": (max(np.mean(dispatch_wait_durs or 0), 0.0) * 1000),
        "mean_arrival_to_prefill_ms": (
            max(np.mean(arrival_to_prefill_durs or 0), 0.0) * 1000
        ),
        "mean_tpot_ms": np.mean(tpots or 0) * 1000,
        "median_tpot_ms": np.median(tpots or 0) * 1000,
        "std_tpot_ms": np.std(tpots or 0) * 1000,
        "p90_tpot_ms": np.percentile(tpots or 0, 90) * 1000,
        "p95_tpot_ms": np.percentile(tpots or 0, 95) * 1000,
        "p99_tpot_ms": np.percentile(tpots or 0, 99) * 1000,
        "mean_itl_ms": np.mean(itls or 0) * 1000,
        "median_itl_ms": np.median(itls or 0) * 1000,
        "std_itl_ms": np.std(itls or 0) * 1000,
        "p90_itl_ms": np.percentile(itls or 0, 90) * 1000,
        "p95_itl_ms": np.percentile(itls or 0, 95) * 1000,
        "p99_itl_ms": np.percentile(itls or 0, 99) * 1000,
        "max_itl_ms": np.max(itls or 0) * 1000,
        "mean_e2e_latency_ms": np.mean(e2e_latencies or 0) * 1000,
        "median_e2e_latency_ms": np.median(e2e_latencies or 0) * 1000,
        "std_e2e_latency_ms": np.std(e2e_latencies or 0) * 1000,
        "p90_e2e_latency_ms": np.percentile(e2e_latencies or 0, 90) * 1000,
        "p95_e2e_latency_ms": np.percentile(e2e_latencies or 0, 95) * 1000,
        "p99_e2e_latency_ms": np.percentile(e2e_latencies or 0, 99) * 1000,
        "concurrency": np.sum(e2e_latencies or 0) / total_dur_s,
        "max_output_tokens_per_s": max_output_tokens_per_s,
        "max_concurrent_requests": max_concurrent_requests,
        "time_cost": -1,  # Updated by external benchmark caller
    }
