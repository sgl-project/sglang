"""Deprecated import path for ``sglang.benchmark.one_batch``.

``python -m sglang.bench_one_batch`` and ``from sglang.bench_one_batch import ...``
still work, but the implementation now lives in ``sglang.benchmark.one_batch``.
Update references to the new path.
"""

import warnings

from sglang.benchmark.one_batch import *  # noqa: F401,F403
from sglang.benchmark.one_batch import cli_main

warnings.warn(
    "`sglang.bench_one_batch` is deprecated and will be removed in a future "
    "release; use `sglang.benchmark.one_batch` instead "
    "(e.g. `python -m sglang.benchmark.one_batch`).",
    FutureWarning,
    stacklevel=1,
)
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.layers.moe import initialize_moe_config
from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler_components.dp_attn import prepare_mlp_sync_batch_raw
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.model_executor.cuda_graph_config import Phase
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import (
    configure_logger,
    get_bool_env_var,
    maybe_reindex_device_id,
    require_mlp_sync,
    require_mlp_tp_gather,
    set_gpu_proc_affinity,
    suppress_other_loggers,
)
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.srt.utils.tensor_bridge import use_mlx


def start_profile(
    profile_activities,
    profile_record_shapes=False,
    rank_print=print,
    trace_filename=None,
):
    """
    Abstracted function to start profiling based on profile_activities.
    Returns profiler object (or None).
    """
    if use_mlx():
        import mlx.core as mx

        if trace_filename:
            mlx_trace_filename = trace_filename.replace(".trace.json.gz", ".gputrace")
            mx.metal.start_capture(mlx_trace_filename)
            rank_print(f"MLX Metal capture started directly to {mlx_trace_filename}")
        return "mlx"

    if "CUDA_PROFILER" in profile_activities:
        try:
            torch.cuda.cudart().cudaProfilerStart()
            rank_print("CUDA Profiler started (nsys will begin capturing)")
        except Exception as e:
            rank_print(f"Failed to start CUDA profiler: {e}")
        return None
    else:
        activities = []
        if "CPU" in profile_activities:
            activities.append(torch.profiler.ProfilerActivity.CPU)
        if "GPU" in profile_activities:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        if "XPU" in profile_activities:
            activities.append(torch.profiler.ProfilerActivity.XPU)
        if activities:
            profiler = torch.profiler.profile(
                activities=activities,
                with_stack=True,
                record_shapes=profile_record_shapes,
            )
            profiler.start()
            return profiler
        return None


def stop_profile(
    profiler,
    profile_activities,
    rank_print=print,
    save_trace=False,
    trace_filename=None,
    stage=None,
):
    """
    Abstracted function to stop profiling based on profile_activities.
    Optionally saves trace results and prints completion messages.
    """
    if profiler == "mlx":
        import mlx.core as mx

        mx.metal.stop_capture()

        if save_trace and trace_filename:
            # Change SGLang's default torch extension to Apple's .gputrace extension
            mlx_trace_filename = trace_filename.replace(".trace.json.gz", ".gputrace")

            stage_desc = f"for {stage}" if stage else ""
            rank_print(f"MLX Metal gputrace {stage_desc} saved to {mlx_trace_filename}")
        return

    if "CUDA_PROFILER" in profile_activities:
        try:
            torch.cuda.cudart().cudaProfilerStop()
            rank_print("CUDA Profiler stopped (nsys should dump traces)")
        except Exception as e:
            rank_print(f"Failed to stop CUDA profiler: {e}")
    elif profiler is not None:
        profiler.stop()

    if save_trace:
        if profiler is not None:
            if trace_filename:
                _save_profile_trace_results(
                    profiler, profile_activities, trace_filename
                )
                stage_desc = f"for {stage}" if stage else ""
                rank_print(
                    f"torch profiler chrome trace {stage_desc} saved to {trace_filename}"
                )
        if "CUDA_PROFILER" in profile_activities:
            rank_print(f"CUDA profiler trace for {stage} completed")


@dataclasses.dataclass
class BenchArgs:
    run_name: str = "default"
    batch_size: Tuple[int] = (1,)
    input_len: Tuple[int] = (1024,)
    skewed_input_lens: Tuple[int] = ()
    output_len: Tuple[int] = (16,)
    prompt_filename: str = ""
    result_filename: str = "result.jsonl"
    correctness_test: bool = False
    # This is only used for correctness test
    cut_len: int = 4
    log_decode_step: int = 0
    profile: bool = False
    profile_record_shapes: bool = False
    profile_activities: Tuple[str] = ("CPU", "GPU")
    profile_stage: str = "all"
    profile_filename_prefix: str = "profile"
    profile_start_step: Optional[int] = None
    profile_steps: Optional[int] = None

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--run-name", type=str, default=BenchArgs.run_name)
        parser.add_argument(
            "--batch-size", type=int, nargs="+", default=BenchArgs.batch_size
        )
        parser.add_argument(
            "--input-len", type=int, nargs="+", default=BenchArgs.input_len
        )
        parser.add_argument(
            "--skewed-input-lens",
            type=int,
            nargs="+",
            default=BenchArgs.skewed_input_lens,
            help=(
                "Use one batch with per-request input lengths, e.g. "
                "`--skewed-input-lens 128 512 2048 4096`. This is useful for "
                "paged-attention benchmarking where padded decode waste matters."
            ),
        )
        parser.add_argument(
            "--output-len", type=int, nargs="+", default=BenchArgs.output_len
        )
        parser.add_argument(
            "--prompt-filename", type=str, default=BenchArgs.prompt_filename
        )
        parser.add_argument(
            "--result-filename", type=str, default=BenchArgs.result_filename
        )
        parser.add_argument("--correctness-test", action="store_true")
        parser.add_argument("--cut-len", type=int, default=BenchArgs.cut_len)
        parser.add_argument(
            "--log-decode-step",
            type=int,
            default=BenchArgs.log_decode_step,
            help="Log decode latency by step, default is set to zero to disable.",
        )
        parser.add_argument("--profile", action="store_true", help="Enable profiling.")
        parser.add_argument(
            "--profile-record-shapes",
            action="store_true",
            help="Record tensor shapes in profiling results.",
        )
        parser.add_argument(
            "--profile-activities",
            type=str,
            nargs="+",
            default=["CPU", "GPU"],
            choices=["CPU", "GPU", "CUDA_PROFILER", "XPU"],
            help="Profiler activities: CPU, GPU, XPU, CUDA_PROFILER. If CPU/GPU/XPU, use torch profiler. If CUDA_PROFILER, use CUDA profiler.",
        )
        parser.add_argument(
            "--profile-stage",
            type=str,
            default=BenchArgs.profile_stage,
            choices=["all", "prefill", "decode"],
            help="Which stage to profile: all, prefill, or decode only.",
        )
        parser.add_argument(
            "--profile-filename-prefix",
            type=str,
            default=BenchArgs.profile_filename_prefix,
            help="Prefix of the profiling file names. The full profiling result file(s) be "
            '"[profile_filename_prefix]_batch[batch_size]_input[input_len]_output[output_len].trace.json.gz"',
        )
        parser.add_argument(
            "--profile-start-step",
            type=int,
            default=None,
            help="Decode step at which to start profiling (0-indexed). If not specified, defaults to output_len // 2.",
        )
        parser.add_argument(
            "--profile-steps",
            type=int,
            default=None,
            help="Number of decode steps to profile starting from profile-start-step. If not specified, profiles only one step.",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # use the default value's type to cast the args into correct types.
        attrs = [(attr.name, type(attr.default)) for attr in dataclasses.fields(cls)]
        result = {}
        for attr, attr_type in attrs:
            value = getattr(args, attr)
            # Handle None values - don't try to cast them
            if value is None or attr_type == type(None):
                result[attr] = value
            else:
                result[attr] = attr_type(value)
        return cls(**result)


def load_model(server_args, port_args, gpu_id, tp_rank):
    suppress_other_loggers()
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None
    moe_ep_rank = tp_rank // (server_args.tp_size // server_args.ep_size)

    model_config = ModelConfig.from_server_args(server_args)
    runner_kwargs = dict(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=gpu_id,
        tp_rank=tp_rank,
        tp_size=server_args.tp_size,
        moe_ep_rank=moe_ep_rank,
        moe_ep_size=server_args.ep_size,
        pp_rank=0,
        pp_size=1,
        nccl_port=port_args.nccl_port,
        server_args=server_args,
    )

    _use_mlx = use_mlx()
    if _use_mlx:
        from sglang.srt.hardware_backend.mlx.model_runner_stub import (
            MlxModelRunnerStub,
        )

        model_runner = MlxModelRunnerStub(**runner_kwargs)
    else:
        model_runner = ModelRunner(**runner_kwargs)
        model_runner.alloc_memory_pool()
        model_runner.init_backends()
    rank_print(f"max_total_num_tokens={model_runner.max_total_num_tokens}")
    tokenizer = get_tokenizer(
        server_args.tokenizer_path,
        tokenizer_mode=server_args.tokenizer_mode,
        trust_remote_code=server_args.trust_remote_code,
    )
    if server_args.tp_size > 1:
        dist.barrier()

    if _use_mlx:
        model_runner = _MlxBenchRunner(model_runner, server_args)
    else:
        model_runner = _TorchBenchRunner(model_runner)

    return model_runner, tokenizer


def prepare_inputs_for_correctness_test(bench_args, tokenizer, custom_prompts):
    if custom_prompts:
        custom_input_len = len(custom_prompts)
        bs = bench_args.batch_size[0]
        if custom_input_len > bs:
            logging.warning(
                f"Custom input size ({custom_input_len}) is larger than batch_size ({bs}). "
                f"Using the first {bs} prompts."
            )
            custom_prompts = custom_prompts[:bs]

    prompts = (
        custom_prompts
        if custom_prompts
        else [
            "The capital of France is",
            "The capital of the United Kindom is",
            "Today is a sunny day and I like",
        ]
    )
    input_ids = [tokenizer.encode(p) for p in prompts]
    sampling_params = SamplingParams(
        temperature=0,
        max_new_tokens=BenchArgs.output_len,
    )

    reqs = []
    for i in range(len(prompts)):
        assert len(input_ids[i]) > bench_args.cut_len

        tmp_input_ids = input_ids[i][: bench_args.cut_len]
        req = Req(
            rid=i,
            origin_input_text=prompts[i],
            origin_input_ids=array("q", tmp_input_ids),
            sampling_params=sampling_params,
        )
        req.full_untruncated_fill_ids = req.origin_input_ids
        req.fill_len = len(req.full_untruncated_fill_ids)
        req.logprob_start_len = -1
        req.set_extend_input_len(req.fill_len - len(req.prefix_indices))
        reqs.append(req)

    return input_ids, reqs


def prepare_extend_inputs_for_correctness_test(
    bench_args, input_ids, reqs, model_runner
):
    for i in range(len(reqs)):
        req: Req = reqs[i]
        req.full_untruncated_fill_ids += input_ids[i][bench_args.cut_len :]
        req.fill_len = len(req.full_untruncated_fill_ids)
        if model_runner is not None:
            # Use req.req_pool_idx instead of i to handle slot 0 padding correctly
            req.prefix_indices = model_runner.req_to_token_pool.req_to_token[
                req.req_pool_idx, : bench_args.cut_len
            ].to(req.prefix_indices.dtype)
            req.logprob_start_len = -1
            req.set_extend_input_len(req.fill_len - len(req.prefix_indices))
    return reqs


def prepare_synthetic_inputs_for_latency_test(
    batch_size, input_len, custom_inputs=None
):
    input_ids = (
        custom_inputs
        if custom_inputs
        else np.random.randint(0, 10000, (batch_size, input_len), dtype=np.int32)
    )
    sampling_params = SamplingParams(
        temperature=0,
        max_new_tokens=BenchArgs.output_len,
    )

    reqs = []
    for i in range(len(input_ids)):
        req = Req(
            rid=i,
            origin_input_text="",
            origin_input_ids=array("q", input_ids[i]),
            sampling_params=sampling_params,
        )
        req.full_untruncated_fill_ids = req.origin_input_ids
        req.fill_len = len(req.full_untruncated_fill_ids)
        req.logprob_start_len = -1
        req.set_extend_input_len(req.fill_len - len(req.prefix_indices))
        reqs.append(req)

    return reqs


def prepare_skewed_synthetic_inputs_for_latency_test(input_lens):
    input_ids = [
        np.random.randint(0, 10000, input_len, dtype=np.int32)
        for input_len in input_lens
    ]
    return prepare_synthetic_inputs_for_latency_test(
        len(input_ids),
        max(input_lens),
        custom_inputs=input_ids,
    )


class TreeCacheNamespace(SimpleNamespace):
    def supports_swa(self) -> bool:
        return False

    def supports_mamba(self) -> bool:
        return False

    def is_chunk_cache(self) -> bool:
        return False

    def is_tree_cache(self) -> bool:
        return not self.is_chunk_cache()

    def evict(self, params: EvictParams):
        pass


@torch.no_grad
def extend(reqs, model_runner):
    # Create dummy tree_cache for benchmarks (no prefix caching, just allocation)
    dummy_tree_cache = TreeCacheNamespace(
        page_size=model_runner.server_args.page_size,
        device=model_runner.device,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
    )

    batch = ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        tree_cache=dummy_tree_cache,
        model_config=model_runner.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )
    batch.prepare_for_extend()
    _maybe_prepare_mlp_sync_batch(batch, model_runner)
    if (
        batch.input_ids is None
        and getattr(batch, "prefill_input_ids_cpu", None) is not None
    ):
        batch.input_ids = batch.prefill_input_ids_cpu.to(
            batch.device, non_blocking=True
        )
        batch.prefill_input_ids_cpu = None

    forward_batch = ForwardBatch.init_new(batch, model_runner)
    logits_output = model_runner.forward(forward_batch).logits_output
    next_token_ids = model_runner.sample(logits_output, forward_batch)
    return next_token_ids, logits_output.next_token_logits, batch


@torch.no_grad
def decode(input_token_ids, batch, model_runner):
    batch.input_ids = input_token_ids.to(torch.int64)
    batch.prepare_for_decode()
    _maybe_prepare_mlp_sync_batch(batch, model_runner)
    forward_batch = ForwardBatch.init_new(batch, model_runner)
    logits_output = model_runner.forward(forward_batch).logits_output
    next_token_ids = model_runner.sample(logits_output, forward_batch)
    return next_token_ids, logits_output.next_token_logits


def _maybe_prepare_mlp_sync_batch(batch: ScheduleBatch, model_runner):
    if require_mlp_sync(model_runner.server_args):
        prepare_mlp_sync_batch_raw(
            batch,
            dp_size=model_runner.server_args.dp_size,
            attn_tp_size=get_attention_tp_size(),
            attn_cp_size=model_runner.attn_cp_size,
            tp_group=model_runner.tp_group,
            get_idle_batch=None,
            disable_cuda_graph=model_runner.server_args.disable_cuda_graph,
            require_mlp_tp_gather=require_mlp_tp_gather(model_runner.server_args),
            disable_overlap_schedule=model_runner.server_args.disable_overlap_schedule,
            offload_tags=set(),
        )


class _TorchBenchRunner:
    """Wraps ModelRunner for the standard PyTorch benchmark path."""

    def __init__(self, model_runner):
        self.torch_runner = model_runner

    def clear(self):
        self.torch_runner.req_to_token_pool.clear()
        self.torch_runner.token_to_kv_pool_allocator.clear()

    def extend(self, reqs):
        return extend(reqs, self.torch_runner)

    def decode(self, next_token_ids, batch):
        return decode(next_token_ids, batch, self.torch_runner)

    def cleanup(self, batch):
        pass

    def synchronize(self):
        synchronize(self.torch_runner.device)

    def max_batch_size(self, input_len, output_len):
        return self.torch_runner.max_total_num_tokens // (input_len + output_len)


class _MlxBenchRunner:
    """Wraps MlxModelRunner for the MLX benchmark path."""

    def __init__(self, model_runner, server_args):
        from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner

        # Radix cache requires the scheduler's allocator/trie; disable in
        # standalone bench mode where no scheduler is present.
        self.use_req_to_token_pool = get_bool_env_var("SGLANG_MLX_USE_PAGED_ATTENTION")
        self.fake_torch_runner = model_runner
        init_kwargs = dict(
            model_path=server_args.model_path,
            trust_remote_code=server_args.trust_remote_code,
            disable_radix_cache=not self.use_req_to_token_pool,
            mem_fraction_static=server_args.mem_fraction_static,
            quantization=server_args.quantization,
        )
        if server_args.max_total_tokens is not None:
            init_kwargs["pool_size"] = server_args.max_total_tokens
        self.mlx_runner = MlxModelRunner(**init_kwargs)
        self.req_to_token_pool = (
            self.fake_torch_runner.req_to_token_pool
            if self.use_req_to_token_pool
            else None
        )
        self.req_pool_idx_by_req_id = {}
        self.next_kv_slot = 1
        self.mlx_runner.init_cache_pools(req_to_token_pool=self.req_to_token_pool)

    def clear(self):
        self.mlx_runner.clear()
        self.req_pool_idx_by_req_id.clear()
        self.next_kv_slot = 1
        if self.req_to_token_pool is not None:
            self.req_to_token_pool.clear()

    def _alloc_slots(self, count):
        slots = list(range(self.next_kv_slot, self.next_kv_slot + count))
        self.next_kv_slot += count
        return slots

    def _write_req_slots(self, req_pool_idx, start, slots):
        if self.req_to_token_pool is None or not slots:
            return
        end = start + len(slots)
        if end > self.req_to_token_pool.max_context_len:
            raise ValueError(
                "MLX skewed benchmark exceeded req_to_token capacity: "
                f"start={start}, slots={len(slots)}, "
                f"max_context_len={self.req_to_token_pool.max_context_len}. "
                "Use smaller --skewed-input-lens, smaller --output-len, or a "
                "larger model context length."
            )
        self.req_to_token_pool.req_to_token[req_pool_idx, start:end] = torch.tensor(
            slots, dtype=torch.int32
        )

    def extend(self, reqs):
        req_ids = [str(req.rid) for req in reqs]
        results = []
        for rid, req in zip(req_ids, reqs):
            token_ids = [int(t) for t in req.get_fill_ids()]
            req_pool_idx = 0
            new_slot_ids = []
            if self.req_to_token_pool is not None:
                req_pool_idx = self.req_to_token_pool.free_slots.pop(0)
                self.req_pool_idx_by_req_id[rid] = req_pool_idx
                new_slot_ids = self._alloc_slots(len(token_ids))
                self._write_req_slots(req_pool_idx, 0, new_slot_ids)
            next_token = self.mlx_runner.prefill(
                req_id=rid,
                new_token_ids=token_ids,
                full_token_ids=token_ids,
                prefix_slot_ids=[],
                new_slot_ids=new_slot_ids,
                req_pool_idx=req_pool_idx,
            )
            results.append(next_token)
        return torch.tensor(results), None, req_ids

    def decode(self, next_token_ids, req_ids):
        if self.req_to_token_pool is not None:
            for req_id in req_ids:
                cache = self.mlx_runner._req_caches[req_id]
                offset = self.mlx_runner._first_attention_cache(cache).offset
                req_pool_idx = self.req_pool_idx_by_req_id[req_id]
                slot = self._alloc_slots(1)
                self._write_req_slots(req_pool_idx, offset, slot)
        next_token_ids = self.mlx_runner.decode_batch(req_ids)
        return torch.tensor(next_token_ids), None

    def cleanup(self, batch):
        if isinstance(batch, list):
            for req_id in batch:
                if self.req_to_token_pool is not None:
                    req_pool_idx = self.req_pool_idx_by_req_id.pop(req_id, None)
                    if req_pool_idx is not None:
                        self.req_to_token_pool.free_slots.append(req_pool_idx)
            # The standalone benchmark owns synthetic slot bookkeeping. Avoid
            # the production remove_request() final KV sync, which assumes the
            # scheduler has populated every remaining decode slot.
            self.mlx_runner.clear()

    def synchronize(self):
        pass

    def max_batch_size(self, input_len, output_len):
        return self.fake_torch_runner.max_total_num_tokens // (input_len + output_len)


def _read_prompts_from_file(prompt_file, rank_print):
    """Read custom prompts from the file specified by `--prompt-filename`."""
    if not prompt_file:
        return []
    if not os.path.exists(prompt_file):
        rank_print(
            f"Custom prompt file {prompt_file} not found. Using default inputs..."
        )
        return []
    with open(prompt_file, "r") as pf:
        return pf.readlines()


def _get_torch_profiler_output_dir():
    return os.environ.get("SGLANG_TORCH_PROFILER_DIR", "/tmp")


def _create_torch_profiler_filename(
    profile_filename_prefix, batch_size, input_len, output_len, stage
):
    output_dir = _get_torch_profiler_output_dir()
    filename = f"{profile_filename_prefix}_batch{batch_size}_input{input_len}_output{output_len}_{stage}.trace.json.gz"
    return os.path.join(output_dir, filename)


def _save_profile_trace_results(profiler, profile_activities, filename):
    parent_dir = os.path.dirname(os.path.abspath(filename))
    os.makedirs(parent_dir, exist_ok=True)
    profiler.export_chrome_trace(filename)
    if "GPU" in profile_activities:
        sort_by = "self_cuda_time_total"
    elif "XPU" in profile_activities:
        sort_by = "self_xpu_time_total"
    else:
        sort_by = "self_cpu_time_total"
    print(profiler.key_averages(group_by_input_shape=True).table(sort_by=sort_by))


def correctness_test(
    server_args,
    port_args,
    bench_args,
    gpu_id,
    tp_rank,
):
    # Configure the logger
    configure_logger(server_args, prefix=f" TP{tp_rank}")
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    # Load the model
    model_runner, tokenizer = load_model(server_args, port_args, gpu_id, tp_rank)

    # Prepare inputs
    custom_prompts = _read_prompts_from_file(bench_args.prompt_filename, rank_print)
    input_ids, reqs = prepare_inputs_for_correctness_test(
        bench_args, tokenizer, custom_prompts
    )
    rank_print(f"\n{input_ids=}\n")

    if bench_args.cut_len > 0:
        # Prefill
        next_token_ids, next_token_logits, batch = model_runner.extend(reqs)
        rank_print(f"prefill logits (first half): {next_token_logits} \n")

        # Prepare extend inputs
        torch_runner = getattr(model_runner, "torch_runner", None)
        reqs = prepare_extend_inputs_for_correctness_test(
            bench_args, input_ids, reqs, torch_runner
        )

    # Extend (prefill w/ KV cache)
    next_token_ids, next_token_logits, batch = model_runner.extend(reqs)
    rank_print(f"prefill logits (final): {next_token_logits} \n")

    # Decode
    output_ids = [input_ids[i] + [next_token_ids[i]] for i in range(len(input_ids))]
    for _ in range(bench_args.output_len[0] - 1):
        next_token_ids, _ = model_runner.decode(next_token_ids, batch)
        next_token_ids_list = next_token_ids.tolist()
        for i in range(len(reqs)):
            output_ids[i].append(next_token_ids_list[i])

    # Clean up
    model_runner.cleanup(batch)

    # Print output texts
    for i in range(len(reqs)):
        rank_print(f"========== Prompt {i} ==========")
        rank_print(tokenizer.decode(output_ids[i]), "\n")


def synchronize(device):
    torch.get_device_module(device).synchronize()


def latency_test_run_once(
    run_name,
    model_runner,
    rank_print,
    reqs,
    batch_size,
    input_len,
    output_len,
    log_decode_step,
    profile,
    profile_record_shapes,
    profile_activities,
    profile_filename_prefix,
    profile_stage,
    tp_rank,
    profile_start_step=None,
    profile_steps=None,
    input_token_count=None,
    input_lens=None,
):
    max_batch_size = model_runner.max_batch_size(input_len, output_len)
    if batch_size > max_batch_size:
        rank_print(
            f"skipping ({batch_size}, {input_len}, {output_len}) due to max batch size limit"
        )
        return

    model_runner.clear()

    measurement_results = {
        "run_name": run_name,
        "batch_size": batch_size,
        "input_len": input_len,
        "output_len": output_len,
    }
    if input_lens is not None:
        measurement_results["input_lens"] = list(input_lens)
    if input_token_count is None:
        input_token_count = input_len * batch_size

    tot_latency = 0

    profiler = None
    enable_profile_prefill = profile and profile_stage in ["all", "prefill"]
    trace_filename_prefill = None
    if enable_profile_prefill:
        trace_filename_prefill = _create_torch_profiler_filename(
            profile_filename_prefix, batch_size, input_len, output_len, "prefill"
        )
        profiler = start_profile(
            profile_activities,
            profile_record_shapes=profile_record_shapes,
            rank_print=rank_print,
            trace_filename=trace_filename_prefill,  # pass it in here for the MLX path only
        )

    model_runner.synchronize()
    tic = time.perf_counter()
    next_token_ids, _, batch = model_runner.extend(reqs)
    model_runner.synchronize()
    prefill_latency = time.perf_counter() - tic

    if enable_profile_prefill:
        stop_profile(
            profiler,
            profile_activities,
            rank_print=rank_print,
            save_trace=True,
            trace_filename=trace_filename_prefill,
            stage="prefill",
        )

    tot_latency += prefill_latency
    throughput = input_token_count / prefill_latency
    rank_print(
        f"Prefill. latency: {prefill_latency:6.5f} s, throughput: {throughput:9.2f} token/s"
    )
    measurement_results["prefill_latency"] = prefill_latency
    measurement_results["prefill_throughput"] = throughput

    decode_latencies = []
    # Determine profiling start step and end step
    profile_start = (
        profile_start_step if profile_start_step is not None else (output_len // 2)
    )
    profile_end = profile_start + (profile_steps if profile_steps is not None else 1)
    enable_profile_decode = profile and profile_stage in ["all", "decode"]
    trace_filename_decode = None
    profiler = None
    for i in range(output_len - 1):
        model_runner.synchronize()
        # Start profiler at the specified step
        if enable_profile_decode and i == profile_start:
            trace_filename_decode = _create_torch_profiler_filename(
                profile_filename_prefix, batch_size, input_len, output_len, "decode"
            )
            profiler = start_profile(
                profile_activities,
                profile_record_shapes=profile_record_shapes,
                rank_print=rank_print,
                trace_filename=trace_filename_decode,
            )

        tic = time.perf_counter()
        next_token_ids, _ = model_runner.decode(next_token_ids, batch)
        model_runner.synchronize()
        latency = time.perf_counter() - tic

        # Stop profiler after the specified number of steps
        if enable_profile_decode and profiler is not None and i >= profile_end - 1:
            stop_profile(
                profiler,
                profile_activities,
                rank_print=rank_print,
                save_trace=True,
                trace_filename=trace_filename_decode,
                stage="decode",
            )
            profiler = None

        tot_latency += latency
        throughput = batch_size / latency
        decode_latencies.append(latency)
        if i < 5 or (log_decode_step > 0 and i % log_decode_step == 0):
            rank_print(
                f"Decode {i}. Batch size: {batch_size}, latency: {latency:6.5f} s, throughput: {throughput:9.2f} token/s"
            )

    # Record decode timing from 2nd output
    if output_len > 1:
        med_decode_latency = np.median(decode_latencies)
        med_decode_throughput = batch_size / med_decode_latency
        rank_print(
            f"Decode.  median latency: {med_decode_latency:6.5f} s, median throughput: {med_decode_throughput:9.2f} token/s"
        )
        measurement_results["median_decode_latency"] = med_decode_latency
        measurement_results["median_decode_throughput"] = med_decode_throughput

    throughput = (input_token_count + output_len * batch_size) / tot_latency
    rank_print(
        f"Total. latency: {tot_latency:6.3f} s, throughput: {throughput:9.2f} token/s"
    )
    measurement_results["total_latency"] = tot_latency
    measurement_results["overall_throughput"] = throughput

    model_runner.cleanup(batch)
    return measurement_results


def latency_test(
    server_args,
    port_args,
    bench_args,
    gpu_id,
    tp_rank,
):
    initialize_moe_config(server_args)
    initialize_fp8_gemm_config(server_args)
    initialize_fp4_gemm_config(server_args)

    # Set CPU affinity
    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(
            server_args.pp_size, server_args.tp_size, server_args.nnodes, tp_rank
        )

    # Configure the logger
    configure_logger(server_args, prefix=f" TP{tp_rank}")
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    # Load the model
    model_runner, tokenizer = load_model(server_args, port_args, gpu_id, tp_rank)

    # Prepare inputs for warm up
    if bench_args.skewed_input_lens:
        warmup_lens = bench_args.skewed_input_lens
        reqs = prepare_skewed_synthetic_inputs_for_latency_test(warmup_lens)
        warmup_bs = len(warmup_lens)
        warmup_input_len = max(warmup_lens)
        warmup_input_token_count = sum(warmup_lens)
    else:
        reqs = prepare_synthetic_inputs_for_latency_test(
            bench_args.batch_size[0], bench_args.input_len[0]
        )
        warmup_bs = bench_args.batch_size[0]
        warmup_input_len = bench_args.input_len[0]
        warmup_input_token_count = warmup_bs * warmup_input_len

    # Warm up
    rank_print("Warmup ...")
    latency_test_run_once(
        bench_args.run_name,
        model_runner,
        rank_print,
        reqs,
        warmup_bs,
        warmup_input_len,
        min(32, bench_args.output_len[0]),  # shorter decoding to speed up the warmup
        log_decode_step=0,
        profile=False,
        profile_record_shapes=False,
        profile_activities=("CPU", "GPU"),
        profile_filename_prefix="",
        profile_stage="all",
        tp_rank=tp_rank,
        profile_start_step=None,
        profile_steps=None,
        input_token_count=warmup_input_token_count,
        input_lens=warmup_lens if bench_args.skewed_input_lens else None,
    )

    rank_print("Benchmark ...")

    custom_inputs = _read_prompts_from_file(bench_args.prompt_filename, rank_print)
    custom_inputs = [tokenizer.encode(p.strip()) for p in custom_inputs]
    custom_input_len = len(custom_inputs)

    # Run the sweep
    result_list = []
    if bench_args.skewed_input_lens:
        if custom_inputs:
            raise ValueError(
                "--skewed-input-lens cannot be combined with --prompt-filename"
            )
        skewed_lens = bench_args.skewed_input_lens
        sweep_cases = [
            (len(skewed_lens), max(skewed_lens), ol, skewed_lens)
            for ol in bench_args.output_len
        ]
    else:
        sweep_cases = [
            (bs, il, ol, None)
            for bs, il, ol in itertools.product(
                bench_args.batch_size, bench_args.input_len, bench_args.output_len
            )
        ]

    for bs, il, ol, per_req_lens in sweep_cases:
        bs_aligned_inputs = []
        if custom_inputs:
            if custom_input_len == bs:
                bs_aligned_inputs = custom_inputs
            elif custom_input_len > bs:
                rank_print(
                    f"Custom input size ({custom_input_len}) is larger than batch_size ({bs}). "
                    f"Using the first {bs} prompts."
                )
                bs_aligned_inputs = copy.deepcopy(custom_inputs[:bs])
            else:
                rank_print(
                    f"Custom input size ({custom_input_len}) is smaller than batch_size ({bs}). "
                    f"Pad to the desired batch_size with the last prompt."
                )
                bs_aligned_inputs = copy.deepcopy(custom_inputs)
                bs_aligned_inputs.extend(
                    [bs_aligned_inputs[-1]] * (bs - custom_input_len)
                )

        if per_req_lens is not None:
            rank_print(f"Skewed input lengths: {list(per_req_lens)}")
            reqs = prepare_skewed_synthetic_inputs_for_latency_test(per_req_lens)
            input_token_count = sum(per_req_lens)
        else:
            reqs = prepare_synthetic_inputs_for_latency_test(bs, il, bs_aligned_inputs)
            input_token_count = bs * il
        ret = latency_test_run_once(
            bench_args.run_name,
            model_runner,
            rank_print,
            reqs,
            bs,
            il,
            ol,
            bench_args.log_decode_step,
            bench_args.profile if tp_rank == 0 else None,
            bench_args.profile_record_shapes if tp_rank == 0 else None,
            bench_args.profile_activities,
            bench_args.profile_filename_prefix,
            bench_args.profile_stage,
            tp_rank,
            bench_args.profile_start_step,
            bench_args.profile_steps,
            input_token_count=input_token_count,
            input_lens=per_req_lens,
        )
        if ret is not None:
            result_list.append(ret)

    # Write results in jsonlines format on rank 0.
    if tp_rank == 0 and bench_args.result_filename:
        with open(bench_args.result_filename, "a") as fout:
            for result in result_list:
                fout.write(json.dumps(result) + "\n")

    if server_args.tp_size > 1:
        destroy_model_parallel()
        destroy_distributed_environment()


def main(server_args, bench_args):
    # Post-init write to the legacy cuda_graph_max_bs_decode field would
    # not propagate to cuda_graph_config; update the decode phase directly.
    if server_args.cuda_graph_config is not None:
        server_args.cuda_graph_config[Phase.DECODE].max_bs = max(bench_args.batch_size)

    _set_envs_and_config(server_args)

    if server_args.model_path:
        if bench_args.correctness_test:
            work_func = correctness_test
        else:
            work_func = latency_test
    else:
        raise ValueError(
            "Provide --model-path for running the tests or "
            "provide --result-filename for plotting the results"
        )

    port_args = PortArgs.init_new(server_args)

    # Calculate local ranks for multi-node setup
    nranks_per_node = server_args.tp_size // server_args.nnodes
    local_rank_start = server_args.node_rank * nranks_per_node
    local_rank_end = local_rank_start + nranks_per_node

    if server_args.tp_size == 1:
        work_func(server_args, port_args, bench_args, 0, 0)
    else:
        workers = []
        for tp_rank in range(local_rank_start, local_rank_end):
            with maybe_reindex_device_id(tp_rank - local_rank_start) as gpu_id:
                proc = multiprocessing.Process(
                    target=work_func,
                    args=(
                        server_args,
                        port_args,
                        bench_args,
                        gpu_id,
                        tp_rank,
                    ),
                )
                proc.start()
                workers.append(proc)

        for proc in workers:
            proc.join()

        proc.terminate()


if __name__ == "__main__":
    cli_main()
