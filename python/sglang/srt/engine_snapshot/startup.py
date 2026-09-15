# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Engine-side startup barriers and configuration validation for snapshots."""

import os
import sys
from pathlib import Path

import msgspec

from sglang.srt.engine_snapshot import control
from sglang.srt.engine_snapshot.errors import (
    SnapshotRuntimeFailure,
    SnapshotUsageError,
)
from sglang.srt.engine_snapshot.manifest import (
    SnapshotCanary,
    load_manifest,
    write_json_atomic,
)

# Fixed greedy prompt sampled at create time and re-checked after the restore
# weight reload; a mismatch means the reloaded engine computes different logits.
CANARY_PROMPT = "The first month of the year is"
_CANARY_MAX_NEW_TOKENS = 1

_REQUIRED_SERVER_ARGS = {
    "tp_size": 1,
    "pp_size": 1,
    "dp_size": 1,
    "nnodes": 1,
    "tokenizer_worker_num": 1,
    "detokenizer_worker_num": 1,
    "disaggregation_mode": "null",
    "weight_cache_mode": "off",
    "speculative_algorithm": None,
    "quantization": None,
    "enable_memory_saver": True,
    "enable_lora": False,
    "enable_hierarchical_cache": False,
    "use_ray": False,
    "encoder_only": False,
    "smg_grpc_mode": False,
}

_RELOADABLE_LOAD_FORMATS = ("auto", "pt", "safetensors")


def validate_startup(args):
    from sglang.srt.arg_groups.overrides import resolving_view
    from sglang.srt.environ import envs

    args = resolving_view(args)
    for name, value in _REQUIRED_SERVER_ARGS.items():
        actual = getattr(args, name)
        if value is False:
            actual = bool(actual)
        if actual != value:
            raise SnapshotUsageError(f"Initialized snapshots require {name}={value!r}")
    if envs.SGLANG_RUST_SERVER.get() or args.grpc_port is not None:
        raise SnapshotUsageError(
            "Initialized snapshots require the Python HTTP server only"
        )
    if not Path(args.model_path).is_dir():
        raise SnapshotUsageError(
            "Initialized snapshots require a local model directory"
        )
    if args.load_format not in _RELOADABLE_LOAD_FORMATS:
        raise SnapshotUsageError(
            "Initialized snapshots require auto, pt or safetensors load_format"
        )
    if args.device != "cuda" or args.base_gpu_id != 0:
        raise SnapshotUsageError(
            "Initialized snapshots require CUDA with base_gpu_id=0"
        )
    if args.ssl_keyfile is not None or args.ssl_certfile is not None:
        raise SnapshotUsageError("Initialized snapshots currently require plain HTTP")
    if not 0 < args.port < 65536:
        raise SnapshotUsageError(
            "Initialized snapshots require a fixed listen port (1-65535)"
        )


def validate_server_args(argv):
    from sglang.srt.plugins import load_plugins
    from sglang.srt.server_args import prepare_server_args

    load_plugins()
    args = prepare_server_args(argv)
    args.resolve_once()
    validate_startup(args)
    return args


def _run_canary_forward(scheduler):
    """Run one greedy single-token prefill and return the sampled token id.

    Leaves no KV-cache slot, req-pool slot or queued request behind: the engine
    is dumped (create) or starts serving (restore) right after.
    """
    from array import array

    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
    from sglang.srt.mem_cache.common import release_kv_cache
    from sglang.srt.runtime_context import get_schedule
    from sglang.srt.sampling.sampling_params import SamplingParams

    if not scheduler.is_generation:
        raise SnapshotUsageError("Initialized snapshots require a generation model")
    tokenizer = scheduler.tokenizer
    if tokenizer is None:
        raise SnapshotUsageError("Initialized snapshots require the tokenizer")
    input_ids = tokenizer.encode(CANARY_PROMPT)
    if not input_ids:
        raise SnapshotRuntimeFailure("Snapshot canary prompt tokenized empty")
    sampling_params = SamplingParams(
        max_new_tokens=_CANARY_MAX_NEW_TOKENS, temperature=0
    )
    sampling_params.normalize(tokenizer)
    req = Req(
        "snapshot-canary",
        CANARY_PROMPT,
        array("q", input_ids),
        sampling_params,
        eos_token_ids=scheduler.model_config.hf_eos_token_id,
    )
    req.tokenizer = tokenizer
    req.init_next_round_input(scheduler.tree_cache)

    attn_backend = scheduler.tp_worker.model_runner.attn_backend
    adder = PrefillAdder(
        scheduler.page_size,
        scheduler.tree_cache,
        scheduler.token_to_kv_pool_allocator,
        scheduler.running_batch,
        scheduler.new_token_ratio_tracker.current,
        scheduler.max_prefill_tokens,
        scheduler.chunked_prefill_size,
        0,
        scheduler.priority_scheduling_preemption_threshold,
        max_prefill_bs=int(scheduler.max_prefill_bs),
        max_running_requests=scheduler.max_running_requests,
        prefill_max_requests=get_schedule().prefill_max_requests,
        dllm_config=scheduler.dllm_config,
        waiting_queue_len=1,
        prefill_tile_block_m=getattr(attn_backend, "extend_attention_block_m", 64),
    )
    admitted = adder.add_one_req(
        req,
        has_chunked_req=False,
        truncation_align_size=scheduler.truncation_align_size,
    )
    if admitted != AddReqResult.CONTINUE or adder.can_run_list != [req]:
        raise SnapshotRuntimeFailure("Snapshot canary was not admitted for prefill")

    batch = ScheduleBatch.init_new(
        adder.can_run_list,
        scheduler.req_to_token_pool,
        scheduler.token_to_kv_pool_allocator,
        scheduler.tree_cache,
        scheduler.model_config,
        scheduler.enable_overlap,
        scheduler.spec_algorithm,
    )
    batch.prepare_for_extend()
    # prepare_for_extend defers the input_ids H2D copy to resolve_forward_inputs,
    # which only runs inside the (not yet running) event loop; do it here instead.
    batch.input_ids = batch.prefill_input_ids_cpu.to(batch.device, non_blocking=True)
    batch.prefill_input_ids_cpu = None
    try:
        result = scheduler.model_worker.forward_batch_generation(batch)
        if result.delay_sample_func is not None:
            result = result.delay_sample_func()
        scheduler.device_module.synchronize()
        token_id = int(result.next_token_ids[0].item())
        req.output_ids.append(token_id)
    finally:
        allocator = scheduler.token_to_kv_pool_allocator
        allocator.free_group_begin()
        try:
            release_kv_cache(req, scheduler.tree_cache, is_insert=False)
        finally:
            allocator.free_group_end()
    return token_id


def _record_canary(scheduler, control_dir):
    token_id = _run_canary_forward(scheduler)
    canary = SnapshotCanary(prompt=CANARY_PROMPT, token_id=token_id)
    write_json_atomic(control_dir / control.CANARY, msgspec.to_builtins(canary))


def _verify_canary(scheduler, artifact_path):
    manifest = load_manifest(Path(artifact_path))
    expected = manifest.canary.token_id
    observed = _run_canary_forward(scheduler)
    if observed != expected:
        raise SnapshotRuntimeFailure(
            "Snapshot canary mismatch: reloaded engine sampled token "
            f"{observed} but the snapshot recorded {expected}"
        )


def scheduler_barrier(scheduler, artifact_path):
    """Release weights and KV memory, park, then resume and re-check the canary.

    The barrier is where the engine becomes snapshot-safe: nothing is queued,
    scheduled or allocated after it, so the dumped process image resumes with a
    quiescent engine.
    """
    from sglang.srt.arg_groups.overrides import resolving_view
    from sglang.srt.managers.io_struct import (
        ReleaseMemoryOccupationReqInput,
        ResumeMemoryOccupationReqInput,
        UpdateWeightFromDiskReqInput,
    )

    control_dir = Path(artifact_path) / control.CONTROL_DIRNAME
    server_args_view = resolving_view(scheduler.server_args)
    try:
        if scheduler.device_module.device_count() != 1:
            raise SnapshotUsageError(
                "Initialized snapshots require exactly one visible CUDA device"
            )
        _record_canary(scheduler, control_dir)
        updater = scheduler.weight_updater
        tags = ["weights", "kv_cache"]
        scheduler.device_module.synchronize()
        updater.release_memory_occupation(ReleaseMemoryOccupationReqInput(tags=tags))
        uuid = f"GPU-{scheduler.device_module.get_device_properties(0).uuid}"
        write_json_atomic(
            control_dir / control.SCHEDULER,
            msgspec.to_builtins(control.SchedulerInfo(gpu_uuid=uuid)),
        )
        control.wait_for(control_dir, control.RELEASE)
        updater.resume_memory_occupation(ResumeMemoryOccupationReqInput(tags=tags))
        result = updater.update_weights_from_disk(
            UpdateWeightFromDiskReqInput(
                model_path=server_args_view.model_path,
                load_format=server_args_view.load_format,
            )
        )
        if not result.success:
            raise SnapshotRuntimeFailure(
                f"Snapshot weight reload failed: {result.message}"
            )
        scheduler.device_module.synchronize()
        _verify_canary(scheduler, artifact_path)
        (control_dir / control.RESUMED).touch()
    except BaseException as exc:
        control.write_error(control_dir, exc)
        raise


def server_barrier(server_args, artifact_path):
    """Publish the initialized engine, then wait for release and the address.

    The HTTP listener is opened after this barrier, so the release marker is
    where a restore may redirect it: the captured address stays the default, and
    an explicit override changes only this process's resolved serving config.
    """
    from sglang.srt.arg_groups.overrides import resolving_view
    from sglang.srt.runtime_context import get_context

    server_args_view = resolving_view(server_args)
    control_dir = Path(artifact_path) / control.CONTROL_DIRNAME
    scheduler_info = control.wait_and_read(
        control_dir, control.SCHEDULER, control.SchedulerInfo
    )
    write_json_atomic(
        control_dir / control.READY,
        msgspec.to_builtins(
            control.EngineInfo(
                gpu_uuid=scheduler_info.gpu_uuid,
                model_path=server_args_view.model_path,
                host=server_args_view.host,
                port=server_args_view.port,
            )
        ),
    )
    release = control.wait_and_read(control_dir, control.RELEASE, control.ReleaseInfo)
    overrides = {
        name: value
        for name, value in (("host", release.host), ("port", release.port))
        if value is not None
    }
    if overrides:
        get_context().override("snapshot-restore", **overrides)


if __name__ == "__main__":
    from sglang.launch_server import run_server
    from sglang.srt.environ import envs
    from sglang.srt.utils import kill_process_tree

    artifact_path = envs.SGLANG_SNAPSHOT_DIR.get()
    if not artifact_path:
        raise SystemExit(
            "SGLANG_SNAPSHOT_DIR is not set; start the engine through "
            "`sglang snapshot create`"
        )
    control_dir = Path(artifact_path) / control.CONTROL_DIRNAME
    try:
        args = validate_server_args(sys.argv[1:])
        run_server(args)
    except BaseException as exc:
        control.write_error(control_dir, exc)
        raise
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
