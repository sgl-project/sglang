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
# The regions the barrier releases and rebuilds: everything the engine
# allocates after initialization and reads back from disk.
_MEMORY_TAGS = ("weights", "kv_cache")

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


def validate_listen_host(host):
    """Refuse listen addresses the snapshot path cannot carry.

    The probes and sockets behind a restore speak plain IPv4-or-hostname
    addresses; an IPv6 literal would have to be bracketed for some of them and
    unbracketed for others. Refusing it where the address enters turns a later
    probe timeout into a message that names the value.
    """
    if ":" in host:
        raise SnapshotUsageError(
            "Initialized snapshots require an IPv4 address or hostname, "
            f"not an IPv6 literal: {host}"
        )


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
    validate_listen_host(args.host)


def validate_server_args(argv):
    from sglang.srt.plugins import load_plugins
    from sglang.srt.server_args import prepare_server_args

    load_plugins()
    args = prepare_server_args(argv)
    args.resolve_once()
    validate_startup(args)
    return args


def _run_canary(scheduler):
    """Run one greedy single-token prefill and describe the token it sampled.

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
        logits_output = result.logits_output
        if result.delay_sample_func is not None:
            result = result.delay_sample_func()
        scheduler.device_module.synchronize()
        token_id = int(result.next_token_ids[0].item())
        req.output_ids.append(token_id)
        logprob = _sampled_logprob(logits_output, token_id)
    finally:
        allocator = scheduler.token_to_kv_pool_allocator
        allocator.free_group_begin()
        try:
            release_kv_cache(req, scheduler.tree_cache, is_insert=False)
        finally:
            allocator.free_group_end()
    return SnapshotCanary(prompt=CANARY_PROMPT, token_id=token_id, logprob=logprob)


def _sampled_logprob(logits_output, token_id):
    """Log-probability of the sampled token, from the raw next-token logits.

    Read here rather than from the sampler because the canary asks for no
    logprobs, and from the raw logits rather than the sampling distribution so
    that both sides of the snapshot compute it the same way.
    """
    import torch

    logits = getattr(logits_output, "next_token_logits", None)
    if logits is None:
        raise SnapshotRuntimeFailure("Snapshot canary produced no next-token logits")
    return float(torch.log_softmax(logits[0].float(), dim=-1)[token_id].item())


def _verify_canary(scheduler, expected):
    """Re-run the canary and require the reloaded engine to reproduce it."""
    observed = _run_canary(scheduler)
    if not expected.matches(observed.token_id, observed.logprob):
        raise SnapshotRuntimeFailure(
            "Snapshot canary mismatch: the reloaded engine sampled token "
            f"{observed.token_id} at logprob {observed.logprob:.6g}, but the "
            f"snapshot recorded token {expected.token_id} at "
            f"{expected.logprob:.6g}"
        )
    return observed


def _release_memory(scheduler):
    """Wait for the device to settle, then give weights and KV memory back."""
    from sglang.srt.managers.io_struct import ReleaseMemoryOccupationReqInput

    scheduler.device_module.synchronize()
    scheduler.weight_updater.release_memory_occupation(
        ReleaseMemoryOccupationReqInput(tags=list(_MEMORY_TAGS))
    )


def _reload_and_verify(scheduler, server_args_view, expected):
    """Rebuild the released state and prove the recorded canary still holds.

    Shared by the create-time rehearsal and the restore path, so both sides
    exercise one implementation of the state transition they must agree on.
    """
    from sglang.srt.managers.io_struct import (
        ResumeMemoryOccupationReqInput,
        UpdateWeightFromDiskReqInput,
    )

    updater = scheduler.weight_updater
    updater.resume_memory_occupation(
        ResumeMemoryOccupationReqInput(tags=list(_MEMORY_TAGS))
    )
    result = updater.update_weights_from_disk(
        UpdateWeightFromDiskReqInput(
            model_path=server_args_view.model_path,
            load_format=server_args_view.load_format,
        )
    )
    if not result.success:
        raise SnapshotRuntimeFailure(f"Snapshot weight reload failed: {result.message}")
    scheduler.device_module.synchronize()
    return _verify_canary(scheduler, expected)


def active_artifact_path():
    """The artifact directory when this process tree is a snapshot engine.

    The snapshot entry marks the tree it starts, so a plain ``sglang serve``
    that inherited ``SGLANG_SNAPSHOT_DIR`` from the operator's shell (or a
    shared ``.env``) never enters the barrier: the variable only says where an
    artifact lives, while the marker says a controller asked for this engine.
    """
    from sglang.srt.environ import envs

    if not envs.SGLANG_SNAPSHOT_ENGINE.get():
        return None
    artifact_path = envs.SGLANG_SNAPSHOT_DIR.get()
    if not artifact_path:
        raise SnapshotUsageError(
            "SGLANG_SNAPSHOT_ENGINE is set but SGLANG_SNAPSHOT_DIR is not"
        )
    return artifact_path


def _apply_release_address(release):
    """Apply a restore's listen-address override in this process.

    The engine tree is captured once and every process keeps its own resolved
    config, so each process that wakes on the release applies the override
    itself: patching only the HTTP process would leave the scheduler on the
    captured address (disaggregation endpoints, metrics, registration).
    """
    overrides = {
        name: value
        for name, value in (("host", release.host), ("port", release.port))
        if value is not None
    }
    if overrides:
        from sglang.srt.runtime_context import get_context

        get_context().override("snapshot-restore", **overrides)


def scheduler_barrier(scheduler, artifact_path):
    """Rehearse the reload, park released, then resume and re-verify on restore.

    The barrier is where the engine becomes snapshot-safe: nothing is queued,
    scheduled or allocated after it, so the dumped process image resumes with a
    quiescent engine.
    """
    from sglang.srt.arg_groups.overrides import resolving_view

    control_dir = Path(artifact_path) / control.CONTROL_DIRNAME
    server_args_view = resolving_view(scheduler.server_args)
    try:
        if scheduler.device_module.device_count() != 1:
            raise SnapshotUsageError(
                "Initialized snapshots require exactly one visible CUDA device"
            )
        canary = _run_canary(scheduler)
        _release_memory(scheduler)
        # Rehearsal: prove before the capture that the released engine comes
        # back to the same canary. A restore that would fail here fails while
        # the source container and the model path are still known good; the
        # engine is released again below, so the captured image stays quiescent.
        _reload_and_verify(scheduler, server_args_view, canary)
        _release_memory(scheduler)
        uuid = f"GPU-{scheduler.device_module.get_device_properties(0).uuid}"
        write_json_atomic(
            control_dir / control.SCHEDULER,
            msgspec.to_builtins(control.SchedulerInfo(gpu_uuid=uuid, canary=canary)),
        )
        release = control.wait_and_read(
            control_dir, control.RELEASE, control.ReleaseInfo
        )
        _apply_release_address(release)
        expected = load_manifest(Path(artifact_path)).canary
        observed = _reload_and_verify(scheduler, server_args_view, expected)
        # This run's evidence, so replacing a leftover marker is intended.
        write_json_atomic(
            control_dir / control.RESUMED,
            msgspec.to_builtins(
                control.ResumedInfo(
                    token_id=observed.token_id, logprob=observed.logprob
                )
            ),
            overwrite=True,
        )
    except BaseException as exc:
        control.write_error(control_dir, exc)
        raise


def server_barrier(server_args, artifact_path):
    """Publish the initialized engine, then wait for release and the address.

    The HTTP listener is opened after this barrier, so the release marker is
    where a restore may redirect it: the captured address stays the default, and
    an explicit override replaces it wherever this engine resolves it.
    """
    from sglang.srt.arg_groups.overrides import resolving_view

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
    _apply_release_address(release)


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
    # The barriers key off this marker rather than off SGLANG_SNAPSHOT_DIR:
    # only this entry - the one a controller launches - knows that a snapshot
    # run asked for the engine, and the tree inherits the marker from here.
    envs.SGLANG_SNAPSHOT_ENGINE.set(True)
    control_dir = Path(artifact_path) / control.CONTROL_DIRNAME
    try:
        args = validate_server_args(sys.argv[1:])
        run_server(args)
    except BaseException as exc:
        control.write_error(control_dir, exc)
        raise
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
