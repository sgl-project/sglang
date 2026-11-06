"""SGLang workspace and expert execution for FlashInfer SM90 MXFP4 MoE.

Single-node pure EP only. Model routing and shared experts remain in the model.
Workspace allocation and collective tuning finish before KV-pool sizing.
"""

import functools
import logging

import torch
import torch.distributed as dist

from sglang.srt.arg_groups.model_override_base import resolved_view
from sglang.srt.runtime_context import get_exec, get_parallel, get_server_args

logger = logging.getLogger(__name__)
_BUFFERS = {}
_BUFFER_STREAMS = {}
_REPORTED_WORKSPACES = set()
_BUFFER_READY = {}
_OUTPUT_WORKSPACES = {}


def enabled():
    return get_exec().moe.enable_mxfp4_mega


@functools.lru_cache(maxsize=1)
def init_nvshmem():
    import numpy as np
    import nvshmem.core
    from cuda.core import Device

    from sglang.srt.distributed import get_moe_ep_group

    group = get_moe_ep_group()
    world = dist.get_world_size(group.device_group)
    rank = dist.get_rank(group.device_group)
    if (
        get_parallel().pp_size != 1
        or get_parallel().moe_tp_size != 1
        or world != dist.get_world_size()
        or torch.cuda.get_device_capability() != (9, 0)
    ):
        raise ValueError(
            "SM90 MXFP4 experiment requires PP1, pure EP over the full world, and SM90"
        )
    uid = nvshmem.core.get_unique_id(empty=rank != 0)
    raw = uid._data.view(np.uint8).copy()
    tensor = torch.from_numpy(raw)
    dist.broadcast(tensor, src=0, group=group.cpu_group)
    uid._data[:] = tensor.numpy().view(uid._data.dtype)
    device = Device(torch.cuda.current_device())
    device.set_current()
    nvshmem.core.init(
        device=device, uid=uid, rank=rank, nranks=world, initializer_method="uid"
    )
    return rank, world


def workspace_capacity(args, context_len):
    """Bound rank-local tokens for both contiguous GDN and attention layouts.

    A request can round its contiguous CP shard up to a 64-token boundary.
    Without chunking the first admitted request can exceed max_prefill_tokens.
    Use the same scheduler bound on every EP rank, including empty ranks.
    """
    cp, requests = args.attn_cp_size, args.max_running_requests
    if cp <= 1 or not requests or requests <= 0 or context_len <= 0:
        raise ValueError(
            "Mega workspace requires CP and explicit positive request/context bounds"
        )
    chunk = args.chunked_prefill_size
    total = (
        chunk
        if chunk is not None and chunk > 0
        else max(args.max_prefill_tokens, context_len)
    )
    local = min(total, (total + cp - 1) // cp + 63 * requests)
    return (local + 255) // 256 * 256


@torch.inference_mode(False)
@torch.no_grad()
def warmup_experts(experts, *, device, dtype, context_len):
    """Warm mutable workspaces reused outside inference mode by the scheduler."""
    from flashinfer.moe_ep._sm90_mxfp4.staging_fused import stage_mega_moe_inputs_fused
    from flashinfer.moe_ep.kernel_src.sm90 import pull_style_cutedsl_megakernel as sm90
    from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim.autotune import (
        autotune_hopper_mxfp4_mega_moe,
    )

    from sglang.srt.distributed import get_moe_ep_group

    args = resolved_view(get_server_args())
    rank, world = init_nvshmem()
    group = get_moe_ep_group()
    capacity = workspace_capacity(args, context_len)
    hidden = experts.hidden_size
    key = (
        device.index,
        id(group.device_group),
        world,
        capacity,
        experts.num_experts,
        experts.top_k,
        hidden,
        experts.intermediate_size_per_partition,
    )
    if key not in _BUFFERS:
        # CPU cache files may differ between hosts. Resolve once and distribute
        # either the complete tactic or the error before symmetric allocation.
        selection = [None]
        if rank == 0:
            try:
                selection[0] = (
                    sm90.resolve_hopper_mxfp4_knobs(
                        None,
                        world_size=world,
                        hidden=hidden,
                        intermediate=experts.intermediate_size_per_partition,
                        num_total_experts=experts.num_experts,
                        num_topk=experts.top_k,
                        num_max_tokens=capacity,
                    ),
                    None,
                )
            except Exception as exc:
                selection[0] = (None, str(exc))
        dist.broadcast_object_list(selection, src=0, group=group.cpu_group)
        tactic, error = selection[0]
        if error is not None:
            raise RuntimeError(f"MXFP4 tactic resolution failed: {error}")
        signature = (key[2:], tactic, args.disable_flashinfer_autotune)
        signatures = [None] * world
        dist.all_gather_object(signatures, signature, group=group.cpu_group)
        if any(peer != signature for peer in signatures):
            raise ValueError("EP ranks disagree on Mega workspace geometry or autotune")
        buf = sm90.get_symm_buffer_for_hopper_mxfp4_mega_moe(
            experts.num_experts,
            capacity,
            experts.top_k,
            hidden,
            experts.intermediate_size_per_partition,
            rank,
            world,
            knobs=tactic,
        )
        x = torch.zeros((capacity, hidden), device=device, dtype=dtype)
        ids = (
            torch.arange(capacity * experts.top_k, device=device)
            .reshape(capacity, experts.top_k)
            .remainder(experts.num_experts)
            .int()
        )
        weights = torch.full(
            ids.shape, 1.0 / experts.top_k, device=device, dtype=torch.float32
        )
        stage_mega_moe_inputs_fused(
            x,
            weights,
            ids,
            buf.x,
            buf.x_sf,
            buf.topk_idx,
            buf.topk_weights,
            quantize_input=True,
            safe_quantization=True,
        )
        if not args.disable_flashinfer_autotune:
            autotune_hopper_mxfp4_mega_moe(
                buf.output_activation,
                experts.mega_l1_weights,
                experts.mega_l2_weights,
                buf,
                process_group=group.device_group,
            )
        if args.attn_cp_size > 1 and args.attn_cp_size == world:
            from sglang.srt.distributed.device_communicators.torch_symm_mem import (
                TorchSymmMemCommunicator,
            )

            communicator = TorchSymmMemCommunicator.get(group.cpu_group, device)
            output = communicator.allocate_peer_workspace((capacity, hidden), dtype)
            destinations = torch.full(
                (capacity, 2), -1, device=device, dtype=torch.int64
            )
            shared = torch.zeros((capacity, hidden), device=device, dtype=dtype)
            peers = output.peer_ptrs.view(torch.int64)
            _OUTPUT_WORKSPACES[key] = (
                output,
                destinations,
                peers,
                shared,
                torch.ones(capacity, device=device, dtype=torch.float32),
            )
        _BUFFERS[key] = buf
        _BUFFER_STREAMS[key] = torch.cuda.current_stream(device)
        logger.info(
            "MXFP4 Mega workspace warmed: EP=%d capacity=%d autotune=%s fused_stage=True",
            world,
            capacity,
            not args.disable_flashinfer_autotune,
        )
    buf = _BUFFERS[key]
    # Compile each layer's weight arguments on the loading stream. At first
    # use, rebind only the launch arguments to the scheduler's execution stream.
    sm90.hopper_mxfp4_mega_launch_thunk(
        experts.mega_l1_weights, experts.mega_l2_weights, buf
    )
    if key in _OUTPUT_WORKSPACES:
        _, destinations, peers, shared, gates = _OUTPUT_WORKSPACES[key]
        sm90.hopper_mxfp4_mega_launch_thunk(
            experts.mega_l1_weights,
            experts.mega_l2_weights,
            buf,
            output_destinations=destinations,
            output_peer_ptrs=peers,
            shared_output=shared,
            shared_gate=gates,
        )
    _BUFFER_READY[key] = torch.cuda.current_stream(device).record_event()
    experts._sm90_mega_workspace = (buf, key)


def prepare_workspace(experts, hidden_states, *, num_experts, top_k):
    """Reuse warmed storage; never allocate or tune on a request's forward path."""
    workspace = getattr(experts, "_sm90_mega_workspace", None)
    if workspace is None:
        raise RuntimeError("MXFP4 Mega workspace must be warmed before serving")
    buf, key = workspace
    if (num_experts, top_k, hidden_states.shape[1]) != (key[4], key[5], key[6]):
        raise ValueError("MXFP4 Mega geometry differs from its warmup")
    if hidden_states.shape[0] > key[3]:
        raise ValueError(
            f"MXFP4 local tokens exceed the scheduler workspace bound {key[3]}"
        )
    current = torch.cuda.current_stream(hidden_states.device)
    ready = _BUFFER_READY.pop(key, None)
    if ready is not None:
        current.wait_event(ready)
        _BUFFER_STREAMS[key] = current
    if current != _BUFFER_STREAMS[key]:
        raise ValueError(
            "SM90 Mega workspace must be reused on its bound execution stream"
        )
    return workspace


def run_experts(
    experts,
    hidden_states,
    routing,
    workspace,
    *,
    output_layout=None,
    shared_output=None,
    shared_gate=None,
):
    """Execute already-routed tokens and return a borrowed view of the workspace.

    The caller must copy/add into owned storage before another layer reuses it.
    Empty local ranks must still call this function to participate in EP.
    """
    from flashinfer.moe_ep._sm90_mxfp4.staging_fused import stage_mega_moe_inputs_fused
    from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel import (
        hopper_mxfp4_mega_launch_thunk,
    )

    buf, key = workspace
    stage_mega_moe_inputs_fused(
        hidden_states,
        routing.topk_weights,
        routing.topk_ids,
        buf.x,
        buf.x_sf,
        buf.topk_idx,
        buf.topk_weights,
        quantize_input=True,
        safe_quantization=True,
    )
    mapped = output_layout is not None
    thunk_key = (key, mapped)
    extra = {}
    if mapped:
        if key not in _OUTPUT_WORKSPACES:
            raise ValueError("Mapped combine requires a shared pure CP/EP group")
        output, destinations, peers, shared, gates = _OUTPUT_WORKSPACES[key]
        plan = output_layout.exchange_plan(
            output_layout.current, output_layout.mlp_destination
        )
        count = plan.output_destinations.shape[0]
        if count > hidden_states.shape[0]:
            raise ValueError("MoE output map exceeds the local input rows")
        destinations.fill_(-1)
        destinations[:count].copy_(plan.output_destinations)
        if shared_output is None:
            shared.zero_()
        else:
            shared[:count].copy_(shared_output[:count])
        if shared_gate is None:
            gates.fill_(1)
        else:
            gates[:count].copy_(shared_gate[:count])
        extra = dict(
            output_destinations=destinations,
            output_peer_ptrs=peers,
            shared_output=shared,
            shared_gate=gates,
        )
        output.data_handle.barrier(channel=0)
    if getattr(experts, "_sm90_mega_thunk_key", None) != thunk_key:
        experts._sm90_mega_thunk = hopper_mxfp4_mega_launch_thunk(
            experts.mega_l1_weights, experts.mega_l2_weights, buf, **extra
        )
        experts._sm90_mega_thunk_key = thunk_key
    if key not in _REPORTED_WORKSPACES:
        workspace = buf._frontend._mega
        logger.info(
            "SM90 Mega workspace: allocator=NVSHMEM cached_buffers=%d "
            "input_bytes=%d communication_bytes=%d local_bytes=%d "
            "reuse=across_layers_on_construction_stream output_storage=borrowed",
            len(_BUFFERS),
            sum(t.numel() * t.element_size() for t in buf._sym_roots),
            workspace.shared_workspace.numel(),
            workspace.local_workspace.numel(),
        )
        _REPORTED_WORKSPACES.add(key)
    experts._sm90_mega_thunk()
    if mapped:
        output.data_handle.barrier(channel=1)
        output_layout.mapped_moe_output = True
        rows = len(
            output_layout.token_layouts[output_layout.mlp_destination][
                output_layout.rank
            ]
        )
        return output.buffers[:rows]
    return buf.output_activation[: hidden_states.shape[0]]
