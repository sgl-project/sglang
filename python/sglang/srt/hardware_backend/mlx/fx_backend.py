"""Serving executors for the exported whole-model MLX region.

The op-level ATen-to-MLX lowering layer (registry, per-op lowerings, generic
executor, export planning, Dynamo validation backend) lives in
``fx_lowering``; this module builds on it and owns only the serving-state
contract: the export executors bound to the KV pools through zero-copy views,
the deferred Radix attention dispatch, the KV-delta commit wiring, and the
Torch reference runners the validation harness compares against.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.fx

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.mlx.fx_lowering import (
    MlxFxLoweringRegistry,
    UnsupportedMlxFxGraphError,
    _lower_mlx_node,
    _resolve_fx_value,
    build_mlx_fx_plan,
)


def _make_mlx_export_executor(
    exported_program: Any,
    example_inputs: tuple[Any, ...],
    *,
    mode: str,
) -> Callable[..., torch.Tensor]:
    """Lower one admitted strict serving export to MLX plus deferred KV commit."""

    import mlx.core as mx

    from sglang.kernels.ops.attention.mlx_kv_commit import (
        commit_deferred_kv,
        verify_deferred_kv_commit,
    )
    from sglang.kernels.ops.attention.mlx_radix_attention import (
        DeferredAttentionSpec,
        causal_gqa,
        radix_decode_deferred,
        radix_prefill_deferred,
    )
    from sglang.srt.utils.tensor_bridge import MlxTensorView, mlx_call_multi

    graph_module = exported_program.module()
    for node in tuple(graph_module.graph.nodes):
        if node.op == "call_module" and str(node.target) == "_guards_fn":
            if node.users:
                raise UnsupportedMlxFxGraphError("export guard unexpectedly has users")
            graph_module.graph.erase_node(node)
    graph_module.recompile()

    registry = MlxFxLoweringRegistry.standard_export_decoder()
    attention_target = torch.ops.sglang.unified_attention_with_output.default
    registry.register_function(attention_target, "radix_decode")
    plan = build_mlx_fx_plan(graph_module, registry)
    plan.require_fully_supported()

    attr_nodes = tuple(
        node for node in graph_module.graph.nodes if node.op == "get_attr"
    )
    attr_tensors = tuple(_get_graph_tensor(graph_module, node) for node in attr_nodes)
    attr_views = tuple(MlxTensorView(tensor) for tensor in attr_tensors)
    placeholder_nodes = tuple(
        node for node in graph_module.graph.nodes if node.op == "placeholder"
    )
    if len(placeholder_nodes) != len(example_inputs):
        raise UnsupportedMlxFxGraphError(
            "decode export placeholder and example-input counts do not match"
        )
    if mode not in {"decode", "prefill"}:
        raise ValueError(f"unsupported MLX serving export mode: {mode}")
    tensor_positions = tuple(
        index
        for index, value in enumerate(example_inputs)
        if isinstance(value, torch.Tensor)
    )
    tensor_placeholders = tuple(placeholder_nodes[index] for index in tensor_positions)
    used_non_tensors = tuple(
        placeholder_nodes[index]
        for index, value in enumerate(example_inputs)
        if not isinstance(value, torch.Tensor) and placeholder_nodes[index].users
    )
    if used_non_tensors:
        raise UnsupportedMlxFxGraphError(
            "decode export contains used non-tensor runtime inputs"
        )

    attention_nodes = tuple(
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function" and node.target is attention_target
    )
    k_pools = tuple(
        _get_graph_tensor(graph_module, node.kwargs["k_cache"])
        for node in attention_nodes
    )
    v_pools = tuple(
        _get_graph_tensor(graph_module, node.kwargs["v_cache"])
        for node in attention_nodes
    )
    if k_pools:
        # Loud launch-time failure beats silently serving a pool whose
        # commits vanish (compile_shader write loss near the working-set
        # limit); probes the real buffers through the same kernel.
        verify_deferred_kv_commit(
            list(k_pools),
            list(v_pools),
            num_kv_heads=int(k_pools[0].shape[1]),
            head_dim=int(k_pools[0].shape[2]),
        )
    from sglang.srt.hardware_backend.mlx.export_validation import ServingForwardArg

    out_cache_position = ServingForwardArg.OUT_CACHE_LOC
    prefix_lens_position = ServingForwardArg.EXTEND_PREFIX_LENS
    debug_attention = envs.SGLANG_DEBUG_MLX_EXPORT_ATTENTION.get()

    def make_mlx_graph(prefill_attention: str):
        def mlx_graph(*arrays):
            return _run_mlx_graph(prefill_attention, *arrays)

        return mlx_graph

    def _run_mlx_graph(prefill_attention, *arrays):
        runtime_arrays = arrays[: len(tensor_placeholders)]
        captured_arrays = arrays[len(tensor_placeholders) :]
        values: dict[torch.fx.Node, Any] = dict(
            zip(tensor_placeholders, runtime_arrays)
        )
        values.update(zip(attr_nodes, captured_arrays))
        new_keys = []
        new_values = []
        attention_outputs = []
        first_query = None
        first_key = None
        first_value = None
        for node, node_plan in zip(graph_module.graph.nodes, plan.nodes):
            if node.op in {"placeholder", "get_attr"}:
                continue
            args = _resolve_fx_value(node.args, values)
            kwargs = _resolve_fx_value(node.kwargs, values)
            if node.op == "output":
                result = args[0]
                outputs = (
                    tuple(result) if isinstance(result, (tuple, list)) else (result,)
                )
                if debug_attention:
                    return (
                        *outputs,
                        mx.stack(attention_outputs),
                        first_query,
                        first_key,
                        first_value,
                        mx.stack(new_keys),
                        mx.stack(new_values),
                    )
                return (*outputs, mx.stack(new_keys), mx.stack(new_values))
            if node.target is attention_target:
                query, key, value, output = args[:4]
                k_cache = kwargs["k_cache"]
                num_kv_heads = k_cache.shape[1]
                head_dim = k_cache.shape[2]
                spec = DeferredAttentionSpec(
                    num_q_heads=query.shape[-1] // head_dim,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                )
                query = query.reshape(query.shape[0], spec.num_q_heads, head_dim)
                key = key.reshape(key.shape[0], num_kv_heads, head_dim)
                value = value.reshape(value.shape[0], num_kv_heads, head_dim)
                if mode == "decode":
                    query = mx.contiguous(query)
                    key = mx.contiguous(key)
                    value = mx.contiguous(value)
                    attention = radix_decode_deferred(
                        query,
                        key,
                        value,
                        k_cache,
                        kwargs["v_cache"],
                        kwargs["req_to_token"],
                        kwargs["req_pool_indices"],
                        kwargs["seq_lens"],
                        spec=spec,
                    )
                elif prefill_attention == "causal":
                    # Single request, no cached prefix: plain causal SDPA is
                    # exactly the same math and ~15x faster than the radix
                    # kernel at prefill shapes. Selection happens per call in
                    # execute(); this graph is only run when it applies.
                    query = mx.contiguous(query)
                    key = mx.contiguous(key)
                    value = mx.contiguous(value)
                    attention = causal_gqa(query, key, value, spec=spec)
                else:
                    query = mx.contiguous(query)
                    key = mx.contiguous(key)
                    value = mx.contiguous(value)
                    attention = radix_prefill_deferred(
                        query,
                        key,
                        value,
                        k_cache,
                        kwargs["v_cache"],
                        kwargs["req_to_token"],
                        kwargs["req_pool_indices"],
                        kwargs["extend_prefix_lens"],
                        kwargs["extend_seq_lens"],
                        spec=spec,
                    )
                output[:] = attention.reshape(output.shape)
                attention_outputs.append(attention)
                if first_query is None:
                    first_query = query
                    first_key = key
                if first_value is None:
                    first_value = value
                new_keys.append(key)
                new_values.append(value)
                values[node] = None
                continue
            if node_plan.lowering is None:
                raise UnsupportedMlxFxGraphError(
                    f"missing lowering for admitted node {node.target}"
                )
            values[node] = _lower_mlx_node(node_plan.lowering, args, kwargs)
        raise UnsupportedMlxFxGraphError("decode export has no output node")

    compiled_graph = mx.compile(make_mlx_graph("radix"), shapeless=False)
    # Packed multi-request batches and radix-prefix reuse need the custom
    # kernel; the plain-causal fast path is only valid for one request with
    # no cached prefix, so it exists only for batch-size-1 exports.
    causal_prefill_graph = None
    if (
        mode == "prefill"
        and example_inputs[ServingForwardArg.REQ_POOL_INDICES].shape[0] == 1
    ):
        causal_prefill_graph = mx.compile(make_mlx_graph("causal"), shapeless=False)

    def execute(*runtime_inputs):
        tensor_inputs = tuple(runtime_inputs[index] for index in tensor_positions)
        graph = compiled_graph
        if (
            causal_prefill_graph is not None
            and int(runtime_inputs[prefix_lens_position].max()) == 0
        ):
            graph = causal_prefill_graph
        results = mlx_call_multi(
            graph,
            *tensor_inputs,
            *attr_views,
            device="mps",
        )
        expected_results = 7 if debug_attention else 3
        if len(results) != expected_results:
            raise RuntimeError(
                "MLX graph expected logits, optional debug attention, and two KV "
                f"deltas; got {len(results)}"
            )
        if debug_attention:
            logits, all_attention, first_query, first_key, first_value, new_k, new_v = (
                results
            )
        else:
            logits, new_k, new_v = results
        if envs.SGLANG_DEBUG_MLX_KV_DELTAS.get():
            # new_k/new_v are zero-copy views of MLX-owned buffers; MLX may
            # reuse those buffers on the next region run, so a stash held
            # across runs must own its storage.
            execute.last_new_k = new_k.clone()
            execute.last_new_v = new_v.clone()
        first_k_pool = k_pools[0]
        spec = DeferredAttentionSpec(
            num_q_heads=(
                attention_nodes[0].args[0].meta["val"].shape[-1]
                // first_k_pool.shape[2]
            ),
            num_kv_heads=first_k_pool.shape[1],
            head_dim=first_k_pool.shape[2],
        )
        commit_deferred_kv(
            new_k,
            new_v,
            runtime_inputs[out_cache_position],
            k_pools,
            v_pools,
            num_kv_heads=spec.num_kv_heads,
            head_dim=spec.head_dim,
        )
        return (
            (logits, all_attention, first_query, first_key, first_value)
            if debug_attention
            else logits
        )

    return execute


def make_mlx_decode_export_executor(
    exported_program: Any,
    example_inputs: tuple[Any, ...],
) -> Callable[..., torch.Tensor]:
    """Build the single-region MLX executor for one decode export bucket."""
    return _make_mlx_export_executor(exported_program, example_inputs, mode="decode")


def make_mlx_prefill_export_executor(
    exported_program: Any,
    example_inputs: tuple[Any, ...],
) -> Callable[..., torch.Tensor]:
    """Build the single-region MLX executor for one packed Radix prefill bucket."""
    return _make_mlx_export_executor(exported_program, example_inputs, mode="prefill")


def _get_graph_tensor(
    graph_module: torch.fx.GraphModule, node: torch.fx.Node
) -> torch.Tensor:
    if node.op != "get_attr":
        raise UnsupportedMlxFxGraphError(
            f"expected a bound graph tensor, found {node.op}:{node.target}"
        )
    target = str(node.target)
    try:
        return graph_module.get_parameter(target)
    except AttributeError:
        pass
    try:
        return graph_module.get_buffer(target)
    except AttributeError:
        pass
    # Non-strict export stores lifted closure constants (`lifted_tensor_*`)
    # as plain module attributes, reachable only by attribute traversal.
    value = graph_module
    for part in target.split("."):
        value = getattr(value, part)
    if not isinstance(value, torch.Tensor):
        raise UnsupportedMlxFxGraphError(
            f"graph attribute {target} is not a tensor: {type(value)}"
        )
    return value


def run_torch_decode_export_reference(
    exported_program: Any,
    runtime_inputs: tuple[Any, ...],
    *,
    return_attention: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Execute the same decode export with a small Torch attention reference."""
    graph_module = exported_program.module()
    attention_target = torch.ops.sglang.unified_attention_with_output.default
    for node in graph_module.graph.nodes:
        if node.op == "call_function" and node.target is attention_target:
            node.target = _torch_radix_decode_reference
    graph_module.recompile()
    _TORCH_DECODE_DEBUG_ATTENTION.clear()
    logits = graph_module(*runtime_inputs)
    if return_attention:
        return logits, torch.stack(_TORCH_DECODE_DEBUG_ATTENTION)
    return logits


_TORCH_DECODE_DEBUG_ATTENTION: list[torch.Tensor] = []
_TORCH_PREFILL_DEBUG_ATTENTION: list[torch.Tensor] = []
_TORCH_PREFILL_DEBUG_QKV: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []


def run_torch_prefill_export_reference(
    exported_program: Any,
    runtime_inputs: tuple[Any, ...],
    *,
    return_first_attention: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Execute a prefill export with the deferred Radix attention contract."""
    graph_module = exported_program.module()
    attention_target = torch.ops.sglang.unified_attention_with_output.default
    for node in graph_module.graph.nodes:
        if node.op == "call_function" and node.target is attention_target:
            node.target = _torch_radix_prefill_reference
    graph_module.recompile()
    _TORCH_PREFILL_DEBUG_ATTENTION.clear()
    _TORCH_PREFILL_DEBUG_QKV.clear()
    logits = graph_module(*runtime_inputs)
    if return_first_attention:
        return (
            logits,
            torch.stack(_TORCH_PREFILL_DEBUG_ATTENTION),
            _TORCH_PREFILL_DEBUG_QKV[0],
        )
    return logits


def _torch_radix_prefill_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    save_kv_cache: bool,
    layer_id: int,
    *,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    out_cache_loc: torch.Tensor,
    extend_prefix_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    **_: Any,
) -> None:
    del layer_id
    num_kv_heads, head_dim = k_cache.shape[1:]
    num_q_heads = query.shape[-1] // head_dim
    q = query.reshape(query.shape[0], num_q_heads, head_dim)
    current_k = key.reshape(key.shape[0], num_kv_heads, head_dim)
    current_v = value.reshape(value.shape[0], num_kv_heads, head_dim)
    _TORCH_PREFILL_DEBUG_QKV.append(
        (q.detach().clone(), current_k.detach().clone(), current_v.detach().clone())
    )
    q_per_kv = num_q_heads // num_kv_heads
    rows = []
    request_start = 0
    for batch in range(req_pool_indices.shape[0]):
        request = int(req_pool_indices[batch].item())
        prefix_length = int(extend_prefix_lens[batch].item())
        extension_length = int(extend_seq_lens[batch].item())
        prefix_slots = req_to_token[request, :prefix_length].long()
        cached_k = k_cache[prefix_slots]
        cached_v = v_cache[prefix_slots]
        for offset in range(extension_length):
            output_index = request_start + offset
            keys = torch.cat(
                (cached_k, current_k[request_start : output_index + 1])
            ).repeat_interleave(q_per_kv, dim=1)
            values = torch.cat(
                (cached_v, current_v[request_start : output_index + 1])
            ).repeat_interleave(q_per_kv, dim=1)
            scores = torch.einsum("hd,thd->ht", q[output_index].float(), keys.float())
            probabilities = torch.softmax(scores * (head_dim**-0.5), dim=-1)
            rows.append(
                torch.einsum("ht,thd->hd", probabilities, values.float()).to(q.dtype)
            )
        request_start += extension_length
    output.copy_(torch.stack(rows).reshape_as(output))
    _TORCH_PREFILL_DEBUG_ATTENTION.append(output.detach().clone())
    if save_kv_cache:
        k_cache.index_copy_(0, out_cache_loc, current_k)
        v_cache.index_copy_(0, out_cache_loc, current_v)


def _torch_radix_decode_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    save_kv_cache: bool,
    layer_id: int,
    *,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    out_cache_loc: torch.Tensor,
    **_: Any,
) -> None:
    del layer_id
    num_kv_heads, head_dim = k_cache.shape[1:]
    num_q_heads = query.shape[-1] // head_dim
    q = query.reshape(query.shape[0], num_q_heads, head_dim)
    current_k = key.reshape(key.shape[0], num_kv_heads, head_dim)
    current_v = value.reshape(value.shape[0], num_kv_heads, head_dim)
    rows = []
    q_per_kv = num_q_heads // num_kv_heads
    for batch in range(q.shape[0]):
        sequence_length = int(seq_lens[batch].item())
        request = int(req_pool_indices[batch].item())
        prefix_slots = req_to_token[request, : sequence_length - 1].long()
        keys = torch.cat((k_cache[prefix_slots], current_k[batch : batch + 1]))
        values = torch.cat((v_cache[prefix_slots], current_v[batch : batch + 1]))
        keys = keys.repeat_interleave(q_per_kv, dim=1)
        values = values.repeat_interleave(q_per_kv, dim=1)
        scores = torch.einsum("hd,thd->ht", q[batch].float(), keys.float())
        probabilities = torch.softmax(scores * (head_dim**-0.5), dim=-1)
        rows.append(
            torch.einsum("ht,thd->hd", probabilities, values.float()).to(q.dtype)
        )
    output.copy_(torch.stack(rows).reshape_as(output))
    _TORCH_DECODE_DEBUG_ATTENTION.append(output.detach().clone())
    if save_kv_cache:
        k_cache.index_copy_(0, out_cache_loc, current_k)
        v_cache.index_copy_(0, out_cache_loc, current_v)
