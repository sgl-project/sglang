"""Validation harness for exporting a real SGLang serving forward.

This is intentionally not a runtime integration. It proves whether the model
body, shared attention boundary, and Torch-owned serving buffers can form one
ExportedProgram without a model-architecture adapter.
"""

from __future__ import annotations

import inspect
import json
from collections import Counter
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class DecoderTopology(msgspec.Struct, frozen=True):
    """Where a CausalLM keeps its decoder body and transformer layer stack.

    ``body_name`` is the direct child attribute the top-level model forwards
    hidden states through (``model`` for Llama-family, ``transformer`` for
    GPT-2, the OPT wrapper around its ``decoder``); ``layers`` is the stack of
    transformer blocks wherever it nests below that child.
    """

    body_name: str
    layers: Any


def resolve_decoder_topology(model: torch.nn.Module) -> DecoderTopology:
    """Structurally locate the decoder layer stack of any SGLang model class.

    Model classes disagree on attribute names (``model.layers``,
    ``model.decoder.layers``, ``transformer.h``), so discovery keys on the one
    invariant they share: the transformer blocks live in a single
    ``nn.ModuleList``/``nn.ModuleDict`` whose blocks contain the shared
    :class:`RadixAttention` boundary. Raises ``RuntimeError`` naming the model
    class when zero or several such stacks exist; callers turn that into a
    pre-launch rejection to the eager Torch path.
    """
    from torch import nn

    from sglang.srt.layers.radix_attention import RadixAttention

    candidates = []
    for name, module in model.named_modules():
        if not isinstance(module, (nn.ModuleList, nn.ModuleDict)):
            continue
        blocks = tuple(module.values() if isinstance(module, nn.ModuleDict) else module)
        if any(
            isinstance(submodule, RadixAttention)
            for block in blocks
            if isinstance(block, nn.Module)
            for submodule in block.modules()
        ):
            candidates.append((name, module))
    if len(candidates) != 1:
        found = ", ".join(name for name, _ in candidates) or "none"
        raise RuntimeError(
            f"MLX region cannot locate the decoder layer stack of "
            f"{type(model).__name__}: expected exactly one "
            f"nn.ModuleList/ModuleDict of transformer blocks containing "
            f"RadixAttention, found {found}. The model serves on the eager "
            f"Torch path."
        )
    layers_name, layers = candidates[0]
    body_name, _, remainder = layers_name.partition(".")
    if not remainder:
        raise RuntimeError(
            f"MLX region requires the layer stack of {type(model).__name__} "
            f"to live under a direct child module callable as the decoder "
            f"body, but '{layers_name}' hangs off the top-level model itself."
        )
    return DecoderTopology(body_name=body_name, layers=layers)


def resolve_body_call_kwargs(
    model: torch.nn.Module, body: torch.nn.Module
) -> dict[str, Any]:
    """Map the serving-forward roles onto one decoder body's parameter names.

    Body signatures only agree on names, not position (Phi declares
    ``(input_ids, forward_batch, positions)``, GPT-2 calls its positions
    ``position_ids``), so the export wrapper must call by keyword. Values are
    the role each parameter receives: ``"input_ids"``/``"positions"``/
    ``"forward_batch"``, or ``None`` for a PP proxy that single-rank serving
    always passes as None. Raises ``RuntimeError`` naming the model class
    when a role cannot be placed.
    """
    parameters = inspect.signature(body.forward).parameters
    positions_names = [
        name for name in ("positions", "position_ids") if name in parameters
    ]
    if (
        "input_ids" not in parameters
        or "forward_batch" not in parameters
        or len(positions_names) != 1
    ):
        raise RuntimeError(
            f"MLX region cannot bind the decoder body of "
            f"{type(model).__name__}: expected parameters 'input_ids', "
            f"'forward_batch', and one of 'positions'/'position_ids', found "
            f"({', '.join(parameters)}). The model serves on the eager "
            f"Torch path."
        )
    call_kwargs: dict[str, Any] = {
        "input_ids": "input_ids",
        positions_names[0]: "positions",
        "forward_batch": "forward_batch",
    }
    if "pp_proxy_tensors" in parameters:
        # PP-aware bodies (e.g. OPT) declare it without a default;
        # single-rank serving always passes None.
        call_kwargs["pp_proxy_tensors"] = None
    return call_kwargs


@dataclass(frozen=True)
class ServingExportResult:
    exported_program: Any
    report: dict[str, Any]


@dataclass(frozen=True)
class ServingMlxExecutor:
    exported_program: Any
    wrapper: torch.nn.Module
    args: tuple[Any, ...]
    execution_mode: str
    execute: Any


class ServingForwardExportWrapper(torch.nn.Module):
    """Expose changing serving metadata as a small tensor-only signature."""

    def __init__(
        self,
        model: torch.nn.Module,
        model_runner: Any,
        forward_batch: ForwardBatch,
    ) -> None:
        super().__init__()
        self.model = model
        self.forward_mode = forward_batch.forward_mode
        self.seq_lens_sum = forward_batch.seq_lens_sum
        self.extend_num_tokens = forward_batch.extend_num_tokens
        self.num_token_non_padded_cpu = forward_batch.num_token_non_padded_cpu
        # The decoder body is resolved structurally (its attribute name varies
        # by model class) and re-read through ``self.model`` at forward time so
        # the shared submodule is not registered twice on this wrapper.
        self._body_name = resolve_decoder_topology(model).body_name
        self._body_call_kwargs = resolve_body_call_kwargs(
            model, getattr(model, self._body_name)
        )
        # ``LogitsProcessor`` narrows the lm_head's padded width back to the
        # real vocabulary before sampling (``_copy_logits_to_buffer``). The
        # region bypasses the processor entirely, so it has to apply the same
        # narrowing: SGLang rounds the lm_head up to a multiple of 64 with
        # zero rows, whose logit is exactly 0.0, and any row whose real logits
        # are all negative would otherwise argmax onto a pad column and emit
        # an out-of-vocabulary token id. Attribute access is deliberate --
        # a model without a LogitsProcessor is already rejected before export
        # by ``_nontrivial_logits_reason``.
        self.vocab_size = model.logits_processor.vocab_size
        backend = model_runner.attn_backend
        self.register_buffer(
            "req_to_token",
            backend.req_to_token_pool.req_to_token,
            persistent=False,
        )
        for layer_id in range(len(model_runner.attention_layers)):
            k_cache, v_cache = backend.token_to_kv_pool.get_kv_buffer(layer_id)
            self.register_buffer(f"k_cache_{layer_id}", k_cache, persistent=False)
            self.register_buffer(f"v_cache_{layer_id}", v_cache, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        extend_seq_lens: Optional[torch.Tensor],
        extend_prefix_lens: Optional[torch.Tensor],
        extend_start_loc: Optional[torch.Tensor],
        num_token_non_padded: Optional[torch.Tensor],
    ) -> torch.Tensor:
        forward_batch = ForwardBatch(
            forward_mode=self.forward_mode,
            batch_size=req_pool_indices.shape[0],
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=self.seq_lens_sum,
            positions=positions,
            extend_num_tokens=self.extend_num_tokens,
            extend_seq_lens=extend_seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            extend_start_loc=extend_start_loc,
            num_token_non_padded=num_token_non_padded,
            num_token_non_padded_cpu=self.num_token_non_padded_cpu,
        )
        roles = {
            "input_ids": input_ids,
            "positions": positions,
            "forward_batch": forward_batch,
        }
        hidden_states = getattr(self.model, self._body_name)(
            **{
                name: None if role is None else roles[role]
                for name, role in self._body_call_kwargs.items()
            }
        )
        logits = torch.matmul(
            hidden_states.to(self.model.lm_head.weight.dtype),
            self.model.lm_head.weight.T,
        )
        # Mirrors LogitsProcessor's padded-vocab truncation; see __init__.
        # Static shapes make this a trace-time branch, so an unpadded model
        # exports the same graph it did before.
        if logits.shape[-1] > self.vocab_size:
            logits = logits[:, : self.vocab_size]
        return logits


class ServingForwardArg(IntEnum):
    """Positions in the flat serving-forward signature.

    ``serving_forward_args`` builds its tuple in this member order, so the
    enum is the single source of truth for the signature layout. Consumers
    that index runtime inputs (the MLX executor's ``out_cache_loc`` and
    ``extend_prefix_lens`` reads) must use these names, never literals.
    """

    INPUT_IDS = 0
    POSITIONS = 1
    REQ_POOL_INDICES = 2
    SEQ_LENS = 3
    OUT_CACHE_LOC = 4
    EXTEND_SEQ_LENS = 5
    EXTEND_PREFIX_LENS = 6
    EXTEND_START_LOC = 7
    NUM_TOKEN_NON_PADDED = 8


def serving_forward_args(forward_batch: ForwardBatch) -> tuple[Any, ...]:
    """The flat tensor signature shared by export, validation, and serving.

    Order is load-bearing: the exported program's placeholders and the MLX
    executor's runtime-input reads are derived from it, so every caller must
    build the tuple through this function, whose layout follows
    :class:`ServingForwardArg`.
    """
    values = {
        ServingForwardArg.INPUT_IDS: forward_batch.input_ids,
        ServingForwardArg.POSITIONS: forward_batch.positions,
        ServingForwardArg.REQ_POOL_INDICES: forward_batch.req_pool_indices,
        ServingForwardArg.SEQ_LENS: forward_batch.seq_lens,
        ServingForwardArg.OUT_CACHE_LOC: forward_batch.out_cache_loc,
        ServingForwardArg.EXTEND_SEQ_LENS: forward_batch.extend_seq_lens,
        ServingForwardArg.EXTEND_PREFIX_LENS: forward_batch.extend_prefix_lens,
        ServingForwardArg.EXTEND_START_LOC: forward_batch.extend_start_loc,
        ServingForwardArg.NUM_TOKEN_NON_PADDED: forward_batch.num_token_non_padded,
    }
    ordered: list[Any] = [None] * len(ServingForwardArg)
    for arg in ServingForwardArg:
        # Index by member value, not declaration order, so reordering the
        # enum declarations cannot silently permute the signature.
        ordered[arg] = values[arg]
    return tuple(ordered)


def ensure_model_layers(model_runner: Any) -> None:
    """Populate the layer maps the export wrapper reads, once per runner."""
    from sglang.srt.model_executor.model_runner_components.layer_setup import (
        compute_attention_and_moe_layers,
    )

    if hasattr(model_runner, "attention_layers"):
        return
    # ``compute_attention_and_moe_layers`` reads its argument's ``.layers``;
    # hand it the structurally resolved stack instead of assuming where each
    # model class hangs it.
    topology = resolve_decoder_topology(model_runner.model)
    (
        model_runner.attention_layers,
        model_runner.moe_layers,
        model_runner.moe_fusions,
        model_runner.dsa_indexers,
        model_runner.mha_companion_layers,
    ) = compute_attention_and_moe_layers(SimpleNamespace(layers=topology.layers))


def serving_export_context(model_runner: Any, forward_batch: ForwardBatch) -> Any:
    """The forward context torch.export needs to trace the serving forward."""
    from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
        set_tc_piecewise_forward_context,
    )

    ensure_model_layers(model_runner)
    return set_tc_piecewise_forward_context(
        forward_batch,
        model_runner.attention_layers,
        getattr(model_runner.model, "quant_config", None),
        model_runner.moe_layers,
        model_runner.moe_fusions,
        dsa_indexers=model_runner.dsa_indexers,
        mha_companion_layers=model_runner.mha_companion_layers,
        full_graph=True,
    )


def build_serving_forward_wrapper(
    model_runner: Any,
    forward_batch: ForwardBatch,
) -> tuple[ServingForwardExportWrapper, tuple[Any, ...]]:
    wrapper = ServingForwardExportWrapper(
        model_runner.model, model_runner, forward_batch
    ).eval()
    return wrapper, serving_forward_args(forward_batch)


def serving_graph_signature(
    model_runner: Any, forward_batch: ForwardBatch
) -> tuple[str, ...]:
    """The exported graph's op sequence, for constant-independence checks.

    The wrapper bakes Python-scalar batch fields (``seq_lens_sum``,
    ``num_token_non_padded_cpu``) as constants. Reusing one executor across
    steps is only sound if the traced graph does not depend on them; callers
    compare this signature across exports with perturbed constants.
    """
    from sglang.srt.compilation.torch_compile_decoration import _to_torch

    _to_torch(
        model_runner.model,
        reverse=False,
        num_tokens=forward_batch.input_ids.shape[0],
    )
    wrapper, args = build_serving_forward_wrapper(model_runner, forward_batch)
    exported = torch.export.export(wrapper, args, strict=False)
    return tuple(
        f"{node.op}:{node.target}" for node in exported.graph_module.graph.nodes
    )


def build_serving_mlx_executor(
    model_runner: Any,
    forward_batch: ForwardBatch,
) -> ServingMlxExecutor:
    """Build the MLX executor for one already-prepared serving bucket."""
    from sglang.srt.compilation.torch_compile_decoration import _to_torch
    from sglang.srt.hardware_backend.mlx.fx_backend import (
        make_mlx_decode_export_executor,
        make_mlx_prefill_export_executor,
    )

    def build_single(batch: ForwardBatch) -> ServingMlxExecutor:
        _to_torch(
            model_runner.model,
            reverse=False,
            num_tokens=batch.input_ids.shape[0],
        )
        wrapper, args = build_serving_forward_wrapper(model_runner, batch)
        exported = torch.export.export(wrapper, args, strict=False)
        if batch.forward_mode.is_decode():
            execution_mode = "decode"
            executor = make_mlx_decode_export_executor(exported, args)
        elif batch.forward_mode.is_extend():
            execution_mode = "prefill"
            executor = make_mlx_prefill_export_executor(exported, args)
        else:
            raise RuntimeError(
                f"unsupported forward mode for MLX export: {batch.forward_mode}"
            )
        return ServingMlxExecutor(
            exported_program=exported,
            wrapper=wrapper,
            args=args,
            execution_mode=execution_mode,
            execute=executor,
        )

    return build_single(forward_batch)


def export_serving_forward(
    model_runner: Any,
    forward_batch: ForwardBatch,
    report_path: str,
) -> ServingExportResult:
    """Export one real serving bucket and describe its state contract."""
    from sglang.srt.compilation.torch_compile_decoration import _to_torch
    from sglang.srt.hardware_backend.mlx.fx_backend import (
        MlxFxLoweringRegistry,
        build_mlx_fx_plan,
    )

    _to_torch(
        model_runner.model,
        reverse=False,
        num_tokens=forward_batch.input_ids.shape[0],
    )
    wrapper, args = build_serving_forward_wrapper(model_runner, forward_batch)
    exported = torch.export.export(wrapper, args, strict=False)
    graph = exported.graph_module.graph
    attention_nodes = [
        node
        for node in graph.nodes
        if "unified_attention_with_output" in str(node.target)
    ]
    mutation_outputs = [
        str(spec.target)
        for spec in exported.graph_signature.output_specs
        if "MUTATION" in str(spec.kind)
    ]
    node_targets = Counter(
        str(node.target) for node in graph.nodes if node.op == "call_function"
    )
    attention_schema = str(
        torch.ops.sglang.unified_attention_with_output.default._schema
    )
    lowering_registry = MlxFxLoweringRegistry.standard_export_decoder()
    lowering_plan = build_mlx_fx_plan(exported.graph_module, lowering_registry)
    unlifted = exported.module()
    unlifted_node_kinds = Counter(node.op for node in unlifted.graph.nodes)
    unsupported_targets = Counter(
        str(node.target) for node in lowering_plan.unsupported
    )
    report = {
        "forward_mode": str(forward_batch.forward_mode),
        "graph_nodes": len(tuple(graph.nodes)),
        "attention_nodes": len(attention_nodes),
        "attention_schema": attention_schema,
        "kv_cache_is_directly_mutable": all(
            marker in attention_schema
            for marker in ("Tensor(a15!)? k_cache", "Tensor(a16!)? v_cache")
        ),
        "mutation_outputs": mutation_outputs,
        "node_targets": dict(sorted(node_targets.items())),
        "unsupported_node_targets": dict(sorted(unsupported_targets.items())),
        "unlifted_node_kinds": dict(sorted(unlifted_node_kinds.items())),
        "unlifted_call_modules": [
            str(node.target)
            for node in unlifted.graph.nodes
            if node.op == "call_module"
        ],
        "input_specs": [
            {"kind": str(spec.kind), "target": str(spec.target)}
            for spec in exported.graph_signature.input_specs
        ],
        "output_specs": [
            {"kind": str(spec.kind), "target": str(spec.target)}
            for spec in exported.graph_signature.output_specs
        ],
    }
    if envs.SGLANG_DEBUG_MLX_EXPORT_EXECUTE.get():
        from sglang.srt.hardware_backend.mlx.fx_backend import (
            make_mlx_decode_export_executor,
            make_mlx_prefill_export_executor,
            run_torch_decode_export_reference,
            run_torch_prefill_export_reference,
        )

        if forward_batch.forward_mode.is_decode():
            mlx_executor = make_mlx_decode_export_executor(exported, args)
            execution_mode = "decode"
        elif forward_batch.forward_mode.is_extend():
            mlx_executor = make_mlx_prefill_export_executor(exported, args)
            execution_mode = "prefill"
        else:
            mlx_executor = None
            execution_mode = "not_admitted"
        if mlx_executor is None:
            report["execution"] = {"mode": execution_mode}
            Path(report_path).write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n"
            )
            return ServingExportResult(exported_program=exported, report=report)
        report["execution_inputs"] = {
            "req_pool_indices": forward_batch.req_pool_indices.cpu().tolist(),
            "seq_lens": forward_batch.seq_lens.cpu().tolist(),
            "out_cache_loc": forward_batch.out_cache_loc.cpu().tolist(),
            "extend_prefix_lens": (
                None
                if forward_batch.extend_prefix_lens is None
                else forward_batch.extend_prefix_lens.cpu().tolist()
            ),
            "extend_seq_lens": (
                None
                if forward_batch.extend_seq_lens is None
                else forward_batch.extend_seq_lens.cpu().tolist()
            ),
            "extend_start_loc": (
                None
                if forward_batch.extend_start_loc is None
                else forward_batch.extend_start_loc.cpu().tolist()
            ),
        }
        debug_attention = envs.SGLANG_DEBUG_MLX_EXPORT_ATTENTION.get()
        if execution_mode == "decode":
            torch_reference = run_torch_decode_export_reference(
                exported, args, return_attention=debug_attention
            )
            if debug_attention:
                torch_logits, torch_attention = torch_reference
            else:
                torch_logits = torch_reference
                torch_attention = None
        else:
            torch_reference = run_torch_prefill_export_reference(
                exported, args, return_first_attention=debug_attention
            )
            if debug_attention:
                torch_logits, torch_attention, torch_first_qkv = torch_reference
                torch_first_query, torch_first_key, torch_first_value = torch_first_qkv
            else:
                torch_logits = torch_reference
                torch_attention = None
                torch_first_query = None
                torch_first_key = None
                torch_first_value = None
        mlx_result = mlx_executor(*args)
        if debug_attention:
            (
                mlx_logits,
                mlx_attention,
                mlx_first_query,
                mlx_first_key,
                mlx_first_value,
            ) = mlx_result
        else:
            mlx_logits = mlx_result
            mlx_attention = None
            mlx_first_query = None
            mlx_first_key = None
            mlx_first_value = None
        torch.mps.synchronize()
        difference = (mlx_logits.float() - torch_logits.float()).abs()
        report["execution"] = {
            "mode": execution_mode,
            "max_abs_error": float(difference.max().cpu()),
            "mean_abs_error": float(difference.mean().cpu()),
            "allclose": bool(
                torch.allclose(
                    mlx_logits,
                    torch_logits,
                    atol=0.08,
                    rtol=0.03,
                )
            ),
        }
        if torch_attention is not None and mlx_attention is not None:
            attention_difference = (
                mlx_attention.reshape_as(torch_attention).float()
                - torch_attention.float()
            ).abs()
            per_layer_max = attention_difference.flatten(1).amax(dim=1)
            report["execution"]["first_attention_max_abs_error"] = float(
                per_layer_max[0].cpu()
            )
            report["execution"]["first_attention_mean_abs_error"] = float(
                attention_difference[0].mean().cpu()
            )
            report["execution"]["attention_max_abs_error_by_layer"] = [
                float(value.cpu()) for value in per_layer_max
            ]
            per_layer_row_max = attention_difference.flatten(2).amax(dim=2)
            report["execution"]["attention_max_abs_error_by_layer_and_row"] = [
                [float(value.cpu()) for value in row] for row in per_layer_row_max
            ]
            if execution_mode == "prefill" and bool(
                torch.all(forward_batch.extend_prefix_lens == 0).cpu()
            ):
                q_heads = mlx_attention.shape[2]
                kv_heads = mlx_first_value.shape[1]
                first_token_expected = mlx_first_value[0].repeat_interleave(
                    q_heads // kv_heads, dim=0
                )
                first_token_difference = (
                    mlx_attention[0, 0].float() - first_token_expected.float()
                ).abs()
                report["execution"]["mlx_first_token_causal_max_abs_error"] = float(
                    first_token_difference.max().cpu()
                )
            if torch_first_value is not None and mlx_first_value is not None:
                for name, mlx_value, torch_value in (
                    ("query", mlx_first_query, torch_first_query),
                    ("key", mlx_first_key, torch_first_key),
                    ("value", mlx_first_value, torch_first_value),
                ):
                    difference = (mlx_value.float() - torch_value.float()).abs()
                    report["execution"][f"first_{name}_max_abs_error"] = float(
                        difference.max().cpu()
                    )
                    report["execution"][f"first_{name}_mean_abs_error"] = float(
                        difference.mean().cpu()
                    )
    Path(report_path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if (
        envs.SGLANG_DEBUG_MLX_EXPORT_SAVE.get()
        and forward_batch.forward_mode.is_decode()
    ):
        torch.export.save(exported, report_path + ".pt2")
    return ServingExportResult(exported_program=exported, report=report)


__all__ = [
    "DecoderTopology",
    "ServingExportResult",
    "ServingForwardExportWrapper",
    "export_serving_forward",
    "resolve_body_call_kwargs",
    "resolve_decoder_topology",
]
