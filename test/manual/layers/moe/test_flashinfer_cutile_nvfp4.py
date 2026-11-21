"""Manual SM120/SM121 checks for the opt-in cuTile NVFP4 adapter.

Run this file in an installed SGLang GPU development environment. Set
SGLANG_TEST_MINIMAX_LAYER30_DIR to a directory containing config.json,
model.safetensors.index.json and the shards holding layer-30 experts 0-31
to enable real-weight checks. SGLANG_TEST_CUTILE_LARGE_BATCH=1 additionally
enables the full 128-expert checkpoint case and workspace measurements; the
checkpoint directory must then contain all layer-30 expert shards.

These are layer correctness tests, not a full-model quality evaluation.
Latency measurements belong in benchmark/kernels/flashinfer_cutile/.
No model server or downloads are started.
"""

import gc
import json
import unittest
import weakref
from pathlib import Path

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.runtime_context import (
    get_context,
    get_exec,
    get_flags,
    get_schedule,
    reset_context,
)
from sglang.test.test_utils import CustomTestCase

ALPHA, BETA, LIMIT = 1.702, 1.0, 7.0
E2M1 = (0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6)


def _dequantize(packed, block_scale, global_scale):
    """Unpack two E2M1 values per byte; each E4M3 scale covers 16 values."""
    lut = torch.tensor(E2M1, device=packed.device)
    values = torch.stack((lut[(packed & 15).long()], lut[(packed >> 4).long()]), -1)
    values = values.reshape(packed.shape[0], -1)
    return values * block_scale.float().repeat_interleave(16, dim=-1) / global_scale


def _quantize_weight(weight):
    from flashinfer import fp4_quantize

    global_scale = (448 * 6 / weight.abs().amax()).float().reshape(1)
    packed, scale = fp4_quantize(weight, global_scale, 16, False, False)
    scale = scale.view(torch.float8_e4m3fn).reshape(weight.shape[0], -1)
    return packed.view(torch.uint8), scale, global_scale.squeeze(0)


def _activation_reference(values):
    """Model cuTile's NVFP4 activation quantization using independent torch ops.

    FlashInfer 0.7.4's cuTile encoder selects the smaller magnitude at exact
    E2M1 midpoints. The generic fp4_quantize API uses round-to-nearest-even,
    so it is not an interchangeable reference: e.g. at unit block scale,
    0.75 becomes 0.5 here but 1.0 with the generic quantizer. Both inputs and
    SwiGLU intermediates must use the cuTile rule to isolate integration error.
    This is a backend-specific reference, not a universal NVFP4 rounding rule.
    """
    groups = values.bfloat16().float().reshape(values.shape[0], -1, 16)
    scale = (groups.abs().amax(dim=-1) / 6).to(torch.float8_e4m3fn).float()
    # Match cuTile's minimum nonzero E4M3 divisor. Keep the original scale
    # for dequantization so a block whose scale rounds to zero stays zero.
    normalized = groups / scale.clamp_min(2**-9).unsqueeze(-1)
    codes = torch.zeros_like(normalized, dtype=torch.long)
    for midpoint in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0):
        codes += normalized.abs() > midpoint
    codes += (normalized < 0).long() * 8
    lut = torch.tensor(E2M1, device=values.device)
    return (lut[codes] * scale.unsqueeze(-1)).reshape_as(values)


def _layer(parameters, *, top_k):
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig

    layer = torch.nn.Module()
    for name, value in parameters.items():
        layer.register_parameter(name, torch.nn.Parameter(value, requires_grad=False))
    experts, rows, packed_hidden = layer.w13_weight_packed.shape
    layer.num_experts = layer.num_local_experts = experts
    layer.hidden_size = packed_hidden * 2
    layer.intermediate_size_per_partition = rows // 2
    layer.params_dtype = torch.bfloat16
    layer.moe_ep_size = layer.moe_tp_size = 1
    layer.moe_ep_rank = layer.moe_tp_rank = 0
    layer.moe_runner_config = MoeRunnerConfig(
        num_experts=experts,
        num_local_experts=experts,
        hidden_size=layer.hidden_size,
        intermediate_size_per_partition=layer.intermediate_size_per_partition,
        top_k=top_k,
        num_fused_shared_experts=0,
        params_dtype=torch.bfloat16,
        gemm1_alpha=ALPHA,
        gemm1_beta=BETA,
        gemm1_clamp_limit=LIMIT,
        gate_up_interleaved=False,
        layer=layer,
    )
    return layer


def _synthetic_layer(
    *,
    experts=4,
    hidden_size=256,
    intermediate_size=256,
    top_k=2,
    distinct_gate_up_scales=False,
):
    w13 = (
        torch.randn(
            experts,
            2 * intermediate_size,
            hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
        )
        / 4
    )
    w2 = (
        torch.randn(
            experts, hidden_size, intermediate_size, device="cuda", dtype=torch.bfloat16
        )
        / 8
    )
    if distinct_gate_up_scales:
        # Quantize the two branches independently. Equal globals can hide an
        # accidental swap of weights without their corresponding global scales.
        gate, up = w13.chunk(2, dim=1)
        up = up * 0.25
        qg, sg, gg = zip(*(_quantize_weight(weight) for weight in gate))
        qu, su, gu = zip(*(_quantize_weight(weight) for weight in up))
        q13 = [torch.cat((g, u)) for g, u in zip(qg, qu)]
        s13 = [torch.cat((g, u)) for g, u in zip(sg, su)]
        globals13 = torch.stack((torch.stack(gg), torch.stack(gu)), dim=1)
    else:
        q13, s13, g13 = zip(*(_quantize_weight(weight) for weight in w13))
        globals13 = torch.stack(g13)[:, None].repeat(1, 2)
    q2, s2, g2 = zip(*(_quantize_weight(weight) for weight in w2))
    return _layer(
        dict(
            w13_weight_packed=torch.stack(q13),
            w13_weight_scale=torch.stack(s13),
            w13_weight_global_scale=globals13,
            w13_input_global_scale=torch.ones(experts, 2, device="cuda"),
            w2_weight_packed=torch.stack(q2),
            w2_weight_scale=torch.stack(s2),
            w2_weight_global_scale=torch.stack(g2),
            w2_input_global_scale=torch.ones(experts, device="cuda"),
        ),
        top_k=top_k,
    )


def _packed_synthetic_layer(seed=0):
    """Large raw-NVFP4 fixture matching the public comparison's stress geometry.

    Generate packed codes, E4M3 scales and unequal branch globals directly;
    this complements the existing quantized-Gaussian fixture rather than
    replacing it. The generator is local so tests do not depend on run order.
    """
    generator = torch.Generator(device="cuda").manual_seed(seed)
    experts, hidden, intermediate = 16, 6144, 3072

    def packed(rows, columns):
        return torch.randint(
            0,
            256,
            (experts, rows, columns // 2),
            dtype=torch.uint8,
            device="cuda",
            generator=generator,
        )

    def scales(rows, columns):
        return (
            torch.rand(experts, rows, columns // 16, device="cuda", generator=generator)
            + 0.5
        ).to(torch.float8_e4m3fn)

    def globals_(shape):
        return torch.rand(shape, device="cuda", generator=generator) * 200 + 300

    return _layer(
        dict(
            w13_weight_packed=packed(2 * intermediate, hidden),
            w13_weight_scale=scales(2 * intermediate, hidden),
            w13_weight_global_scale=globals_((experts, 2)),
            w13_input_global_scale=torch.ones(experts, 2, device="cuda"),
            w2_weight_packed=packed(hidden, intermediate),
            w2_weight_scale=scales(hidden, intermediate),
            w2_weight_global_scale=globals_((experts,)),
            w2_input_global_scale=torch.ones(experts, device="cuda"),
        ),
        top_k=4,
    )


def _peer_dispatch(layer, tokens, *, seed, input_std=0.5, skewed=False):
    """CPU RNG and unnormalized weights; no simulated router is being timed."""
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    cfg = layer.moe_runner_config
    generator = torch.Generator().manual_seed(seed)
    x = (
        torch.randn(tokens, cfg.hidden_size, generator=generator) * input_std
    ).bfloat16()
    ids = torch.stack(
        [
            torch.randperm(cfg.num_experts, generator=generator)[: cfg.top_k]
            for _ in range(tokens)
        ]
    ).int()
    if skewed:
        ids[:] = torch.arange(cfg.top_k, dtype=torch.int32)
    weights = torch.rand(tokens, cfg.top_k, generator=generator)
    # The layer API carries full routing scores even when top-k is precomputed.
    # Build a consistent dense representation without drawing extra RNG values.
    logits = torch.zeros(tokens, cfg.num_experts).scatter_(1, ids.long(), weights)
    device = layer.w13_weight_packed.device
    return StandardDispatchOutput(
        hidden_states=x.to(device),
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(
            topk_ids=ids.to(device),
            topk_weights=weights.to(device),
            router_logits=logits.to(device),
        ),
    )


def _relative_l2(actual, expected):
    return (
        (actual.float() - expected.float()).norm()
        / expected.float().norm().clamp_min(1e-8)
    ).item()


def _weight_storage_inventory(value):
    """Count unique live tensor storage, not tensor aliases or allocator reserves."""
    seen, tensors, storages = set(), [], {}

    def visit(obj, path):
        if isinstance(obj, torch.Tensor):
            storage = obj.untyped_storage()
            key = (str(obj.device), storage.data_ptr(), storage.nbytes())
            storages[key] = storage.nbytes()
            tensors.append(
                dict(
                    path=path,
                    shape=list(obj.shape),
                    dtype=str(obj.dtype),
                    device=str(obj.device),
                    storage_ptr=storage.data_ptr(),
                    storage_bytes=storage.nbytes(),
                )
            )
            return
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, dict):
            for key, child in obj.items():
                visit(child, path + "." + str(key))
        elif isinstance(obj, (list, tuple)):
            for index, child in enumerate(obj):
                visit(child, path + "." + str(index))
        elif hasattr(obj, "__dict__"):
            visit(vars(obj), path)

    visit(value, "weight_pack")
    return dict(
        tensors=tensors,
        unique_storage_count=len(storages),
        unique_storage_bytes=sum(storages.values()),
    )


def _direct_flashinfer(layer, *, capacity):
    """Build a public-API reference without the SGLang adapter helpers."""
    from flashinfer.fused_moe import (
        BackendOptions,
        CuTileNvfp4Config,
        ExecutionConfig,
        ExpertConfig,
        MoEConfig,
        MoELayer,
        MoEWeightPack,
        QuantConfig,
        QuantFormat,
        RoutingConfig,
        SwiGLU,
    )

    cfg = layer.moe_runner_config
    gate, up = layer.w13_weight_packed.chunk(2, dim=1)
    gate_scale, up_scale = layer.w13_weight_scale.chunk(2, dim=1)
    activation = SwiGLU(alpha=ALPHA, beta=BETA, limit=LIMIT)
    view = CuTileNvfp4Config.prepare_weights(
        w1_fp4=torch.cat((up, gate), dim=1),
        w1_block_scale=torch.cat((up_scale, gate_scale), dim=1),
        w1_global_scale=layer.w13_weight_global_scale[:, [1, 0]].contiguous(),
        w2_fp4=layer.w2_weight_packed,
        w2_block_scale=layer.w2_weight_scale,
        w2_global_scale=layer.w2_weight_global_scale,
        num_local_experts=cfg.num_experts,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size_per_partition,
        activation=activation,
        source_format="compressed_tensors",
        device=torch.device("cuda", torch.cuda.current_device()),
    )
    weights = MoEWeightPack()
    weights.prepare_for("cutile_nvfp4", view)
    config = MoEConfig(
        routing=RoutingConfig(num_experts=cfg.num_experts, top_k=cfg.top_k),
        quant=QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4),
        experts=ExpertConfig(intermediate_size=cfg.intermediate_size_per_partition),
        activation=activation,
        backend=BackendOptions(candidates=(CuTileNvfp4Config(),)),
        execution=ExecutionConfig(tune_max_num_tokens=capacity),
    )
    return MoELayer(config=config, device=layer.w13_weight_packed.device), weights


def _reference(
    layer,
    x,
    ids,
    weights,
    *,
    quantize_activations=True,
    return_stats=False,
    reference_mode="corrected",
    swap_gate_up=False,
    apply_clamp=True,
):
    """FP32 GEMMs with unpacked weights and either NVFP4 or BF16 activations.

    The NVFP4 path isolates integration error from intentional W4A4 loss.
    The BF16 path makes that loss visible, but is not bitwise Marlin output.
    Negative IDs represent absent routes and therefore contribute nothing.
    """
    if reference_mode not in ("corrected", "peer-diagnostic"):
        raise ValueError("unknown reference mode")
    # The peer diagnostic omits GEMM BF16 boundaries and moves encode globals
    # into dequantization. It explains report differences; it is NOT a new gate.
    rounded_gemms = quantize_activations and reference_mode == "corrected"
    xq = _activation_reference(x) if quantize_activations else x.bfloat16().float()
    output = torch.zeros_like(x, dtype=torch.float32)
    clamped = torch.zeros((), device=x.device, dtype=torch.int64)
    total = 0
    for expert in range(layer.num_experts):
        token, choice = (ids == expert).nonzero(as_tuple=True)
        if token.numel() == 0:
            continue
        qg, qu = layer.w13_weight_packed[expert].chunk(2, dim=0)
        sg, su = layer.w13_weight_scale[expert].chunk(2, dim=0)
        gg, gu = layer.w13_weight_global_scale[expert]
        if swap_gate_up:
            qg, qu, sg, su, gg, gu = qu, qg, su, sg, gu, gg
        if rounded_gemms:
            # cuTile accumulates block-scaled FP4 products in FP32, applies
            # the expert global scale, then stores BF16 before SwiGLU.
            # Moving the global scale into the GEMM or omitting this rounding
            # can change which side of an FP4 midpoint the activation lands on.
            gate = ((xq[token] @ _dequantize(qg, sg, 1.0).T) / gg).bfloat16().float()
            up = ((xq[token] @ _dequantize(qu, su, 1.0).T) / gu).bfloat16().float()
        else:
            # Preserve the separate BF16-activation quality comparison.
            gate = xq[token] @ _dequantize(qg, sg, gg).T
            up = xq[token] @ _dequantize(qu, su, gu).T
        if return_stats:
            clamped += (gate > LIMIT).sum() + (up.abs() > LIMIT).sum()
            total += gate.numel() + up.numel()
        if apply_clamp:
            gate = gate.clamp(max=LIMIT)
            up = up.clamp(-LIMIT, LIMIT)
        hidden = gate * torch.sigmoid(ALPHA * gate) * (up + BETA)
        hidden = (
            _activation_reference(hidden)
            if quantize_activations
            else hidden.bfloat16().float()
        )
        global_scale = layer.w2_weight_global_scale[expert]
        down = _dequantize(
            layer.w2_weight_packed[expert],
            layer.w2_weight_scale[expert],
            1.0 if rounded_gemms else global_scale,
        )
        expert_output = hidden @ down.T
        if rounded_gemms:
            # GEMM2 also stores BF16 before the FP32 routing-weight reduction.
            expert_output = (expert_output / global_scale).bfloat16().float()
        output.index_add_(0, token, expert_output * weights[token, choice, None])
    if return_stats:
        return output, {"clamped_fraction": clamped.item() / max(total, 1)}
    return output


def _slice_layer_for_tp(layer, *, tp_rank, tp_size):
    """Keep every expert and partition gate/up rows and down input columns."""
    intermediate = layer.intermediate_size_per_partition
    if tp_size < 1 or not 0 <= tp_rank < tp_size:
        raise ValueError("invalid TP rank/size")
    if intermediate % tp_size or (intermediate // tp_size) % 64:
        raise ValueError("each TP intermediate partition must be divisible by 64")
    width = intermediate // tp_size
    start, end = tp_rank * width, (tp_rank + 1) * width
    sliced_names = {
        "w13_weight_packed",
        "w13_weight_scale",
        "w2_weight_packed",
        "w2_weight_scale",
    }
    # Clone only the small per-expert globals here. Cloning full packed weights
    # before replacing them with slices needlessly doubles peak GPU memory.
    parameters = {
        name: value.detach().clone()
        for name, value in layer.named_parameters()
        if name not in sliced_names
    }
    for name in ("w13_weight_packed", "w13_weight_scale"):
        gate, up = getattr(layer, name).chunk(2, dim=1)
        parameters[name] = torch.cat((gate[:, start:end], up[:, start:end]), dim=1)
    # FP4 packs two columns per byte; one E4M3 block scale covers 16 columns.
    parameters["w2_weight_packed"] = layer.w2_weight_packed[
        :, :, start // 2 : end // 2
    ].contiguous()
    parameters["w2_weight_scale"] = layer.w2_weight_scale[
        :, :, start // 16 : end // 16
    ].contiguous()
    part = _layer(parameters, top_k=layer.moe_runner_config.top_k)
    part.moe_tp_rank, part.moe_tp_size = tp_rank, tp_size
    part.moe_runner_config.routed_scaling_factor = (
        layer.moe_runner_config.routed_scaling_factor
    )
    return part


def _load_checkpoint_layer(
    directory, *, experts=32, layer_index=30, tp_rank=0, tp_size=1
):
    """Load only the requested expert tensors and TP slice; never download.

    Slicing is done on CPU before transfer, so a TP rank need not fit a full
    unsliced layer on its GPU. This is tensor parallelism, not expert sharding.
    Checkpoint activation globals are preserved when present rather than
    silently replaced by ones; the adapter then validates its unit-scale rule.
    """
    from safetensors import safe_open

    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    config = config.get("text_config", config)
    if not 1 <= experts <= config["num_local_experts"]:
        raise ValueError("requested expert count is outside the checkpoint")
    if tp_size < 1 or not 0 <= tp_rank < tp_size:
        raise ValueError("invalid TP rank/size")
    for key, expected in (("swiglu_alpha", ALPHA), ("swiglu_limit", LIMIT)):
        if key in config and config[key] != expected:
            raise ValueError(f"{key}={config[key]} differs from the reference")
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    suffix = f"layers.{layer_index}.block_sparse_moe.experts.0.w1.weight_packed"
    matches = [name for name in weight_map if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"expected one checkpoint key ending with {suffix}")
    prefix = matches[0].rsplit(".0.w1.", 1)[0]
    tensors = {}
    intermediate = None
    for expert in range(experts):
        for projection in ("w1", "w2", "w3"):
            for field in (
                "weight_packed",
                "weight_scale",
                "weight_global_scale",
                "input_global_scale",
            ):
                name = f"{prefix}.{expert}.{projection}.{field}"
                if field == "input_global_scale" and name not in weight_map:
                    tensor = torch.ones(())
                else:
                    if name not in weight_map:
                        raise ValueError(f"checkpoint is missing {name}")
                    shard = directory / weight_map[name]
                    if not shard.is_file():
                        raise FileNotFoundError(
                            f"missing shard {shard} required by expert {expert}"
                        )
                    with safe_open(shard, framework="pt", device="cpu") as handle:
                        tensor = handle.get_tensor(name)
                if field in ("weight_packed", "weight_scale"):
                    if projection in ("w1", "w3"):
                        width = tensor.shape[0]
                        if intermediate is None:
                            intermediate = width
                        if width != intermediate:
                            raise ValueError("gate/up intermediate sizes disagree")
                        if width % tp_size or (width // tp_size) % 64:
                            raise ValueError("TP intermediate must be divisible by 64")
                        part = width // tp_size
                        tensor = tensor[tp_rank * part : (tp_rank + 1) * part]
                    else:
                        storage_width = tensor.shape[-1]
                        if storage_width % tp_size:
                            raise ValueError("down storage width cannot be TP sliced")
                        part = storage_width // tp_size
                        tensor = tensor[:, tp_rank * part : (tp_rank + 1) * part]
                if field.endswith("global_scale"):
                    if tensor.numel() != 1:
                        raise ValueError(f"expected one global scale for {name}")
                    # Checkpoints may encode scalar scales as [] or [1].
                    # Normalize them before mixing present and default fields.
                    tensor = tensor.reshape(())
                tensors[expert, projection, field] = tensor.contiguous().to("cuda")

    def stack(projection, field):
        return torch.stack([tensors[e, projection, field] for e in range(experts)])

    parameters = dict(
        w13_weight_packed=torch.cat(
            (stack("w1", "weight_packed"), stack("w3", "weight_packed")), dim=1
        ).view(torch.uint8),
        w13_weight_scale=torch.cat(
            (stack("w1", "weight_scale"), stack("w3", "weight_scale")), dim=1
        ),
        w13_weight_global_scale=torch.stack(
            (
                stack("w1", "weight_global_scale").reshape(experts),
                stack("w3", "weight_global_scale").reshape(experts),
            ),
            dim=1,
        ).float(),
        w13_input_global_scale=torch.stack(
            (
                stack("w1", "input_global_scale").reshape(experts),
                stack("w3", "input_global_scale").reshape(experts),
            ),
            dim=1,
        ).float(),
        w2_weight_packed=stack("w2", "weight_packed").view(torch.uint8),
        w2_weight_scale=stack("w2", "weight_scale"),
        w2_weight_global_scale=stack("w2", "weight_global_scale")
        .reshape(experts)
        .float(),
        w2_input_global_scale=stack("w2", "input_global_scale")
        .reshape(experts)
        .float(),
    )
    layer = _layer(parameters, top_k=config["num_experts_per_tok"])
    layer.moe_tp_rank, layer.moe_tp_size = tp_rank, tp_size
    return layer


def _dispatch(layer, tokens, *, input_std=1.0):
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    config = layer.moe_runner_config
    x = (
        torch.randn(tokens, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        * input_std
    )
    ids = torch.arange(tokens * config.top_k, device="cuda", dtype=torch.int32)
    ids = ids.reshape(tokens, config.top_k) % config.num_experts
    # Deliberately not normalized: an adapter must preserve the caller's weights.
    weights = (
        torch.linspace(0.25, 1.75, config.top_k, device="cuda", dtype=torch.float32)
        .expand(tokens, -1)
        .contiguous()
    )
    return StandardDispatchOutput(
        hidden_states=x,
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(
            topk_ids=ids, topk_weights=weights, router_logits=None
        ),
    )


def _reference_token_indices(tokens, limit):
    """Sample both sides of every 8192 boundary and the final row.

    Fill the remaining budget across the batch, rather than checking only a
    prefix that cannot detect a misplaced tail. Reject an insufficient budget.
    """
    if tokens < 1 or limit < 1:
        raise ValueError("tokens and reference limit must be positive")
    count = min(tokens, limit)
    required = {0, tokens - 1}
    for boundary in range(8192, tokens, 8192):
        required.update((boundary - 1, boundary))
    if len(required) > count:
        raise ValueError("reference budget cannot cover all chunk boundaries")
    # Include evenly spaced rows first, then fill any deduplicated positions.
    if count > 1:
        candidates = [i * (tokens - 1) // (count - 1) for i in range(count)]
    else:
        candidates = [0]
    for index in candidates:
        if len(required) == count:
            break
        required.add(index)
    for index in range(tokens):
        if len(required) == count:
            break
        required.add(index)
    return sorted(required)


def _select_dispatch_rows(dispatch, indices):
    """Gather activation and routing rows together, preserving optional logits."""
    indices = torch.tensor(indices, device=dispatch.hidden_states.device)
    topk = dispatch.topk_output
    logits = topk.router_logits
    return dispatch._replace(
        hidden_states=dispatch.hidden_states.index_select(0, indices),
        topk_output=topk._replace(
            topk_ids=topk.topk_ids.index_select(0, indices),
            topk_weights=topk.topk_weights.index_select(0, indices),
            router_logits=None if logits is None else logits.index_select(0, indices),
        ),
    )


def _direct_dispatch(runner, prepared, dispatch, *, chunk_size=None):
    """Independent public-API oracle: concatenate results without adapter code."""
    from flashinfer.fused_moe import MoEActivationPack
    from flashinfer.fused_moe.core import RoutingInputMode

    tokens = dispatch.hidden_states.shape[0]
    if tokens < 1:
        raise ValueError("direct oracle requires a nonempty batch")
    step = tokens if chunk_size is None else chunk_size
    outputs = []
    for start in range(0, tokens, step):
        end = min(start + step, tokens)
        ids = dispatch.topk_output.topk_ids[start:end]
        weights = dispatch.topk_output.topk_weights[start:end]
        outputs.append(
            runner(
                MoEActivationPack(
                    hidden_states_q=dispatch.hidden_states[start:end].contiguous(),
                    hidden_states_scale=None,
                    topk_ids=ids.clamp_min(0).int().contiguous(),
                    topk_weights=weights.masked_fill(ids < 0, 0).float().contiguous(),
                    routing_input_mode=RoutingInputMode.PackedPrecomputed,
                ),
                prepared,
            )
        )
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)


class TestFlashInferCuTileNvfp4(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not torch.cuda.is_available():
            raise unittest.SkipTest("requires an SM120/SM121 GPU")
        if torch.cuda.get_device_capability() not in ((12, 0), (12, 1)):
            raise unittest.SkipTest("cuTile NVFP4 is restricted to SM120/SM121")
        from sglang.srt.layers.moe.moe_runner import flashinfer_cutile

        cls.adapter = flashinfer_cutile

    def setUp(self):
        super().setUp()
        torch.manual_seed(17)
        reset_context()
        self.addCleanup(reset_context)
        self.enterContext(
            get_context().override_server_args(
                max_prefill_tokens=512, disable_cuda_graph=True
            )
        )
        self.enterContext(get_exec().graph.override(cuda_graph_config=None))
        self.enterContext(
            get_flags().moe.override(
                runner_backend=MoeRunnerBackend.FLASHINFER_CUTILE,
                a2a_backend=MoeA2ABackend.NONE,
            )
        )

    def _run_case(
        self,
        layer,
        tokens,
        *,
        dispatch=None,
        capacity=None,
        expect_clamp=False,
        reference_tokens=None,
        compare_unchunked=False,
    ):
        capacity = max(512, tokens) if capacity is None else capacity
        self.enterContext(get_schedule().override(max_prefill_tokens=capacity))
        dispatch = _dispatch(layer, tokens) if dispatch is None else dispatch
        x = dispatch.hidden_states
        ids = dispatch.topk_output.topk_ids
        weights = dispatch.topk_output.topk_weights
        factor = layer.moe_runner_config.routed_scaling_factor
        factor = 1.0 if factor is None else factor
        reference_indices = _reference_token_indices(
            tokens, tokens if reference_tokens is None else reference_tokens
        )
        selected = _select_dispatch_rows(dispatch, reference_indices)
        reference, stats = _reference(
            layer,
            selected.hidden_states,
            selected.topk_output.topk_ids,
            selected.topk_output.topk_weights,
            return_stats=True,
        )
        reference = reference * factor
        bf16_reference = (
            _reference(
                layer,
                selected.hidden_states,
                selected.topk_output.topk_ids,
                selected.topk_output.topk_weights,
                quantize_activations=False,
            )
            * factor
        )
        if expect_clamp:
            self.assertGreater(stats["clamped_fraction"], 0.01)
        # Use identical per-call shapes for the bit-exact integration oracle.
        # The unchunked oracle below is a separate, cross-shape comparison.
        direct, direct_weights = _direct_flashinfer(layer, capacity=min(capacity, 8192))
        unsplit = (
            _direct_flashinfer(layer, capacity=capacity) if compare_unchunked else None
        )
        prepared = self.adapter.prepare_flashinfer_cutile_nvfp4_weights(
            layer, layer.moe_runner_config
        )
        runner = self.adapter.FlashInferCuTileRunnerCore(layer.moe_runner_config)

        def run():
            return runner.run_from_dispatch(
                dispatch, prepared, layer.moe_runner_config
            ).hidden_states

        def direct_run():
            result = _direct_dispatch(direct, direct_weights, dispatch, chunk_size=8192)
            return result if factor == 1.0 else result * factor

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            expected = direct_run().clone()
            actual = run().clone()
            self.assertTrue(torch.isfinite(actual).all().item())
            # Same kernel through two entry points must be bit-identical.
            self.assertTrue(
                torch.equal(actual.view(torch.int16), expected.view(torch.int16))
            )
            sampled_actual = actual[reference_indices].float()
            relative = (sampled_actual - reference).norm() / reference.norm().clamp_min(
                1e-8
            )
            bf16_relative = (
                sampled_actual - bf16_reference
            ).norm() / bf16_reference.norm().clamp_min(1e-8)
            unchunked_relative = None
            if unsplit is not None:
                unchunked = _direct_dispatch(*unsplit, dispatch) * factor
                unchunked_relative = (
                    (actual.float() - unchunked.float()).norm()
                    / unchunked.float().norm().clamp_min(1e-8)
                ).item()
                self.assertLess(unchunked_relative, 0.05)
            # Predeclared integration bound, separate from W4A4-vs-W4A16 loss.
            # Do not loosen it to accommodate a failing GPU run without first
            # investigating the numerical reference and reporting both errors.
            self.assertLess(relative.item(), 0.05)
            for repeat in range(20):
                with self.subTest(repeat=repeat):
                    self.assertTrue(
                        torch.equal(actual.view(torch.int16), run().view(torch.int16))
                    )
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured = run()
            graph.replay()
            self.assertTrue(
                torch.equal(captured.view(torch.int16), actual.view(torch.int16))
            )
            # Mutate storage, not Python tensor references: a graph must read
            # fresh activations AND routes on replay, including changed padding.
            for replay in range(4):
                replacement = _dispatch(layer, tokens, input_std=0.5 + replay * 0.25)
                replacement.topk_output.topk_ids.add_(replay + 1).remainder_(
                    layer.num_experts
                )
                replacement.topk_output.topk_ids[::3, -1] = -1
                x.copy_(replacement.hidden_states)
                ids.copy_(replacement.topk_output.topk_ids)
                weights.copy_(replacement.topk_output.topk_weights * (replay + 1) / 4)
                eager_changed = run().clone()
                direct_changed = direct_run().clone()
                graph.replay()
                torch.testing.assert_close(captured, eager_changed, rtol=0, atol=0)
                self.assertTrue(
                    torch.equal(
                        captured.view(torch.int16), direct_changed.view(torch.int16)
                    )
                )
            stream.synchronize()
            # Captured workspaces stay alive until graph destruction; no
            # reset_context or cache eviction is performed while this is live.
            del graph
        torch.cuda.current_stream().wait_stream(stream)
        print(
            json.dumps(
                {
                    "tokens": tokens,
                    "experts": layer.num_experts,
                    "relative_l2_nvfp4_reference": relative.item(),
                    "relative_l2_bf16_reference": bf16_relative.item(),
                    "clamped_fraction": stats["clamped_fraction"],
                    "repeatability_calls": 20,
                    "changed_input_graph_replays": 4,
                    "reference_indices": reference_indices,
                    "direct_chunked_bit_identity": True,
                    "relative_l2_unchunked": unchunked_relative,
                    "chunk_size": 8192,
                },
                sort_keys=True,
            )
        )
        return actual

    def test_large_geometry_skewed_routes_and_swap_negative_control(self):
        for seed in (0, 1):
            layer = _packed_synthetic_layer(seed)
            for skewed in (False, True):
                with self.subTest(seed=seed, skewed=skewed):
                    dispatch = _peer_dispatch(layer, 412, seed=seed, skewed=skewed)
                    x, routes = dispatch.hidden_states, dispatch.topk_output
                    normal = _reference(layer, x, routes.topk_ids, routes.topk_weights)
                    wrong = _reference(
                        layer,
                        x,
                        routes.topk_ids,
                        routes.topk_weights,
                        swap_gate_up=True,
                    )
                    diagnostic = _reference(
                        layer,
                        x,
                        routes.topk_ids,
                        routes.topk_weights,
                        reference_mode="peer-diagnostic",
                    )
                    if skewed:
                        self.assertEqual(torch.unique(routes.topk_ids).numel(), 4)
                    actual = self._run_case(layer, 412, dispatch=dispatch)
                    good_error = _relative_l2(actual, normal)
                    bad_error = _relative_l2(actual, wrong)
                    self.assertLess(good_error, 0.05)
                    self.assertGreater(bad_error, 5 * good_error)
                    print(
                        json.dumps(
                            dict(
                                test="skew_and_swap",
                                seed=seed,
                                skewed=skewed,
                                experts=16,
                                empty_experts=12 if skewed else 0,
                                reference_l2=good_error,
                                swapped_l2=bad_error,
                                peer_diagnostic_l2=_relative_l2(actual, diagnostic),
                                swap_ratio_limit=5,
                            ),
                            sort_keys=True,
                        )
                    )

    def test_large_geometry_unclamped_negative_control(self):
        layer = _packed_synthetic_layer(0)
        dispatch = _peer_dispatch(layer, 412, seed=0, input_std=32)
        x, routes = dispatch.hidden_states, dispatch.topk_output
        reference, stats = _reference(
            layer, x, routes.topk_ids, routes.topk_weights, return_stats=True
        )
        wrong = _reference(
            layer, x, routes.topk_ids, routes.topk_weights, apply_clamp=False
        )
        self.assertGreater(stats["clamped_fraction"], 0.01)
        self.assertGreater(_relative_l2(wrong, reference), 0.2)
        actual = self._run_case(layer, 412, dispatch=dispatch, expect_clamp=True)
        self.assertLess(_relative_l2(actual, reference), 0.05)
        self.assertGreater(_relative_l2(actual, wrong), 0.2)
        print(
            json.dumps(
                dict(
                    test="unclamped_negative_control",
                    input_std=32,
                    **stats,
                    reference_l2=_relative_l2(actual, reference),
                    unclamped_l2=_relative_l2(actual, wrong),
                    unclamped_reference_l2=_relative_l2(wrong, reference),
                ),
                sort_keys=True,
            )
        )

    def test_padding_preserves_real_rows_across_batch_shapes(self):
        layer = _packed_synthetic_layer(0)
        cfg = layer.moe_runner_config
        prepared = self.adapter.prepare_flashinfer_cutile_nvfp4_weights(layer, cfg)
        runner = self.adapter.FlashInferCuTileRunnerCore(cfg)
        dispatch = _peer_dispatch(layer, 64, seed=0)
        small = _select_dispatch_rows(dispatch, list(range(48)))
        expected = runner.run_from_dispatch(small, prepared, cfg).hidden_states.clone()
        dispatch.topk_output.topk_ids[48:] = -1
        actual = runner.run_from_dispatch(dispatch, prepared, cfg).hidden_states.clone()
        self.assertLess(_relative_l2(actual[:48], expected), 0.01)
        self.assertEqual(torch.count_nonzero(actual[48:]).item(), 0)
        dispatch.topk_output.topk_ids[0, -1] = -1
        partial = runner.run_from_dispatch(
            dispatch, prepared, cfg
        ).hidden_states.clone()
        dispatch.topk_output.topk_ids[0, -1] = 0
        dispatch.topk_output.topk_weights[0, -1] = 0
        neutral = runner.run_from_dispatch(dispatch, prepared, cfg).hidden_states
        self.assertTrue(
            torch.equal(partial.view(torch.int16), neutral.view(torch.int16))
        )
        print(
            json.dumps(
                dict(
                    test="padding_cross_batch",
                    real_rows=48,
                    padded_rows=16,
                    relative_l2=_relative_l2(actual[:48], expected),
                )
            )
        )

    @unittest.skipUnless(
        envs.SGLANG_TEST_CUTILE_LARGE_BATCH.get(),
        "enable large-batch tests for workspace invariants",
    )
    def test_warm_workspace_is_stable_and_bounded(self):
        layer = _packed_synthetic_layer(0)
        cfg = layer.moe_runner_config
        self.enterContext(get_schedule().override(max_prefill_tokens=16384))
        prepared = self.adapter.prepare_flashinfer_cutile_nvfp4_weights(layer, cfg)
        runner = self.adapter.FlashInferCuTileRunnerCore(cfg)
        large = _peer_dispatch(layer, 8193, seed=0)
        dispatches = {
            n: _select_dispatch_rows(large, list(range(n))) for n in (8192, 8193)
        }
        # Warm both sizes first. All inputs/weights live before the baseline.
        for dispatch in dispatches.values():
            result = runner.run_from_dispatch(dispatch, prepared, cfg).hidden_states
            del result
        torch.cuda.synchronize()
        gc.collect()
        held = torch.cuda.memory_allocated()
        peaks = {}
        for tokens, dispatch in dispatches.items():
            peaks[tokens] = []
            for repeat in range(3):
                torch.cuda.reset_peak_memory_stats()
                result = runner.run_from_dispatch(dispatch, prepared, cfg).hidden_states
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated()
                live = torch.cuda.memory_allocated()
                del result
                torch.cuda.synchronize()
                self.assertEqual(torch.cuda.memory_allocated(), held)
                peaks[tokens].append(peak)
                print(
                    json.dumps(
                        dict(
                            test="warm_workspace",
                            tokens=tokens,
                            repeat=repeat,
                            held_bytes=held,
                            output_live_allocated_bytes=live,
                            output_released_allocated_bytes=torch.cuda.memory_allocated(),
                            peak_bytes=peak,
                            reserved_bytes=torch.cuda.memory_reserved(),
                        ),
                        sort_keys=True,
                    )
                )
        self.assertLess(
            max(peaks[8193]), max(peaks[8192]) + 3 * 8193 * cfg.hidden_size * 2
        )

    def test_same_geometry_layers_share_one_execution(self):
        one, two = _synthetic_layer(), _synthetic_layer()
        first = self.adapter.FlashInferCuTileRunnerCore(one.moe_runner_config)
        second = self.adapter.FlashInferCuTileRunnerCore(two.moe_runner_config)
        weights = [
            self.adapter.prepare_flashinfer_cutile_nvfp4_weights(x, x.moe_runner_config)
            for x in (one, two)
        ]
        inputs = _dispatch(one, 64)
        first.run_from_dispatch(inputs, weights[0], one.moe_runner_config)
        first_owner = first._get_execution(inputs.hidden_states.device, 64)
        second.run_from_dispatch(inputs, weights[1], two.moe_runner_config)
        second_owner = second._get_execution(inputs.hidden_states.device, 64)
        self.assertIs(first_owner, second_owner)
        self.assertIsNot(weights[0].weight_pack, weights[1].weight_pack)
        print(
            json.dumps(
                dict(
                    test="multi_layer_owner",
                    layers=2,
                    execution_owner_ids=[id(first_owner), id(second_owner)],
                )
            )
        )

    @unittest.skipUnless(
        envs.SGLANG_TEST_MINIMAX_LAYER30_DIR.get(),
        "checkpoint required for production weight ownership audit",
    )
    def test_real_prepared_weight_storage_has_no_retained_source_copy(self):
        from sglang.srt.layers.quantization.minimax_m3_nvfp4 import MiniMaxM3Nvfp4MoE

        for experts in (32, 128):
            if experts == 128 and not envs.SGLANG_TEST_CUTILE_LARGE_BATCH.get():
                continue
            layer = _load_checkpoint_layer(
                envs.SGLANG_TEST_MINIMAX_LAYER30_DIR.get(), experts=experts
            )
            names = (
                "w13_weight_packed",
                "w2_weight_packed",
                "w13_weight_scale",
                "w2_weight_scale",
            )
            source = {
                name: dict(
                    ref=weakref.ref(getattr(layer, name)),
                    ptr=getattr(layer, name).untyped_storage().data_ptr(),
                    bytes=getattr(layer, name).untyped_storage().nbytes(),
                )
                for name in names
            }
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            scheme = MiniMaxM3Nvfp4MoE()
            scheme.create_moe_runner(layer, layer.moe_runner_config)
            scheme.process_weights_after_loading(layer)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            gc.collect()
            inventory = _weight_storage_inventory(scheme.cutile_quant_info.weight_pack)
            self.assertTrue(inventory["tensors"])
            owned = {t["storage_ptr"] for t in inventory["tensors"]}
            for name, original in source.items():
                self.assertIsNone(getattr(layer, name))
                # A source tensor may be the very tensor used by the prepared
                # view (not a duplicate); all other source objects must die.
                if original["ref"]() is not None:
                    self.assertIn(original["ptr"], owned)
            packed = {
                t["storage_ptr"]
                for t in inventory["tensors"]
                if t["dtype"] == "torch.uint8"
            }
            self.assertEqual(len(packed), 2)
            print(
                json.dumps(
                    dict(
                        test="production_weight_storage",
                        experts=experts,
                        **inventory,
                        source_large_storage_bytes=sum(
                            x["bytes"] for x in source.values()
                        ),
                        allocated_before=before,
                        allocated_after=torch.cuda.memory_allocated(),
                        preparation_peak_bytes=peak,
                        reserved_bytes=torch.cuda.memory_reserved(),
                        source_objects_alive={
                            n: x["ref"]() is not None for n, x in source.items()
                        },
                    ),
                    sort_keys=True,
                )
            )
            del layer, scheme
            gc.collect()

    def test_activation_reference_midpoints(self):
        # +/-6 anchors each 16-value block to the chosen E4M3 scale. Explicit
        # expected values distinguish midpoint ties from either neighbor;
        # they do not call FlashInfer or the quantizer under test.
        midpoints = torch.tensor(
            [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
            device="cuda",
            dtype=torch.bfloat16,
        )
        anchor = torch.tensor([6.0], device="cuda", dtype=torch.bfloat16)
        lower = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6], device="cuda", dtype=torch.float32
        )
        upper = torch.tensor(
            [0.5, 1, 1.5, 2, 3, 4, 6, 6], device="cuda", dtype=torch.float32
        )
        for block_scale in (0.5, 1.0, 2.0):
            for offset, expected_magnitudes in (
                (-0.0625, lower),
                (0.0, lower),
                (0.0625, upper),
            ):
                with self.subTest(block_scale=block_scale, offset=offset):
                    positive = torch.cat((midpoints + offset, anchor))
                    values = torch.cat((positive, -positive))[None] * block_scale
                    expected = (
                        torch.cat((expected_magnitudes, -expected_magnitudes))[None]
                        * block_scale
                    )
                    torch.testing.assert_close(
                        _activation_reference(values), expected, rtol=0, atol=0
                    )
        zeros = torch.zeros(2, 32, device="cuda", dtype=torch.bfloat16)
        torch.testing.assert_close(
            _activation_reference(zeros), zeros.float(), rtol=0, atol=0
        )

    def test_synthetic_eager_graph_and_repeatability(self):
        for tokens in (1, 64, 129):
            with self.subTest(tokens=tokens):
                self._run_case(_synthetic_layer(), tokens)

    @unittest.skipUnless(
        envs.SGLANG_TEST_CUTILE_LARGE_BATCH.get(),
        "set SGLANG_TEST_CUTILE_LARGE_BATCH=1 for chunk-boundary Graph checks",
    )
    def test_chunk_boundaries_eager_graph_and_padding(self):
        for tokens in (8191, 8192, 8193, 16384, 16385):
            with self.subTest(tokens=tokens):
                layer = _synthetic_layer(distinct_gate_up_scales=True)
                layer.moe_runner_config.routed_scaling_factor = 1.75
                dispatch = _dispatch(layer, tokens)
                ids = dispatch.topk_output.topk_ids.long()
                for index in (0, 8191, 8192, 16383, 16384):
                    if index < tokens:
                        ids[index, -1] = -1
                ids[-1] = -1
                # Strided rows exercise per-chunk normalization, including tail.
                dispatch = dispatch._replace(
                    hidden_states=dispatch.hidden_states.repeat_interleave(2, dim=1)[
                        :, ::2
                    ],
                    topk_output=dispatch.topk_output._replace(
                        topk_ids=ids.repeat_interleave(2, dim=1)[:, ::2],
                        topk_weights=dispatch.topk_output.topk_weights.bfloat16(),
                    ),
                )
                actual = self._run_case(
                    layer,
                    tokens,
                    dispatch=dispatch,
                    capacity=32768,
                    reference_tokens=64,
                    compare_unchunked=True,
                )
                self.assertEqual(torch.count_nonzero(actual[-1]).item(), 0)

    @unittest.skipUnless(
        envs.SGLANG_TEST_CUTILE_LARGE_BATCH.get(),
        "set SGLANG_TEST_CUTILE_LARGE_BATCH=1 for cold-tail capture rejection",
    )
    def test_chunked_capture_requires_tail_warmup(self):
        layer = _synthetic_layer()
        cfg = layer.moe_runner_config
        prepared = self.adapter.prepare_flashinfer_cutile_nvfp4_weights(layer, cfg)
        runner = self.adapter.FlashInferCuTileRunnerCore(cfg)
        self.enterContext(get_schedule().override(max_prefill_tokens=16384))
        full = _dispatch(layer, 8193)
        main = full._replace(
            hidden_states=full.hidden_states[:8192],
            topk_output=full.topk_output._replace(
                topk_ids=full.topk_output.topk_ids[:8192],
                topk_weights=full.topk_output.topk_weights[:8192],
            ),
        )
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            runner.run_from_dispatch(main, prepared, cfg)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            try:
                # Main block is warm, but no 1-token call has run on this stream.
                # Rejection must precede any work from the main block.
                with self.assertRaisesRegex(RuntimeError, "warmup"):
                    with torch.cuda.graph(graph, stream=stream):
                        runner.run_from_dispatch(full, prepared, cfg)
            finally:
                graph.reset()
            expected = runner.run_from_dispatch(
                full, prepared, cfg
            ).hidden_states.clone()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph, stream=stream):
                    actual = runner.run_from_dispatch(full, prepared, cfg).hidden_states
                graph.replay()
                self.assertTrue(
                    torch.equal(actual.view(torch.int16), expected.view(torch.int16))
                )
                stream.synchronize()
            finally:
                graph.reset()
        torch.cuda.current_stream().wait_stream(stream)

    def test_distinct_gate_up_global_and_block_scales(self):
        layer = _synthetic_layer(distinct_gate_up_scales=True)
        globals13 = layer.w13_weight_global_scale
        self.assertTrue((globals13[:, 0] != globals13[:, 1]).all().item())
        self._run_case(layer, 64)

    def test_clamp_is_exercised(self):
        layer = _synthetic_layer()
        self._run_case(
            layer, 64, dispatch=_dispatch(layer, 64, input_std=4.0), expect_clamp=True
        )

    def test_partial_and_all_padding_routes(self):
        for all_padding in (False, True):
            with self.subTest(all_padding=all_padding):
                layer = _synthetic_layer()
                dispatch = _dispatch(layer, 64)
                ids = dispatch.topk_output.topk_ids
                if all_padding:
                    ids.fill_(-1)
                else:
                    ids[:, -1] = -1
                    ids[0].fill_(-1)
                # Int64 IDs and BF16 weights additionally exercise the casts.
                dispatch = dispatch._replace(
                    topk_output=dispatch.topk_output._replace(
                        topk_ids=ids.long(),
                        topk_weights=dispatch.topk_output.topk_weights.bfloat16(),
                    )
                )
                actual = self._run_case(layer, 64, dispatch=dispatch)
                if all_padding:
                    self.assertEqual(torch.count_nonzero(actual).item(), 0)
                else:
                    self.assertEqual(torch.count_nonzero(actual[0]).item(), 0)

    def test_scheme_loading_and_apply_use_cutile(self):
        from sglang.srt.layers.quantization import minimax_m3_nvfp4

        layer = _synthetic_layer()
        dispatch = _dispatch(layer, 8)
        reference_runner = self.adapter.FlashInferCuTileRunnerCore(
            layer.moe_runner_config
        )
        reference_weights = self.adapter.prepare_flashinfer_cutile_nvfp4_weights(
            layer, layer.moe_runner_config
        )
        expected = reference_runner.run_from_dispatch(
            dispatch, reference_weights, layer.moe_runner_config
        ).hidden_states.clone()
        # A factor on the runner contract belongs after the MoE call. Applying
        # it to quantized inputs or twice would change this exact result.
        layer.moe_runner_config.routed_scaling_factor = 1.75
        expected = expected * 1.75
        scheme = minimax_m3_nvfp4.MiniMaxM3Nvfp4MoE()
        scheme.create_moe_runner(layer, layer.moe_runner_config)
        scheme.process_weights_after_loading(layer)
        actual = scheme.apply_weights(layer, dispatch).hidden_states
        self.assertTrue(scheme.use_flashinfer_cutile)
        self.assertFalse(scheme.use_marlin)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_routed_scaling_factor_two_is_applied_once(self):
        layer = _synthetic_layer()
        layer.moe_runner_config.routed_scaling_factor = 2.0
        self._run_case(layer, 64)

    def test_two_streams_keep_execution_workspaces_separate(self):
        """Interleaved warmed calls must keep each stream's scratch space separate."""
        layer = _synthetic_layer()
        cfg = layer.moe_runner_config
        prepared = self.adapter.prepare_flashinfer_cutile_nvfp4_weights(layer, cfg)
        runner = self.adapter.FlashInferCuTileRunnerCore(cfg)
        inputs = [_dispatch(layer, 64, input_std=std) for std in (0.5, 1.0)]
        expected = [
            runner.run_from_dispatch(item, prepared, cfg).hidden_states.clone()
            for item in inputs
        ]
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        executions = []
        # Compile and initialize each stream before the interleaved submissions:
        # cold-start work could otherwise serialize the only calls being checked.
        for stream, dispatch in zip(streams, inputs):
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    runner.run_from_dispatch(dispatch, prepared, cfg)
                executions.append(runner._get_execution(prepared.device, 64))
        torch.cuda.synchronize()
        self.assertIsNot(executions[0], executions[1])

        actual = []
        # Submit both streams in every round without synchronizing between calls.
        # This exercises isolation under interleaved submissions; scheduling on
        # the device may still serialize them, so it does not prove GPU overlap.
        for _ in range(20):
            for stream_index, (stream, dispatch) in enumerate(zip(streams, inputs)):
                with torch.cuda.stream(stream):
                    output = runner.run_from_dispatch(dispatch, prepared, cfg)
                    # Retain independent storage for every call until validation;
                    # later workspace reuse must not overwrite an earlier result.
                    actual.append((stream_index, output.hidden_states.clone()))
        torch.cuda.synchronize()
        for stream_index, observed in actual:
            torch.testing.assert_close(observed, expected[stream_index], rtol=0, atol=0)

    def test_context_reset_rebuilds_execution_and_warmup_state(self):
        """Keep a runner alive while replacing its resource-owning context."""
        layer = _synthetic_layer()
        cfg = layer.moe_runner_config
        prepared = self.adapter.prepare_flashinfer_cutile_nvfp4_weights(layer, cfg)
        runner = self.adapter.FlashInferCuTileRunnerCore(cfg)
        dispatch = _dispatch(layer, 64)
        expected = runner.run_from_dispatch(
            dispatch, prepared, cfg
        ).hidden_states.clone()
        old_execution = runner._get_execution(prepared.device, 64)
        self.assertIn(64, old_execution.warmed_token_counts)
        torch.cuda.synchronize()
        # No live CUDA graph exists here. A reset with a live graph would be an
        # invalid resource-lifetime experiment because the graph owns pointers.
        reset_context()
        with (
            get_context().override_server_args(
                max_prefill_tokens=512, disable_cuda_graph=True
            ),
            get_exec().graph.override(cuda_graph_config=None),
            get_flags().moe.override(
                runner_backend=MoeRunnerBackend.FLASHINFER_CUTILE,
                a2a_backend=MoeA2ABackend.NONE,
            ),
        ):
            new_execution = runner._get_execution(prepared.device, 64)
            self.assertIsNot(old_execution, new_execution)
            self.assertNotIn(64, new_execution.warmed_token_counts)
            actual = runner.run_from_dispatch(dispatch, prepared, cfg).hidden_states
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertIn(64, new_execution.warmed_token_counts)

    def test_context_reset_releases_execution_after_graph_destruction(self):
        """A live runner must not retain an execution after its context is reset."""
        layer = _synthetic_layer()
        cfg = layer.moe_runner_config
        prepared = self.adapter.prepare_flashinfer_cutile_nvfp4_weights(layer, cfg)
        runner = self.adapter.FlashInferCuTileRunnerCore(cfg)
        dispatch = _dispatch(layer, 64)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.stream(stream):
                expected = runner.run_from_dispatch(
                    dispatch, prepared, cfg
                ).hidden_states.clone()
                old_execution = runner._get_execution(prepared.device, 64)
                execution_ref = weakref.ref(old_execution)
                stream.synchronize()
                with torch.cuda.graph(graph, stream=stream):
                    captured = runner.run_from_dispatch(
                        dispatch, prepared, cfg
                    ).hidden_states
                graph.replay()
                torch.testing.assert_close(captured, expected, rtol=0, atol=0)
        finally:
            # Even a failed comparison must not leave a live graph pointing into
            # executions that test cleanup will release through reset_context().
            torch.cuda.synchronize()
            graph.reset()

        del graph, captured, expected, old_execution
        gc.collect()
        # The context still owns the execution after graph destruction.
        self.assertIsNotNone(execution_ref())
        reset_context()
        gc.collect()
        torch.cuda.synchronize()
        # Keep runner alive to detect an accidental runner-local strong cache.
        # Assert owner reclamation, not reserved bytes: the CUDA allocator may
        # retain freed blocks, and unrelated JIT caches have separate lifetimes.
        self.assertIsNone(execution_ref())

    def test_tp_weight_slices_match_unsharded_layer(self):
        """Check TP slice arithmetic on one GPU; distributed all-reduce is separate."""
        self._check_tp_weight_slices(_synthetic_layer(intermediate_size=256), 64)

    def _check_tp_weight_slices(self, full, tokens, *, input_std=1.0):
        self.enterContext(get_schedule().override(max_prefill_tokens=max(512, tokens)))
        dispatch = _dispatch(full, tokens, input_std=input_std)
        cfg = full.moe_runner_config
        full_runner = self.adapter.FlashInferCuTileRunnerCore(cfg)
        full_weights = self.adapter.prepare_flashinfer_cutile_nvfp4_weights(full, cfg)
        expected = full_runner.run_from_dispatch(
            dispatch, full_weights, cfg
        ).hidden_states.clone()
        results = []
        for rank in range(2):
            part = _slice_layer_for_tp(full, tp_rank=rank, tp_size=2)
            part_cfg = part.moe_runner_config
            runner = self.adapter.FlashInferCuTileRunnerCore(part_cfg)
            weights = self.adapter.prepare_flashinfer_cutile_nvfp4_weights(
                part, part_cfg
            )
            results.append(
                runner.run_from_dispatch(dispatch, weights, part_cfg)
                .hidden_states.float()
                .clone()
            )
        combined = torch.stack(results).sum(dim=0)
        relative = (combined - expected.float()).norm() / expected.float().norm()
        self.assertLess(relative.item(), 0.02)
        print(
            json.dumps(
                {
                    "test": "tp2_partition_sum_vs_tp1",
                    "tokens": tokens,
                    "experts": full.num_experts,
                    "relative_l2": relative.item(),
                    "relative_l2_limit": 0.02,
                    "distributed_nccl": False,
                },
                sort_keys=True,
            )
        )

    @unittest.skipUnless(
        envs.SGLANG_TEST_CUTILE_LARGE_BATCH.get(),
        "set SGLANG_TEST_CUTILE_LARGE_BATCH=1 for workspace boundary measurements",
    )
    def test_workspace_bucket_boundary(self):
        # Match the issue's local TP2 geometry; this is a single-device layer test.
        layer = _synthetic_layer(
            experts=16, hidden_size=6144, intermediate_size=1536, top_k=4
        )
        cfg = layer.moe_runner_config
        prepared = self.adapter.prepare_flashinfer_cutile_nvfp4_weights(layer, cfg)
        # The scheduler accepts 16384; each FlashInfer call is bounded to 8192.
        self.enterContext(get_schedule().override(max_prefill_tokens=16384))
        self.assertEqual(self.adapter._get_max_num_tokens(), 16384)
        # Retain one runtime context and runner across every bucket/stream.
        # Resetting the context per step would hide cumulative allocations.
        runner = self.adapter.FlashInferCuTileRunnerCore(cfg)
        dispatch = _dispatch(layer, 8193)
        # Inputs are allocated once, so differences below include accumulated
        # workspaces and temporaries rather than different-sized input tensors.
        streams = (torch.cuda.current_stream(), torch.cuda.Stream())
        torch.cuda.synchronize()
        initial_allocated = torch.cuda.memory_allocated()
        initial_reserved = torch.cuda.memory_reserved()
        for stream_index, stream in enumerate(streams):
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for step, tokens in enumerate((8192, 8193, 8192)):
                    with self.subTest(stream=stream_index, step=step, tokens=tokens):
                        current = dispatch._replace(
                            hidden_states=dispatch.hidden_states[:tokens],
                            topk_output=dispatch.topk_output._replace(
                                topk_ids=dispatch.topk_output.topk_ids[:tokens],
                                topk_weights=dispatch.topk_output.topk_weights[:tokens],
                            ),
                        )
                        torch.cuda.synchronize()
                        allocated_before = torch.cuda.memory_allocated()
                        reserved_before = torch.cuda.memory_reserved()
                        torch.cuda.reset_peak_memory_stats()
                        output = runner.run_from_dispatch(
                            current, prepared, cfg
                        ).hidden_states
                        torch.cuda.synchronize()
                        allocated_after = torch.cuda.memory_allocated()
                        reserved_after = torch.cuda.memory_reserved()
                        peak_allocated = torch.cuda.max_memory_allocated()
                        peak_reserved = torch.cuda.max_memory_reserved()
                        self.assertEqual(output.shape, (tokens, cfg.hidden_size))
                        # Numerical checks live in the other cases: allocating
                        # a finite-check tensor here would pollute the next
                        # step's reserved-memory baseline.
                        print(
                            json.dumps(
                                {
                                    "stream": stream_index,
                                    "step": step,
                                    "tokens": tokens,
                                    "capacity": 16384,
                                    "flashinfer_call_ceiling": 8192,
                                    "allocated_mib": allocated_after / 2**20,
                                    "reserved_mib": reserved_after / 2**20,
                                    "step_allocated_delta_mib": (
                                        allocated_after - allocated_before
                                    )
                                    / 2**20,
                                    "step_reserved_delta_mib": (
                                        reserved_after - reserved_before
                                    )
                                    / 2**20,
                                    "cumulative_allocated_delta_mib": (
                                        allocated_after - initial_allocated
                                    )
                                    / 2**20,
                                    "cumulative_reserved_delta_mib": (
                                        reserved_after - initial_reserved
                                    )
                                    / 2**20,
                                    "peak_allocated_mib": peak_allocated / 2**20,
                                    "peak_reserved_mib": peak_reserved / 2**20,
                                    "step_peak_allocated_delta_mib": (
                                        peak_allocated - allocated_before
                                    )
                                    / 2**20,
                                },
                                sort_keys=True,
                            )
                        )
                        del output
            torch.cuda.current_stream().wait_stream(stream)
        # Do not empty the allocator or evict executions between steps: both
        # bucket and stream allocations must remain visible in this experiment.

    @unittest.skipUnless(
        envs.SGLANG_TEST_MINIMAX_LAYER30_DIR.get(),
        "set SGLANG_TEST_MINIMAX_LAYER30_DIR to test checkpoint weights",
    )
    def test_real_layer30_checkpoint(self):
        """The issue's first 32 experts with synthetic routing, not full service."""
        layer = _load_checkpoint_layer(envs.SGLANG_TEST_MINIMAX_LAYER30_DIR.get())
        for tokens in (1, 64, 412):
            with self.subTest(tokens=tokens):
                self._run_case(
                    layer, tokens, dispatch=_dispatch(layer, tokens, input_std=0.5)
                )

    @unittest.skipUnless(
        envs.SGLANG_TEST_MINIMAX_LAYER30_DIR.get()
        and envs.SGLANG_TEST_CUTILE_LARGE_BATCH.get(),
        "set checkpoint directory and SGLANG_TEST_CUTILE_LARGE_BATCH=1 for 128 experts",
    )
    def test_real_layer30_full_expert_count(self):
        layer = _load_checkpoint_layer(
            envs.SGLANG_TEST_MINIMAX_LAYER30_DIR.get(), experts=128
        )
        self._run_case(layer, 64, dispatch=_dispatch(layer, 64, input_std=0.5))

    @unittest.skipUnless(
        envs.SGLANG_TEST_MINIMAX_LAYER30_DIR.get()
        and envs.SGLANG_TEST_CUTILE_LARGE_BATCH.get(),
        "set checkpoint directory and large-batch flag for real chunk checks",
    )
    def test_real_layer30_chunk_boundaries(self):
        for experts in (32, 128):
            layer = _load_checkpoint_layer(
                envs.SGLANG_TEST_MINIMAX_LAYER30_DIR.get(), experts=experts
            )
            for tokens in (8191, 8192, 8193, 16384, 16385):
                with self.subTest(experts=experts, tokens=tokens):
                    self._run_case(
                        layer,
                        tokens,
                        capacity=32768,
                        dispatch=_dispatch(layer, tokens, input_std=0.5),
                        reference_tokens=64,
                        compare_unchunked=True,
                    )

    @unittest.skipUnless(
        envs.SGLANG_TEST_MINIMAX_LAYER30_DIR.get()
        and envs.SGLANG_TEST_CUTILE_LARGE_BATCH.get(),
        "set checkpoint directory and large-batch flag for real TP slice comparison",
    )
    def test_real_layer30_tp2_partition_sum_vs_tp1(self):
        # This isolates partition arithmetic; torchrun exercises actual NCCL.
        layer = _load_checkpoint_layer(envs.SGLANG_TEST_MINIMAX_LAYER30_DIR.get())
        for tokens in (64, 8193):
            with self.subTest(tokens=tokens):
                self._check_tp_weight_slices(layer, tokens, input_std=0.5)

    @unittest.skipUnless(
        envs.SGLANG_TEST_MINIMAX_LAYER30_DIR.get(),
        "set checkpoint directory for real-weight clamp coverage",
    )
    def test_real_layer30_clamp(self):
        layer = _load_checkpoint_layer(envs.SGLANG_TEST_MINIMAX_LAYER30_DIR.get())
        self._run_case(
            layer, 64, dispatch=_dispatch(layer, 64, input_std=4.0), expect_clamp=True
        )


if __name__ == "__main__":
    unittest.main()
