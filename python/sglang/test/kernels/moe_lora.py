"""Shared helpers for the MoE LoRA runner numerics tests."""

import torch


def cutedsl_ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0):
        return False
    try:
        import cuda.bindings.driver  # noqa: F401
        import cutlass  # noqa: F401
    except Exception:
        return False
    return True


def make_token_lora_mapping(traffic: str, num_tokens: int) -> torch.Tensor:
    if traffic == "active":
        return torch.zeros(num_tokens, dtype=torch.int32)
    if traffic == "mixed":
        # Batch preparation normalizes every inactive resident assignment
        # to -1 before any MoE layer consumes the mapping.
        pattern = torch.tensor([0, -1, -1, 0, -1, -1], dtype=torch.int32)
        return pattern.repeat((num_tokens + pattern.numel() - 1) // pattern.numel())[
            :num_tokens
        ].contiguous()
    if traffic == "base_only":
        # Config choice and graph shape stay fixed while every token follows
        # the base sentinel; one topology serves every batch.
        return torch.full((num_tokens,), -1, dtype=torch.int32)
    raise AssertionError(f"unknown traffic pattern {traffic}")


def standalone_output_allocation(runner, *, num_tokens, dtype, device):
    """Match eager output geometry without requiring a serving TP group."""
    return torch.empty((num_tokens, runner.hidden_size), dtype=dtype, device=device)


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).norm() / b.norm().clamp(min=1e-12)


def gather_pairs(rows: torch.Tensor, pair_to_row: torch.Tensor, pair_idx):
    """The provider rows (flat or [E, m_max, K] slabs) of the given pairs."""
    return rows.reshape(-1, rows.shape[-1])[pair_to_row[pair_idx].long()]


def check_marlin_lora_replay(
    monkeypatch,
    selected,
    provider,
    hidden,
    topk_ids,
    batch,
    w13,
    w2,
    *,
    tol,
    check_distinct_slots=False,
    check_base_only=False,
):
    """Check Marlin MoE LoRA against FP32 before and after changing graph inputs."""
    from sglang.kernels.ops.lora.moe.activation import ActivationFn
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.lora.moe.runner import MoeLoraRunner
    from sglang.srt.lora.utils import Phase
    from sglang.srt.runtime_context import get_context

    n_tokens, top_k = topk_ids.shape
    n_experts, _, intermediate = w2.shape
    slices = w13.shape[1] // intermediate
    rank = batch.down_lora_a.shape[-2]
    device = hidden.device
    phase = Phase.PREFILL if batch.is_prefill else Phase.DECODE
    mapping = batch.token_lora_mapping
    weights = torch.tensor([0.65, 0.35], device=device).repeat(n_tokens, 1)

    monkeypatch.setattr(MoeLoraRunner, "_allocate_output", standalone_output_allocation)
    runner = MoeLoraRunner(
        providers={selected.base_gemm_rows: provider},
        top_k=top_k,
        routed_scaling_factor=0.75,
        activation=ActivationFn.SILU if slices == 2 else ActivationFn.RELU2,
        is_gated=slices == 2,
    )
    runner.validate_plan(selected.plan, base_gemm_rows=selected.base_gemm_rows)
    runner.plans = {phase: selected}
    dispatch = StandardDispatchOutput(
        hidden_states=hidden,
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(
            topk_weights=weights,
            topk_ids=topk_ids,
            router_logits=torch.zeros((n_tokens, n_experts), device=device),
        ),
    )

    def reference(token_slots):
        pair_tokens = torch.arange(n_tokens, device=device).repeat_interleave(top_k)
        experts = topk_ids.flatten().long()
        routed = (experts >= 0).float()[:, None]
        experts = experts.clamp(min=0)
        outer_experts = 0 if batch.gate_up_lora_a.shape[1] == 1 else experts
        slots = token_slots[pair_tokens].long()
        adapted = (slots >= 0).float()[:, None]
        slots = slots.clamp(min=0)
        inputs = hidden.float()[pair_tokens]
        gate_up = torch.einsum("ph,pnh->pn", inputs, w13[experts])
        outer = torch.einsum(
            "ph,prh->pr", inputs, batch.gate_up_lora_a.float()[slots, outer_experts]
        )
        inner = batch.gate_up_lora_b.float()[slots, experts]
        delta = torch.cat(
            [
                torch.einsum(
                    "pir,pr->pi",
                    inner[:, s * intermediate : (s + 1) * intermediate],
                    outer[:, s * rank : (s + 1) * rank],
                )
                for s in range(slices)
            ],
            dim=1,
        )
        gate_up = gate_up + adapted * delta
        if slices == 2:
            act = (
                torch.nn.functional.silu(gate_up[:, :intermediate])
                * gate_up[:, intermediate:]
            )
        else:
            act = gate_up.relu().square()
        pairs = torch.einsum("pi,phi->ph", act, w2[experts])
        down_rank = torch.einsum(
            "pi,pri->pr", act, batch.down_lora_a.float()[slots, experts]
        )
        pairs = pairs + adapted * torch.einsum(
            "pr,phr->ph", down_rank, batch.down_lora_b.float()[slots, outer_experts]
        )
        pairs = pairs * routed
        return (pairs.view(n_tokens, top_k, -1) * weights[..., None]).sum(1) * 0.75

    def forward():
        with get_context().override_server_args():
            return runner.run(dispatch, batch).hidden_states

    output = forward()
    torch.cuda.synchronize()
    expected = reference(mapping)
    assert rel_l2(output.float(), expected) < tol, selected.name
    assert rel_l2(reference(torch.full_like(mapping, -1)), expected) > 0.1
    if check_distinct_slots:
        assert rel_l2(reference(mapping.clamp(max=0)), expected) > 0.02

    batch.use_cuda_graph = True
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def capture(run):
        with torch.cuda.stream(stream):
            for _ in range(2):
                run()
        torch.cuda.synchronize()
        runner.workspace.routes.clear()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = run()
        return graph, output

    graph, captured = capture(forward)
    graph.replay()
    torch.cuda.synchronize()
    assert rel_l2(captured.float(), expected) < tol, selected.name

    mapping.copy_(mapping.roll(1))
    topk_ids.copy_(topk_ids.flip(1))
    hidden.mul_(-0.5)
    graph.replay()
    torch.cuda.synchronize()
    assert rel_l2(captured.float(), reference(mapping)) < tol, selected.name

    if check_base_only:
        rows = "route_major" if batch.is_prefill else "expert_major"
        runner.providers[rows] = provider

        def forward_base_only():
            with get_context().override_server_args():
                return runner.run_base_only(
                    dispatch, is_prefill=batch.is_prefill, use_cuda_graph=True
                ).hidden_states

        base_graph, base_output = capture(forward_base_only)
        topk_ids[0, 0] = -1
        hidden.mul_(0.5)
        base_graph.replay()
        torch.cuda.synchronize()
        assert (
            rel_l2(base_output.float(), reference(torch.full_like(mapping, -1))) < tol
        )
