"""CUDA graph and numerical tests for direct Inkling decode LoRA kernels."""

from __future__ import annotations

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-c", runner_config="4-gpu-b200")

_B200_AVAILABLE = bool(
    torch.cuda.is_available()
    and torch.version.hip is None
    and torch.cuda.get_device_capability()[0] == 10
)

E = 256
TOPK = 6
RANK = 32
DTYPE = torch.bfloat16


def _make_topk_ids(
    num_tokens: int, *, device: torch.device, offset: int
) -> torch.Tensor:
    tokens = torch.arange(num_tokens, device=device, dtype=torch.int32)[:, None]
    routes = torch.arange(TOPK, device=device, dtype=torch.int32)[None, :]
    topk_ids = (tokens * 11 + routes * 17 + offset).remainder(E - 1)
    topk_ids[:, 0] = E - 1
    return topk_ids.contiguous()


def _gate_reference(
    shared: torch.Tensor,
    gate_b: torch.Tensor,
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
) -> torch.Tensor:
    intermediate = gate_b.shape[2] // 2
    gate_width = gate_b.shape[2]
    flat_ids = topk_ids.reshape(-1).to(torch.long)
    flat_slots = token_lora_mapping[:, None].expand(-1, TOPK).reshape(-1).to(torch.long)
    active = flat_slots >= 0
    routed_b = gate_b[flat_slots.clamp_min(0), flat_ids].to(torch.float32)
    if shared.ndim == 3:
        token = torch.arange(shared.shape[1], device=shared.device)
        selected_shared = shared[token_lora_mapping.clamp_min(0).long(), token]
    else:
        selected_shared = shared
    routed_shared = (
        selected_shared[:, None, :].expand(-1, TOPK, -1).reshape(-1, 2 * RANK)
    ).float()
    gate = torch.bmm(routed_b[:, :intermediate], routed_shared[:, :RANK, None]).squeeze(
        -1
    )
    up = torch.bmm(routed_b[:, intermediate:], routed_shared[:, RANK:, None]).squeeze(
        -1
    )
    result = torch.cat((gate, up), dim=1)
    result[~active] = 0
    return result.view(topk_ids.shape[0], TOPK, gate_width)


def _down_reference(
    activation: torch.Tensor,
    down_a: torch.Tensor,
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
) -> torch.Tensor:
    flat_ids = topk_ids.reshape(-1).to(torch.long)
    flat_slots = token_lora_mapping[:, None].expand(-1, TOPK).reshape(-1).to(torch.long)
    active = flat_slots >= 0
    routed_a = down_a[flat_slots.clamp_min(0), flat_ids].to(torch.float32)
    result = torch.bmm(routed_a, activation.to(torch.float32).unsqueeze(-1)).squeeze(-1)
    result[~active] = 0
    return result.view(topk_ids.shape[0], TOPK, RANK)


def _make_operands(
    num_tokens: int, intermediate: int, num_slots: int, device: torch.device
):
    gate_width = 2 * intermediate
    generator = torch.Generator(device=device).manual_seed(9000 + num_tokens)
    shared_shape = (
        (num_slots, num_tokens, RANK) if num_slots > 1 else (num_tokens, RANK)
    )
    gate_half = torch.randn(
        shared_shape, device=device, dtype=DTYPE, generator=generator
    )
    # Deliberately unrelated halves regression-protect the gated split.
    up_half = (
        torch.randn(shared_shape, device=device, dtype=DTYPE, generator=generator)
        * -0.75
        + 0.25
    )
    shared = torch.cat((gate_half, up_half), dim=-1).contiguous()
    gate_b = (
        torch.randn(
            (num_slots, E, gate_width, RANK),
            device=device,
            dtype=DTYPE,
            generator=generator,
        )
        / RANK**0.5
    ).contiguous()
    activation = torch.randn(
        (num_tokens * TOPK, intermediate),
        device=device,
        dtype=DTYPE,
        generator=generator,
    )
    down_a = (
        torch.randn(
            (num_slots, E, RANK, intermediate),
            device=device,
            dtype=DTYPE,
            generator=generator,
        )
        / intermediate**0.5
    ).contiguous()
    topk_ids = _make_topk_ids(num_tokens, device=device, offset=0)
    token_lora_mapping = torch.arange(
        num_tokens, device=device, dtype=torch.int32
    ).remainder(num_slots)
    if num_tokens > 1:
        token_lora_mapping[-1] = -1
    gate_output = torch.empty(
        (num_tokens, TOPK, gate_width), device=device, dtype=DTYPE
    )
    down_output = torch.empty((num_tokens, TOPK, RANK), device=device, dtype=DTYPE)
    return (
        shared,
        gate_b,
        activation,
        down_a,
        topk_ids,
        token_lora_mapping,
        gate_output,
        down_output,
    )


@pytest.mark.skipif(
    not _B200_AVAILABLE,
    reason="direct Inkling decode kernels are currently gated to B200",
)
@pytest.mark.parametrize(
    ("num_slots", "num_tokens", "intermediate"),
    [(1, 1, 384), (2, 4, 768), (3, 4, 384), (4, 32, 768)],
)
def test_direct_decode_cuda_graph_replay_and_base_weights(
    num_tokens: int, intermediate: int, num_slots: int
):
    from sglang.srt.lora.marlin_lora_temp.direct_decode import (
        direct_decode_down_shrink,
        direct_decode_gate_expand,
    )

    device = torch.device("cuda")
    (
        shared,
        gate_b,
        activation,
        down_a,
        topk_ids,
        token_lora_mapping,
        gate_output,
        down_output,
    ) = _make_operands(num_tokens, intermediate, num_slots, device)

    def invoke() -> None:
        direct_decode_gate_expand(
            shared, gate_b, topk_ids, token_lora_mapping, gate_output
        )
        direct_decode_down_shrink(
            activation, down_a, topk_ids, token_lora_mapping, down_output
        )

    # Compile and initialize CUDA state outside capture.
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            invoke()
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()

    assert int(topk_ids[0, 0]) == 255
    torch.testing.assert_close(
        gate_output.float(),
        _gate_reference(shared, gate_b, topk_ids, token_lora_mapping),
        rtol=0.03,
        atol=0.01,
    )
    torch.testing.assert_close(
        down_output.float(),
        _down_reference(activation, down_a, topk_ids, token_lora_mapping),
        rtol=0.03,
        atol=0.01,
    )

    stable_tensors = (
        shared,
        gate_b,
        activation,
        down_a,
        topk_ids,
        token_lora_mapping,
        gate_output,
        down_output,
    )
    stable_addresses = tuple(tensor.data_ptr() for tensor in stable_tensors)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        invoke()

    # Mutate every captured input in place. Replay must follow stable addresses
    # and read the new expert ids and values, including expert 255.
    shared.mul_(-0.5).add_(0.125)
    gate_b.mul_(0.75).add_(0.001)
    activation.mul_(0.625).sub_(0.03125)
    down_a.mul_(-0.875).add_(0.0005)
    topk_ids.copy_(_make_topk_ids(num_tokens, device=device, offset=29))
    token_lora_mapping.copy_((token_lora_mapping + 1).remainder(num_slots))
    if num_tokens > 1:
        token_lora_mapping[0] = -1
    gate_output.fill_(float("nan"))
    down_output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()

    assert tuple(tensor.data_ptr() for tensor in stable_tensors) == stable_addresses
    assert int(topk_ids[0, 0]) == 255
    torch.testing.assert_close(
        gate_output.float(),
        _gate_reference(shared, gate_b, topk_ids, token_lora_mapping),
        rtol=0.03,
        atol=0.01,
    )
    torch.testing.assert_close(
        down_output.float(),
        _down_reference(activation, down_a, topk_ids, token_lora_mapping),
        rtol=0.03,
        atol=0.01,
    )

    # Base/None replay retains the same captured pointers and zeroes the loaded
    # adapter weights in place. Both kernels must fully overwrite their output.
    gate_b.zero_()
    down_a.zero_()
    gate_output.fill_(float("nan"))
    down_output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()

    assert tuple(tensor.data_ptr() for tensor in stable_tensors) == stable_addresses
    assert torch.count_nonzero(gate_output).item() == 0
    assert torch.count_nonzero(down_output).item() == 0
    assert torch.isfinite(gate_output).all().item()
    assert torch.isfinite(down_output).all().item()
