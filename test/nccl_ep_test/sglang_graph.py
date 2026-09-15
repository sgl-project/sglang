"""Graph backend and routing fixtures for production runner tests."""

from types import SimpleNamespace

import torch

from .oracle import RoutingBatch, validate_capacity


def backend_for(coordinator, capacity):
    from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
        FullCudaGraphBackend,
    )

    return FullCudaGraphBackend(
        SimpleNamespace(
            device_module=torch.cuda, model_runner=SimpleNamespace(tp_group=coordinator)
        ),
        nccl_ep_capacity=capacity,
    )


def runner_routing(batch, rank):
    """Synthetic GPU router feeding the real SGLang CUDA padding-mask kernel."""
    from sglang.srt.layers.moe.topk import TopKConfig, select_experts

    values = batch.input_ids
    pattern = (2.0 ** (torch.arange(2048, device=values.device) // 128 % 4)).bfloat16()
    x = values[:, None].bfloat16() * pattern
    if not values.numel():
        # Real MoE layers bypass select_experts on IDLE and use an empty
        # top-k output. The mask kernel requires a positive launch grid.
        return (
            x,
            torch.empty((0, 2), dtype=torch.int32, device=values.device),
            torch.empty((0, 2), dtype=torch.float32, device=values.device),
        )

    def router(**kwargs):
        first = (values + rank) % 4
        routes = torch.stack((first, (first + 2) % 4), dim=1).int()
        left = torch.where(values % 2 == 1, 0.75, 0.25)
        return torch.stack((left, 1 - left), dim=1), routes

    config = TopKConfig(
        top_k=2, custom_routing_function=router, allow_routed_experts_capture=False
    )
    topk = select_experts(
        x,
        torch.zeros(len(values), 4, device=values.device),
        config,
        num_token_non_padded=batch.num_token_non_padded,
    )
    return x, topk.topk_ids, topk.topk_weights


def runner_fixture(bucket, values, valid_rows, *, layer=0):
    """Literal CPU routing table, independent of the GPU router arithmetic."""
    tables = (
        {0: [0, 2], 1: [1, 3], 2: [2, 0], 4: [0, 2], 8: [0, 2]},
        {0: [1, 3], 1: [2, 0], 2: [3, 1], 4: [1, 3], 8: [1, 3]},
    )
    tokens, ids, weights = [], [], []
    pattern = torch.tensor([1, 2, 4, 8] * 4).repeat_interleave(128)
    for rank in range(2):
        padded = list(values[rank]) + [0] * (bucket - len(values[rank]))
        tokens.append((torch.tensor(padded)[:, None] * pattern * 2**layer).bfloat16())
        routes = torch.tensor([tables[rank][value] for value in padded])
        routes[valid_rows[rank] :] = -1
        ids.append(routes)
        weights.append(
            torch.tensor(
                [[0.75, 0.25] if value == 1 else [0.25, 0.75] for value in padded]
            )
        )
    batch = RoutingBatch(tuple(tokens), tuple(ids), tuple(weights), 4)
    validate_capacity(batch, bucket)
    return batch
