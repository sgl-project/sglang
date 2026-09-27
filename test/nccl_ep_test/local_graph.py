"""Single-GPU experiment: changing tensor contents preserves Graph topology.

This contains no EP operations and provides no evidence for EP graph safety.
"""

from itertools import product

import torch

from .environment import Unavailable
from .oracle import assert_combine_matches, make_fixture


def exercise():
    if not torch.cuda.is_available():
        raise Unavailable("The non-EP Graph experiment requires one CUDA device")
    graphs, inputs, outputs, addresses = {}, {}, {}, {}
    shared = (
        torch.empty(32, 2048, dtype=torch.bfloat16, device="cuda"),
        torch.empty(32, 2, dtype=torch.int64, device="cuda"),
        torch.empty(32, 2, dtype=torch.float32, device="cuda"),
    )
    shared_views = {}
    stream = torch.cuda.Stream()
    for bucket in (8, 16, 32):
        batch = make_fixture(bucket)
        static = tuple(
            items[0].cuda() for items in (batch.tokens, batch.expert_ids, batch.weights)
        )
        x, ids, weights = static
        shared_x, shared_ids, shared_weights = (tensor[:bucket] for tensor in shared)
        shared_views[str(bucket)] = [
            tensor.data_ptr() for tensor in (shared_x, shared_ids, shared_weights)
        ]

        def forward():
            shared_x.copy_(x)
            shared_ids.copy_(ids)
            shared_weights.copy_(weights)
            factors = (shared_ids + 1).float() * (shared_ids >= 0)
            return (
                shared_x.float()[:, None, :]
                * factors[:, :, None]
                * shared_weights[:, :, None]
            ).sum(1)

        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                forward()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result = forward()
        graphs[bucket], inputs[bucket], outputs[bucket] = graph, static, result
        addresses[bucket] = [tensor.data_ptr() for tensor in static]
    checked = 0
    for change, case, step, bucket in product(
        ("tokens", "routing", "weights", "all"),
        ("balanced", "hotspot", "padding", "empty_rank", "all_masked"),
        (0, 1, 0),
        (8, 16, 32),
    ):
        batch = make_fixture(bucket, case=case, step=step, change=change)
        static = inputs[bucket]
        for target, items in zip(
            static, (batch.tokens, batch.expert_ids, batch.weights)
        ):
            target.copy_(items[0])
        assert addresses[bucket] == [tensor.data_ptr() for tensor in static]
        graphs[bucket].replay()
        assert_combine_matches(batch, 0, outputs[bucket])
        checked += 1
    torch.cuda.synchronize()
    return {
        "checked": checked,
        "ep_tested": False,
        "changed_components": ["tokens", "routing", "weights", "all"],
        "static_addresses": addresses,
        "shared_buffer_views": shared_views,
    }
