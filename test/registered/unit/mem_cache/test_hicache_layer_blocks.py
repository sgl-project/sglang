from types import SimpleNamespace

import pytest
import torch

from sglang.srt.managers.cache_controller import LayerDoneCounter
from sglang.srt.mem_cache.l2_transfer import L2Transfer, L2TransferEngine
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
@pytest.mark.parametrize("block_layers", [1, 2, 3, 8])
@pytest.mark.parametrize("captured", [False, True])
def test_block_events_order_each_restore_generation(block_layers, captured):
    layers = 5
    counter = LayerDoneCounter(layers, block_layers=block_layers)
    if captured:
        counter.init_graph_events()
    device = torch.full((layers, 4), 7.0, device="cuda")
    output = torch.zeros_like(device)
    host = torch.empty_like(device, device="cpu", pin_memory=True)
    forward = torch.cuda.Stream()
    engine = L2TransferEngine("direct")

    def load_layer(pool, host_ids, device_ids, layer, backend, *, is_draft):
        # Make a missing boundary wait observe the previous generation.
        torch.cuda._sleep(2_000_000)
        device[layer].copy_(host[layer], non_blocking=True)

    host_pool = SimpleNamespace(
        layer_num=layers,
        prepare_transfer_indices=lambda host_ids, device_ids, backend: (
            host_ids,
            device_ids,
        ),
        load_to_device_per_layer_physical=load_layer,
    )
    ids = torch.arange(4)
    transfer = L2Transfer(host_pool, None, ids, ids)

    def consume():
        for layer in range(layers):
            counter.wait_until(layer)
            output[layer].copy_(device[layer])

    torch.cuda._sleep(1)
    torch.cuda.synchronize()
    if captured:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=forward):
            consume()
        consume = graph.replay
    with torch.cuda.stream(forward):
        consume()
    forward.synchronize()
    assert torch.all(output == 7)

    # Wrap the producer ring and reuse captured event handles.
    for generation in range(1, 7):
        host.fill_(generation)
        producer_id = counter.update_producer()
        producer = counter.events[producer_id]
        completion = engine.submit_host_to_device(
            [transfer],
            transfer_layer_id_max=layers,
            layer_block_ends=counter.block_ends,
            on_layer_done=producer.complete,
        )
        producer.finish_event = completion.finish_event
        counter.set_consumer(producer_id)
        with torch.cuda.stream(forward):
            consume()
        forward.synchronize()
        torch.testing.assert_close(output.cpu(), host, rtol=0, atol=0)
        assert completion.finish_event.query()
