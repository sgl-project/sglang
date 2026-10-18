"""Deterministic NCCL integration gate for distributed input/Top-N logprobs.

Run with a GPU lease:
  PYTHONPATH=python torchrun --standalone --nproc-per-node=4 \
    test/manual/test_distributed_input_logprobs_tp.py
"""

import json
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.logprob_processor import (
    DistributedLogprobContext,
    InputLogprobProcessor,
)


class Group:
    def __init__(self):
        self.device_group = dist.group.WORLD

    def all_reduce(self, value):
        dist.all_reduce(value)
        return value

    def all_gather(self, value, dim):
        outputs = [torch.empty_like(value) for _ in range(dist.get_world_size())]
        dist.all_gather(outputs, value)
        return torch.cat(outputs, dim=dim)


def check(vocab, dtype, chunk, topk):
    rank, world = dist.get_rank(), dist.get_world_size()
    generator = torch.Generator(device="cuda").manual_seed(314159)
    full = torch.randn((9, vocab), device="cuda", generator=generator).to(dtype).float()
    width = (vocab + world - 1) // world
    start, end = rank * width, min((rank + 1) * width, vocab)
    local = torch.full((9, width), 1e4, device="cuda", dtype=dtype)
    local[:, : end - start] = full[:, start:end]
    group = Group()
    samples = torch.tensor([3, 7, 8], device="cuda")
    inputs = torch.arange(8, device="cuda")
    targets = torch.tensor([0, vocab - 1, width, 3, 7, 9, 11, 13], device="cuda")
    probes = [[0, width, vocab - 1], [], None]
    metadata = SimpleNamespace(
        sample_indices_cpu=[3, 7, 8],
        input_logprob_indices_cpu=list(range(8)),
        extend_return_top_logprob=topk,
        extend_token_ids_logprob=True,
        top_logprobs_nums=[topk, 0, 3] if topk else [0, 0, 0],
        extend_logprob_pruned_lens_cpu=[4, 4, 0],
        extend_input_logprob_token_ids_gpu=targets,
        token_ids_logprobs=probes,
    )
    calls = []

    def gather_sampled(value):
        calls.append(value.shape[0])
        return group.all_gather(value, 1)[:, :vocab]

    context = DistributedLogprobContext(
        tp_group=group,
        vocab_start_index=start,
        vocab_end_index=end,
        vocab_size=vocab,
        get_local_logits_fn=lambda states, *_: states,
        gather_sampled_logits_fn=gather_sampled,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("gathered prompt-logit path was used")

    processor = InputLogprobProcessor(vocab_size=vocab)
    processor.logprobs_chunk_size = chunk
    result, sampled = processor.forward(
        local,
        samples,
        inputs,
        [0] * 4 + [1] * 4 + [2],
        None,
        forbidden,
        metadata,
        distributed_context=context,
    )
    assert result.input_copy_done is not None, "CUDA probes must use staged spans"
    output = LogitsProcessorOutput(next_token_logits=sampled)
    result.write_input_to(output)
    output.finalize_input_logprobs()
    reference = full.log_softmax(-1)
    assert sampled.dtype == torch.float32
    torch.testing.assert_close(
        result.token_logprobs, reference[inputs, targets], rtol=1e-5, atol=2e-6
    )
    torch.testing.assert_close(sampled, full[samples], rtol=0, atol=0)
    assert sum(calls) == len(samples), calls
    torch.testing.assert_close(
        torch.tensor(result.token_ids_logprobs_val[0], device="cuda"),
        reference[:4, probes[0]],
        rtol=1e-5,
        atol=2e-6,
    )
    assert result.token_ids_logprobs_val[1:] == [[[]] * 4, []]
    assert result.token_ids_logprobs_idx == [[probes[0]] * 4, [[]] * 4, []]
    if topk:
        ids = torch.tensor(result.top_logprobs_idx[0], device="cuda")
        values = torch.tensor(result.top_logprobs_val[0], device="cuda")
        assert bool(((ids >= 0) & (ids < vocab)).all())
        assert all(len(set(row)) == topk for row in ids.tolist())
        torch.testing.assert_close(
            values, reference[:4].gather(1, ids), rtol=1e-5, atol=2e-6
        )
        torch.testing.assert_close(
            values, reference[:4].topk(topk).values, rtol=1e-5, atol=2e-6
        )
        assert result.top_logprobs_val[1:] == [[[]] * 4, []]


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    try:
        count = 0
        for vocab in (131, 154881):
            for dtype in (torch.float32, torch.bfloat16, torch.float16):
                for chunk in (1, 4, 32):
                    for topk in (0, 5, 20):
                        check(vocab, dtype, chunk, topk)
                        count += 1
        torch.cuda.synchronize()
        dist.barrier()
        if dist.get_rank() == 0:
            print(
                json.dumps(
                    {"status": "PASS", "cases": count, "tp": dist.get_world_size()}
                ),
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
