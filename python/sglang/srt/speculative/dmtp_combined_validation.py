"""Opt-in real-weight/artificial-hidden validation; never consumes first checks."""

import json
import time
from pathlib import Path

import torch

from sglang.srt.speculative.dmtp_head_conditioner_graph import conditioned_embeddings


@torch.no_grad()
def validate_combined(controller, directory):
    c, combo = controller, controller.t1_combined
    if combo is None or not combo.graphs:
        raise ValueError("Combined diagnostic requires captured graphs")
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    device, dtype = combo.hidden.device, combo.hidden.dtype
    rank = c.worker.tp_rank
    generator = torch.Generator(device=device).manual_seed(20261111)
    cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state(device)
    stream = torch.cuda.current_stream(device)
    checks = []

    def combined(hidden, block, temps, mask, coins):
        bs = block.shape[0]
        stream.wait_event(combo.done)
        combo.pending_ids = block
        combo._stage_inputs(bs, hidden[:, None], block[:, 0], temps, mask, coins)
        combo.graphs[bs].replay()
        result = tuple(value.clone() for value in combo.outputs[bs])
        combo.done.record(stream)
        combo.pending_ids = None
        return result

    def separate(hidden, block, temps, mask, coins):
        bs = block.shape[0]
        head, cond = c.t1_selector, c.conditioner_graphs
        stream.wait_event(head.done)
        head._stage_inputs(bs, hidden[:, None], block[:, 0], temps, mask, coins)
        head.graphs[bs].replay()
        tokens, ids, q = (value.clone() for value in head.outputs[bs])
        head.done.record(stream)
        ids_block = block.clone()
        ids_block[:, 1].copy_(tokens[:, 0])
        blocks = conditioned_embeddings(
            c.worker, combo.embedding, lambda a, b: a, ids_block, hidden
        )
        cond.blocks[:bs].copy_(blocks)
        cond.hidden[:bs].copy_(hidden)
        cond.graphs[bs].replay()
        return tokens, ids, q, cond.outputs[bs].clone(), blocks

    try:
        for bs in sorted(combo.graphs):
            for mode in ("sampled", "greedy", "mixed"):
                for turn in range(2):
                    hidden = torch.randn(
                        (bs, combo.hidden.shape[-1]),
                        device=device,
                        dtype=dtype,
                        generator=generator,
                    )
                    block = torch.randint(
                        c.vocab_size,
                        (bs, 8),
                        device=device,
                        dtype=torch.int64,
                        generator=generator,
                    )
                    block[:, 2:].fill_(int(c.worker._mask_token_id))
                    temps = torch.linspace(0.7, 1.3, bs, device=device)
                    mask = (
                        torch.ones(bs, dtype=torch.bool, device=device)
                        if mode == "greedy"
                        else (
                            torch.arange(bs, device=device) % 2 == 0
                            if mode == "mixed"
                            else torch.zeros(bs, dtype=torch.bool, device=device)
                        )
                    )
                    coins = torch.rand((bs, 1), device=device, generator=generator)
                    before = block.clone()
                    actual = combined(hidden, block, temps, mask, coins)
                    reference = separate(hidden, block, temps, mask, coins)
                    eager = c.t1_selector._reference(
                        hidden[:, None], block[:, 0], temps, mask, coins
                    )
                    for a, b in zip(actual, reference):
                        assert torch.equal(
                            a.contiguous().view(torch.uint8),
                            b.contiguous().view(torch.uint8),
                        )
                    for a, b in zip(actual[:3], eager):
                        assert torch.equal(
                            a.contiguous().view(torch.uint8),
                            b.contiguous().view(torch.uint8),
                        )
                    direct = c.conditioner_graphs.original(reference[4], hidden)
                    assert torch.equal(
                        actual[3].view(torch.uint8), direct.view(torch.uint8)
                    )
                    assert torch.equal(actual[3][:, 0], reference[4][:, 0])
                    assert torch.equal(actual[3][:, 2:], reference[4][:, 2:])
                    assert torch.equal(block, before)
                    checks.append(dict(B=bs, mode=mode, turn=turn, byte_exact=True))
                    if bs in (1, max(combo.graphs)) and turn == 0:
                        torch.save(
                            dict(
                                B=bs,
                                mode=mode,
                                artificial_hidden=True,
                                hidden=hidden.cpu(),
                                input_ids=block.cpu(),
                                coins=coins.cpu(),
                                actual=[v.cpu() for v in actual],
                                separate=[v.cpu() for v in reference[:4]],
                                eager_conditioner=direct.cpu(),
                            ),
                            path / f"rank{rank}_B{bs}_{mode}.pt",
                        )
        timing = {}
        for bs in sorted({1, min(16, max(combo.graphs)), max(combo.graphs)}):
            hidden = torch.randn(
                (bs, combo.hidden.shape[-1]),
                device=device,
                dtype=dtype,
                generator=generator,
            )
            block = torch.randint(
                c.vocab_size,
                (bs, 8),
                device=device,
                dtype=torch.int64,
                generator=generator,
            )
            block[:, 2:].fill_(int(c.worker._mask_token_id))
            temps = torch.ones(bs, device=device)
            mask = torch.zeros(bs, dtype=torch.bool, device=device)
            coins = torch.rand((bs, 1), device=device, generator=generator)
            samples = {"separate": [], "combined": []}
            for i in range(16):
                order = (("separate", separate), ("combined", combined))
                if i % 2:
                    order = tuple(reversed(order))
                for name, function in order:
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                        enable_timing=True
                    )
                    host = time.perf_counter()
                    begin.record()
                    function(hidden, block, temps, mask, coins)
                    end.record()
                    end.synchronize()
                    samples[name].append(
                        dict(
                            stream_ms=begin.elapsed_time(end),
                            host_sync_ms=(time.perf_counter() - host) * 1000,
                        )
                    )
            timing[bs] = samples
        assert (
            not combo.checked
            and not c.t1_selector.checked
            and not c.conditioner_graphs.verified
        )
        (path / f"rank{rank}.json").write_text(
            json.dumps(
                dict(
                    kind="real HF weights, artificial hidden; not teacher or E2E throughput",
                    checks=checks,
                    count=len(checks),
                    first_use_checks_unconsumed=True,
                    timing_scope="captured operators + staging/clones; excludes guards, RNG draw/broadcast",
                    timing=timing,
                ),
                indent=2,
            )
        )
        print("DMTP_SAMPLED_COMBINED_FIXTURE_PASS", rank, len(checks), flush=True)
    finally:
        combo.pending_ids = None
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(gpu_rng, device)
