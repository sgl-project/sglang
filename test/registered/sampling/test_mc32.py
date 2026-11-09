"""MC32 correctness and typed-array transport; GPU tests require CUDA."""

from types import SimpleNamespace
from typing import List, Optional

import msgspec
import numpy as np
import pytest
import torch

from sglang.srt.layers.mc_logprobs import compute_mc_scores
from sglang.srt.managers.io_struct import build_flat_top_logprobs_arrays
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.utils.msgpack_utils import dec_hook, enc_hook, ext_hook


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(score_centering_samples=31),
        dict(score_centering_samples=32),
        dict(
            score_centering_samples=32,
            array_output_logprobs=True,
            sampling_seed=17,
            top_p=0.9,
        ),
        dict(
            score_centering_samples=32,
            array_output_logprobs=True,
            sampling_seed=17,
            temperature=0,
        ),
        dict(score_centering_samples=32, array_output_logprobs=True),
    ],
)
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        SamplingParams(**kwargs).verify(128)


def test_array_transport_preserves_duplicates():
    vals = np.full((7, 32), -2.5, dtype=np.float32)
    ids = np.full((7, 32), 13, dtype=np.int32)
    for value, typ in (
        (vals, Optional[List[Optional[np.ndarray]]]),
        (ids, Optional[List[Optional[np.ndarray]]]),
    ):
        # The exact annotated fields used on both scheduler/detokenizer hops.
        decoder = msgspec.msgpack.Decoder(
            type=typ, dec_hook=dec_hook, ext_hook=ext_hook
        )
        packet = msgspec.msgpack.encode([value], enc_hook=enc_hook)
        decoded = decoder.decode(packet)[0]
        assert isinstance(decoded, np.ndarray) and decoded.dtype == value.dtype
        np.testing.assert_array_equal(decoded, value)
        assert len(packet) < value.nbytes + 100
    v, i, prefix = build_flat_top_logprobs_arrays(list(vals), list(ids), 32)
    assert prefix == 0 and i.dtype == np.int32 and v.dtype == np.float32
    np.testing.assert_array_equal(i, ids)


def test_probe_batch_filter_merge():
    def batch(flags, seeds):
        n = len(flags)
        return SamplingBatchInfo(
            temperatures=torch.ones(n, 1),
            top_ps=torch.ones(n),
            top_ks=torch.full((n,), 1 << 30),
            min_ps=torch.zeros(n),
            is_all_greedy=False,
            is_any_greedy=False,
            need_top_p_sampling=False,
            need_top_k_sampling=False,
            need_min_p_sampling=False,
            vocab_size=128,
            mc32_flags=flags,
            array_logprob_flags=flags.copy(),
            probe_seeds=torch.tensor(seeds),
            device="cpu",
            penalizer_orchestrator=SimpleNamespace(
                filter=lambda _: None, merge=lambda _: None
            ),
        )

    a = batch([True, False, True], [5, 6, 7])
    a.filter_batch([2, 0], torch.tensor([2, 0]))
    assert a.probe_seeds.tolist() == [7, 5] and a.mc32_flags == [True, True]
    a.merge_batch(batch([False], [9]))
    assert a.probe_seeds.tolist() == [7, 5, 9] and a.mc32_flags == [True, True, False]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("vocab", [11, 4097, 100352])
def test_gpu_exact_logprobs_and_rng(dtype, vocab):
    torch.manual_seed(18)
    x = torch.randn(5, vocab, device="cuda", dtype=dtype)
    # Exercise completely masked tiles and a last, partial tile.
    x[0, :] = -torch.inf
    x[0, -1] = 0
    seeds = torch.arange(5, device="cuda", dtype=torch.int64) + 19
    positions = torch.arange(5, device="cuda", dtype=torch.int64) + 101
    rng = torch.cuda.get_rng_state().clone()
    values, ids = compute_mc_scores(x, seeds, positions)
    assert torch.equal(rng, torch.cuda.get_rng_state())
    assert ids.dtype == torch.int32 and values.dtype == torch.float32
    torch.testing.assert_close(
        values,
        torch.log_softmax(x.float(), dim=-1).gather(1, ids.long()),
        atol=3e-5,
        rtol=1e-6,
    )
    assert (ids[0] == vocab - 1).all() and (values[0] == 0).all()
    v2, i2 = compute_mc_scores(x, seeds, positions)
    assert torch.equal(ids, i2) and torch.equal(values, v2)
    _, changed = compute_mc_scores(x, seeds, positions + 1)
    assert not torch.equal(ids[1:], changed[1:])
    # Request reordering must not change draws.
    order = torch.tensor([3, 0, 4, 1, 2], device="cuda")
    v3, i3 = compute_mc_scores(x[order], seeds[order], positions[order])
    assert torch.equal(ids[order], i3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_categorical_frequencies_and_independence():
    n = 8192
    p = torch.tensor([0.1, 0.2, 0.3, 0.4], device="cuda")
    x = p.log().expand(n, -1).contiguous()
    _, ids = compute_mc_scores(
        x,
        torch.arange(n, device="cuda", dtype=torch.int64),
        torch.full((n,), 7, device="cuda"),
    )
    freq = torch.bincount(ids.flatten().long(), minlength=4).float() / (n * 32)
    torch.testing.assert_close(freq, p, atol=0.004, rtol=0)
    # Independent replacement draws repeat with probability sum(p**2).
    repeat = (ids[:, :-1] == ids[:, 1:]).float().mean()
    assert abs(repeat.item() - (p * p).sum().item()) < 0.006


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_sampler_keeps_response_rng_and_mixed_rows():
    # Exercise the real forward and PyTorch response sampler, without model/TP initialization.
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.layers.logprob_processor import OutputLogprobProcessor
    from sglang.srt.layers.sampler import Sampler

    sampler = Sampler.__new__(Sampler)
    torch.nn.Module.__init__(sampler)
    sampler.enable_deterministic = False
    sampler.use_log_softmax_logprob = False
    sampler.use_ascend_backend = False
    sampler.output_logprob_processor = OutputLogprobProcessor()
    n, vocab = 4, 8193
    x = torch.randn(n, vocab, device="cuda")
    info = SamplingBatchInfo(
        temperatures=torch.full((n, 1), 0.7, device="cuda"),
        top_ps=torch.ones(n, device="cuda"),
        top_ks=torch.full((n,), 1 << 30, device="cuda"),
        min_ps=torch.zeros(n, device="cuda"),
        is_all_greedy=False,
        is_any_greedy=False,
        need_top_p_sampling=False,
        need_top_k_sampling=False,
        need_min_p_sampling=False,
        vocab_size=vocab,
        probe_seeds=torch.arange(n, device="cuda", dtype=torch.int64),
        mc32_flags=[False] * n,
        array_logprob_flags=[False] * n,
    )
    positions = torch.arange(n, device="cuda", dtype=torch.int64) + 100
    torch.manual_seed(23)
    state = torch.cuda.get_rng_state()
    original = LogitsProcessorOutput(next_token_logits=x.clone())
    tokens = sampler(original, info, True, [0] * n, [None] * n, positions)
    after = torch.cuda.get_rng_state()
    torch.cuda.set_rng_state(state)
    info.mc32_flags = [True, False, True, False]
    info.array_logprob_flags = [True, True, True, False]
    output = LogitsProcessorOutput(next_token_logits=x.clone())
    actual = sampler(output, info, True, [32, 64, 32, 0], [None] * n, positions)
    assert torch.equal(tokens, actual)
    assert torch.equal(after, torch.cuda.get_rng_state())
    reference = torch.log_softmax(x / 0.7, dim=-1)
    for row in (0, 2):
        ids = output.next_token_top_logprobs_idx[row]
        torch.testing.assert_close(
            output.next_token_top_logprobs_val[row],
            reference[row, ids.long()],
            atol=3e-5,
            rtol=1e-6,
        )
    torch.testing.assert_close(
        output.next_token_top_logprobs_val[1],
        reference[1].topk(64).values,
        atol=3e-5,
        rtol=1e-6,
    )
