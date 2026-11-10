"""MC32/64/128 correctness and typed-array transport; GPU tests require CUDA."""

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


@pytest.mark.parametrize("count", [32, 64, 128])
def test_array_transport_preserves_duplicates(count):
    vals = np.full((7, count), -2.5, dtype=np.float32)
    ids = np.full((7, count), 13, dtype=np.int32)
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
    v, i, prefix = build_flat_top_logprobs_arrays(list(vals), list(ids), count)
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
            mc_sample_counts=flags,
            array_logprob_flags=[bool(count) for count in flags],
            probe_seeds=torch.tensor(seeds),
            device="cpu",
            penalizer_orchestrator=SimpleNamespace(
                filter=lambda _: None, merge=lambda _: None
            ),
        )

    a = batch([32, 0, 128], [5, 6, 7])
    a.filter_batch([2, 0], torch.tensor([2, 0]))
    assert a.probe_seeds.tolist() == [7, 5] and a.mc_sample_counts == [128, 32]
    a.merge_batch(batch([64], [9]))
    assert a.probe_seeds.tolist() == [7, 5, 9] and a.mc_sample_counts == [128, 32, 64]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("vocab", [11, 4097, 100352])
@pytest.mark.parametrize("count", [32, 64, 128])
def test_gpu_exact_logprobs_and_rng(dtype, vocab, count):
    torch.manual_seed(18)
    x = torch.randn(5, vocab, device="cuda", dtype=dtype)
    # Exercise completely masked tiles and a last, partial tile.
    x[0, :] = -torch.inf
    x[0, -1] = 0
    seeds = torch.arange(5, device="cuda", dtype=torch.int64) + 19
    positions = torch.arange(5, device="cuda", dtype=torch.int64) + 101
    rng = torch.cuda.get_rng_state().clone()
    values, ids = compute_mc_scores(x, seeds, positions, num_samples=count)
    assert torch.equal(rng, torch.cuda.get_rng_state())
    assert ids.shape == (5, count)
    assert ids.dtype == torch.int32 and values.dtype == torch.float32
    torch.testing.assert_close(
        values,
        torch.log_softmax(x.float(), dim=-1).gather(1, ids.long()),
        atol=3e-5,
        rtol=1e-6,
    )
    assert (ids[0] == vocab - 1).all() and (values[0] == 0).all()
    v2, i2 = compute_mc_scores(x, seeds, positions, num_samples=count)
    assert torch.equal(ids, i2) and torch.equal(values, v2)
    _, changed = compute_mc_scores(x, seeds, positions + 1, num_samples=count)
    assert not torch.equal(ids[1:], changed[1:])
    # Request reordering must not change draws.
    order = torch.tensor([3, 0, 4, 1, 2], device="cuda")
    v3, i3 = compute_mc_scores(
        x[order], seeds[order], positions[order], num_samples=count
    )
    assert torch.equal(ids[order], i3)
    # Increasing the requested count preserves earlier draws.
    _, prefix = compute_mc_scores(x, seeds, positions, num_samples=32)
    assert torch.equal(ids[:, :32], prefix)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("count", [32, 64, 128])
def test_gpu_categorical_frequencies_and_independence(count):
    n = 8192
    p = torch.tensor([0.1, 0.2, 0.3, 0.4], device="cuda")
    x = p.log().expand(n, -1).contiguous()
    _, ids = compute_mc_scores(
        x,
        torch.arange(n, device="cuda", dtype=torch.int64),
        torch.full((n,), 7, device="cuda"),
        num_samples=count,
    )
    freq = torch.bincount(ids.flatten().long(), minlength=4).float() / (n * count)
    torch.testing.assert_close(freq, p, atol=0.004, rtol=0)
    # Independent replacement draws repeat with probability sum(p**2).
    repeat = (ids[:, :-1] == ids[:, 1:]).float().mean()
    assert abs(repeat.item() - (p * p).sum().item()) < 0.006


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("counts", [(32, 32), (64, 128), (128, 64)])
def test_gpu_sampler_keeps_response_rng_and_mixed_rows(counts):
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
        mc_sample_counts=[0] * n,
        array_logprob_flags=[False] * n,
    )
    positions = torch.arange(n, device="cuda", dtype=torch.int64) + 100
    torch.manual_seed(23)
    state = torch.cuda.get_rng_state()
    original = LogitsProcessorOutput(next_token_logits=x.clone())
    tokens = sampler(original, info, True, [0] * n, [None] * n, positions)
    after = torch.cuda.get_rng_state()
    torch.cuda.set_rng_state(state)
    info.mc_sample_counts = [counts[0], 0, counts[1], 0]
    info.array_logprob_flags = [True, True, True, False]
    output = LogitsProcessorOutput(next_token_logits=x.clone())
    actual = sampler(
        output, info, True, [counts[0], 64, counts[1], 0], [None] * n, positions
    )
    assert torch.equal(tokens, actual)
    assert torch.equal(after, torch.cuda.get_rng_state())
    reference = torch.log_softmax(x / 0.7, dim=-1)
    for row in (0, 2):
        ids = output.next_token_top_logprobs_idx[row]
        assert ids.numel() == info.mc_sample_counts[row]
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

    # All-MC batches may still contain different widths; each row keeps its
    # own shape and draws when the scheduler combines requests.
    info.mc_sample_counts = [32, 64, 128, 64]
    info.array_logprob_flags = [True] * n
    torch.cuda.set_rng_state(state)
    all_mc = LogitsProcessorOutput(next_token_logits=x.clone())
    actual = sampler(all_mc, info, True, info.mc_sample_counts, [None] * n, positions)
    assert torch.equal(tokens, actual)
    assert torch.equal(after, torch.cuda.get_rng_state())
    for row, count in enumerate(info.mc_sample_counts):
        values, ids = compute_mc_scores(
            x[row : row + 1] / 0.7,
            info.probe_seeds[row : row + 1],
            positions[row : row + 1],
            num_samples=count,
        )
        assert torch.equal(all_mc.next_token_top_logprobs_idx[row], ids[0])
        torch.testing.assert_close(all_mc.next_token_top_logprobs_val[row], values[0])


@pytest.mark.parametrize("phase", ["prefill", "decode"])
@pytest.mark.parametrize("count", [32, 64, 128])
def test_candidate_arrays_never_become_python_scalars(monkeypatch, phase, count):
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.managers.scheduler_components.batch_result_processor import (
        SchedulerBatchResultProcessor,
    )

    original = torch.Tensor.tolist

    def forbid_candidate_lists(tensor):
        assert tensor.numel() < 32, (
            "candidate tensors must never become Python scalar lists"
        )
        return original(tensor)

    monkeypatch.setattr(torch.Tensor, "tolist", forbid_candidate_lists)
    output = LogitsProcessorOutput(
        next_token_logits=None,
        next_token_logprobs=torch.tensor([-0.1, -0.2]),
        next_token_top_logprobs_val=torch.full((2, count), -1.0, dtype=torch.float32),
        next_token_top_logprobs_idx=torch.ones((2, count), dtype=torch.int32),
    )
    batch = SimpleNamespace(
        return_logprob=True,
        reqs=[
            SimpleNamespace(sampling_params=SamplingParams(array_output_logprobs=True))
            for _ in range(2)
        ],
        spec_algorithm=SimpleNamespace(is_none=lambda: True),
    )
    if phase == "prefill":
        SchedulerBatchResultProcessor.move_logprobs_to_cpu(
            None, batch=batch, logits_output=output
        )
    else:
        SchedulerBatchResultProcessor._normalize_decode_outputs(
            None,
            batch=batch,
            result=None,
            logits_output=output,
            next_token_ids=torch.tensor([1, 2]),
        )
    assert all(
        isinstance(row, np.ndarray) and row.dtype == np.float32
        for row in output.next_token_top_logprobs_val
    )
    assert all(
        isinstance(row, np.ndarray) and row.dtype == np.int32
        for row in output.next_token_top_logprobs_idx
    )


@pytest.mark.parametrize("count", [32, 64, 128])
def test_valid_probe_counts(count):
    SamplingParams(
        score_centering_samples=count, array_output_logprobs=True, sampling_seed=17
    ).verify(128)


@pytest.mark.parametrize("count", [0, 31, 33, 129])
def test_invalid_kernel_probe_counts(count):
    with pytest.raises(ValueError, match="num_samples"):
        compute_mc_scores(
            torch.empty(1, 10), torch.tensor([1]), torch.tensor([1]), num_samples=count
        )
