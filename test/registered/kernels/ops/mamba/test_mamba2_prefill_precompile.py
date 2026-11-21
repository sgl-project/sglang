"""The Mamba2 prefill precompile sweep must leave nothing to compile later.

After ``sweep_mamba2_prefill`` runs, prefill batches of serving-like shapes
(any request count, any lengths, with and without initial states and state
tracking) must not load a single new Triton kernel binary. A load here is a
compile that would otherwise land on live traffic (issue #51628).
"""

import itertools
import random
import sys

import pytest
import torch

from sglang.srt.layers.attention.mamba.mamba2_precompile import (
    Mamba2PrefillDims,
    Mamba2PrefillParams,
    _tracked_requests,
    prefill_layouts,
    run_mamba2_prefill,
    sweep_mamba2_prefill,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=240, stage="base-b-kernel-unit", runner_config="1-gpu-small")

CHUNK_SIZE = 256


def _int_class(value: int) -> str:
    # The integer classes Triton specializes kernel arguments on.
    if value == 1:
        return "one"
    return "div16" if value % 16 == 0 else "other"


def test_layouts_cover_integer_classes():
    token_classes, chunk_classes, request_classes = set(), set(), set()
    for seq_lens in prefill_layouts(CHUNK_SIZE):
        num_tokens = sum(seq_lens)
        token_classes.add(_int_class(num_tokens))
        chunk_classes.add(_int_class(-(-num_tokens // CHUNK_SIZE)))
        request_classes.add(_int_class(len(seq_lens)))
    expected = {"one", "div16", "other"}
    assert token_classes == expected
    assert chunk_classes == expected
    assert request_classes == expected


def test_layouts_cover_logical_chunk_counts():
    # _state_passing_fwd_kernel takes the token count, the chunk count and,
    # with initial states, the logical chunk count as separate integer
    # arguments, so the sweep must cover combinations of their classes, not
    # just each class on its own.
    covered = set()
    for seq_lens in prefill_layouts(CHUNK_SIZE):
        num_tokens = sum(seq_lens)
        num_chunks = -(-num_tokens // CHUNK_SIZE)
        ends = itertools.accumulate(seq_lens[:-1])
        num_logical = num_chunks + sum(end % CHUNK_SIZE != 0 for end in ends)
        covered.add(
            (_int_class(num_logical), _int_class(num_chunks), _int_class(num_tokens))
        )
    # (logical chunks, chunks, tokens). The first entry is a 5-request batch
    # of 2988 tokens: 12 chunks plus 4 boundaries inside chunks.
    required = {
        ("div16", "other", "other"),
        ("div16", "other", "div16"),
        ("other", "other", "other"),
        ("div16", "div16", "other"),
        ("other", "div16", "div16"),
        ("div16", "one", "div16"),
        ("other", "one", "other"),
        ("one", "one", "one"),
    }
    assert required <= covered


def test_layouts_exercise_state_tracking():
    tracked_counts = {
        len(_tracked_requests(seq_lens, CHUNK_SIZE)[0])
        for seq_lens in prefill_layouts(CHUNK_SIZE)
    }
    assert {1, 2} <= tracked_counts


def test_tracked_requests_start_off_grid():
    seq_idx, end_locs = _tracked_requests((100, 300, 50, 512), CHUNK_SIZE)
    # Request 1 starts at 100 and covers one full chunk; request 2 is shorter
    # than a chunk; request 3 starts at 450 and covers two chunks.
    assert seq_idx == [1, 3]
    assert end_locs == [100 + 256, 450 + 512]


def _small_model(device: torch.device):
    num_heads, head_dim, n_groups, state_size, conv_kernel = 8, 64, 2, 64, 4
    intermediate_size = num_heads * head_dim
    conv_dim = intermediate_size + 2 * n_groups * state_size
    dims = Mamba2PrefillDims(
        num_heads=num_heads,
        head_dim=head_dim,
        n_groups=n_groups,
        state_size=state_size,
        intermediate_size=intermediate_size,
        conv_dim=conv_dim,
        conv_kernel=conv_kernel,
        chunk_size=CHUNK_SIZE,
        dtype=torch.bfloat16,
        conv_state_dtype=torch.bfloat16,
        ssm_state_dtype=torch.float32,
    )
    params = Mamba2PrefillParams(
        A=-torch.rand(num_heads, dtype=torch.float32, device=device),
        D=torch.ones(num_heads, dtype=torch.bfloat16, device=device),
        dt_bias=torch.ones(num_heads, dtype=torch.bfloat16, device=device),
        conv_weight=torch.randn(
            conv_dim, conv_kernel, dtype=torch.bfloat16, device=device
        ),
        conv_bias=torch.zeros(conv_dim, dtype=torch.bfloat16, device=device),
        activation="silu",
    )
    return dims, params


def _serving_like_batches(rng: random.Random, count: int):
    max_tokens = 16 * CHUNK_SIZE  # the default chunked prefill size
    for _ in range(count):
        num_seqs = rng.choice([1, 1, 2, 3, 5, 8, 16, 17, 31])
        seq_lens = []
        budget = max_tokens
        for i in range(num_seqs):
            remaining = num_seqs - i - 1
            seq_len = rng.randint(1, max(1, min(3 * CHUNK_SIZE, budget - remaining)))
            seq_lens.append(seq_len)
            budget -= seq_len
            if budget <= remaining:
                break
        yield tuple(seq_lens), rng.random() < 0.5, rng.random() < 0.5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_no_kernel_loads_after_sweep():
    triton_knobs = pytest.importorskip("triton.knobs")
    load_hook = getattr(
        getattr(triton_knobs, "runtime", None), "kernel_load_start_hook", None
    )
    if load_hook is None or not hasattr(load_hook, "remove"):
        pytest.skip("this Triton build has no kernel load hook")

    device = torch.device("cuda")
    dims, params = _small_model(device)
    with torch.inference_mode():
        assert sweep_mamba2_prefill(dims, params, use_triton_causal_conv=True) > 0
    torch.cuda.synchronize()

    late_loads = []

    def on_load(module, function, name, metadata_group, hash):
        late_loads.append(name)

    load_hook.add(on_load)
    try:
        rng = random.Random(0)
        with torch.inference_mode():
            for seq_lens, with_initial_states, with_track in _serving_like_batches(
                rng, count=60
            ):
                run_mamba2_prefill(
                    dims,
                    params,
                    seq_lens,
                    with_initial_states=with_initial_states,
                    with_track=with_track,
                    use_triton_causal_conv=True,
                )
        torch.cuda.synchronize()
    finally:
        load_hook.remove(on_load)

    assert late_loads == [], (
        f"kernels loaded after the sweep: {sorted(set(late_loads))}"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
