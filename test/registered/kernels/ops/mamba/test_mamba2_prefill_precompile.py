"""The Mamba2 prefill precompile sweep must leave nothing to compile later.

After ``sweep_mamba2_prefill`` runs, prefill batches of serving-like shapes
(any request count, any lengths, with and without initial states and state
tracking) must not load a single new Triton kernel binary. A load here is a
compile that would otherwise land on live traffic (issue #51628). The CPU
checks of the swept layouts live in
``test/registered/unit/layers/attention/mamba/test_mamba2_precompile.py``.
"""

import random
import sys

import pytest
import torch

from sglang.srt.layers.attention.mamba.mamba2_precompile import (
    Mamba2PrefillDims,
    Mamba2PrefillParams,
    run_mamba2_prefill,
    sweep_mamba2_prefill,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=240, stage="base-b-kernel-unit", runner_config="1-gpu-small")

CHUNK_SIZE = 256


def _small_model(*, device: torch.device):
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


def _serving_like_batches(*, rng: random.Random, count: int):
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
    dims, params = _small_model(device=device)
    with torch.inference_mode():
        assert (
            sweep_mamba2_prefill(dims=dims, params=params, use_triton_causal_conv=True)
            > 0
        )
    torch.cuda.synchronize()

    late_loads = []

    def on_load(module, function, name, metadata_group, hash):
        late_loads.append(name)

    load_hook.add(on_load)
    try:
        rng = random.Random(0)
        with torch.inference_mode():
            # A rare shape first: 16 logical chunks in one chunk, 17 tokens.
            batches = [((1,) * 15 + (2,), True, False)]
            batches += _serving_like_batches(rng=rng, count=60)
            for seq_lens, with_initial_states, with_track in batches:
                run_mamba2_prefill(
                    dims=dims,
                    params=params,
                    seq_lens=seq_lens,
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
