import unittest

import torch
import triton

from sglang.kernels.ops.mamba.triton_ops import (
    mamba_chunk_scan_combined,
    ssd_bmm,
    ssd_chunk_scan,
    ssd_chunk_state,
    ssd_state_passing,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")

# Mamba2 shapes with one group, as Falcon-H1-7B serves them at tp 1. The defect
# this guards showed from a few dozen tokens on, in the second chunk and beyond.
DIMS = dict(nheads=24, headdim=128, ngroups=1, dstate=256, chunk_size=256)
SEQLENS = (89, 607)
# bf16 inputs through fp32 accumulation: the reference below is the same
# recurrence in fp32, so a correct kernel lands well inside this, while an
# output that is zero, stale or overflowed does not.
TOL = 5e-2


def _spec(config):
    """A config's identity, without importing a helper the fix adds."""
    return (tuple(sorted(config.kwargs.items())), config.num_warps, config.num_stages)


def _prune(kernel):
    """The kernel's `early_config_prune`, wherever this Triton keeps it."""
    prune = getattr(kernel, "early_config_prune", None)
    if prune is None:
        prune = (getattr(kernel, "prune_configs_by", None) or {}).get(
            "early_config_prune"
        )
    assert prune is not None, "this kernel's autotuner exposes no early_config_prune"
    return prune


def _named(**over):
    named = dict(
        chunk_size=DIMS["chunk_size"],
        hdim=DIMS["headdim"],
        dstate=DIMS["dstate"],
        K=DIMS["dstate"],
        dim=DIMS["nheads"] * DIMS["headdim"] * DIMS["dstate"],
        nheads=DIMS["nheads"],
        IS_CAUSAL=True,
    )
    named.update(over)
    return named


def _selectable(kernel):
    """The configs this kernel's autotuner may choose between for these shapes."""
    try:
        return list(_prune(kernel)(list(kernel.configs), _named()))
    except KeyError:
        return list(kernel.configs)


class _Pinned:
    """Run kernels with one config each: no benchmarking, no cached winner."""

    def __init__(self, *pairs):
        self.pairs = pairs

    def __enter__(self):
        self.saved = [(k, k.configs, dict(k.cache)) for k, _ in self.pairs]
        for kernel, config in self.pairs:
            kernel.configs = [config]
            kernel.cache.clear()

    def __exit__(self, *exc):
        for kernel, configs, cache in self.saved:
            kernel.configs = configs
            kernel.cache.clear()
            kernel.cache.update(cache)


def _siblings():
    """One fixed config for each kernel the scan also runs, so a measurement is
    attributed to the scan config under test and not to a benchmark that
    happened to pick differently. A wrong sibling config does not pass
    unnoticed: the reference comparison below covers the whole scan."""
    return [
        (k, _selectable(k)[0])
        for k in (
            ssd_chunk_state._chunk_state_fwd_kernel,
            ssd_bmm._bmm_chunk_fwd_kernel,
            ssd_state_passing._state_passing_fwd_kernel,
        )
    ]


def _inputs(seqlen, *, dtype=torch.bfloat16, seed=1107):
    g = torch.Generator(device="cpu").manual_seed(seed)

    def rand(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float32).to(
            device="cuda", dtype=dtype
        )

    nheads, headdim, ngroups, dstate = (
        DIMS[k] for k in ("nheads", "headdim", "ngroups", "dstate")
    )
    return {
        "x": rand(1, seqlen, nheads, headdim),
        "dt": rand(1, seqlen, nheads),
        # A is -exp(A_log) in the model: strictly negative, so the state decays.
        "A": (-torch.rand(nheads, generator=g, dtype=torch.float32) - 0.1).to("cuda"),
        "B": rand(1, seqlen, ngroups, dstate),
        "C": rand(1, seqlen, ngroups, dstate),
        "D": torch.rand(nheads, generator=g, dtype=torch.float32).to("cuda"),
        "dt_bias": torch.rand(nheads, generator=g, dtype=torch.float32).to("cuda"),
        "seqlen": seqlen,
    }


def _scan(inputs, *, fill=0.0):
    """The chunk scan as the Mamba2 mixer calls it, into a prefilled buffer."""
    seqlen = inputs["seqlen"]
    out = torch.full_like(inputs["x"], fill)
    mamba_chunk_scan_combined(
        inputs["x"],
        inputs["dt"],
        inputs["A"],
        inputs["B"],
        inputs["C"],
        chunk_size=DIMS["chunk_size"],
        D=inputs["D"],
        z=None,
        dt_bias=inputs["dt_bias"],
        seq_idx=torch.zeros(1, seqlen, dtype=torch.int32, device="cuda"),
        cu_seqlens=torch.tensor([0, seqlen], dtype=torch.int32, device="cuda"),
        initial_states=None,
        chunk_indices=None,
        chunk_offsets=None,
        dt_softplus=True,
        dt_limit=(0.0, float("inf")),
        out=out,
        return_varlen_states=True,
        state_dtype=torch.float32,
    )
    torch.cuda.synchronize()
    return out


def _reference(inputs):
    """The same selective-scan recurrence, stepped in fp32: for each head,
    S <- S * exp(dt * A) + dt * x B^T and y <- S C + D x. Independent of the
    kernel's chunking, so it catches an output that is zero, stale or
    overflowed as well as one that is merely imprecise."""
    x, dt, A = inputs["x"][0].float(), inputs["dt"][0].float(), inputs["A"].float()
    B, C = inputs["B"][0].float(), inputs["C"][0].float()
    D, dt_bias = inputs["D"].float(), inputs["dt_bias"].float()
    seqlen, nheads, headdim = x.shape
    per_group = nheads // B.shape[1]
    dt = torch.nn.functional.softplus(dt + dt_bias)
    state = torch.zeros(
        nheads, headdim, B.shape[-1], device=x.device, dtype=torch.float32
    )
    out = torch.empty(seqlen, nheads, headdim, device=x.device, dtype=torch.float32)
    for t in range(seqlen):
        decay = torch.exp(dt[t] * A)
        b = B[t].repeat_interleave(per_group, dim=0)
        c = C[t].repeat_interleave(per_group, dim=0)
        state = (
            state * decay[:, None, None]
            + dt[t][:, None, None] * x[t][:, :, None] * b[:, None, :]
        )
        out[t] = (state * c[:, None, :]).sum(-1) + D[:, None] * x[t]
    return out


def _relative(actual, reference):
    diff = (actual[0].float() - reference).abs()
    diff = torch.where(torch.isfinite(diff), diff, torch.full_like(diff, float("inf")))
    return float(diff.max()) / max(float(reference.abs().max()), 1e-6)


class TestSsdChunkScanConfigs(CustomTestCase):
    """The chunk scan's autotuner chooses by measured time, so every config it
    may choose has to compute the same, right answer: one wrong config in the
    list makes a model's output depend on which config won a benchmark."""

    def test_every_selectable_config_computes_the_reference_scan(self):
        kernel = ssd_chunk_scan._chunk_scan_fwd_kernel
        configs = _selectable(kernel)
        self.assertTrue(
            configs, "every config was pruned; the autotuner has nothing to choose"
        )
        for seqlen in SEQLENS:
            inputs = _inputs(seqlen)
            reference = _reference(inputs)
            self.assertGreater(
                float(reference.abs().max()), 0.0, "the reference scan is all zero"
            )
            for config in configs:
                with self.subTest(
                    seqlen=seqlen,
                    config=str(
                        config.kwargs | {"w": config.num_warps, "s": config.num_stages}
                    ),
                ):
                    with _Pinned((kernel, config), *_siblings()):
                        out = _scan(inputs)
                        again = _scan(inputs)
                        poisoned = _scan(inputs, fill=float("nan"))
                    self.assertTrue(
                        torch.isfinite(out).all(), "non-finite scan from bounded inputs"
                    )
                    self.assertLessEqual(
                        _relative(out, reference),
                        TOL,
                        "the scan departs from the reference",
                    )
                    self.assertTrue(
                        torch.equal(out, again),
                        "the same inputs gave two different results",
                    )
                    # Rows a kernel never writes keep whatever the buffer held.
                    self.assertTrue(
                        torch.equal(out, poisoned),
                        "the result depends on what the output buffer held",
                    )

    def test_every_excluded_config_still_computes_a_wrong_scan_here(self):
        """Opt-in, and host-specific: it says the exclusion still has cause on
        the GPU and Triton build being measured. A host where these configs are
        correct is a finding to record, not a reason to drop the exclusion,
        which still protects the hosts where they are not."""
        # Looked up by name, as _WRONG_CONFIGS is: on a tree without the fix the switch does
        # not exist, and this opt-in case must skip there rather than error.
        switch = getattr(envs, "SGLANG_TEST_MAMBA_SSD_DIAGNOSE", None)
        if switch is None or not switch.get():
            self.skipTest(
                "set SGLANG_TEST_MAMBA_SSD_DIAGNOSE=1 to re-measure the exclusion"
            )
        excluded = getattr(ssd_chunk_scan, "_WRONG_CONFIGS", ())
        self.assertTrue(excluded, "this tree excludes no config")
        kernel = ssd_chunk_scan._chunk_scan_fwd_kernel
        for entry in excluded:
            config = triton.Config(
                {k: v for k, v in entry.items() if k.startswith("BLOCK_")},
                num_warps=entry["num_warps"],
                num_stages=entry["num_stages"],
            )
            with self.subTest(config=str(entry)):
                wrong = False
                for seqlen in SEQLENS:
                    inputs = _inputs(seqlen)
                    with _Pinned((kernel, config), *_siblings()):
                        out = _scan(inputs)
                        again = _scan(inputs)
                    wrong = (
                        wrong
                        or not bool(torch.isfinite(out).all())
                        or not torch.equal(out, again)
                        or _relative(out, _reference(inputs)) > TOL
                    )
                self.assertTrue(
                    wrong,
                    f"{entry} is excluded but computes the reference scan on this host",
                )

    def test_the_autotuner_cannot_choose_a_config_measured_wrong(self):
        excluded = getattr(ssd_chunk_scan, "_WRONG_CONFIGS", ())
        if not excluded:
            self.skipTest("this tree excludes no config")
        specs = {_spec(c) for c in _selectable(ssd_chunk_scan._chunk_scan_fwd_kernel)}
        for entry in excluded:
            blocks = tuple(
                sorted((k, v) for k, v in entry.items() if k.startswith("BLOCK_"))
            )
            self.assertNotIn(
                (blocks, entry["num_warps"], entry["num_stages"]),
                specs,
                f"{entry} is still a candidate",
            )
        self.assertTrue(
            specs, "every config was pruned; the autotuner has nothing to choose"
        )

    def test_a_pinned_config_survives_pruning_so_it_stays_measurable(self):
        excluded = getattr(ssd_chunk_scan, "_WRONG_CONFIGS", ())
        if not excluded:
            self.skipTest("this tree excludes no config")
        entry = excluded[0]
        config = triton.Config(
            {k: v for k, v in entry.items() if k.startswith("BLOCK_")},
            num_warps=entry["num_warps"],
            num_stages=entry["num_stages"],
        )
        kept = _prune(ssd_chunk_scan._chunk_scan_fwd_kernel)([config], _named())
        self.assertEqual([_spec(c) for c in kept], [_spec(config)])


if __name__ == "__main__":
    unittest.main()
