"""The GDN in_proj algorithm pin runs the GEMM it names: every heuristic algorithm
at a generic M sweep writes every output, repeats bit for bit, and agrees with the
stock BF16 GEMM dispatch to BF16 rounding. Which algorithms are bit-identical to the
stock GEMM at each M (what a tuning-file entry must be checked for on its own stack)
is recorded, not asserted, since it depends on the cuBLAS build and device.

Outputs are NaN-poisoned, and a seeded positive control (one perturbed input row)
must change the pinned output.

Opt-in: with SGLANG_FLASHINFER_TACTIC_PIN_FILE naming a tuning file whose metadata
matches this stack, the file's in_proj entry must be bit-identical to the stock GEMM
at every M of its range, and forcing a listed index that is not must fail the same
comparison. Without such a file that case is skipped.
"""

import json

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

flashinfer = pytest.importorskip("flashinfer")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
        reason="needs an SM100+ GPU",
    ),
]

# Qwen3.8-Flash-Next fused in_proj (qkvz + ba) at TP1; a generic M sweep.
N, K = 16480, 2560
M_RANGE = range(33, 257, 37)
HEURISTIC_M = 256


def _pin_file(tmp_path, algo_index):
    path = tmp_path / f"pins_{algo_index}.json"
    from sglang.srt.layers.moe.flashinfer_tactic_pins import _current_metadata

    path.write_text(
        json.dumps(
            {
                "_metadata": _current_metadata(),
                "gdn_in_proj_cublaslt": {
                    "n": N,
                    "k": K,
                    "m_min": M_RANGE.start,
                    "m_max": M_RANGE.stop - 1,
                    "heuristic_m": HEURISTIC_M,
                    "algo_index": algo_index,
                },
            }
        )
    )
    return str(path)


def _pinned(pin_module, path, inputs, weight, monkeypatch):
    """Run every M through the pinned GEMM from a fresh plan; None if refused."""
    n = weight.shape[0]
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe import flashinfer_tactic_pins

    for cached in (flashinfer_tactic_pins.read_tuning_file, pin_module._pin_of):
        cached.cache_clear()
    monkeypatch.setattr(pin_module, "_plans", {})
    monkeypatch.setattr(pin_module, "_declined_m", set())
    monkeypatch.setattr(pin_module, "_disabled", False)
    outputs = {}
    with envs.SGLANG_FLASHINFER_TACTIC_PIN_FILE.override(path):
        for m, x in inputs.items():
            poison = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device="cuda")
            del poison
            outputs[m] = pin_module.gdn_in_proj_pinned_gemm(x, weight)
    torch.cuda.synchronize()
    return outputs


def test_pinned_algorithms_compute_the_gemm(tmp_path, monkeypatch, record_property):
    from sglang.srt.layers import gdn_in_proj_algo_pin as pin_module
    from sglang.srt.layers.moe.flashinfer_tactic_pins import (
        _INDETERMINATE_METADATA,
        _current_metadata,
    )
    from sglang.srt.layers.quantization.unquant import bf16_gemm_dispatch

    if any(value in _INDETERMINATE_METADATA for value in _current_metadata().values()):
        pytest.skip("FlashInfer cannot identify this stack, so no tuning file applies")

    gen = torch.Generator(device="cuda").manual_seed(0)
    weight = (0.02 * torch.randn(N, K, generator=gen, device="cuda")).to(torch.bfloat16)
    inputs = {
        m: torch.randn(m, K, generator=gen, device="cuda").to(torch.bfloat16)
        for m in M_RANGE
    }
    stock = {m: bf16_gemm_dispatch(x, weight, None) for m, x in inputs.items()}

    bitwise = {m: [] for m in M_RANGE}
    tried = []
    for algo_index in range(8):
        path = _pin_file(tmp_path, algo_index)
        outputs = _pinned(pin_module, path, inputs, weight, monkeypatch)
        if any(out is None for out in outputs.values()):
            break  # The heuristic returned fewer algorithms.
        tried.append(algo_index)
        again = _pinned(pin_module, path, inputs, weight, monkeypatch)
        for m in M_RANGE:
            assert not torch.isnan(outputs[m]).any(), "an output was never written"
            assert torch.equal(outputs[m].view(torch.int16), again[m].view(torch.int16))
            torch.testing.assert_close(outputs[m], stock[m], rtol=2e-2, atol=2e-2)
            if torch.equal(outputs[m].view(torch.int16), stock[m].view(torch.int16)):
                bitwise[m].append(algo_index)
    record_property("bitwise_algo_indices_per_m", bitwise)
    assert tried, "the heuristic returned no algorithm"

    m = M_RANGE.start
    perturbed = dict(inputs)
    perturbed[m] = inputs[m].clone()
    perturbed[m][0] = perturbed[m][0] * 2 + 1
    pinned = _pinned(pin_module, _pin_file(tmp_path, tried[0]), inputs, weight, monkeypatch)
    control = _pinned(
        pin_module, _pin_file(tmp_path, tried[0]), perturbed, weight, monkeypatch
    )
    assert not torch.equal(control[m].view(torch.int16), pinned[m].view(torch.int16))


def _bitwise(outputs, stock):
    return all(
        outputs[m] is not None
        and torch.equal(outputs[m].view(torch.int16), stock[m].view(torch.int16))
        for m in stock
    )


def test_supplied_tuning_file_entry_is_bitwise(tmp_path, monkeypatch, record_property):
    from sglang.srt.environ import envs
    from sglang.srt.layers import gdn_in_proj_algo_pin as pin_module
    from sglang.srt.layers.moe import flashinfer_tactic_pins
    from sglang.srt.layers.quantization.unquant import bf16_gemm_dispatch

    supplied = envs.SGLANG_FLASHINFER_TACTIC_PIN_FILE.get()
    if not supplied:
        pytest.skip("set SGLANG_FLASHINFER_TACTIC_PIN_FILE to check a tuning file's entry")
    flashinfer_tactic_pins.read_tuning_file.cache_clear()
    document = flashinfer_tactic_pins.read_tuning_file(supplied)
    entry = pin_module.parse_gdn_in_proj_pin(document)
    if entry is None:
        pytest.skip("the tuning file has no gdn_in_proj_cublaslt entry")
    if not flashinfer_tactic_pins.tuning_file_applies(supplied, need_cache_key_layout=False):
        pytest.skip("the tuning file's _metadata does not match this stack")

    gen = torch.Generator(device="cuda").manual_seed(1)
    weight = (0.02 * torch.randn(entry.n, entry.k, generator=gen, device="cuda")).to(
        torch.bfloat16
    )
    inputs = {
        m: torch.randn(m, entry.k, generator=gen, device="cuda").to(torch.bfloat16)
        for m in range(entry.m_min, entry.m_max + 1)
    }
    stock = {m: bf16_gemm_dispatch(x, weight, None) for m, x in inputs.items()}

    pinned = _pinned(pin_module, supplied, inputs, weight, monkeypatch)
    assert all(out is not None for out in pinned.values()), "the pin was refused"
    assert not any(torch.isnan(out).any() for out in pinned.values())
    differing = [
        m
        for m in stock
        if not torch.equal(pinned[m].view(torch.int16), stock[m].view(torch.int16))
    ]
    assert not differing, f"entry index {entry.algo_index} differs from stock at M {differing}"

    # Control: an index the heuristic lists that is not bit-identical must fail.
    wrong = None
    for algo_index in range(16):
        if algo_index == entry.algo_index:
            continue
        entry_json = dict(document["gdn_in_proj_cublaslt"], algo_index=algo_index)
        forced = dict(document, gdn_in_proj_cublaslt=entry_json)
        path = tmp_path / f"forced_{algo_index}.json"
        path.write_text(json.dumps(forced))
        outputs = _pinned(pin_module, str(path), inputs, weight, monkeypatch)
        if any(out is None for out in outputs.values()):
            break  # Past the heuristic's list.
        if not _bitwise(outputs, stock):
            wrong = algo_index
            break
    record_property("forced_wrong_index", wrong)
    if wrong is None:
        # Every listed index is bit-identical here; check the comparison can fail.
        m = entry.m_min
        perturbed = dict(inputs)
        perturbed[m] = inputs[m].clone()
        perturbed[m][0] = perturbed[m][0] * 2 + 1
        assert not _bitwise(_pinned(pin_module, supplied, perturbed, weight, monkeypatch), stock)
