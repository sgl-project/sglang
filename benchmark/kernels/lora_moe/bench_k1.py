"""K1 staged-rounding accuracy study (P6; plan section 30).

Question: does materializing any of the five LoRA bridges in FP32 buy
output-boundary accuracy? This study answers it with the THEORETICAL-MAX
form, deliberately without kernels: every hop computes in FP32 torch and
each bridge rounds to its DECLARED dtype exactly where production
materializes it. A real kernel consumer can never extract more from an
FP32 bridge than an FP32-computing consumer does, so:

* if an arm shows NO output-boundary error movement here, the bridge is
  closed — FP32 materialization would be pure bandwidth at any
  implementation (section 30's decision rule, one-sided);
* if an arm DOES move error, kernel-side dtype support and perf arms
  become warranted (P6 phase 2) with a measured accuracy prize.

The base path rounds at production's own materialization points in EVERY
arm (gateup_out, act_out, down_out are BF16 provider contracts — not part
of the K1 axis), so arms differ ONLY on the declared bridge. The
production kernels' FP32 accumulators are modeled by FP32 torch compute;
the section 7.4 BF16-ACCUMULATION diagnostic arm bounds the axis from the
other side by accumulating the four LoRA GEMMs in chunked BF16 partials.

Error metric: relative error of the complete-local-MoE output against the
all-FP32 reference (reference_local_moe), reported as max / p99 / mean
over token-hidden elements, per arm per case.

CPU-only by design (no GPU, no kernels); runs anywhere torch does.

Usage::

    python3 -m benchmark.kernels.lora_moe.bench_k1 \
        --output k1_accuracy.json --source-revision <sha>
"""

from __future__ import annotations

import argparse
import json

import torch

from benchmark.kernels.lora_moe.cases import (
    AdapterCell,
    Topology,
    build_case,
    materialize_case_tensors,
)
from benchmark.kernels.lora_moe.reference import (
    _activate,
    _coefficients,
    _resolve_pair_domain,
    reference_local_moe,
)

BRIDGES = (
    "gate_a_out",
    "gate_up_delta",
    "activation_lora_input",
    "down_a_out",
    "down_delta",
)
ARMS = ("control",) + BRIDGES + ("all_fp32", "bf16_acc_diag")
SEEDS = (11, 137, 997)


def _round(x: torch.Tensor, fp32: bool) -> torch.Tensor:
    """Materialization point: BF16 round-trip unless the bridge is FP32."""
    return x if fp32 else x.to(torch.bfloat16).to(torch.float32)


def _bf16_acc_einsum(equation: str, a: torch.Tensor, b: torch.Tensor, chunk=128):
    """Batched einsum with BF16 partial-sum accumulation over the K axis.

    Models a BF16-accumulator kernel (section 7.4 diagnostic control): each
    K-chunk's FP32 partial product folds into a BF16 running sum. The K axis
    is the last axis of BOTH operands by construction of the two equations
    this study uses ("nrk,nk->nr" and "nik,nk->ni").
    """
    k_size = a.shape[-1]
    acc = None
    for k0 in range(0, k_size, chunk):
        part = torch.einsum(equation, a[..., k0 : k0 + chunk], b[..., k0 : k0 + chunk])
        folded = part if acc is None else acc.to(torch.float32) + part
        acc = folded.to(torch.bfloat16)
    return acc.to(torch.float32)


def staged_local_moe(case, tensors, *, fp32_bridges: set[str], bf16_acc: bool):
    """Complete-local-MoE output with per-bridge staged rounding."""
    slices = 2 if case.expert_form == "gated_two_slice" else 1
    h = case.moe_hidden_size
    i_local = case.intermediate_size_local
    r_phys = case.physical_rank
    num_pairs = case.num_tokens * case.top_k

    expert_valid, pair_expert, pair_adapter = _resolve_pair_domain(case, tensors)
    lora_valid = pair_adapter >= 0
    token_of_pair = torch.arange(num_pairs) // case.top_k

    hidden = tensors.hidden_states.to(torch.float32)
    w13 = tensors.w13.to(torch.float32)
    w2 = tensors.w2.to(torch.float32)
    a_gu = tensors.lora_a_gate_up.to(torch.float32)
    b_gu = tensors.lora_b_gate_up.to(torch.float32)
    a_dn = tensors.lora_a_down.to(torch.float32)
    b_dn = tensors.lora_b_down.to(torch.float32)
    shared_gate = a_gu.shape[1] == 1
    shared_down = b_dn.shape[1] == 1

    def lora_gemm(equation: str, a: torch.Tensor, b: torch.Tensor):
        """The four LoRA GEMMs; the diagnostic arm accumulates them in BF16.
        The base path stays FP32-computed in every arm (its BF16
        materialization points are constant; only LoRA kernels are the
        section 7.4 accumulator question)."""
        if bf16_acc:
            return _bf16_acc_einsum(equation, a, b)
        return torch.einsum(equation, a, b)

    pair_out = torch.zeros(num_pairs, h)
    for expert in range(case.num_experts_local):
        rows = torch.nonzero(expert_valid & (pair_expert == expert)).reshape(-1)
        if rows.numel() == 0:
            continue
        x = hidden[token_of_pair[rows]]
        # Base materialization points are production BF16 contracts in
        # EVERY arm — the K1 axis is only the five LoRA bridges.
        gate_up_base = _round(x @ w13[expert].T, fp32=False)

        gate_up_delta = torch.zeros_like(gate_up_base)
        lora_rows_mask = lora_valid[rows]
        lora_rows = rows[lora_rows_mask]
        if lora_rows.numel():
            adapters = pair_adapter[lora_rows]
            x_lora = hidden[token_of_pair[lora_rows]]
            a_sel = a_gu[adapters, 0 if shared_gate else expert]
            b_sel = b_gu[adapters, expert]
            a_out = lora_gemm("nrk,nk->nr", a_sel, x_lora)
            a_out = _round(a_out, fp32="gate_a_out" in fp32_bridges)
            delta = torch.zeros(lora_rows.numel(), slices * i_local)
            for s in range(slices):
                delta[:, s * i_local : (s + 1) * i_local] = lora_gemm(
                    "nik,nk->ni",
                    b_sel[:, s * i_local : (s + 1) * i_local, :],
                    a_out[:, s * r_phys : (s + 1) * r_phys],
                )
            gate_up_delta[lora_rows_mask] = _round(
                delta, fp32="gate_up_delta" in fp32_bridges
            )

        activated = _activate(case, gate_up_base + gate_up_delta)
        act_base = _round(activated, fp32=False)  # act_out, production BF16
        down_base = _round(act_base @ w2[expert].T, fp32=False)

        down_delta = torch.zeros(rows.numel(), h)
        if lora_rows.numel():
            adapters = pair_adapter[lora_rows]
            act_lora = _round(
                activated[lora_rows_mask],
                fp32="activation_lora_input" in fp32_bridges,
            )
            a_out_dn = lora_gemm("nrk,nk->nr", a_dn[adapters, expert], act_lora)
            a_out_dn = _round(a_out_dn, fp32="down_a_out" in fp32_bridges)
            dd = lora_gemm(
                "nik,nk->ni",
                b_dn[adapters, 0 if shared_down else expert],
                a_out_dn,
            )
            down_delta[lora_rows_mask] = _round(dd, fp32="down_delta" in fp32_bridges)
        pair_out[rows] = down_base + down_delta

    coeff = _coefficients(case, tensors).reshape(-1, 1)
    weighted = (pair_out * coeff).reshape(case.num_tokens, case.top_k, h)
    accumulator = torch.zeros(case.num_tokens, h, dtype=torch.float32)
    for slot in range(case.top_k):
        accumulator = accumulator + weighted[:, slot]
    return accumulator * case.routed_scaling_factor


def _errors(output: torch.Tensor, reference: torch.Tensor) -> dict:
    denom = reference.abs().clamp_min(1e-3)
    rel = ((output - reference).abs() / denom).reshape(-1)
    return {
        "max": float(rel.max()),
        "p99": float(rel.quantile(0.99)),
        "mean": float(rel.mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-revision", required=True)
    arguments = parser.parse_args()
    torch.manual_seed(0)

    results = []
    grids = [
        # (shared_signature, ranks): per-expert and shared-outer sites both
        # carry the same five bridges; the shared form changes which factor
        # the gate A reads, not the rounding structure.
        ("per_expert", (16, 64)),
        ("shared_gate_up_a", (16, 64)),
    ]
    for shared, ranks in grids:
        for rank in ranks:
            for num_tokens in (64, 2048):
                for seed in SEEDS:
                    case = build_case(
                        device="cpu",
                        model_preset="qwen35_35b",
                        topology=Topology(tp_size=8, ep_size=8),
                        adapter_cell=AdapterCell(
                            active_adapters=4,
                            include_base_rows=True,
                            slot_capacity=8,
                        ),
                        route_generator="iid",
                        num_tokens=num_tokens,
                        active_rank=rank,
                        shared_factor_signature=shared,
                        seed=seed,
                        source_revision=arguments.source_revision,
                    )
                    tensors = materialize_case_tensors(case)
                    reference = reference_local_moe(case, tensors)
                    label = f"{shared:16s} " f"r{rank:<4d} T={num_tokens:<5d} s{seed}"
                    for arm in ARMS:
                        fp32_bridges: set[str] = set()
                        bf16_acc = arm == "bf16_acc_diag"
                        if arm in BRIDGES:
                            fp32_bridges = {arm}
                        elif arm == "all_fp32":
                            fp32_bridges = set(BRIDGES)
                        output = staged_local_moe(
                            case,
                            tensors,
                            fp32_bridges=fp32_bridges,
                            bf16_acc=bf16_acc,
                        )
                        errors = _errors(output, reference)
                        results.append(
                            {
                                "case_id": case.case_id,
                                "shared": shared != "per_expert",
                                "rank": rank,
                                "T": num_tokens,
                                "seed": seed,
                                "arm": arm,
                                **errors,
                            }
                        )
                        print(
                            f"{label} {arm:22s} max={errors['max']:.3e} "
                            f"p99={errors['p99']:.3e} mean={errors['mean']:.3e}"
                        )

    from benchmark.kernels.lora_moe.timing import (
        content_fingerprint,
        resolve_source_revision,
    )

    payload = {
        "study": "k1_staged_rounding_accuracy",
        "source_revision": arguments.source_revision,
        "observed_revision": resolve_source_revision(),
        "source_digest": content_fingerprint(),
        "torch_version": str(torch.__version__),
        "results": results,
    }
    with open(arguments.output, "w") as handle:
        json.dump(payload, handle, indent=1)
    print(f"{len(results)} rows -> {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
