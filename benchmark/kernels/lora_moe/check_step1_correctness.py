"""Step-1 GPU correctness driver (plan §14 Step 1 required-correctness list).

Runs the curated early-guardrail matrix through ``serial_materialized_control``
with NaN-poisoned workspaces and validates every case against the independent
FP32 reference under the §21.1 signal-relative gates.  Emits one JSON record
per case for post-hoc re-adjudication.

Usage (single GPU, eager only):

    PYTHONPATH=python python benchmark/kernels/lora_moe/check_step1_correctness.py \
        --device cuda:0 --output step1_correctness.json
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import msgspec
import torch

from benchmark.kernels.lora_moe.cases import (
    COMPACT_ADAPTER_CELLS,
    AdapterCell,
    MoeLoraBenchCase,
    Topology,
    build_case,
    capture_source_revision,
    materialize_case_tensors,
)
from benchmark.kernels.lora_moe.production_runner import (
    prepare_production_forward,
    production_runner_skip_reason,
    run_production_runner,
)
from benchmark.kernels.lora_moe.reference import (
    reference_local_moe,
    reference_pair_stages,
)
from benchmark.kernels.lora_moe.serial_control import (
    run_base_only_torch,
    run_serial_materialized_control,
)
from benchmark.kernels.lora_moe.signal_gates import (
    SignalCheckRecord,
    check_delta,
    require_bitwise_equal,
    require_finite,
    resolve_signal_gates,
)

EP8 = Topology(tp_size=8, ep_size=8)


class CaseResult(msgspec.Struct, kw_only=True):
    case: MoeLoraBenchCase
    checks: list[SignalCheckRecord]
    bitwise_checks: list[str]
    replays: int
    passed: bool
    failure: str = ""
    # Gate-2 review: a record must say WHICH provider it observed and carry
    # elementwise output digests, so cross-provider bitwise identity is a
    # checkable property of two archives rather than an inference from equal
    # gate scalars. `production_provider` is the contract key read OFF THE
    # CONSTRUCTED RUNNER (and asserted against the env request), not the
    # request itself.
    control_output_sha256: str | None = None
    production_provider: str | None = None
    production_output_sha256: str | None = None


def _tensor_digest(tensor: torch.Tensor) -> str:
    """SHA256 over dtype, shape, and raw element bytes (bf16 viewed as int16)."""
    t = tensor.detach().cpu().contiguous()
    header = f"{t.dtype}|{tuple(t.shape)}|".encode()
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    return hashlib.sha256(header + t.numpy().tobytes()).hexdigest()


def step1_matrix(device: str, source_revision: str) -> list[MoeLoraBenchCase]:
    """The curated Step-1 correctness matrix (early guardrails, eager)."""

    def case(**kw) -> MoeLoraBenchCase:
        keywords = dict(
            device=device,
            # A2 RULED (plan section 48): canonical = s x FP32(w), the form the
            # production combine physically computes and every surveyed backend
            # shares. bf16_rounded remains a diagnostic axis value.
            route_coeff_precision="fp32",
            source_revision=source_revision,
            seed=11,
        )
        keywords.update(kw)
        return build_case(**keywords)

    def cell(l_active: int, base: bool, capacity: int, **kw) -> AdapterCell:
        return AdapterCell(
            active_adapters=l_active,
            include_base_rows=base,
            slot_capacity=capacity,
            **kw,
        )

    cases: list[MoeLoraBenchCase] = []

    # Canaries: tiny geometries, gated and non-gated.
    for preset in ("tiny_smoke", "tiny_smoke_relu2"):
        cases.append(
            case(
                model_preset=preset,
                adapter_cell=cell(2, True, 4),
                route_generator="iid",
                num_tokens=16,
                active_rank=16,
            )
        )

    # Primary anchor: qwen35 EP8-local shape across the §12.2 adapter cells.
    for adapter_cell in COMPACT_ADAPTER_CELLS:
        cases.append(
            case(
                model_preset="qwen35_35b",
                topology=EP8,
                adapter_cell=adapter_cell,
                route_generator="iid",
                num_tokens=32,
                active_rank=16,
            )
        )

    qwen = dict(model_preset="qwen35_35b", topology=EP8, active_rank=16)
    cases += [
        # Route families and token extremes.
        case(
            **qwen,
            adapter_cell=cell(1, True, 8),
            route_generator="balanced",
            num_tokens=32,
        ),
        case(
            **qwen,
            adapter_cell=cell(1, True, 8),
            route_generator="one_hot",
            num_tokens=32,
        ),
        case(
            **qwen,
            adapter_cell=cell(3, False, 8),
            route_generator="hotset_80_20",
            num_tokens=256,
        ),
        case(
            **qwen, adapter_cell=cell(1, True, 8), route_generator="iid", num_tokens=1
        ),
        # Rank spread at scale (§14 Step-3 evidence: ranks 32/64/128).
        case(
            model_preset="qwen35_35b",
            topology=EP8,
            adapter_cell=cell(1, True, 8),
            route_generator="iid",
            num_tokens=256,
            active_rank=32,
        ),
        case(
            model_preset="qwen35_35b",
            topology=EP8,
            adapter_cell=cell(1, True, 8),
            route_generator="iid",
            num_tokens=256,
            active_rank=64,
        ),
        case(
            model_preset="qwen35_35b",
            topology=EP8,
            adapter_cell=cell(1, True, 8),
            route_generator="iid",
            num_tokens=256,
            active_rank=128,
        ),
        # Caller-selected FP32 destination.
        case(
            **qwen,
            adapter_cell=cell(1, True, 8),
            route_generator="iid",
            num_tokens=32,
            output_dtype="float32",
        ),
        # Partial slice targets.
        case(
            **qwen,
            adapter_cell=cell(1, True, 8),
            route_generator="iid",
            num_tokens=32,
            slice_target="gate_only",
        ),
        case(
            **qwen,
            adapter_cell=cell(1, True, 8),
            route_generator="iid",
            num_tokens=32,
            slice_target="up_only",
        ),
        # Shared-outer control form.
        case(
            **qwen,
            adapter_cell=cell(1, True, 8),
            route_generator="iid",
            num_tokens=32,
            shared_factor_signature="shared_both",
        ),
        case(
            **qwen,
            adapter_cell=cell(4, True, 5),
            route_generator="iid",
            num_tokens=32,
            shared_factor_signature="shared_both",
        ),
        # Active rank below physical rank (R8 masked to physical 16).
        case(
            model_preset="qwen35_35b",
            topology=EP8,
            adapter_cell=cell(1, True, 8),
            route_generator="iid",
            num_tokens=32,
            active_rank=8,
            max_rank=16,
            physical_rank=16,
        ),
        # Non-unit routed scaling (BF16 destination; fold documented in the
        # serial control).
        case(
            **qwen,
            adapter_cell=cell(1, True, 8),
            route_generator="iid",
            num_tokens=32,
            routed_scaling_factor=1.75,
        ),
        # Sparse/permuted slot occupancy.
        case(
            model_preset="qwen35_35b",
            topology=EP8,
            adapter_cell=cell(3, True, 8, active_slot_ids=(1, 4, 7)),
            route_generator="iid",
            num_tokens=32,
            active_rank=16,
        ),
        # Global expert IDs with a nonzero EP-rank offset map.
        case(
            model_preset="qwen35_35b",
            topology=Topology(tp_size=8, ep_size=8, ep_rank=1),
            adapter_cell=cell(1, True, 8),
            route_generator="iid",
            num_tokens=32,
            active_rank=16,
            expert_id_domain="global",
        ),
        # No locally owned route at all: output must be exactly zero.
        # Covered at EP rank 0 AND a nonzero rank (owned interval offset).
        case(
            model_preset="qwen35_35b",
            topology=EP8,
            adapter_cell=cell(1, True, 8),
            route_generator="no_local",
            num_tokens=32,
            active_rank=16,
            expert_id_domain="global",
        ),
        case(
            model_preset="qwen35_35b",
            topology=Topology(tp_size=8, ep_size=8, ep_rank=3),
            adapter_cell=cell(1, True, 8),
            route_generator="no_local",
            num_tokens=32,
            active_rank=16,
            expert_id_domain="global",
        ),
        # Literal -1 sentinel inputs in the local ID domain (the standard
        # dispatcher form for non-owned routes).
        case(
            **qwen,
            adapter_cell=cell(1, True, 8),
            route_generator="iid_with_sentinels",
            num_tokens=32,
        ),
        # MoE-DP topology local-shape proxy (moe_tp = tp/ep/dp).
        case(
            model_preset="qwen35_35b",
            topology=Topology(tp_size=8, ep_size=2, moe_dp_size=2),
            adapter_cell=cell(1, True, 8),
            route_generator="iid",
            num_tokens=32,
            active_rank=16,
        ),
    ]

    # Cross-model semantic guardrails (incl. non-gated ReLU2 and odd-I).
    for preset in (
        "qwen35_397b",
        "kimi_k25",
        "glm_52",
        "nemotron3_super",
        "nemotron3_nano",
    ):
        for adapter_cell in (cell(1, True, 8), cell(3, False, 8)):
            cases.append(
                case(
                    model_preset=preset,
                    topology=EP8,
                    adapter_cell=adapter_cell,
                    route_generator="iid",
                    num_tokens=32,
                    active_rank=16,
                )
            )
    return cases


def _check_case(
    case: MoeLoraBenchCase,
    *,
    device: torch.device,
    replays: int,
    deep_replays: int,
    production_deep_replays: int = 0,
) -> CaseResult:
    tensors = materialize_case_tensors(case)
    checks: list[SignalCheckRecord] = []
    bitwise: list[str] = []

    result = run_serial_materialized_control(
        case, tensors, device=device, poison_workspaces=True
    )
    require_finite(result.output, label="output poison hygiene")
    bitwise.append("output finite under poisoned workspaces")
    expected_dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[
        case.output_dtype
    ]
    if result.output.dtype != expected_dtype:
        raise AssertionError(
            f"declared output dtype {case.output_dtype} but got "
            f"{result.output.dtype}"
        )

    stages = reference_pair_stages(case, tensors)
    lora_rows = (stages.pair_adapter >= 0).cpu()

    # Pairs without LoRA work must leave EXACT zeros in both delta buffers
    # (sentinel-block overwrite contract) — checked on the poisoned run.
    sentinel_rows = (~lora_rows).to(result.output.device)
    if bool(sentinel_rows.any()):
        for label, buffer in (
            ("gate/up delta", result.gate_up_delta),
            ("down delta", result.down_delta),
        ):
            require_bitwise_equal(
                buffer[sentinel_rows],
                torch.zeros_like(buffer[sentinel_rows]),
                label=f"{label} exact zero at sentinel pairs",
            )
        bitwise.append("delta buffers exactly zero at sentinel pairs")

    # Signal presence is decided at the OUTPUT boundary: a gate/up-only
    # adapter produces zero down_delta yet changes the final output.
    base_reference = reference_local_moe(case, tensors, include_lora=False)
    full_reference = reference_local_moe(case, tensors, stages=stages)
    has_signal = float((full_reference - base_reference).abs().max()) > 0.0

    if has_signal:
        # Subtract each side's OWN matched base so base-pipeline BF16 noise
        # cancels and the gate sees only the LoRA contribution (§21.1). The
        # rel-L2 gate uses BF16 computation precision even for an FP32
        # destination: the pipeline computes in BF16 throughout.
        base_torch = run_base_only_torch(case, tensors, device=device)
        gates = resolve_signal_gates(
            full_reference - base_reference,
            gate_dtype=torch.bfloat16,
            base_reference=base_reference,
        )
        checks.append(
            check_delta(
                result.output.cpu().float() - base_torch.cpu().float(),
                full_reference - base_reference,
                gates,
                label="complete local MoE",
            )
        )
        # Per-site boundaries on the valid LoRA rows.
        for label, observed, reference in (
            ("gate/up LoRA A", result.gate_up_lora_a, stages.gate_up_lora_a),
            ("gate/up delta", result.gate_up_delta, stages.gate_up_delta),
            ("down LoRA A", result.down_lora_a, stages.down_lora_a),
            ("down delta (LoRA B)", result.down_delta, stages.down_delta),
        ):
            site_reference = reference[lora_rows]
            if float(site_reference.abs().max()) == 0.0:
                continue
            site_gates = resolve_signal_gates(site_reference, gate_dtype=torch.bfloat16)
            checks.append(
                check_delta(
                    observed.cpu().float()[lora_rows],
                    site_reference,
                    site_gates,
                    label=label,
                )
            )
        # Mixed traffic: base rows must be untouched by the LoRA launches.
        base_rows = (tensors.token_lora_mapping == -1).to(result.output.device)
        if bool(base_rows.any()):
            require_bitwise_equal(
                result.output[base_rows],
                base_torch[base_rows],
                label="mixed base rows bitwise",
            )
            bitwise.append("mixed base rows bitwise-equal to pure-base pipeline")
    else:
        base_torch = run_base_only_torch(case, tensors, device=device)
        require_bitwise_equal(
            result.output, base_torch, label="zero-LoRA bitwise parity"
        )
        bitwise.append("zero-LoRA bitwise parity vs pure-base torch pipeline")
        if case.p_valid == 0 and case.route_generator == "no_local":
            require_bitwise_equal(
                result.output,
                torch.zeros_like(result.output),
                label="no-local route zero output",
            )
            bitwise.append("no-local route produces exact zeros")

    total_replays = deep_replays if case.active_adapters == 4 else replays
    for replay in range(total_replays):
        repeat = run_serial_materialized_control(
            case, tensors, device=device, poison_workspaces=False
        )
        require_bitwise_equal(repeat.output, result.output, label=f"replay {replay}")
    bitwise.append(f"{total_replays} replays bitwise stable")

    # ---- Production runner as a SECOND observed backend (plan 31.6). ----
    # Same reference, same matched-base discipline, same gates: the shipping
    # path is held to the identical contract as the lab control, so a
    # control-vs-production divergence cannot hide behind either being
    # internally consistent.
    skip_reason = production_runner_skip_reason(case)
    production_provider_key = None
    if skip_reason is not None:
        bitwise.append(f"production runner skipped: {skip_reason}")
    else:
        # OBSERVE the provider off the constructed runner and assert it is
        # the one the environment requested — recording the env value alone
        # would not have caught the harness that hard-coded DeepGEMM while
        # the env said cutedsl (third review pass).
        from sglang.srt.environ import envs

        runner, dispatcher, batch, dispatch_output = prepare_production_forward(
            case, tensors, device=device
        )
        production_provider_key = runner.provider.contract.key
        requested = envs.SGLANG_LORA_MOE_BASE_PROVIDER.get()
        expected_key = {"deepgemm": "deepgemm_bf16", "cutedsl": "cutedsl_bf16"}[
            requested
        ]
        if production_provider_key != expected_key:
            raise AssertionError(
                f"provider selection drift: env requested {requested!r} but the "
                f"runner constructed {production_provider_key!r}"
            )
        production = dispatcher.combine(
            runner.run(dispatch_output, batch, output_dtype=expected_dtype)
        )
        require_finite(production, label="production output finite")
        if production.dtype != expected_dtype:
            raise AssertionError(
                f"production runner output dtype {production.dtype} != declared "
                f"{case.output_dtype}"
            )
        production_base = run_production_runner(
            case, tensors, device=device, disable_lora=True
        )
        if has_signal:
            checks.append(
                check_delta(
                    production.cpu().float() - production_base.cpu().float(),
                    full_reference - base_reference,
                    gates,
                    label="production runner e2e",
                )
            )
            base_rows = (tensors.token_lora_mapping == -1).to(production.device)
            if bool(base_rows.any()):
                require_bitwise_equal(
                    production[base_rows],
                    production_base[base_rows],
                    label="production mixed base rows bitwise",
                )
                bitwise.append(
                    "production base rows bitwise-equal to its all-base forward"
                )
        else:
            require_bitwise_equal(
                production,
                production_base,
                label="production zero-LoRA parity",
            )
            bitwise.append("production zero-LoRA bitwise parity")
        production_replays = max(replays, production_deep_replays)
        for replay in range(production_replays):
            repeat = run_production_runner(case, tensors, device=device)
            require_bitwise_equal(
                repeat, production, label=f"production replay {replay}"
            )
        bitwise.append(
            f"production runner: {production_replays} replays bitwise stable"
        )

    passed = all(record.passed for record in checks)
    return CaseResult(
        case=case,
        checks=checks,
        bitwise_checks=bitwise,
        replays=total_replays,
        passed=passed,
        control_output_sha256=_tensor_digest(result.output),
        production_provider=production_provider_key,
        production_output_sha256=(
            _tensor_digest(production) if skip_reason is None else None
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="step1_correctness.json")
    parser.add_argument("--filter", default="", help="substring of preset/case_id")
    parser.add_argument("--replays", type=int, default=3)
    parser.add_argument("--deep-replays", type=int, default=100)
    parser.add_argument(
        "--production-deep-replays",
        type=int,
        default=0,
        help="when > 0, replay the PRODUCTION runner this many times bitwise "
        "on every case (Step-3 determinism evidence for schedule arms — the "
        "default 0 keeps the Step-1/2 matrix cost and semantics)",
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument(
        "--source-revision",
        default="",
        help="explicit source identity when running from an exported tree",
    )
    arguments = parser.parse_args()

    device = torch.device(arguments.device)
    source_revision = arguments.source_revision or capture_source_revision()
    cases = [
        candidate
        for candidate in step1_matrix(str(device), source_revision)
        if arguments.filter in candidate.model_preset
        or arguments.filter in candidate.case_id
    ]
    if arguments.list:
        for candidate in cases:
            print(
                f"{candidate.case_id} {candidate.model_preset} "
                f"T={candidate.num_tokens} R={candidate.active_rank} "
                f"L=({candidate.active_adapters},{int(candidate.include_base_rows)},"
                f"{candidate.slot_capacity}) {candidate.route_generator} "
                f"{candidate.shared_factor_signature} {candidate.slice_target} "
                f"{candidate.output_dtype}"
            )
        return 0

    results: list[CaseResult] = []
    failures = 0
    for index, candidate in enumerate(cases):
        label = (
            f"[{index + 1}/{len(cases)}] {candidate.model_preset} {candidate.case_id}"
        )
        try:
            outcome = _check_case(
                candidate,
                device=device,
                replays=arguments.replays,
                deep_replays=arguments.deep_replays,
                production_deep_replays=arguments.production_deep_replays,
            )
        except (AssertionError, RuntimeError, ValueError) as error:
            outcome = CaseResult(
                case=candidate,
                checks=[],
                bitwise_checks=[],
                replays=0,
                passed=False,
                failure=f"{type(error).__name__}: {error}",
            )
        results.append(outcome)
        failures += 0 if outcome.passed else 1
        status = "PASS" if outcome.passed else f"FAIL ({outcome.failure})"
        print(f"{label}: {status}", flush=True)

    Path(arguments.output).write_bytes(
        msgspec.json.format(msgspec.json.encode(results), indent=2)
    )
    print(
        f"{len(results) - failures}/{len(results)} cases passed; "
        f"records written to {arguments.output}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
