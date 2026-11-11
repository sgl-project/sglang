"""The test-only launch recorder on existing MoE LoRA tests.

Operands: the provider pipeline of the prepared-input reuse tests (BF16 and
128-block FP8 base weights on the CuTeDSL providers, masked and contiguous
rows) runs twice on identical inputs, with a fresh dispatch each time and with
dropped pairs and padding rows in the batch. At every launch boundary the valid
data of every live buffer must be the same in both forwards. Buffers in
provider order are compared by logical pair, so the test does not depend on
the dispatch giving a pair the same physical row twice (on the boxes this was
written on it never did); padding rows and storage that no producer has written
are never compared.

Structure: an existing runner test (BF16 and FP8 on CuTeDSL, the FP8 families
on Marlin W8A16 with mixed adapter ranks, the NVFP4 fused down-B on SM100,
eager and under graph capture) runs three times under the recorder; the second
and third launch sequences must be identical: kernels, grids, constants and
each operand's view (shape, dtype, strides, buffer name, byte offset in its
storage). The first run warms the compile caches and is not compared.

Controls: one corrupted valid row, a changed batch size, and a kernel handed
the other half of the same allocation must each show, at the call and the
buffer or operand where they enter.

Before and after a change: a run that keeps a junit record (``--junitxml``)
also keeps, beside it, one artifact per case (``<junit name>.launches.json``):
the calls, the boundary and output digests of the provider pipeline cases
(``numeric``: they repeat on one device), the structure alone of the reused
runner tests (``structure``: those tests own their buffers), with the fixture,
revision, device and library versions. Two such files compare with
``python -m sglang.test.kernels.moe_lora_launch_recorder before after``, which
names the first call, operand, declared buffer or output where a case parts.
The record covers what the recorder module's ``COVERAGE`` states and nothing
else: it is not a statement about every operand of every kernel.

The graph, overlap and fresh-routing numerical checks of the reused tests run
unchanged inside these calls; the recorder adds no code to the serving path.
"""

from __future__ import annotations

import json
import pathlib

import pytest
import test_moe_lora_blockscaled_provider_numerics as B
import test_moe_lora_fp8_marlin_provider_numerics as M
import test_moe_lora_moe_runner_numerics as N
import test_moe_lora_prepared_input_reuse as R
import torch

from sglang.srt.lora.utils import Phase
from sglang.srt.lora.workspace import LoraWorkspace
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.moe_lora import cutedsl_ready
from sglang.test.kernels.moe_lora_launch_recorder import (
    UNRECORDED,
    first_difference,
    main,
    record_launches,
    rows_by_pair,
    save_artifact,
)

# The run time is mostly the Marlin kernel build of the first Marlin case.
register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="1-gpu-large")
# SM100 runs its own plan rows and base GEMMs. The test runs on one GPU:
# 4-gpu-b200 is the kernel lane's only SM100 pool, so its four GPUs are
# allocation granularity, not multi-GPU coverage.
register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not cutedsl_ready(), reason="the reused tests run the CuTeDSL providers"
)

provider_case = R.provider_case
_TOKENS = 257
_SEED = 20261109
_started: set[pathlib.Path] = set()


def _keep(request, artifact: dict) -> dict:
    """Save the case's artifact beside the junit record, if the run keeps one."""
    xml = request.config.option.xmlpath
    if xml:
        path = pathlib.Path(xml).with_suffix(".launches.json")
        if path not in _started:  # one file per run, not appended across runs
            _started.add(path)
            path.unlink(missing_ok=True)
        save_artifact(path, artifact)
    return artifact


def _inputs(tokens: int = _TOKENS, hidden_size: int = R._H):
    generator = torch.Generator(device="cuda").manual_seed(_SEED)
    hidden = (
        torch.randn(tokens, hidden_size, device="cuda", generator=generator) * 0.2
    ).to(torch.bfloat16)
    ids = torch.empty(tokens, R._K, device="cuda", dtype=torch.int32)
    R._traffic(ids, "padded")  # dropped pairs, tokens without any expert
    weights = torch.tensor([0.65, 0.35], device="cuda").expand(tokens, R._K)
    return hidden, ids, weights.contiguous()


def _forward(
    provider,
    workspace,
    recorder,
    hidden,
    ids,
    weights,
    *,
    graph_mode: bool = False,
    separate_quant: bool = False,
    corrupt_gateup: bool = False,
    act_half: int | None = None,
    prepared=None,
):
    """The base pipeline of the reuse tests with the recorder told, stage by
    stage, which buffers hold this forward's data and where their valid rows
    are. ``act_half`` allocates the activation buffer twice as large and
    hands the kernels that half of it. ``prepared`` maps the row state to the
    tensors of it that identical inputs must rebuild byte for byte, each with
    its comparable region (None: all of it)."""
    workspace.begin_forward(graph_mode=graph_mode, is_prefill_graph=False)
    recorder.bind("hidden", hidden)  # the fixture: defined before any launch
    recorder.define("hidden")
    state = provider.prepare(hidden, ids, R._K, workspace)
    for name, (tensor, region) in (prepared(state) if prepared else {}).items():
        recorder.bind(name, tensor, region)
        recorder.define(name)
    valid = ids >= 0

    def by_pair(buffer):
        return rows_by_pair(buffer, state.pair_to_row, valid)

    def buffer(name, shape):
        return workspace.tensor(name, shape, dtype=torch.bfloat16, device="cuda")

    gateup = buffer("test:gateup", provider.gateup_out_shape(state))
    if act_half is None:
        act = buffer("test:act", provider.act_out_shape(state))
    else:
        act = buffer("test:act", (2, *provider.act_out_shape(state)))[act_half]
    pairs = buffer("test:pairs", (hidden.shape[0], R._K, provider.intermediate_size))
    recorder.bind("gateup", gateup, by_pair)
    recorder.bind("act", act, by_pair)
    recorder.bind("pairs", pairs)  # pair-major activation: every pair is written

    provider.gateup(state, gateup)
    provider.release_prepared_inputs(state)
    recorder.define("gateup")
    if corrupt_gateup:
        row = state.pair_to_row.reshape(-1)[valid.reshape(-1)][0].long()
        gateup.reshape(-1, gateup.shape[-1])[row, 0] += 1.0
    provider.act_with_delta(state, gateup, None, ids, act, pairs)
    recorder.define("act", "pairs")
    if separate_quant:
        state.act_quant_ready = False
    down = provider.down_output(state, workspace)
    recorder.bind("down", down, by_pair)
    provider.down(state, act, down)
    recorder.define("down")
    output = torch.empty_like(hidden)
    provider.finalize(state, down, ids, weights, 0.75, output)
    return state, pairs, output


def test_fresh_forwards_agree_at_every_launch_boundary(
    provider_case, record_property, request
) -> None:
    provider, w13, w2, family, separate_quant = provider_case
    fresh_forwards_case(
        provider,
        w13,
        w2,
        family,
        record_property,
        request,
        kinds={"triton", "cutedsl"},
        fixture=dict(seed=_SEED, tokens=_TOKENS, traffic="padded", family=family),
        separate_quant=separate_quant,
    )


def fresh_forwards_case(
    provider,
    w13,
    w2,
    family,
    record_property,
    request,
    *,
    kinds: set[str],
    fixture: dict,
    separate_quant: bool = False,
    prepared=None,
    check=R._check,
) -> None:
    """Two fresh forwards on identical inputs: the same launches, the same
    bytes at every boundary and the same outputs; the first one's record is
    kept for another tree to compare with. ``check`` holds each output to the
    case's reference."""
    hidden, ids, weights = _inputs(hidden_size=provider.hidden_size)
    valid = (ids >= 0).reshape(-1)
    runs = []
    for _ in range(2):
        with record_launches() as recorder:
            state, pairs, output = _forward(
                provider,
                LoraWorkspace(),
                recorder,
                hidden,
                ids,
                weights,
                separate_quant=separate_quant,
                prepared=prepared,
            )
        check(output, hidden, ids, weights, w13, w2, family)
        runs.append(
            (
                recorder,
                state.pair_to_row.reshape(-1)[valid].clone(),
                pairs.clone(),
                output.clone(),
            )
        )
    (first, rows_a, pairs_a, out_a), (second, rows_b, pairs_b, out_b) = runs
    assert first.launches and first.structure() == second.structure()
    assert kinds <= first.kinds()
    assert first.boundaries() == second.boundaries()
    assert torch.equal(pairs_a, pairs_b)
    assert torch.equal(out_a, out_b)
    # Reported, not required: whether the two dispatches used the same rows.
    record_property("same_physical_rows", bool(torch.equal(rows_a, rows_b)))
    record_property("launches", len(first.launches))
    record_property("structure", first.structure_digest())
    # The record another tree compares with; the second forward's is the same.
    kept = _keep(
        request,
        first.artifact(
            request.node.name,
            check="numeric",
            fixture=fixture,
            outputs=dict(pairs=pairs_a, output=out_a),
        ),
    )
    again = second.artifact(
        request.node.name,
        check="numeric",
        fixture=fixture,
        outputs=dict(pairs=pairs_b, output=out_b),
    )
    assert first_difference(kept, again) is None
    assert kept["calls"][-1]["boundary"] and kept["outputs"]


def test_graph_capture_launches_what_the_warm_forward_launched(provider_case) -> None:
    provider, w13, w2, family, separate_quant = provider_case
    hidden, ids, weights = _inputs()
    workspace = LoraWorkspace()
    stream = torch.cuda.Stream()

    def run(recorder):
        return _forward(
            provider,
            workspace,
            recorder,
            hidden,
            ids,
            weights,
            graph_mode=True,
            separate_quant=separate_quant,
        )

    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        with record_launches() as discarded:
            run(discarded)
        with record_launches() as warm:
            _, _, warm_output = run(warm)
        warm_output = warm_output.clone()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with record_launches() as captured:
        with torch.cuda.graph(graph, stream=stream):
            _, _, output = run(captured)
    torch.cuda.current_stream().wait_stream(stream)
    assert warm.structure() == captured.structure()
    assert all(not boundary for _, boundary in captured.boundaries())

    output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    R._check(output, hidden, ids, weights, w13, w2, family)
    assert torch.equal(output, warm_output)


def test_recorder_shows_a_corrupted_row_a_changed_batch_and_a_moved_view(
    provider_case, tmp_path
) -> None:
    provider, _, _, _, separate_quant = provider_case
    hidden, ids, weights = _inputs()

    def artifact(recorder):
        return recorder.artifact("control", check="numeric", fixture={})

    def record(**kwargs):
        with record_launches() as recorder:
            _forward(
                provider,
                LoraWorkspace(),
                recorder,
                *kwargs.pop("inputs", (hidden, ids, weights)),
                separate_quant=separate_quant,
                **kwargs,
            )
        return recorder

    clean, corrupted = record(), record(corrupt_gateup=True)
    assert clean.structure() == corrupted.structure()
    differs = [
        a != b for (_, a), (_, b) in zip(clean.boundaries(), corrupted.boundaries())
    ]
    first = differs.index(True)  # raises if the corruption went unnoticed
    assert all(differs[first:]) and not any(differs[:first])
    assert (
        clean.boundaries()[first][1]["gateup"]
        != (corrupted.boundaries()[first][1]["gateup"])
    )
    assert "gateup" not in clean.boundaries()[first - 1][1] or first == 0
    found = first_difference(artifact(clean), artifact(corrupted))
    assert (found["field"], found["buffer"], found["call"]) == (
        "buffer",
        "gateup",
        first,
    )

    smaller = record(inputs=_inputs(_TOKENS - 128))
    assert smaller.structure() != clean.structure()
    found = first_difference(artifact(clean), artifact(smaller))
    assert found["field"] in ("grid", "constant", "operand")

    # The same allocation, the other half of it: the data and every other
    # field of every operand are the same, only the view's offset moves.
    lower, upper = record(act_half=0), record(act_half=1)
    assert lower.boundaries() == upper.boundaries()
    assert lower.structure() != upper.structure()
    found = first_difference(artifact(lower), artifact(upper))
    before, after = found["before"], found["after"]
    assert found["field"] == "operand" and before[4] == after[4] == "act"
    assert before[:5] == after[:5] and before[6:] == after[6:]
    assert before[5] == 0 and after[5] > 0

    def act_offsets(recorder, kind):
        return [
            operand[5]
            for launch in recorder.launches
            if launch["kind"] == kind
            for operand in launch["operands"]
            if operand[4] == "act"
        ]

    for kind in ("triton", "cutedsl"):
        assert set(act_offsets(lower, kind)) <= {0}
        assert act_offsets(upper, kind) == [after[5]] * len(act_offsets(lower, kind))
    assert act_offsets(lower, "triton")

    # What a CuTeDSL launch does not expose is marked, not left empty: no
    # grid; an operand outside a named buffer has no storage offset.
    cute = [launch for launch in clean.launches if launch["kind"] == "cutedsl"]
    assert cute and all(launch["grid"] == UNRECORDED for launch in cute)
    tensors = [op for launch in cute for op in launch["operands"] if op[6] is not None]
    assert tensors and all(op[3] and UNRECORDED not in (op[2], op[3]) for op in tensors)
    assert all(UNRECORDED not in op[6] for op in tensors)
    assert all((op[4] is None) == (op[5] == UNRECORDED) for op in tensors)
    assert {op[5] == UNRECORDED for op in tensors} == {True, False}
    coverage = artifact(clean)["coverage"]
    assert coverage["launches_without_grid"] == len(cute)
    assert 0 < coverage["tensor_operands_without_offset"] < coverage["tensor_operands"]

    # Saved and compared as two trees' records would be.
    same, moved = tmp_path / "same.json", tmp_path / "moved.json"
    save_artifact(same, artifact(lower))
    save_artifact(moved, artifact(upper))
    assert json.loads(same.read_text())["control"]["revision"]
    assert main([str(same), str(same)]) == 0 and main([str(same), str(moved)]) == 1


_RUNNER_CASES = {
    "bf16-cutedsl-decode": (
        lambda mp: N.test_selected_pipeline_matches_reference_in_eager_and_cuda_graph(
            mp, Phase.DECODE, 8, N._HIDDEN, N._INTERMEDIATE, N._PHYSICAL_RANK
        ),
        {"triton", "cutedsl"},
    ),
    "bf16-cutedsl-prefill": (
        lambda mp: N.test_selected_pipeline_matches_reference_in_eager_and_cuda_graph(
            mp, Phase.PREFILL, 64, N._HIDDEN, N._INTERMEDIATE, N._PHYSICAL_RANK
        ),
        {"triton", "cutedsl"},
    ),
    "fp8-cutedsl-decode": (
        N.test_fp8_base_pipeline_matches_reference_under_graph_replay,
        {"triton", "cutedsl"},
    ),
    "fp8-marlin-per-expert-prefill": (
        lambda mp: M.test_fp8_marlin_plans_replay_mixed_adapter_ranks(
            mp, "fp8", False, Phase.PREFILL
        ),
        {"triton", "marlin"},
    ),
    "fp8-marlin-per-expert-decode": (
        lambda mp: M.test_fp8_marlin_plans_replay_mixed_adapter_ranks(
            mp, "fp8", False, Phase.DECODE
        ),
        {"triton", "marlin"},
    ),
    "fp8_channel-marlin-shared-outer-decode": (
        lambda mp: M.test_fp8_marlin_plans_replay_mixed_adapter_ranks(
            mp, "fp8_channel", True, Phase.DECODE
        ),
        {"triton", "marlin"},
    ),
    "mxfp8-marlin-per-expert-prefill": (
        lambda mp: M.test_fp8_marlin_plans_replay_mixed_adapter_ranks(
            mp, "mxfp8", False, Phase.PREFILL
        ),
        {"triton", "marlin"},
    ),
    # SM100 only (the reused test skips elsewhere): the static NVFP4 prefill
    # row with its down-B fused into the base GEMM.
    "nvfp4-cutedsl-fused-prefill": (
        lambda mp: B.test_fused_static_epilogue_plan_at_the_checkpoint_scale_extremes(
            mp, 16
        ),
        {"triton", "cutedsl"},
    ),
}


@pytest.mark.parametrize("case", _RUNNER_CASES)
def test_runner_tests_launch_the_same_sequence_again(
    monkeypatch: pytest.MonkeyPatch, case: str, record_property, request
) -> None:
    reused_test_case(*_RUNNER_CASES[case], case, monkeypatch, record_property, request)


def reused_test_case(
    body, kinds: set[str], case: str, monkeypatch, record_property, request
) -> None:
    """An existing test run three times under the recorder: the second and
    third launch sequences are identical, and the structure is kept."""
    structures = []
    for _ in range(3):
        with record_launches() as recorder:
            body(monkeypatch)
        structures.append(recorder.structure())
    assert structures[1] and structures[1] == structures[2]
    assert kinds <= recorder.kinds()
    record_property("launches", len(structures[1]))
    record_property("first_run_same", structures[0] == structures[1])
    record_property("structure", recorder.structure_digest())
    # The reused test owns its buffers: its record is the structure alone.
    _keep(
        request,
        recorder.artifact(
            request.node.name, check="structure", fixture=dict(reused_test=case)
        ),
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
