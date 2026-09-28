"""Whole-role graph strategy: the seam, the sentinel, and the fail-closed mode."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.afd import config as config_mod
from sglang.srt.afd import contracts as contracts
from sglang.srt.afd import integration as afd_integration
from sglang.srt.afd import profiles as profiles
from sglang.srt.afd import role_graph as role_graph
from sglang.test.afd.graph_fixtures import (
    CAPTURE_SIZES,
    RecordingDriver,
    RecordingProgram,
)
from sglang.test.afd.graph_fixtures import make_shape as planned_shape
from sglang.test.afd.graph_fixtures import role_service as _service
from sglang.test.afd.graph_fixtures import (
    role_step,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def tensor_rows(rows):
    return torch.zeros(rows, dtype=torch.bfloat16)


def _shape(*, config, stage_rows=(1, 1)):
    return planned_shape(
        lane=0,
        lane_rows=(stage_rows,),
        hidden_size=1,
        dtype="bfloat16",
        config=config,
    )


def test_fixed_graph_identity_is_part_of_actual_peer_contract_and_digest():
    adapter = SimpleNamespace(
        num_layers=2,
        hidden_size=16,
        attention_capability=lambda **kwargs: {"backend": kwargs["configured_backend"]},
    )
    descriptor = afd_integration._model_descriptor(
        capture_sizes=(8, 16, 32),
        role=contracts.AFDRole.ATTENTION,
        adapter=adapter,
        profile=profiles.QWEN3_PAIRED_C1,
        config=config_mod.AFDConfig(),
        dtype="bfloat16",
    )
    runtime = descriptor["runtime_contract"]
    assert runtime["graph_strategy"] == profiles.ROLE_GRAPH_STRATEGY_ID
    assert (
        runtime["capability_profile"]["graph_strategy"]
        == profiles.ROLE_GRAPH_STRATEGY_ID
    )
    assert descriptor["runtime_contract_digest"] == contracts.contract_digest(runtime)
    changed = dict(runtime, graph_strategy="local-compute-child-graph-v1")
    assert contracts.contract_digest(changed) != descriptor["runtime_contract_digest"]


def test_a_capture_that_writes_nothing_takes_the_role_down(monkeypatch):
    """A captured-but-empty graph replays in ~0us and reads as a huge speedup.

    Fatal rather than downgraded to eager: this role has already spent the step's
    exchange round on the replay, so recovering would need a second round the
    peer does not perform, and the pair would hang one round apart instead of
    reporting the fault.
    """

    def sentinel_static(self):
        raise contracts.AFDError("AFD_ROLE_GRAPH_SELF_TEST_SENTINEL_STATIC")

    monkeypatch.setattr(RecordingProgram, "require_sentinel_written", sentinel_static)
    driver = RecordingDriver()
    cfg, service = _service(driver=driver)
    with pytest.raises(
        contracts.AFDError,
        match="AFD_ROLE_GRAPH_SELF_TEST_SENTINEL_STATIC",
    ):
        role_step(service, cfg, step_id=1)
    assert driver.programs[0].armed == 1
    # It did replay -- that is the round the sentinel proves ran -- and the check
    # sits outside the fallback so no eager path can swallow it.
    assert driver.programs[0].replays == 1


def test_step_rows_must_match_the_shape_the_bucket_was_selected_for():
    """Replaying a graph against different real rows would return wrong tokens."""

    cfg, service = _service()
    service.begin_step(
        capture=True,
        step_id=1,
        shape=_shape(config=cfg, stage_rows=(1, 1)),
        eligible=True,
    )
    with pytest.raises(
        contracts.AFDError,
        match="AFD_ROLE_GRAPH_STEP_ROWS_INVALID",
    ):
        service.execute_step(
            stage_args=((tensor_rows(2),), (tensor_rows(1),)),
            stage_rows=(2, 1),
            forward_batches=(None, None),
            compute=lambda staged: staged,
            metadata_guards=(None, None),
        )


def test_the_usage_snapshot_names_which_strategy_produced_it(caplog):
    """One marker keeps existing operator gates matching; a field disambiguates."""

    cfg, service = _service()
    with caplog.at_level("INFO"):
        role_step(service, cfg, step_id=1)
    snapshots = [
        record.getMessage()
        for record in caplog.records
        if "AFD_GRAPH_USAGE_SNAPSHOT" in record.getMessage()
    ]
    assert snapshots
    assert '"strategy": "role"' in snapshots[-1]
    assert '"expected_operations": 1' in snapshots[-1]


def test_the_ffn_role_captures_although_it_stages_no_tensors():
    """Its inputs arrive over the wire, so it has no entry tensor to ask.

    That is why the capture device is bound at construction: deriving it from the
    step's entry tensors works for attention and is impossible for FFN, which
    would have made the whole-role graph reachable on only one of the two roles.
    """

    driver = RecordingDriver()
    cfg, service = _service(driver=driver, role=contracts.AFDRole.FFN)
    service.begin_step(
        capture=True,
        step_id=1,
        shape=_shape(config=cfg, stage_rows=(1, 1)),
        eligible=True,
    )
    ran = []
    result = service.execute_step(
        # Empty per stage: nothing is staged and nothing is extracted.
        stage_args=((), ()),
        stage_rows=(1, 1),
        forward_batches=(None, None),
        compute=lambda staged: (ran.append(staged), ((), ()))[1],
        metadata_guards=(None, None),
    )
    assert service.end_step() is None
    assert result.reason is contracts.AFDReason.ARMING
    assert len(driver.programs) == 1
    assert driver.programs[0].spec.device == "cuda:0"
    assert driver.programs[0].armed == 1


def test_a_role_graph_without_a_device_refuses_to_be_built():
    """Nothing later in an FFN step can supply one, so this must fail at startup.

    A None device would otherwise travel all the way to the graph pool handle and
    fail somewhere unrelated, on the first eligible step rather than at launch.
    """

    with pytest.raises(
        contracts.AFDError,
        match="AFD_ROLE_GRAPH_DEVICE_REQUIRED",
    ):
        role_graph.AFDRoleGraphService(
            capture_sizes=CAPTURE_SIZES,
            role=contracts.AFDRole.FFN,
            config=config_mod.AFDConfig(),
            num_layers=2,
            driver=RecordingDriver(),
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-x"]))
