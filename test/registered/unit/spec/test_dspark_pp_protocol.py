import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

# Load the dependency-free wire contract without importing CUDA model modules.
_ROOT = Path(__file__).resolve().parents[4]
register_cpu_ci = runpy.run_path(str(_ROOT / "python/sglang/test/ci/ci_register.py"))[
    "register_cpu_ci"
]
register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_contract = runpy.run_path(
    str(_ROOT / "python/sglang/srt/speculative/dspark_components/dspark_pp.py")
)
draft_owner = _contract["draft_owner"]
Identity = _contract["PPDSparkIdentity"]
validate_identities = _contract["validate_identities"]
validate_pd_contract = _contract["validate_pd_contract"]
pack_proposal = _contract["pack_proposal"]
unpack_proposal = _contract["unpack_proposal"]
owned_token_rows = _contract["owned_token_rows"]


def test_owner_partition_is_stable_and_disjoint():
    rids = [f"request-{i}" for i in range(100)]
    partitions = [
        {rid for rid in rids if draft_owner(rid, 2) == rank} for rank in range(2)
    ]
    assert partitions[0] and partitions[1]
    assert not partitions[0] & partitions[1]
    assert partitions[0] | partitions[1] == set(rids)
    assert [draft_owner(rid, 2) for rid in rids] == [
        draft_owner(rid, 2) for rid in rids
    ]
    with pytest.raises(ValueError, match="positive"):
        draft_owner("request", 0)


@pytest.mark.parametrize(
    "received",
    [
        [("other", (3, 0), 4)],
        [("r", (2, 0), 4)],
        [("r", (3, 1), 4)],
        [("r", (3, 0), 3)],
        [],
        [("r", (3, 0), 5)],
    ],
)
def test_identity_rejects_wrong_request_generation_or_round(received):
    with pytest.raises(RuntimeError, match="Stale or mismatched"):
        validate_identities([Identity("r", (3, 0), 4)], received)


def test_identity_round_and_wire_round_trip():
    identity = Identity.from_req(
        SimpleNamespace(
            rid="r", bootstrap_room=321, retraction_count=2, spec_verify_ct=5
        )
    )
    assert identity.to_wire() == ("r", (321, 2), 5)
    assert identity.next_round().to_wire() == ("r", (321, 2), 6)
    validate_identities([identity], [["r", [321, 2], 5]])
    with pytest.raises(ValueError, match="generation"):
        Identity.from_req(
            SimpleNamespace(
                rid="r",
                bootstrap_room=None,
                retraction_count=0,
                spec_verify_ct=0,
            )
        )


@pytest.mark.parametrize("local,remote", [(True, False), (False, True)])
def test_pd_rejects_asymmetric_mode(local, remote):
    with pytest.raises(ValueError, match="both prefill and decode"):
        validate_pd_contract(local, remote, 2)


def test_pd_protocol_topology():
    validate_pd_contract(True, True, 2)
    validate_pd_contract(False, False, 8)
    with pytest.raises(ValueError, match="PP2"):
        validate_pd_contract(True, True, 1)


def test_owner_proposals_remain_flat_and_independent():
    tensor0, tensor1 = object(), object()
    first = {"identities": [("a", (1, 0), 2)], "draft_tokens": tensor0}
    second = {"identities": [("b", (3, 0), 4)], "draft_tokens": tensor1}
    wire = {**pack_proposal(0, first), **pack_proposal(1, second)}
    assert unpack_proposal(0, wire) == first
    assert unpack_proposal(1, wire) == second
    assert wire["dspark_next_0_draft_tokens"] is tensor0
    assert not any(isinstance(value, dict) for value in wire.values())
    with pytest.raises(RuntimeError, match="Missing"):
        unpack_proposal(1, pack_proposal(0, first))


def test_context_selection_covers_each_token_exactly_once():
    rids = [f"req-{i}" for i in range(9)]
    counts = [3, 1, 0, 5, 2, 1, 4, 7, 2]
    selected = [owned_token_rows(rids, counts, rank, 2) for rank in range(2)]
    assert sorted(selected[0][0] + selected[1][0]) == list(range(len(rids)))
    assert sorted(selected[0][1] + selected[1][1]) == list(range(sum(counts)))
    for rank, (rows, tokens) in enumerate(selected):
        assert all(draft_owner(rids[row], 2) == rank for row in rows)
        assert len(tokens) == sum(counts[row] for row in rows)
    assert owned_token_rows([], [], 0, 2) == ([], [])
    with pytest.raises(ValueError, match="token counts"):
        owned_token_rows(["r"], [], 0, 2)
