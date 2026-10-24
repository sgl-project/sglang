"""Empirical progress mechanics only; fixture rows are not native evidence."""

import hashlib
import json
from dataclasses import replace

import pytest
from sglang_simulator.time_predictor.base import PredictorError
from sglang_simulator.time_predictor.dspark_progress import (
    DEFINITION,
    EmpiricalDSparkProgress,
)
from test_dspark_step import step


def raw(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


@pytest.fixture
def artifact(tmp_path, monkeypatch):
    sdk = pytest.importorskip("infercast.sdk.native_sampling")
    from infercast.sdk.native_sources import NATIVE_SOURCE_0522
    from sglang_simulator.time_predictor import dspark_progress

    # Distribution/leakage tests isolate the native source reader. Production
    # always invokes the real shared SDK frame join, tested separately.
    batch_sizes = {}
    monkeypatch.setattr(
        dspark_progress,
        "_verify_native_progress_context",
        lambda document, read: (
            {row["native_key"]: row["native_key"] for row in document["observations"]},
            {
                row["native_key"]: batch_sizes.get(row["native_key"], 1)
                for row in document["observations"]
            },
        ),
    )

    recipe = dict.fromkeys(sdk.SAMPLING_FIELDS)
    recipe.update(
        is_normalized=True, max_new_tokens=20, temperature=1.0, top_k=1073741824
    )
    profile = '{"source":"CPU protocol fixture"}'
    pd = hashlib.sha256(profile.encode()).hexdigest()

    def row(name, trajectory, commit):
        return {
            "native_key": name,
            "forward_ct": 7,
            "rid": name,
            "trajectory_id": trajectory,
            "prefix_len": 512,
            "draft_history_tokens": 128,
            "sampling_params": recipe,
            "verify_width": 6,
            "commit_len": commit,
        }

    train = [row("train:1", "train", 2), row("train:2", "train", 4)]
    validation = [row("heldout:1", "heldout", 1)]
    document = {
        "schema_version": 1,
        "definition": DEFINITION,
        "runtime_profile_sha256": pd,
        "source_pin": NATIVE_SOURCE_0522.name,
        "seed": 7,
        "prefix_edges": [128, 1024],
        "history_edges": [128],
        "training": [],
        "validation": [],
    }
    participants = [{"tp_size": 2, "attn_dp_size": 1, "pp_size": 1, "attn_tp_size": 2}]

    def write(rows, split):
        proof = raw({"scope": "CPU protocol fixture only"})
        proof_digest = hashlib.sha256(proof).hexdigest()
        proof_path = tmp_path / f"source-{proof_digest}.json"
        proof_path.write_bytes(proof)
        value = {
            "schema_version": 1,
            "runtime_profile_sha256": pd,
            "source_pin": NATIVE_SOURCE_0522.name,
            "observations": rows,
            "native": [{"path": proof_path.name, "sha256": proof_digest}],
            "causal_history_evidence": [
                {"path": proof_path.name, "sha256": proof_digest}
            ],
            "participants": participants,
            "phase_policy": "serialized",
        }
        data = raw(value)
        digest = hashlib.sha256(data).hexdigest()
        p = tmp_path / f"progress-observations-{digest}.json"
        p.write_bytes(data)
        document[split] = [{"path": p.name, "sha256": digest}]

    def load():
        write(train, "training")
        write(validation, "validation")
        p = tmp_path / "progress.json"
        data = raw(document)
        p.write_bytes(data)
        return EmpiricalDSparkProgress(
            p,
            expected_sha256=hashlib.sha256(data).hexdigest(),
            runtime_profile_json=profile,
        )

    load.batch_sizes = batch_sizes
    load.participants = participants
    return recipe, profile, train, validation, document, load


def before(recipe, profile):
    s = step()
    r = replace(s.requests[0], sampling_params_json=raw(recipe).decode())
    return replace(
        s, groups=(replace(s.groups[0], requests=(r,)),), runtime_profile_json=profile
    )


def test_training_distribution_excludes_validation_counts_and_reports_full_error(
    artifact,
):
    recipe, profile, _, _, _, load = artifact
    model = load()
    assert next(iter(model.distributions.values())) == {2: 1, 4: 1}
    assert model.validation[0]["mean_error"] == 2
    assert model.validation[0]["total_variation"] == 1
    prediction = model.predict_progress(before(recipe, profile))
    assert prediction.committed_tokens[0][1] in {2, 4}
    assert model.predict_progress(before(recipe, profile)) == prediction


@pytest.mark.parametrize("leak", ["trajectory", "native_key", "rid"])
def test_same_trajectory_or_original_cannot_be_both_train_and_heldout(artifact, leak):
    _, _, train, heldout, _, load = artifact
    field = "trajectory_id" if leak == "trajectory" else leak
    heldout[0][field] = train[0][field]
    with pytest.raises(PredictorError, match="overlap|leakage"):
        load()


def test_no_unvalidated_condition_or_changed_recipe_fallback(artifact):
    recipe, profile, _, _, _, load = artifact
    model = load()
    changed = {**recipe, "temperature": 0.6}
    with pytest.raises(PredictorError, match="independent training/heldout"):
        model.predict_progress(before(changed, profile))
    with pytest.raises(PredictorError, match="runtime profile"):
        model.predict_progress(
            replace(before(recipe, profile), runtime_profile_json='{"source":"other"}')
        )


def test_live_tail_preserves_native_commit_label_for_scheduler_finish_clipping(
    artifact,
):
    recipe, profile, _, _, _, load = artifact
    model = load()
    s = before(recipe, profile)
    request = replace(s.requests[0], output_tokens_before=19)
    s = replace(s, groups=(replace(s.groups[0], requests=(request,)),))
    assert request.remaining_output_tokens == 1
    assert model.predict_progress(s).committed_tokens[0][1] in {2, 4}


def test_phase_preparation_can_read_training_frequencies_without_sampling(
    artifact, monkeypatch
):
    recipe, profile, _, _, _, load = artifact
    model = load()

    def no_sample(_step):
        pytest.fail("conditional phase preparation sampled progress")

    monkeypatch.setattr(model, "predict_progress", no_sample)
    s = before(recipe, profile)
    counts = model.conditional_commit_counts(s)
    assert counts == (("a", ((2, 1), (4, 1))),)
    with pytest.raises(TypeError):
        counts[0][1][0] = (6, 100)
    tail = replace(s.requests[0], output_tokens_before=19)
    assert (
        model.conditional_commit_counts(
            replace(s, groups=(replace(s.groups[0], requests=(tail,)),))
        )
        == counts
    )
    changed = {**recipe, "temperature": 0.6}
    with pytest.raises(PredictorError, match="independent training/heldout"):
        model.conditional_commit_counts(before(changed, profile))


def test_heldout_only_cells_never_appear_as_training_distributions(artifact):
    _, _, _, heldout, _, load = artifact
    heldout[0]["prefix_len"] = 5000
    model = load()
    assert len(model.distributions) == 1
    assert sum(sum(counts.values()) for counts in model.distributions.values()) == 2
    assert not model.validated_keys
    assert len(model.validation) == 2


def test_batch_support_requires_training_and_heldout_without_refitting_marginals(
    artifact,
):
    recipe, profile, train, _, _, load = artifact
    load.batch_sizes.update({row["native_key"]: 2 for row in train})
    model = load()
    assert model.validated_keys
    assert not model.validated_batch_keys
    assert model.validation[0]["training_batch_sizes"] == [2]
    assert model.validation[0]["validation_batch_sizes"] == [1]
    assert model.validation[0]["supported_batch_sizes"] == []
    assert next(iter(model.distributions.values())) == {2: 1, 4: 1}
    with pytest.raises(PredictorError, match="local batch"):
        model.conditional_commit_counts(before(recipe, profile))


def test_batch_support_uses_actual_local_group_not_total_participants(artifact):
    from sglang_simulator.simulation.sglang.dspark import DSparkTopology

    recipe, profile, _, _, _, load = artifact
    load.participants[:] = [
        {"tp_size": 2, "attn_dp_size": 2, "pp_size": 1, "attn_tp_size": 1}
    ]
    model = load()
    s = before(recipe, profile)
    other = replace(s.requests[0], rid="b")
    s = replace(
        s,
        topology=DSparkTopology(2, 2),
        groups=(
            s.groups[0],
            replace(s.groups[0], attention_dp_rank=1, requests=(other,)),
        ),
    )
    assert model.conditional_commit_counts(s) == (
        ("a", ((2, 1), (4, 1))),
        ("b", ((2, 1), (4, 1))),
    )
    unsupported = replace(
        s,
        groups=(
            replace(s.groups[0], requests=(s.requests[0], other)),
            replace(s.groups[1], forward_mode="IDLE", requests=()),
        ),
    )
    with pytest.raises(PredictorError, match="local batch"):
        model.predict_progress(unsupported)


def test_supported_batches_share_training_marginal_and_unobserved_batch_rejects(
    artifact,
):
    import copy

    recipe, profile, train, heldout, _, load = artifact
    second_heldout = copy.deepcopy(heldout[0])
    second_heldout.update(native_key="heldout:2", rid="heldout:2", commit_len=6)
    heldout.append(second_heldout)
    load.batch_sizes.update({train[1]["native_key"]: 2, "heldout:2": 2})
    model = load()
    s = before(recipe, profile)
    one = model.conditional_commit_counts(s)
    two = replace(
        s,
        groups=(
            replace(
                s.groups[0], requests=(s.requests[0], replace(s.requests[0], rid="b"))
            ),
        ),
    )
    assert model.conditional_commit_counts(two) == (
        one[0],
        ("b", one[0][1]),
    )
    assert one[0][1] == ((2, 1), (4, 1))
    heldout[0]["commit_len"] = 5
    assert load().conditional_commit_counts(s) == one
    b16 = replace(
        s,
        groups=(
            replace(
                s.groups[0],
                requests=tuple(replace(s.requests[0], rid=str(i)) for i in range(16)),
            ),
        ),
    )
    with pytest.raises(PredictorError, match="local batch"):
        model.conditional_commit_counts(b16)


def test_batch_seen_in_another_cell_is_not_support_for_query(artifact):
    recipe, profile, train, heldout, _, load = artifact
    train[1]["prefix_len"] = 5000
    heldout.append(
        {
            **heldout[0],
            "native_key": "heldout:2",
            "rid": "heldout:2",
            "prefix_len": 5000,
        }
    )
    load.batch_sizes.update({train[1]["native_key"]: 2, "heldout:2": 2})
    model = load()
    s = before(recipe, profile)
    two = replace(
        s,
        groups=(
            replace(
                s.groups[0], requests=(s.requests[0], replace(s.requests[0], rid="b"))
            ),
        ),
    )
    with pytest.raises(PredictorError, match="local batch"):
        model.predict_progress(two)


def test_missing_recipe_fields_and_current_outcome_inputs_are_rejected(artifact):
    _, _, train, _, _, load = artifact
    train[0]["current_accepted_length"] = 4
    with pytest.raises(PredictorError, match="fields"):
        load()
    train[0].pop("current_accepted_length")
    train[0]["sampling_params"] = {"temperature": 1}
    with pytest.raises(PredictorError, match="complete normalized"):
        load()


def test_native_label_verifier_uses_leader_only_and_checks_complete_peer_recipe(
    monkeypatch,
):
    import copy

    sampling = pytest.importorskip("infercast.sdk.native_sampling")
    from infercast.sdk import native_windows
    from sglang_simulator.time_predictor.dspark_progress import (
        _verify_native_progress_context,
        verify_native_progress_rows,
    )

    recipe = dict.fromkeys(sampling.SAMPLING_FIELDS)
    recipe.update(is_normalized=True, max_new_tokens=20)
    row = {
        "native_key": "first",
        "forward_ct": 7,
        "rid": "a",
        "trajectory_id": "CPU fixture",
        "prefix_len": 512,
        "draft_history_tokens": 128,
        "sampling_params": recipe,
        "verify_width": 6,
        "commit_len": 2,
    }
    peers = []
    for rank in (0, 1):
        peers.append(
            {
                "topology": {"attn_tp_rank": rank},
                "metadata": {
                    "request_ids": ["a"],
                    "mode": "DECODE",
                    "recipes": [copy.deepcopy(recipe)],
                },
                "native_step": {
                    "reqs": [
                        {"rid": "a", "prefix_len": 512, "verify_len": 6, "acc_len": 2}
                    ]
                    if rank == 0
                    else None
                },
            }
        )
    monkeypatch.setattr(
        native_windows,
        "join_native_frames",
        lambda *args, **kwargs: [{"forward_ct": 7, "participants": peers}],
    )
    monkeypatch.setattr(
        sampling, "native_sampling_params", lambda meta: meta["recipes"]
    )
    doc = {
        "native": [{"path": "CPU fixture.json", "sha256": "a" * 64}],
        "source_pin": "CPU fixture",
        "phase_policy": "serialized",
        "participants": [{"attn_tp_size": 2}],
        "observations": [row],
    }
    read = lambda ref: {"records": [], "metadata": []}
    identity = verify_native_progress_rows(doc, read)["first"]
    assert _verify_native_progress_context(doc, read) == (
        {"first": identity},
        {"first": 1},
    )
    row["native_key"] = "renamed"
    assert verify_native_progress_rows(doc, read)["renamed"] == identity
    row["commit_len"] = 3
    with pytest.raises(PredictorError, match="label or before-input"):
        verify_native_progress_rows(doc, read)
    row["commit_len"] = 2
    peers[1]["metadata"]["recipes"][0]["temperature"] = 0.6
    with pytest.raises(PredictorError, match="peer original"):
        verify_native_progress_rows(doc, read)
    peers[1]["metadata"]["recipes"][0]["temperature"] = None
    for peer in peers:
        peer["metadata"]["request_ids"].append("not-selected")
        peer["metadata"]["recipes"].append(copy.deepcopy(recipe))
    peers[0]["native_step"]["reqs"].append(
        {"rid": "not-selected", "prefix_len": 512, "verify_len": 6, "acc_len": 2}
    )
    assert len(doc["observations"]) == 1
    assert _verify_native_progress_context(doc, read)[1] == {"renamed": 2}
    peers[1]["metadata"]["request_ids"].pop()
    peers[1]["metadata"]["recipes"].pop()
    with pytest.raises(PredictorError, match="local batch size"):
        verify_native_progress_rows(doc, read)
