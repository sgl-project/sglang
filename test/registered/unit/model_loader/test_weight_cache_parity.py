"""CPU-only unit tests for the weight-cache parity harness diff layer.

Pins down, without CUDA or a daemon: no false positives on identical
snapshots, detection of every drift class the harness claims to catch,
snapshot completeness (non-persistent buffers, tied and shared-module
aliases), and manifest JSON round-trip.
"""

import unittest

import msgspec
import torch
import torch.nn as nn

from sglang.srt.weight_cache.parity import (
    RankManifest,
    diff_manifests,
    snapshot_state,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

from sglang.test.test_utils import CustomTestCase


def _module(*, tied: bool = False) -> nn.Module:
    torch.manual_seed(0)
    model = nn.Module()
    model.linear = nn.Linear(4, 4, bias=True)
    model.register_buffer("persistent_buf", torch.ones(3))
    model.register_buffer("cache_buf", torch.arange(2.0), persistent=False)
    if tied:
        model.tied = nn.Linear(4, 4, bias=False)
        model.tied.weight = model.linear.weight
    return model


class TestSnapshotState(CustomTestCase):
    def test_snapshot_covers_non_persistent_buffers(self):
        records = snapshot_state(_module())
        self.assertEqual(records["cache_buf"].kind, "buffer")
        self.assertFalse(records["cache_buf"].persistent)
        self.assertTrue(records["persistent_buf"].persistent)
        self.assertEqual(records["linear.weight"].kind, "param")

    def test_snapshot_enumerates_shared_module_aliases(self):
        # A module shared under two parents (rotary-embedding _ROPE_DICT
        # pattern) must appear under both alias paths, or the compared name
        # set silently shrinks on the side that registered it first.
        shared = nn.Linear(2, 2, bias=False)
        model = nn.Module()
        model.layer_a = nn.Module()
        model.layer_b = nn.Module()
        model.layer_a.rope = shared
        model.layer_b.rope = shared
        records = snapshot_state(model)
        self.assertIn("layer_a.rope.weight", records)
        self.assertIn("layer_b.rope.weight", records)
        self.assertEqual(
            records["layer_a.rope.weight"].byte_hash,
            records["layer_b.rope.weight"].byte_hash,
        )

    def test_snapshot_keeps_tied_weight_aliases(self):
        records = snapshot_state(_module(tied=True))
        self.assertIn("linear.weight", records)
        self.assertIn("tied.weight", records)
        self.assertEqual(records["tied.weight"].kind, "param")
        self.assertEqual(
            records["tied.weight"].byte_hash, records["linear.weight"].byte_hash
        )


class TestDiffManifests(CustomTestCase):
    def test_identical_snapshots_diff_clean(self):
        ref = snapshot_state(_module())
        ipc = snapshot_state(_module())
        self.assertTrue(diff_manifests(ref, ipc).is_clean)

    def test_value_change_hits_byte_hash(self):
        ref = snapshot_state(_module())
        perturbed = _module()
        with torch.no_grad():
            perturbed.linear.weight[0, 0] += 1.0
        diff = diff_manifests(ref, snapshot_state(perturbed))
        self.assertEqual(
            [(m.name, m.field) for m in diff.mismatches],
            [("linear.weight", "byte_hash")],
        )

    def test_layout_change_hits_stride(self):
        base = torch.arange(4.0).reshape(2, 2)
        model_a, model_b = nn.Module(), nn.Module()
        model_a.register_buffer("w", base.contiguous())
        model_b.register_buffer("w", base.t())
        diff = diff_manifests(snapshot_state(model_a), snapshot_state(model_b))
        self.assertIn(("w", "stride"), [(m.name, m.field) for m in diff.mismatches])

    def test_missing_and_extra_names(self):
        ref = snapshot_state(_module())
        ipc = dict(ref)
        moved = ipc.pop("persistent_buf")
        ipc["unexpected_buf"] = moved
        diff = diff_manifests(ref, ipc)
        self.assertEqual(diff.missing_in_ipc, ["persistent_buf"])
        self.assertEqual(diff.extra_in_ipc, ["unexpected_buf"])
        self.assertFalse(diff.is_clean)

    def test_persistence_flip_is_detected(self):
        # Guards the bug the harness first caught live: the IPC loader
        # registering a non-persistent buffer (rotary cos_sin_cache) as
        # persistent, changing the loaded model's state_dict shape.
        model_np = nn.Module()
        model_np.register_buffer("cache", torch.ones(2), persistent=False)
        model_p = nn.Module()
        model_p.register_buffer("cache", torch.ones(2), persistent=True)
        diff = diff_manifests(snapshot_state(model_np), snapshot_state(model_p))
        self.assertEqual(
            [(m.name, m.field) for m in diff.mismatches], [("cache", "persistent")]
        )

    def test_param_to_buffer_flip_hits_kind(self):
        model_param = nn.Module()
        model_param.w = nn.Parameter(torch.ones(2), requires_grad=False)
        model_buf = nn.Module()
        model_buf.register_buffer("w", torch.ones(2))
        diff = diff_manifests(snapshot_state(model_param), snapshot_state(model_buf))
        self.assertEqual([(m.name, m.field) for m in diff.mismatches], [("w", "kind")])


class TestManifestRoundTrip(CustomTestCase):
    def test_manifest_json_round_trip(self):
        manifest = RankManifest(
            side="ref",
            tp_rank=1,
            pp_rank=0,
            records=snapshot_state(_module(tied=True)),
        )
        decoded = msgspec.json.decode(msgspec.json.encode(manifest), type=RankManifest)
        self.assertEqual(decoded, manifest)


if __name__ == "__main__":
    unittest.main()
