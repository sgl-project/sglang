"""GDN tree verify without per-node snapshots vs the snapshotting tree verify:
the outputs agree to rounding, and the commit fold of any accepted path
reproduces the snapshot of its last node."""

import random
import unittest
from dataclasses import dataclass

import torch

from sglang.kernels.ops.attention.linear.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.kernels.ops.attention.linear.gdn_replayssm_spec_fold import (
    commit_gdn_replayssm_fold_all_layers,
)
from sglang.kernels.ops.attention.linear.gdn_tree_verify import gdn_tree_verify
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-large")

DEVICE = "cuda"
NUM_SLOTS, NUM_ROWS = 8, 4
SLOTS, TRACK_SLOTS, ROWS = [5, 2, 7], [1, 4, 6], [3, 0, 1]
# The fold and the snapshot kernel normalize k in different ways.
FOLD_TOL = {"rtol": 1e-6, "atol": 2 * torch.finfo(torch.float32).eps}


@dataclass(frozen=True)
class Geometry:
    T: int  # draft tokens per request
    H: int
    HV: int
    K: int
    V: int
    dtype: torch.dtype = torch.bfloat16
    ring_len: int = 0  # 0: as long as the tree
    packed: bool = False  # q, k, v as strided views of one tensor

    @property
    def parents(self) -> list[list[int]]:
        """Per request: a chain, a star, and a random tree. The root's parent
        is 0, as the conv kernel writes it."""
        rng = random.Random(0)
        return [
            [0] + list(range(self.T - 1)),
            [0] * self.T,
            [0] + [rng.randrange(t) for t in range(1, self.T)],
        ]


GEOMETRIES = [
    Geometry(T=8, H=4, HV=8, K=64, V=64, ring_len=16),
    Geometry(T=16, H=2, HV=4, K=128, V=128, packed=True),
    Geometry(T=24, H=2, HV=2, K=64, V=64),
    Geometry(T=5, H=2, HV=4, K=48, V=40),
    Geometry(T=8, H=2, HV=2, K=64, V=64, dtype=torch.float32),
]


def _path(parents, node):
    path = [node]
    while node > 0:
        node = parents[node]
        path.append(node)
    return path[::-1]


class _Case:
    def __init__(self, geo: Geometry, slots=SLOTS, rows=ROWS):
        self.geo = geo
        self.B = B = len(geo.parents)
        T, H, HV, K, V = geo.T, geo.H, geo.HV, geo.K, geo.V
        ring_len = geo.ring_len or T
        gen = torch.Generator(device=DEVICE).manual_seed(0)

        def rand(*shape, dtype=geo.dtype):
            return torch.randn(*shape, device=DEVICE, dtype=dtype, generator=gen)

        self.inputs = {
            "A_log": rand(HV, dtype=torch.float32) * 0.1,
            "dt_bias": rand(HV, dtype=torch.float32) * 0.1,
            "q": rand(1, B * T, H, K),
            "k": rand(1, B * T, H, K),
            "v": rand(1, B * T, HV, V),
        }
        if geo.packed:
            sizes = [H * K, H * K, HV * V]
            packed = torch.cat([self.inputs[n].flatten(2) for n in "qkv"], dim=-1)
            for name, part in zip("qkv", packed.split(sizes, dim=-1)):
                self.inputs[name] = part.unflatten(-1, self.inputs[name].shape[2:])
        self.inputs |= {
            "a": rand(B * T, HV),
            "b": rand(B * T, HV),
        }
        self.state = rand(1, NUM_SLOTS, HV, V, K, dtype=torch.float32)
        self.slots = torch.tensor(slots, dtype=torch.int32, device=DEVICE)
        self.rows = torch.tensor(rows, dtype=torch.int32, device=DEVICE)
        self.parents = torch.tensor(geo.parents, dtype=torch.int64, device=DEVICE)

        def ring(*shape, dtype):
            return torch.zeros(1, NUM_ROWS, *shape, device=DEVICE, dtype=dtype)

        self.rings = {
            "rawv_cache": ring(HV, ring_len, V, dtype=geo.dtype),
            "rawk_cache": ring(H, ring_len, K, dtype=geo.dtype),
            "g_cache": ring(HV, ring_len, dtype=torch.float32),
            "beta_cache": ring(HV, ring_len, dtype=torch.float32),
        }

    def snapshot_verify(self):
        geo, B = self.geo, self.B
        snapshots = torch.zeros(B, geo.T, geo.HV, geo.V, geo.K, device=DEVICE)
        out = fused_sigmoid_gating_delta_rule_update(
            **self.inputs,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            initial_state_source=self.state[0],
            initial_state_indices=self.slots,
            cu_seqlens=torch.arange(0, B * geo.T + 1, geo.T, device=DEVICE).int(),
            use_qk_l2norm_in_kernel=True,
            is_kda=False,
            disable_state_update=True,
            intermediate_states_buffer=snapshots,
            intermediate_state_indices=torch.arange(B, device=DEVICE).int(),
            cache_steps=geo.T,
            retrieve_parent_token=self.parents,
        )
        return out, snapshots

    def tree_verify(self):
        return gdn_tree_verify(
            **self.inputs,
            initial_state_source=self.state[0],
            initial_state_indices=self.slots,
            ring_indices=self.rows,
            retrieve_parent_token=self.parents,
            replayssm_rawv=self.rings["rawv_cache"][0],
            replayssm_rawk=self.rings["rawk_cache"][0],
            replayssm_g=self.rings["g_cache"][0],
            replayssm_beta=self.rings["beta_cache"][0],
        )

    def fold(self, last: int, track_depth: int | None = None):
        """Commit the path to node ``last`` of every request. Returns the state
        and, with ``track_depth``, the node tracked on each path (-1: none)."""
        geo, B = self.geo, self.B
        paths = [_path(parents, last) for parents in geo.parents]
        accept_index = torch.full((B, geo.T), -1, dtype=torch.int64, device=DEVICE)
        for i, path in enumerate(paths):
            accept_index[i, : len(path)] = torch.tensor(path) + i * geo.T
        track = {}
        if track_depth is not None:
            nodes = [p[track_depth] if track_depth < len(p) else -1 for p in paths]
            track = {
                "mamba_track_indices": torch.tensor(TRACK_SLOTS, device=DEVICE).int(),
                "mamba_steps_to_track": torch.tensor(nodes, device=DEVICE).int(),
            }
        state = self.state.clone()
        commit_gdn_replayssm_fold_all_layers(
            checkpoint_state=state,
            **self.rings,
            ssm_state_indices=self.slots,
            accept_lens=torch.tensor([len(p) for p in paths], device=DEVICE).int(),
            max_cache_len=geo.ring_len or geo.T,
            num_k_heads=geo.H,
            ring_indices=self.rows,
            accept_index=accept_index,
            draft_token_num=geo.T,
            **track,
        )
        return state, track.get("mamba_steps_to_track")


class TestGdnTreeVerify(CustomTestCase):
    def test_output_matches_snapshot_verify(self):
        for geo in GEOMETRIES:
            with self.subTest(geo=geo):
                case = _Case(geo)
                state_before = case.state.clone()
                expected, _ = case.snapshot_verify()
                out = case.tree_verify()
                # The same terms summed in another order: a bf16 result may
                # land on a neighbouring value, and only rarely.
                bf16 = geo.dtype == torch.bfloat16
                rtol = 2 * torch.finfo(geo.dtype).eps if bf16 else 1e-4
                torch.testing.assert_close(
                    out.float(), expected.float(), rtol=rtol, atol=1e-5
                )
                if bf16:
                    self.assertLess((out != expected).float().mean().item(), 0.01)
                self.assertTrue(torch.equal(case.state, state_before))

    def test_padded_request_gets_zero_output(self):
        geo = GEOMETRIES[0]
        padded = {"slots": [5, -1, 7], "rows": [3, -1, 1]}
        for name, value in padded.items():
            with self.subTest(padded=name):
                case = _Case(geo, **{name: value})
                out = case.tree_verify().view(case.B, geo.T, geo.HV, geo.V)
                self.assertEqual(out[1].abs().max().item(), 0.0)
                self.assertGreater(out[0].abs().max().item(), 0.0)
                # No live request writes ring rows 0 and 2 in either case.
                for ring in case.rings.values():
                    self.assertEqual(ring[0, [0, 2]].abs().max().item(), 0.0)

    def test_non_finite_node_stays_in_its_subtree(self):
        geo = GEOMETRIES[0]
        clean = _Case(geo).tree_verify().view(-1, geo.T, geo.HV, geo.V)
        case = _Case(geo)
        bad = 1
        rows = torch.arange(case.B, device=DEVICE) * geo.T + bad
        for name in "qkv":
            case.inputs[name][0, rows] = float("nan")
        for name in "ab":
            case.inputs[name][rows] = float("nan")
        out = case.tree_verify().view(-1, geo.T, geo.HV, geo.V)
        for i, parents in enumerate(geo.parents):
            subtree = [bad in _path(parents, t) for t in range(geo.T)]
            outside = [t for t in range(geo.T) if not subtree[t]]
            self.assertTrue(out[i, subtree].isnan().all())
            self.assertTrue(torch.equal(out[i, outside], clean[i, outside]))

    def test_fold_of_any_path_matches_snapshot(self):
        for geo in GEOMETRIES:
            case = _Case(geo)
            _, snapshots = case.snapshot_verify()
            case.tree_verify()
            for last in range(geo.T):
                with self.subTest(geo=geo, last=last):
                    state, _ = case.fold(last)
                    torch.testing.assert_close(
                        state[0, SLOTS], snapshots[:, last], **FOLD_TOL
                    )

    def test_fold_tracks_the_state_after_a_path_node(self):
        geo = GEOMETRIES[0]
        case = _Case(geo)
        _, snapshots = case.snapshot_verify()
        case.tree_verify()
        state, tracked = case.fold(geo.T - 1, track_depth=2)
        self.assertIn(-1, tracked.tolist())
        for i, node in enumerate(tracked.tolist()):
            expected = case.state[0, TRACK_SLOTS[i]] if node < 0 else snapshots[i, node]
            torch.testing.assert_close(state[0, TRACK_SLOTS[i]], expected, **FOLD_TOL)


if __name__ == "__main__":
    unittest.main()
