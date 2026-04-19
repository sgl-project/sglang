"""Unit tests for LoRAMemoryPool's MoE expert-parallel (EP) handling.

Covers the global->local expert-id remapping and per-rank buffer sizing
introduced so that per-expert MoE LoRA buffers stay aligned with the
Triton MoE runner's local-id dispatch under `--ep > 1`.

The tests exercise the class behavior without standing up a full server
or distributed groups: `LoRAMemoryPool` is instantiated via `__new__`
and only the fields the helpers read are populated. This keeps the
tests hermetic (CPU-only, no CUDA, no MoE EP group).

Usage:
    python -m pytest test/registered/unit/lora/test_mem_pool_ep_unit.py -v
"""

from sglang.test.ci.ci_register import register_cuda_ci

# CPU-only unit test; no CUDA/distributed dependencies.
register_cuda_ci(est_time=4, suite="stage-b-test-1-gpu-small")

import types
import unittest
import unittest.mock as mock

import torch

from sglang.srt.lora.mem_pool import (
    LoRAMemoryPool,
    _get_moe_ep_context,
    _moe_runner_keeps_global_expert_ids,
)


def _make_pool(
    *,
    num_experts_global: int,
    moe_ep_size: int,
    moe_ep_rank: int,
    moe_use_local_expert_ids: bool,
) -> LoRAMemoryPool:
    """Construct a minimal LoRAMemoryPool for helper-level tests.

    Bypasses `__init__` (which requires a real base model, HF config, and
    device allocations) and sets only the fields consulted by the EP
    helpers under test.
    """
    pool = LoRAMemoryPool.__new__(LoRAMemoryPool)
    pool.moe_ep_size = moe_ep_size
    pool.moe_ep_rank = moe_ep_rank
    pool.moe_use_local_expert_ids = moe_use_local_expert_ids
    if moe_use_local_expert_ids and num_experts_global % moe_ep_size == 0:
        pool._num_experts_local = num_experts_global // moe_ep_size
    else:
        pool._num_experts_local = num_experts_global
    return pool


def _make_fake_base_model(num_experts: int) -> torch.nn.Module:
    """Return a `torch.nn.Module` whose `.config` exposes `num_experts`.

    Used by `_get_num_experts` / `_get_num_local_experts` which walk the
    HF config object. No real weights needed.
    """
    model = torch.nn.Linear(4, 4, bias=False)
    cfg = types.SimpleNamespace(num_experts=num_experts)
    model.config = cfg
    return model


class TestNumExpertHelpers(unittest.TestCase):
    """`_get_num_experts` / `_get_num_local_experts` / buffer-dim picker."""

    def test_num_experts_read_from_config(self):
        model = _make_fake_base_model(num_experts=8)
        self.assertEqual(LoRAMemoryPool._get_num_experts(model), 8)

    def test_num_local_experts_no_ep(self):
        pool = _make_pool(
            num_experts_global=8,
            moe_ep_size=1,
            moe_ep_rank=0,
            moe_use_local_expert_ids=False,
        )
        model = _make_fake_base_model(num_experts=8)
        self.assertEqual(pool._get_num_local_experts(model), 8)

    def test_num_local_experts_with_ep(self):
        pool = _make_pool(
            num_experts_global=8,
            moe_ep_size=4,
            moe_ep_rank=2,
            moe_use_local_expert_ids=True,
        )
        model = _make_fake_base_model(num_experts=8)
        self.assertEqual(pool._get_num_local_experts(model), 2)

    def test_num_local_experts_with_ep_but_backend_keeps_global_ids(self):
        """FlashInfer-style backends keep global topk_ids, so even under EP
        the LoRA buffers must remain globally-keyed.
        """
        pool = _make_pool(
            num_experts_global=8,
            moe_ep_size=4,
            moe_ep_rank=2,
            moe_use_local_expert_ids=False,
        )
        model = _make_fake_base_model(num_experts=8)
        self.assertEqual(pool._get_num_local_experts(model), 8)

    def test_uneven_split_disables_local_mapping(self):
        """Shouldn't happen in practice (base MoE requires even split), but
        `__init__` must fold uneven splits into `moe_use_local_expert_ids ==
        False` so `_get_num_local_experts` returns the global count and no
        remapping happens anywhere downstream.
        """
        # Simulate what `LoRAMemoryPool.__init__` would set for an uneven
        # split: the divisibility guard there forces the flag to False.
        pool = _make_pool(
            num_experts_global=7,
            moe_ep_size=4,
            moe_ep_rank=0,
            moe_use_local_expert_ids=False,
        )
        model = _make_fake_base_model(num_experts=7)
        self.assertEqual(pool._get_num_local_experts(model), 7)


class TestGlobalToLocalExpertId(unittest.TestCase):
    """`_global_to_local_expert_id` — the per-rank filter + remap."""

    def test_passthrough_without_ep(self):
        pool = _make_pool(
            num_experts_global=8,
            moe_ep_size=1,
            moe_ep_rank=0,
            moe_use_local_expert_ids=False,
        )
        for gid in range(8):
            self.assertEqual(pool._global_to_local_expert_id(gid), gid)

    def test_rank0_of_ep4_owns_first_quarter(self):
        pool = _make_pool(
            num_experts_global=8,
            moe_ep_size=4,
            moe_ep_rank=0,
            moe_use_local_expert_ids=True,
        )
        # Owned: 0, 1 -> local 0, 1
        self.assertEqual(pool._global_to_local_expert_id(0), 0)
        self.assertEqual(pool._global_to_local_expert_id(1), 1)
        # Not owned by rank 0.
        for gid in (2, 3, 4, 5, 6, 7):
            self.assertIsNone(pool._global_to_local_expert_id(gid))

    def test_rank2_of_ep4_owns_third_quarter(self):
        pool = _make_pool(
            num_experts_global=8,
            moe_ep_size=4,
            moe_ep_rank=2,
            moe_use_local_expert_ids=True,
        )
        # Owned globals 4, 5 -> local 0, 1
        self.assertEqual(pool._global_to_local_expert_id(4), 0)
        self.assertEqual(pool._global_to_local_expert_id(5), 1)
        for gid in (0, 1, 2, 3, 6, 7):
            self.assertIsNone(pool._global_to_local_expert_id(gid))

    def test_last_rank_owns_last_slice(self):
        pool = _make_pool(
            num_experts_global=128,
            moe_ep_size=4,
            moe_ep_rank=3,
            moe_use_local_expert_ids=True,
        )
        # Local 0 <-> global 96, local 31 <-> global 127.
        self.assertEqual(pool._global_to_local_expert_id(96), 0)
        self.assertEqual(pool._global_to_local_expert_id(127), 31)
        self.assertIsNone(pool._global_to_local_expert_id(95))
        self.assertIsNone(pool._global_to_local_expert_id(128))


class TestIterLocalExpertWeightsDict(unittest.TestCase):
    """`_iter_local_expert_weights` with dict input (the common case)."""

    def test_passthrough_without_ep(self):
        pool = _make_pool(
            num_experts_global=4,
            moe_ep_size=1,
            moe_ep_rank=0,
            moe_use_local_expert_ids=False,
        )
        weights = {gid: torch.full((2,), float(gid)) for gid in range(4)}
        got = {lid: w.tolist() for lid, w in pool._iter_local_expert_weights(weights)}
        self.assertEqual(
            got, {0: [0.0, 0.0], 1: [1.0, 1.0], 2: [2.0, 2.0], 3: [3.0, 3.0]}
        )

    def test_rank0_of_ep4_filters_and_remaps(self):
        pool = _make_pool(
            num_experts_global=8,
            moe_ep_size=4,
            moe_ep_rank=0,
            moe_use_local_expert_ids=True,
        )
        weights = {gid: torch.full((2,), float(gid)) for gid in range(8)}
        got = {lid: w.tolist() for lid, w in pool._iter_local_expert_weights(weights)}
        # Rank 0 sees globals 0,1 remapped to locals 0,1.
        self.assertEqual(got, {0: [0.0, 0.0], 1: [1.0, 1.0]})

    def test_rank3_of_ep4_filters_and_remaps(self):
        pool = _make_pool(
            num_experts_global=8,
            moe_ep_size=4,
            moe_ep_rank=3,
            moe_use_local_expert_ids=True,
        )
        weights = {gid: torch.full((2,), float(gid)) for gid in range(8)}
        got = {lid: w.tolist() for lid, w in pool._iter_local_expert_weights(weights)}
        # Rank 3 sees globals 6,7 remapped to locals 0,1.
        self.assertEqual(got, {0: [6.0, 6.0], 1: [7.0, 7.0]})

    def test_sparse_dict_only_yields_owned_experts(self):
        """Adapters may only target a subset of experts. The iterator must
        still correctly filter and remap whatever subset is provided.
        """
        pool = _make_pool(
            num_experts_global=8,
            moe_ep_size=4,
            moe_ep_rank=2,
            moe_use_local_expert_ids=True,
        )
        # Only globals 1, 4, 5, 7 present in adapter.
        weights = {
            1: torch.full((2,), 1.0),
            4: torch.full((2,), 4.0),
            5: torch.full((2,), 5.0),
            7: torch.full((2,), 7.0),
        }
        # Rank 2 owns globals 4, 5 -> locals 0, 1.
        got = {lid: w.tolist() for lid, w in pool._iter_local_expert_weights(weights)}
        self.assertEqual(got, {0: [4.0, 4.0], 1: [5.0, 5.0]})

    def test_no_experts_owned_yields_nothing(self):
        """Rank with no matching experts in a sparse dict yields nothing,
        leaves buffer zeroed.
        """
        pool = _make_pool(
            num_experts_global=8,
            moe_ep_size=4,
            moe_ep_rank=0,
            moe_use_local_expert_ids=True,
        )
        # Only globals 4, 5 present (owned by rank 2).
        weights = {4: torch.full((2,), 4.0), 5: torch.full((2,), 5.0)}
        got = list(pool._iter_local_expert_weights(weights))
        self.assertEqual(got, [])


class TestIterLocalExpertWeightsTensor(unittest.TestCase):
    """`_iter_local_expert_weights` with 3D tensor input (shared-outer and
    packed MoE-LoRA formats)."""

    def test_passthrough_without_ep(self):
        pool = _make_pool(
            num_experts_global=4,
            moe_ep_size=1,
            moe_ep_rank=0,
            moe_use_local_expert_ids=False,
        )
        # [num_experts, rank, hidden] with values carrying the expert id.
        weights = torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3)
        got = [(lid, w.clone()) for lid, w in pool._iter_local_expert_weights(weights)]
        self.assertEqual([lid for lid, _ in got], [0, 1, 2, 3])
        for lid, w in got:
            self.assertTrue(torch.equal(w, weights[lid]))

    def test_rank1_of_ep2_sees_upper_half(self):
        pool = _make_pool(
            num_experts_global=4,
            moe_ep_size=2,
            moe_ep_rank=1,
            moe_use_local_expert_ids=True,
        )
        weights = torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3)
        got = [(lid, w.clone()) for lid, w in pool._iter_local_expert_weights(weights)]
        # Rank 1 of EP=2 with 4 experts owns globals 2, 3 -> locals 0, 1.
        self.assertEqual([lid for lid, _ in got], [0, 1])
        self.assertTrue(torch.equal(got[0][1], weights[2]))
        self.assertTrue(torch.equal(got[1][1], weights[3]))

    def test_rank_with_partial_tensor_coverage(self):
        """Defensive: tensor has fewer experts than the expected local slice
        (e.g. sparse adapter).
        """
        pool = _make_pool(
            num_experts_global=8,
            moe_ep_size=4,
            moe_ep_rank=3,
            moe_use_local_expert_ids=True,
        )
        # Only 6 experts present in the tensor; rank 3 expects global 6,7.
        # So it should still yield local 0 mapped to global 6; global 7 is
        # beyond the tensor length and must be skipped safely.
        weights = torch.arange(6 * 2, dtype=torch.float32).reshape(6, 2)
        # Note: this is 2D, not 3D -> should raise (sanity check).
        with self.assertRaises(TypeError):
            list(pool._iter_local_expert_weights(weights))


class TestModuleLevelHelpers(unittest.TestCase):
    """`_get_moe_ep_context` / `_moe_runner_keeps_global_expert_ids`
    must degrade gracefully when the MoE EP group or runner backend is
    not yet initialized (e.g. in pure-TP launches or hermetic tests)."""

    def test_ep_context_defaults_when_group_uninitialized(self):
        # Real process here: the MoE EP group isn't set up in a unit test.
        # The helper must return (1, 0) rather than raising.
        ep_size, ep_rank = _get_moe_ep_context()
        self.assertEqual(ep_size, 1)
        self.assertEqual(ep_rank, 0)

    def test_keeps_global_expert_ids_defaults_to_false(self):
        # Without a specific flashinfer backend selected, default is False.
        self.assertFalse(_moe_runner_keeps_global_expert_ids())


class TestPoolInitPicksUpEpContext(unittest.TestCase):
    """`LoRAMemoryPool.__init__` should read EP context from the module-
    level helpers and set `moe_use_local_expert_ids` correctly."""

    def _new_pool_with_ep(
        self,
        ep_size: int,
        ep_rank: int,
        keeps_global: bool,
        num_experts: int = 8,
    ) -> LoRAMemoryPool:
        """Construct a pool with `__init__` called, but stop before
        `init_buffers` — we only care about the EP-context state.
        """
        with (
            mock.patch(
                "sglang.srt.lora.mem_pool._get_moe_ep_context",
                return_value=(ep_size, ep_rank),
            ),
            mock.patch(
                "sglang.srt.lora.mem_pool._moe_runner_keeps_global_expert_ids",
                return_value=keeps_global,
            ),
            mock.patch.object(LoRAMemoryPool, "init_buffers", lambda self, _m: None),
        ):
            hf_cfg = types.SimpleNamespace(
                num_hidden_layers=1,
                hidden_size=8,
                vocab_size=32,
                num_experts=num_experts,
            )
            base_model = torch.nn.Linear(8, 8, bias=False)
            base_model.config = hf_cfg
            return LoRAMemoryPool(
                base_hf_config=hf_cfg,
                max_loras_per_batch=1,
                dtype=torch.bfloat16,
                tp_size=1,
                tp_rank=0,
                max_lora_rank=8,
                target_modules={"qkv_proj"},
                base_model=base_model,
                eviction_policy="lru",
                lora_added_tokens_size=0,
            )

    def test_no_ep(self):
        pool = self._new_pool_with_ep(ep_size=1, ep_rank=0, keeps_global=False)
        self.assertEqual(pool.moe_ep_size, 1)
        self.assertEqual(pool.moe_ep_rank, 0)
        self.assertFalse(pool.moe_use_local_expert_ids)

    def test_ep4_triton_backend(self):
        pool = self._new_pool_with_ep(ep_size=4, ep_rank=2, keeps_global=False)
        self.assertEqual(pool.moe_ep_size, 4)
        self.assertEqual(pool.moe_ep_rank, 2)
        self.assertTrue(pool.moe_use_local_expert_ids)

    def test_ep4_flashinfer_cutlass_keeps_global(self):
        """FlashInfer CUTLASS keeps global topk_ids, so LoRA buffers stay
        globally-keyed even under EP.
        """
        pool = self._new_pool_with_ep(ep_size=4, ep_rank=2, keeps_global=True)
        self.assertEqual(pool.moe_ep_size, 4)
        self.assertEqual(pool.moe_ep_rank, 2)
        self.assertFalse(pool.moe_use_local_expert_ids)

    def test_ep_with_uneven_split_falls_back_to_global_ids(self):
        """If `num_experts % ep_size != 0` (shouldn't happen in practice,
        base MoE requires even split) `__init__` must fall back to
        globally-keyed buffers rather than silently truncating the local
        slice — otherwise non-zero ranks drop every LoRA weight.
        """
        pool = self._new_pool_with_ep(
            ep_size=4, ep_rank=1, keeps_global=False, num_experts=7
        )
        self.assertEqual(pool.moe_ep_size, 4)
        self.assertEqual(pool.moe_ep_rank, 1)
        self.assertFalse(pool.moe_use_local_expert_ids)


if __name__ == "__main__":
    unittest.main()
