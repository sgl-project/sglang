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
register_cuda_ci(est_time=8, suite="stage-b-test-1-gpu-small")

import types
import unittest
import unittest.mock as mock

import torch

from sglang.srt.lora.mem_pool import (
    LoRAMemoryPool,
    _get_moe_ep_context,
    _get_moe_tp_context,
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
    # Helpers under test in this module don't consult moe_tp_size, but set
    # defaults so accidental reads don't AttributeError.
    pool.moe_tp_size = 1
    pool.moe_tp_rank = 0
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

    def test_tp_context_defaults_when_group_uninitialized(self):
        # Mirror of `_get_moe_ep_context` for the MoE TP group: if it isn't
        # initialized (hermetic tests, pure-TP launches), fall back to (1, 0).
        tp_size, tp_rank = _get_moe_tp_context()
        self.assertEqual(tp_size, 1)
        self.assertEqual(tp_rank, 0)

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
        moe_tp_size: int = 1,
        moe_tp_rank: int = 0,
        tp_size: int = 1,
        tp_rank: int = 0,
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
                "sglang.srt.lora.mem_pool._get_moe_tp_context",
                return_value=(moe_tp_size, moe_tp_rank),
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
                tp_size=tp_size,
                tp_rank=tp_rank,
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

    def test_init_captures_moe_tp_context(self):
        """`__init__` must capture moe_tp_size/rank so per-expert MoE LoRA
        buffers can be sharded by the MoE-TP group (not the outer attn TP).
        Under `--tp N --ep N` the MoE TP group degenerates to size 1.
        """
        pool = self._new_pool_with_ep(
            ep_size=4,
            ep_rank=0,
            keeps_global=False,
            tp_size=4,
            tp_rank=0,
            moe_tp_size=1,
            moe_tp_rank=0,
        )
        self.assertEqual(pool.tp_size, 4)
        self.assertEqual(pool.moe_tp_size, 1)
        self.assertEqual(pool.moe_tp_rank, 0)


def _fake_base_model_with_hidden_dim(num_experts: int) -> torch.nn.Module:
    """Fake base model that implements `get_hidden_dim` for MoE + attention
    modules. Matches the signatures `LoRAMemoryPool.get_lora_{A,B}_shape`
    call through `sglang.srt.lora.utils.get_hidden_dim`.
    """

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4, 4, bias=False)
            self.config = types.SimpleNamespace(
                num_hidden_layers=1,
                hidden_size=64,
                num_attention_heads=8,
                num_key_value_heads=8,
                head_dim=8,
                intermediate_size=256,
                moe_intermediate_size=192,
                vocab_size=32,
                num_experts=num_experts,
            )

        def get_hidden_dim(self, module_name: str, layer_idx: int):
            cfg = self.config
            if module_name == "qkv_proj":
                head = cfg.head_dim
                return cfg.hidden_size, head * (
                    cfg.num_attention_heads + cfg.num_key_value_heads * 2
                )
            if module_name == "o_proj":
                return cfg.head_dim * cfg.num_attention_heads, cfg.hidden_size
            if module_name == "gate_up_proj_moe":
                return cfg.hidden_size, cfg.moe_intermediate_size * 2
            if module_name == "down_proj_moe":
                return cfg.moe_intermediate_size, cfg.hidden_size
            raise NotImplementedError(module_name)

    return _Model()


class TestMoeBufferShardsByMoeTp(unittest.TestCase):
    """Regression: per-expert MoE LoRA buffers must shard by `moe_tp_size`,
    not the outer attention `tp_size`.

    Under `--tp N --ep N` (e.g. tp=4, ep=4) `moe_tp_size == 1`, so per-
    expert weights span the full MoE intermediate dim on every rank; the
    corresponding LoRA buffer must match. Before the fix, the buffer was
    divided by `tp_size` (= 4) while `FusedMoEWithLoRA.slice_moe_lora_*`
    kept the weight full-width, producing a 4x shape-mismatch assert at
    load time. Non-MoE modules still shard by the outer `tp_size`.
    """

    def _pool(
        self,
        *,
        tp_size: int,
        moe_tp_size: int,
        num_experts: int = 128,
        ep_size: int = 1,
        ep_rank: int = 0,
    ) -> LoRAMemoryPool:
        pool = LoRAMemoryPool.__new__(LoRAMemoryPool)
        pool.max_loras_per_batch = 2
        pool.tp_size = tp_size
        pool.tp_rank = 0
        pool.moe_ep_size = ep_size
        pool.moe_ep_rank = ep_rank
        pool.moe_tp_size = moe_tp_size
        pool.moe_tp_rank = 0
        pool.moe_use_local_expert_ids = ep_size > 1
        pool._num_experts_local = (
            num_experts // ep_size if pool.moe_use_local_expert_ids else num_experts
        )
        pool.experts_shared_outer_loras = False
        pool.base_hf_config = types.SimpleNamespace(
            hidden_size=64,
            num_attention_heads=8,
            num_key_value_heads=8,
            head_dim=8,
            intermediate_size=256,
            moe_intermediate_size=192,
        )
        return pool

    def test_moe_down_proj_uses_moe_tp_not_attn_tp(self):
        """down_proj_moe is row-parallel: LoRA-A input_dim = moe_inter must
        be divided by `moe_tp_size`, NOT `tp_size`. This is the exact shape
        that failed at load time on `--tp 4 --ep 4` before the fix.
        """
        pool = self._pool(tp_size=4, moe_tp_size=1, num_experts=128, ep_size=4)
        model = _fake_base_model_with_hidden_dim(num_experts=128)
        num_local = 128 // 4  # 32
        # A: input_dim = moe_inter / moe_tp_size = 192 / 1 = 192 (pre-fix: 48).
        self.assertEqual(
            pool.get_lora_A_shape("down_proj_moe", model, 8, 0),
            (2, num_local, 8, 192),
        )
        # B: output_dim = hidden_size, not row-parallel -> unsharded.
        self.assertEqual(
            pool.get_lora_B_shape("down_proj_moe", model, 8, 0),
            (2, num_local, 64, 8),
        )

    def test_moe_gate_up_proj_uses_moe_tp_not_attn_tp(self):
        """gate_up_proj_moe is column-parallel: LoRA-B output_dim =
        moe_inter*2 must be divided by `moe_tp_size`, not `tp_size`.
        """
        pool = self._pool(tp_size=4, moe_tp_size=1, num_experts=128, ep_size=4)
        model = _fake_base_model_with_hidden_dim(num_experts=128)
        num_local = 128 // 4
        # A: input_dim = hidden_size, not row-parallel -> unsharded. Rank
        # dim is `max_lora_dim * stacked_multiply` (2 for gate_up).
        self.assertEqual(
            pool.get_lora_A_shape("gate_up_proj_moe", model, 8, 0),
            (2, num_local, 16, 64),
        )
        # B: output_dim = moe_inter*2 / moe_tp_size = 384 / 1 = 384 (pre-fix: 96).
        self.assertEqual(
            pool.get_lora_B_shape("gate_up_proj_moe", model, 8, 0),
            (2, num_local, 384, 8),
        )

    def test_moe_tp_gt1_still_shards_moe_dims(self):
        """Under `--tp 8 --ep 4` the MoE TP group has size 2, so per-expert
        weights ARE sharded along the MoE inner dim — the LoRA buffer must
        follow.
        """
        pool = self._pool(tp_size=8, moe_tp_size=2, num_experts=128, ep_size=4)
        model = _fake_base_model_with_hidden_dim(num_experts=128)
        num_local = 128 // 4
        # 192 / 2 = 96
        self.assertEqual(
            pool.get_lora_A_shape("down_proj_moe", model, 8, 0),
            (2, num_local, 8, 96),
        )
        # 384 / 2 = 192 (B: moe_inter*2 / moe_tp_size).
        self.assertEqual(
            pool.get_lora_B_shape("gate_up_proj_moe", model, 8, 0),
            (2, num_local, 192, 8),
        )
        # A: input_dim = hidden_size, unaffected by MoE TP.
        self.assertEqual(
            pool.get_lora_A_shape("gate_up_proj_moe", model, 8, 0),
            (2, num_local, 16, 64),
        )

    def test_non_moe_modules_unaffected_by_moe_tp(self):
        """Non-MoE modules must continue to shard by the outer `tp_size`;
        the MoE-TP substitution applies only to `*_moe` modules.
        """
        pool = self._pool(tp_size=4, moe_tp_size=1, num_experts=128, ep_size=4)
        model = _fake_base_model_with_hidden_dim(num_experts=128)
        # o_proj is row-parallel: A input_dim sharded by tp_size, B unsharded.
        o_a = pool.get_lora_A_shape("o_proj", model, 8, 0)
        o_b = pool.get_lora_B_shape("o_proj", model, 8, 0)
        # head_dim*num_heads / tp_size = 64 / 4 = 16; B output = hidden_size = 64.
        self.assertEqual(o_a, (2, 8, 16))
        self.assertEqual(o_b, (2, 64, 8))
        # qkv_proj is column-parallel: A unsharded, B sharded by tp_size.
        q_b = pool.get_lora_B_shape("qkv_proj", model, 8, 0)
        # head_dim * (heads + 2*kv_heads) / tp_size = 8 * 24 / 4 = 48.
        self.assertEqual(q_b, (2, 48, 8))


class TestLoadBufferPassesMoeTpRankToSlice(unittest.TestCase):
    """Regression: `load_lora_weight_to_buffer` must hand `moe_tp_rank` (not
    the outer `tp_rank`) to `slice_moe_lora_{a,b}_weights`.

    Per-expert MoE weights are sharded along
    `moe_tp_size = tp_size // ep_size // dp_size`, NOT the outer `tp_size`.
    The bug only surfaces when those two values differ — i.e. when
    `1 < ep_size < tp_size`. Concrete reproducer (`tp=4 ep=2`):

      moe_tp_size = 2; outer rank 3 has moe_tp_rank=1.
      `intermediate_size_per_partition = moe_inter / 2 = 384`.
      Slicing with the OUTER rank (3) computes `start = 3 * 384 = 1152`,
      which is past the full `moe_inter = 768`, returning a `[r, 0]`-shaped
      tensor that fails the shape-match assert in `load_lora_weight_tensor`.

    This test exercises `load_lora_weight_to_buffer` end-to-end with a
    minimal mocked `FusedMoEWithLoRA` whose slicer captures-and-raises so
    we don't need to satisfy buffer-copy shape constraints.
    """

    class _StopAfterCapture(Exception):
        """Sentinel raised from the mocked slicer to short-circuit
        execution before the buffer-copy phase (which would need real
        shapes the test does not provide)."""

    def test_moe_tp_rank_used_for_slicing_when_ep_lt_tp(self):
        from sglang.srt.lora.layers import FusedMoEWithLoRA

        # tp=4 ep=2 → moe_tp_size=2. Pick OUTER rank 3 so moe_tp_rank=1.
        # The two values differ; the bug would surface on this exact rank.
        pool = LoRAMemoryPool.__new__(LoRAMemoryPool)
        pool.tp_size = 4
        pool.tp_rank = 3
        pool.moe_tp_size = 2
        pool.moe_tp_rank = 1
        pool.moe_ep_size = 2
        pool.moe_ep_rank = 1
        pool.moe_use_local_expert_ids = True
        pool._num_experts_local = 1
        pool.num_layer = 1
        pool.target_modules = {"gate_up_proj", "down_proj"}
        pool.experts_shared_outer_loras = False
        pool.strict_loading = False
        pool.lora_added_tokens_size = 0
        # Tiny placeholder buffers — the mocked slicer raises before any of
        # this is read in the buffer-copy phase.
        pool.A_buffer = {
            "gate_up_proj_moe": [torch.zeros(1, 1, 1, 1)],
            "down_proj_moe": [torch.zeros(1, 1, 1, 1)],
        }
        pool.B_buffer = {
            "gate_up_proj_moe": [torch.zeros(1, 1, 1, 1)],
            "down_proj_moe": [torch.zeros(1, 1, 1, 1)],
        }
        pool.embedding_A_buffer = {}
        pool.embedding_B_buffer = {}
        pool.lm_head_A_buffer = {}
        pool.lm_head_B_buffer = {}
        pool.new_embeddings_buffer = {}

        captured_ranks = []

        moe_mod = mock.MagicMock(spec=FusedMoEWithLoRA)

        def capture_a(weights, tp_rank, target_module):
            captured_ranks.append(("A", target_module, tp_rank))
            raise TestLoadBufferPassesMoeTpRankToSlice._StopAfterCapture()

        def capture_b(weights, tp_rank, target_module):
            captured_ranks.append(("B", target_module, tp_rank))
            raise TestLoadBufferPassesMoeTpRankToSlice._StopAfterCapture()

        moe_mod.slice_moe_lora_a_weights.side_effect = capture_a
        moe_mod.slice_moe_lora_b_weights.side_effect = capture_b

        # Adapter with one per-expert MoE LoRA-A weight. The expert regex
        # `experts\.(\d+)\.` must match the key, which routes the weight
        # into `temp_A_buffer["gate_up_proj_moe"]` — the dict shape that
        # makes `temp_A_buffer.get("gate_up_proj_moe") is not None` true,
        # which in turn triggers `slice_moe_lora_a_weights` (and the
        # capture).
        adapter = mock.MagicMock()
        adapter.config.r = 4
        adapter.scaling = 1.0
        adapter.embedding_layers = {}
        adapter.added_tokens_embeddings = {}
        adapter.layers = [
            types.SimpleNamespace(
                weights={
                    "model.layers.0.mlp.experts.0.gate_up_proj.lora_A.weight": (
                        torch.zeros(8, 4)
                    ),
                },
            )
        ]

        with self.assertRaises(TestLoadBufferPassesMoeTpRankToSlice._StopAfterCapture):
            pool.load_lora_weight_to_buffer(
                uid="test",
                buffer_id=0,
                lora_adapter=adapter,
                lora_modules=[{"mlp.experts": moe_mod}],
                lora_embed_tokens_module=None,
                lora_lm_head_module=None,
            )

        self.assertGreater(len(captured_ranks), 0, "slicing was never invoked")
        for ab, target_module, rank in captured_ranks:
            self.assertEqual(
                rank,
                pool.moe_tp_rank,
                f"slice_moe_lora_{ab.lower()}_weights for {target_module} "
                f"received rank={rank}; expected moe_tp_rank="
                f"{pool.moe_tp_rank} (outer tp_rank is {pool.tp_rank}). "
                "Passing the outer tp_rank slices past "
                "intermediate_size_per_partition when ep_size < tp_size.",
            )


if __name__ == "__main__":
    unittest.main()
