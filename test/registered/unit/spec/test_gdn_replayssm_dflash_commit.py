"""GDN ReplaySSM spec-verify ring: configuration checks and commit routing.

Under ``--enable-linear-replayssm-spec`` a GDN verify appends the draft window
to a compact request ring of ``--linear-replayssm-cache-len`` positions and the
commit folds the accepted prefix into the checkpoint. DSPARK and DFLASH commit
through ``HybridLinearAttnBackend.update_mamba_state_after_mtp_verify``, which
goes through the same ring commit as the generic spec commit.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.configs.mamba_utils import (
    Mamba2CacheParams,
    Mamba2StateDType,
    Mamba2StateShape,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.layers.attention.linear.gdn_backend import (
    _validate_gdn_target_verify_layout,
)
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_KERNELS = "sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode"
_BACKEND = "sglang.srt.layers.attention.hybrid_linear_attn_backend"
_GDN_COMMIT = f"{_KERNELS}.commit_gdn_replayssm_after_verify"
DRAFT_TOKENS = 8


def _gdn_cache_params(num_layers=2):
    shape = Mamba2StateShape.create(
        tp_world_size=1,
        intermediate_size=64,
        n_groups=2,
        num_heads=4,
        head_dim=16,
        state_size=16,
        conv_kernel=4,
    )
    dtype = Mamba2StateDType(conv=torch.bfloat16, temporal=torch.bfloat16)
    return Mamba2CacheParams(shape=shape, dtype=dtype, layers=list(range(num_layers)))


def _spec_state(*, intermediate_ssm=None, temporal_dtype=torch.bfloat16):
    temporal = torch.zeros(1, 8, 2, 4, 4, dtype=temporal_dtype)
    return MambaPool.SpeculativeState(
        conv=[torch.zeros(1, 8, 3, 2)],
        temporal=temporal,
        intermediate_ssm=intermediate_ssm,
        intermediate_conv_window=[torch.zeros(1, 4, DRAFT_TOKENS, 3, 2)],
        replayssm_d=torch.zeros(1, 4, 2, 16, 4),
    )


def _gdn_pool():
    return SimpleNamespace(
        replayssm_is_kda=False,
        replayssm_spec_fold=False,
        replayssm_spec_write_pos=torch.zeros(40, dtype=torch.int32),
        replayssm_cache_base=torch.zeros(40, dtype=torch.int32),
        replayssm_is_flush=torch.zeros(40, dtype=torch.int8),
    )


def _hybrid_backend(mamba_pool, spec_state, cache_indices):
    """A HybridLinearAttnBackend whose sidecar translates virtual mamba ids by
    +100, so the test can tell translated ids from raw ones."""
    req_to_token_pool = SimpleNamespace(
        mamba_pool=mamba_pool,
        get_speculative_mamba2_params_all_layers=lambda: spec_state,
        get_mamba_indices=lambda rows: rows // 2,
        short_conv_pool=SimpleNamespace(conv_state=None, intermediate_conv_state=None),
        ngram_pool=SimpleNamespace(context=None, intermediate_context=None),
    )
    hybrid = HybridLinearAttnBackend.__new__(HybridLinearAttnBackend)
    hybrid.linear_attn_backend = SimpleNamespace(
        req_to_token_pool=req_to_token_pool,
        forward_metadata=SimpleNamespace(mamba_cache_indices=cache_indices),
        accept_lens_pool=None,
        _translate_mamba_indices=lambda ids: ids + 100,
    )
    return hybrid


class TestGDNReplaySSMRingLength(CustomTestCase):
    @staticmethod
    def _pool(ring_len, draft_tokens=DRAFT_TOKENS):
        return MambaPool(
            size=4,
            spec_state_size=3,
            cache_params=_gdn_cache_params(),
            mamba_layer_ids=[0, 1],
            device="cpu",
            speculative_num_draft_tokens=draft_tokens,
            speculative_eagle_topk=None,
            linear_replayssm_cache_len=ring_len,
            enable_linear_replayssm_spec=True,
        )

    def test_ring_must_hold_the_draft_window(self):
        with self.assertRaisesRegex(ValueError, "ring too small"):
            self._pool(ring_len=16, draft_tokens=24)
        # The compact commit needs at least 16 ring positions.
        with self.assertRaisesRegex(ValueError, "ring too small"):
            self._pool(ring_len=DRAFT_TOKENS)
        with self.assertRaisesRegex(ValueError, "power of two"):
            self._pool(ring_len=12)

    def test_ring_without_snapshots(self):
        for ring_len in (16, 32):
            with self.subTest(ring_len=ring_len):
                pool = self._pool(ring_len=ring_len)
                state = pool.get_speculative_mamba2_params_all_layers()
                hv, v_dim, _ = state.temporal.shape[2:]
                self.assertIsNone(state.intermediate_ssm)
                # Ring rows follow request rows (spec_state_size + 1).
                self.assertEqual(
                    tuple(state.replayssm_d.shape), (2, 4, hv, ring_len, v_dim)
                )
                self.assertEqual(pool.replayssm_cache_base.shape[0], 4)


class TestGDNReplaySSMBackendCommit(CustomTestCase):
    def _commit(self, hybrid, last_correct, req_rows, **kwargs):
        hybrid.update_mamba_state_after_mtp_verify(
            last_correct_step_indices=last_correct,
            mamba_track_indices=kwargs.get("track_indices"),
            mamba_steps_to_track=kwargs.get("track_steps"),
            model=None,
            req_pool_indices=req_rows,
        )

    def test_gdn_ring_commit_replaces_snapshot_scatter(self):
        pool, state = _gdn_pool(), _spec_state()
        # Rows past the request count are graph padding (state -1).
        hybrid = _hybrid_backend(pool, state, torch.tensor([5, 9, 2, -1]))
        req_rows = torch.tensor([17, 23, 31, 0])
        with (
            patch(_GDN_COMMIT) as commit,
            patch(f"{_BACKEND}.scatter_mamba_states_after_mtp_verify") as scatter,
        ):
            self._commit(hybrid, torch.tensor([2, 0, 7]), req_rows)

        scatter.assert_not_called()
        commit.assert_called_once()
        kwargs = commit.call_args.kwargs
        self.assertIs(kwargs["spec_state"], state)
        self.assertEqual(kwargs["state_batch_indices"].tolist(), [5, 9, 2])
        self.assertEqual(kwargs["replay_indices"].tolist(), [17, 23, 31])
        self.assertEqual(kwargs["accept_lens"].tolist(), [3, 1, 8])
        # Every accepted window folds, so `temporal` is the committed state.
        self.assertIs(kwargs["fold_every_commit"], True)

    def test_pp_rows_and_track_slots_are_translated(self):
        pool, state = _gdn_pool(), _spec_state()
        hybrid = _hybrid_backend(pool, state, torch.tensor([7, 7, 7]))
        req_rows = torch.tensor([14, 22, 30])
        track_steps = torch.tensor([1, -1, 5])
        with (
            patch(_GDN_COMMIT) as commit,
            patch(f"{_BACKEND}.pp_spec_stable_rows_enabled", return_value=True),
        ):
            self._commit(
                hybrid,
                torch.tensor([3, 0, 6]),
                req_rows,
                track_indices=torch.tensor([40, 41, 42]),
                track_steps=track_steps,
            )

        kwargs = commit.call_args.kwargs
        # PP stable rows read the slots of the delayed batch's request rows.
        self.assertEqual(kwargs["state_batch_indices"].tolist(), [107, 111, 115])
        self.assertEqual(kwargs["src_indices_raw"].tolist(), [14, 22, 30])
        self.assertEqual(kwargs["replay_indices"].tolist(), [14, 22, 30])
        self.assertEqual(kwargs["mamba_track_indices"].tolist(), [140, 141, 142])
        self.assertIs(kwargs["mamba_steps_to_track"], track_steps)
        self.assertEqual(kwargs["accept_lens"].tolist(), [4, 1, 7])

    def test_empty_batch_is_noop(self):
        hybrid = _hybrid_backend(_gdn_pool(), _spec_state(), torch.tensor([-1]))
        with (
            patch(_GDN_COMMIT) as commit,
            patch(f"{_BACKEND}.scatter_mamba_states_after_mtp_verify") as scatter,
        ):
            self._commit(
                hybrid,
                torch.zeros(0, dtype=torch.int64),
                torch.zeros(0, dtype=torch.int64),
            )
        commit.assert_not_called()
        scatter.assert_not_called()

    def test_snapshot_scatter_requires_snapshots(self):
        pool = SimpleNamespace(replayssm_is_kda=False, replayssm_cache_base=None)
        cache_indices = torch.tensor([5, 9])
        last_correct = torch.tensor([1, 0])
        req_rows = torch.tensor([3, 4])

        hybrid = _hybrid_backend(pool, _spec_state(), cache_indices)
        with (
            patch(f"{_BACKEND}.scatter_mamba_states_after_mtp_verify") as scatter,
            self.assertRaisesRegex(AssertionError, "no ReplaySSM commit"),
        ):
            self._commit(hybrid, last_correct, req_rows)
        scatter.assert_not_called()

        snapshots = torch.zeros(1, 4, DRAFT_TOKENS, 2, 4, 4)
        state = _spec_state(intermediate_ssm=snapshots)
        hybrid = _hybrid_backend(pool, state, cache_indices)
        with patch(f"{_BACKEND}.scatter_mamba_states_after_mtp_verify") as scatter:
            self._commit(hybrid, last_correct, req_rows)
        scatter.assert_called_once()
        self.assertIs(scatter.call_args.args[0], state)


class TestGDNReplaySSMGenericCommit(CustomTestCase):
    def test_cursor_advance_gets_the_state_slots(self):
        """The cursor advance skips the rows the verify and fold skip (state
        slot -1), so it gets the same state slots as the fold."""
        from sglang.srt.speculative.spec_utils import commit_mamba_states_after_verify

        target_worker = MagicMock()
        req_pool = target_worker.model_runner.req_to_token_pool
        req_pool.mamba_pool = _gdn_pool()
        slots = torch.tensor([5, -1])
        req_pool.get_mamba_indices.return_value = slots
        req_pool.get_speculative_mamba2_params_all_layers.return_value = _spec_state()
        batch = MagicMock()
        batch.forward_mode.is_idle.return_value = False
        batch.req_pool_indices = torch.tensor([17, 23])
        batch.mamba_track_indices = None
        accept_lens = torch.tensor([2, 1], dtype=torch.int32)
        accept_index = torch.tensor([[0, 1, -1], [3, -1, -1]], dtype=torch.int32)

        with (
            patch(
                "sglang.srt.speculative.spec_utils.mambaish_config",
                return_value={"some": "config"},
            ),
            patch(f"{_KERNELS}.commit_gdn_replayssm_spec") as advance,
            patch(f"{_KERNELS}.commit_gdn_replayssm_circular") as fold,
            patch(
                "sglang.kernels.ops.mamba.mamba_state_scatter_triton."
                "fused_conv_window_scatter_with_mask"
            ),
        ):
            commit_mamba_states_after_verify(
                target_worker,
                batch,
                accept_lens,
                accept_index,
                draft_token_num=3,
            )

        self.assertIs(advance.call_args.kwargs["state_batch_indices"], slots)
        self.assertIs(fold.call_args.kwargs["state_batch_indices"], slots)


class TestGDNRaggedVerifyRefusal(CustomTestCase):
    @staticmethod
    def _runner(ragged_verify=True, is_draft_worker=False):
        # The check keys on supports_ragged_verify(), not on algorithm names.
        algorithm = SimpleNamespace(supports_ragged_verify=lambda: ragged_verify)
        return SimpleNamespace(
            spec_algorithm=algorithm, is_draft_worker=is_draft_worker
        )

    def test_gdn_target_refuses_compact_verify(self):
        with (
            envs.SGLANG_RAGGED_VERIFY_MODE.override("compact"),
            self.assertRaisesRegex(ValueError, "SGLANG_RAGGED_VERIFY_MODE=compact"),
        ):
            _validate_gdn_target_verify_layout(self._runner())

    def test_dense_verify_layouts_pass(self):
        # Cap-accept verifies the dense window and only trims acceptance.
        for mode in ("static", "cap-accept"):
            with self.subTest(mode=mode), envs.SGLANG_RAGGED_VERIFY_MODE.override(mode):
                _validate_gdn_target_verify_layout(self._runner())
        with envs.SGLANG_RAGGED_VERIFY_MODE.override("compact"):
            # Only ragged-verify algorithms build ragged layouts, and only
            # the target runs the verify.
            _validate_gdn_target_verify_layout(self._runner(ragged_verify=False))
            _validate_gdn_target_verify_layout(self._runner(is_draft_worker=True))


if __name__ == "__main__":
    unittest.main()
