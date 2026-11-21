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
    GDNAttnBackend,
    _validate_gdn_target_verify_layout,
)
from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardStateSlots
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_KERNELS = "sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode"
_BACKEND = "sglang.srt.layers.attention.hybrid_linear_attn_backend"
_GDN_COMMIT = f"{_KERNELS}.commit_gdn_replayssm_after_verify"
DRAFT_TOKENS = 8


class TestDeferredCheckpointConsumers(CustomTestCase):
    def backend(self):
        backend = GDNAttnBackend.__new__(GDNAttnBackend)
        backend.state_pool = SimpleNamespace(
            mamba_layer_ids=[3, 5], replayssm_deferred_state=MagicMock()
        )
        backend.prefill_graph_metadata = None
        backend.forward_metadata = SimpleNamespace(
            mamba_cache_indices=torch.tensor([7, 9, -1]),
            query_start_loc=None,
            retrieve_next_token=None,
            retrieve_next_sibling=None,
            retrieve_parent_token=None,
        )
        return backend

    def test_eager_remains_default(self):
        from sglang.srt.arg_groups.fields.exec_ import ExecMamba

        self.assertEqual(ExecMamba().linear_replayssm_spec_commit_policy, "eager")

    def test_decode_and_extend_drain_before_checkpoint_read(self):
        for mode, method in (
            (ForwardMode.DECODE, "forward_decode"),
            (ForwardMode.EXTEND, "forward_extend"),
        ):
            backend = self.backend()
            batch = SimpleNamespace(forward_mode=mode, batch_size=2)
            deferred = backend.state_pool.replayssm_deferred_state

            def read_checkpoint(layer_id):
                deferred.materialize.assert_called_once()
                raise RuntimeError("checkpoint read reached")

            backend.state_layer_cache = read_checkpoint
            with self.assertRaisesRegex(RuntimeError, "checkpoint read reached"):
                getattr(backend, method)(
                    SimpleNamespace(layer_id=3), batch, torch.zeros(1, 4), None, None
                )
            args, kwargs = deferred.materialize.call_args
            self.assertEqual(args[0].tolist(), [7, 9])
            self.assertEqual(kwargs["reason"], "non_verify_forward")

    def test_prefill_graph_drains_before_graph_checkpoint_consumer(self):
        backend = self.backend()
        backend.prefill_graph_metadata = SimpleNamespace(
            cache_indices=torch.tensor([12, 14, -1])
        )
        deferred = backend.state_pool.replayssm_deferred_state

        def consume(**kwargs):
            deferred.materialize.assert_called_once()
            return "graph output"

        backend._forward_extend_in_graph = consume
        result = backend.forward_extend(
            SimpleNamespace(layer_id=3),
            SimpleNamespace(forward_mode=ForwardMode.EXTEND, batch_size=2),
            torch.zeros(1, 4),
            None,
            None,
        )
        self.assertEqual(result, "graph output")
        self.assertEqual(deferred.materialize.call_args.args[0].tolist(), [12, 14])

    def test_verify_and_later_layers_do_not_drain(self):
        backend = self.backend()
        for layer, mode in ((3, ForwardMode.TARGET_VERIFY), (5, ForwardMode.DECODE)):
            backend._materialize_deferred_before_state_read(
                SimpleNamespace(layer_id=layer),
                SimpleNamespace(forward_mode=mode, batch_size=2),
            )
        backend.state_pool.replayssm_deferred_state.materialize.assert_not_called()

    def test_non_static_pool_is_rejected(self):
        from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator

        configurator = KVCacheConfigurator.__new__(KVCacheConfigurator)
        with (
            patch("sglang.srt.mem_cache.kv_cache_configurator.get_exec") as config,
            self.assertRaisesRegex(ValueError, "static MambaPool"),
        ):
            config.return_value.mamba.linear_replayssm_spec_commit_policy = "deferred"
            configurator._configure_gdn_replayssm_commit(
                SimpleNamespace(mamba_pool=SimpleNamespace())
            )

    def test_deferred_refuses_unsupported_execution_modes(self):
        from sglang.srt.arg_groups.attention_hook import handle_linear_attn_backend

        fields = dict(
            linear_attn_backend="triton",
            linear_attn_decode_backend="triton",
            linear_attn_verify_backend="triton",
            linear_attn_prefill_backend="triton",
            enable_linear_replayssm=False,
            enable_linear_replayssm_spec=True,
            linear_replayssm_spec_commit_policy="deferred",
            speculative_algorithm="DFLASH",
            speculative_eagle_topk=1,
            mamba_ssm_dtype="bfloat16",
            disaggregation_mode="null",
            enable_pd_role_switch=False,
            pp_size=1,
            enable_unified_memory=False,
            mamba_radix_cache_strategy="extra_buffer_lazy",
            enable_int8_mamba_checkpoint=False,
            enable_streaming_session=False,
            enable_session_radix_cache=False,
        )
        with (
            envs.SGLANG_RAGGED_VERIFY_MODE.override("static"),
            patch(
                "sglang.srt.arg_groups.attention_hook.pp_spec_stable_rows_enabled",
                return_value=False,
            ),
        ):
            handle_linear_attn_backend(SimpleNamespace(**fields))
            for key, value in (
                ("enable_linear_replayssm_spec", False),
                ("speculative_algorithm", "DSPARK"),
                ("mamba_ssm_dtype", "float32"),
                ("disaggregation_mode", "decode"),
                ("enable_pd_role_switch", True),
                ("pp_size", 2),
                ("enable_unified_memory", True),
                ("mamba_radix_cache_strategy", "no_buffer"),
                ("enable_int8_mamba_checkpoint", True),
                ("enable_streaming_session", True),
                ("enable_session_radix_cache", True),
            ):
                with (
                    self.subTest(field=key),
                    self.assertRaisesRegex(ValueError, "Deferred GDN commit requires"),
                ):
                    handle_linear_attn_backend(
                        SimpleNamespace(**{**fields, key: value})
                    )
            with (
                envs.SGLANG_RAGGED_VERIFY_MODE.override("compact"),
                self.assertRaisesRegex(ValueError, "Deferred GDN commit requires"),
            ):
                handle_linear_attn_backend(SimpleNamespace(**fields))

    def test_deferred_refuses_non_bf16_and_strided_checkpoints(self):
        from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator

        configurator = KVCacheConfigurator.__new__(KVCacheConfigurator)
        configurator.hybrid_gdn_config = {}
        pool = MambaPool.__new__(MambaPool)
        pool.replayssm_spec_write_pos = torch.zeros(2, dtype=torch.int32)
        for temporal, error in (
            (torch.zeros(1, 4, 1, 2, 2), "BF16 checkpoint"),
            (
                torch.zeros(1, 8, 1, 2, 2, dtype=torch.bfloat16)[:, ::2],
                "contiguous checkpoints",
            ),
        ):
            pool.mamba_cache = SimpleNamespace(temporal=temporal)
            with (
                patch("sglang.srt.mem_cache.kv_cache_configurator.get_exec") as config,
                patch("sglang.srt.mem_cache.kv_cache_configurator.get_spec") as spec,
                envs.SGLANG_GDN_REPLAYSSM_COMMIT_STATS.override(False),
                self.assertRaisesRegex(ValueError, error),
            ):
                config.return_value.mamba.linear_replayssm_spec_commit_policy = (
                    "deferred"
                )
                spec.return_value.speculative_algorithm = "DFLASH"
                configurator._configure_gdn_replayssm_commit(
                    SimpleNamespace(mamba_pool=pool)
                )


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


def _hybrid_backend(mamba_pool, spec_state, cache_indices, track_slots=None):
    """A HybridLinearAttnBackend whose forward resolved ``cache_indices`` (and
    ``track_slots``) as its state slots."""
    mamba_pool.get_speculative_mamba2_params_all_layers = lambda: spec_state

    def forward_state_slots(num_requests, track_ids):
        track = None
        if track_ids is not None:
            track = track_slots[:num_requests]
        ring = hybrid.linear_attn_backend.ring_rows
        return ForwardStateSlots(
            cache_indices[:num_requests],
            track,
            None if ring is None else ring[:num_requests],
        )

    hybrid = HybridLinearAttnBackend.__new__(HybridLinearAttnBackend)
    hybrid.linear_attn_backend = SimpleNamespace(
        state_pool=mamba_pool,
        forward_state_slots=forward_state_slots,
        # The ring rows the verify resolved; on static pools, the request rows.
        ring_rows=None,
        accept_lens_pool=None,
        token_to_kv_pool=SimpleNamespace(
            layer_router=SimpleNamespace(part_view=lambda part: None)
        ),
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
    def test_deferred_route_retains_history_and_passes_capacity_margin(self):
        pool, state = _gdn_pool(), _spec_state()
        pool.replayssm_deferred_state = object()
        pool.replayssm_commit_stats = torch.zeros(40, 9, dtype=torch.int64)
        hybrid = _hybrid_backend(pool, state, torch.tensor([5, 9]))
        with patch(_GDN_COMMIT) as commit:
            self._commit(hybrid, torch.tensor([2, 0]), torch.tensor([17, 23]))
        kwargs = commit.call_args.kwargs
        self.assertIs(kwargs["fold_every_commit"], False)
        self.assertEqual(kwargs["max_spec_len"], DRAFT_TOKENS)
        self.assertIs(kwargs["stats"], pool.replayssm_commit_stats)

    def _commit(self, hybrid, last_correct, req_rows, **kwargs):
        hybrid.linear_attn_backend.ring_rows = req_rows
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

    def test_gdn_ring_commit_takes_the_resolved_ring_rows(self):
        """Under unified memory the ring rows are units of the request sub-pool
        the verify resolved, not the request rows: the commit writes those."""
        pool, state = _gdn_pool(), _spec_state()
        hybrid = _hybrid_backend(pool, state, torch.tensor([5, 9]))
        req_rows = torch.tensor([17, 23])
        with (
            patch(_GDN_COMMIT) as commit,
            patch(f"{_BACKEND}.scatter_mamba_states_after_mtp_verify"),
        ):
            hybrid.linear_attn_backend.ring_rows = torch.tensor([3, 8])
            hybrid.update_mamba_state_after_mtp_verify(
                last_correct_step_indices=torch.tensor([1, 0]),
                mamba_track_indices=None,
                mamba_steps_to_track=None,
                model=None,
                req_pool_indices=req_rows,
            )
        self.assertEqual(commit.call_args.kwargs["replay_indices"].tolist(), [3, 8])

    def test_pp_rows_and_forward_track_slots(self):
        pool, state = _gdn_pool(), _spec_state()
        hybrid = _hybrid_backend(
            pool,
            state,
            torch.tensor([107, 111, 115]),
            track_slots=torch.tensor([140, 141, 142]),
        )
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
        # The commit writes the slots the verify forward resolved; PP stable
        # rows only pick the delayed batch's scratch rows.
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
        pool = _gdn_pool()
        pool.get_speculative_mamba2_params_all_layers = lambda: _spec_state()
        router = target_worker.model_runner.token_to_kv_pool.layer_router
        router.part_view.return_value = pool
        slots = torch.tensor([5, -1])
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
            patch(f"{_BACKEND}.commit_ple_states_after_verify") as ple,
        ):
            commit_mamba_states_after_verify(
                target_worker,
                batch,
                accept_lens,
                accept_index,
                draft_token_num=3,
                state_slots=ForwardStateSlots(slots, None, batch.req_pool_indices),
            )

        for kernel in (advance, fold):
            self.assertEqual(
                kernel.call_args.kwargs["state_batch_indices"].tolist(),
                slots.tolist(),
            )
        # PLE side states roll back with the same slots and accepted steps.
        ple.assert_called_once()
        router_arg, state_arg, steps_arg, track_arg = ple.call_args.args[:4]
        self.assertIs(router_arg, router)
        self.assertEqual(state_arg.tolist(), slots.tolist())
        self.assertEqual(steps_arg.tolist(), [1, 0])
        self.assertIsNone(track_arg)


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
