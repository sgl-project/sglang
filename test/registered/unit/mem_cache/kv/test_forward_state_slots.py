"""Late readers of a forward use the slots it resolved, not the slots of requests
that reused its rows after its shared reads ended."""

import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
from test_declared_parts import _build

from sglang.srt.environ import envs
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
    MambaAttnBackendBase,
)
from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardStateSlots
from sglang.srt.mem_cache.kv.alloc.forward_steps import forward_steps_of
from sglang.srt.mem_cache.kv.alloc.runner_ids import RunnerIds
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.models.qwen4_exp import _commit_ple_batch, _prepare_ple_batch
from sglang.srt.runtime_context import reset_context
from sglang.srt.speculative import eagle_post_verify_graphs, spec_utils
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

_HLA = "sglang.srt.layers.attention.hybrid_linear_attn_backend"
_DRAFT_TOKENS = 2


class _Forward:
    """A forward over request rows 1 and 2 of a unified pool, slots resolved by a
    real Mamba backend (eager or graph replay), counted in flight by the pool."""

    def __init__(self, *, mode, graph, ple=False):
        storage, _, _ = _build(state=True, ple=ple, gamma=_DRAFT_TOKENS)
        self.storage = storage
        alloc = self.alloc = storage.allocator
        pool = self.pool = alloc.get_kvcache()
        alloc.state.bind_routes(pool.layer_router)
        self.band = alloc.mamba_allocator
        # Free the first slots and compact, so virtual and physical ids differ.
        slots = alloc.state_slots.alloc(7)
        alloc.state_slots.free(slots[:2])
        self.band.flush_for_allocation()
        self.spare = slots[2:3].clone()
        self.working, self.track = slots[3:5].clone(), slots[5:7].clone()
        # Holes for the new owners: their slots do not wait on this forward.
        alloc.state_slots.free(alloc.state_slots.alloc(4))
        self.req = storage.requests
        self.rows = torch.tensor([1, 2])
        self.req.req_index_to_mamba_index_mapping[self.rows] = self.working.to(
            self.req.req_index_to_mamba_index_mapping.dtype
        )
        self.reader = RunnerIds(
            req_to_token=self.req.req_to_token,
            token_to_kv_pool_allocator=alloc,
            token_to_kv_pool=pool,
            page_size=4,
            device="cpu",
        )
        backend = self.backend = object.__new__(MambaAttnBackendBase)
        backend.device = "cpu"
        backend.topk = 1
        backend.pad_slot_id = -1
        backend.req_to_token_pool = self.req
        backend.token_to_kv_pool = pool
        backend.state_pool = alloc.state.pool
        backend.accept_lens_pool = None
        backend._state_ids = self.reader.state_ids(self.req.state_sub_pool)
        backend._fused_state_indices_ok = False
        backend._graph_seq_len_fill_value = 1
        for name in (
            "state_indices_list",
            "query_start_loc_list",
            "retrieve_next_token_list",
            "retrieve_next_sibling_list",
            "retrieve_parent_token_list",
        ):
            setattr(backend, name, [])
        self.hybrid = object.__new__(HybridLinearAttnBackend)
        self.hybrid.linear_attn_backend = backend

        self.working_physical = self.translate(self.working)
        self.track_physical = self.translate(self.track)

        steps = forward_steps_of(alloc)
        self.launch = steps.forward_launched()
        steps.forward_recorded(None)

        verify = mode.is_target_verify()
        tokens = _DRAFT_TOKENS if verify else 1
        self.fb = ForwardBatch(
            forward_mode=mode,
            batch_size=2,
            input_ids=torch.arange(2 * tokens) + 90,
            req_pool_indices=self.rows.clone(),
            seq_lens=torch.tensor([9, 9]),
            seq_lens_sum=18,
            seq_lens_cpu=torch.tensor([9, 9]),
            out_cache_loc=torch.arange(2 * tokens) + 1,
            mamba_track_indices=self.track.clone(),
            mamba_track_mask=torch.tensor([True, True]),
            spec_info=(
                SimpleNamespace(draft_token_num=_DRAFT_TOKENS, topk=1)
                if verify
                else None
            ),
        )
        if graph:
            backend.init_cuda_graph_state(2, 2 * _DRAFT_TOKENS)
            backend.forward_metadata = backend._capture_metadata(
                2, self.fb.req_pool_indices, mode, self.fb.spec_info
            )
            backend.forward_metadata = backend._replay_metadata(
                2,
                self.fb.req_pool_indices,
                mode,
                self.fb.spec_info,
                self.fb.seq_lens_cpu,
                num_padding=0,
                mamba_track_indices=self.fb.mamba_track_indices,
            )
        else:
            backend.init_forward_metadata(self.fb)

    def translate(self, virtual):
        return self.backend._translate_mamba_indices(virtual).to(torch.int64)

    def live_working(self):
        return self.translate(self.req.get_mamba_indices(self.rows))

    def reuse_rows(self):
        """The forward's requests finish while it still runs: their slots are
        freed, and their rows go to new requests with slots of their own."""
        self.alloc.state_slots.free(torch.cat([self.working, self.track]))
        fresh = self.alloc.state_slots.alloc(4)
        # Not one of the forward's pages: those stay pending until it is processed.
        assert not set(self.translate(fresh).tolist()) & set(
            torch.cat([self.working_physical, self.track_physical]).tolist()
        )
        self.req.req_index_to_mamba_index_mapping[self.rows] = fresh[:2].to(
            self.req.req_index_to_mamba_index_mapping.dtype
        )
        # The request table and the state map now name the new owners: a late
        # lookup would find none of the forward's slots.
        assert not set(self.live_working().tolist()) & set(
            self.working_physical.tolist()
        )
        assert not set(self.translate(self.track).tolist()) & set(
            self.track_physical.tolist()
        )
        return fresh


class TestLateReadsUseTheForwardSlots(CustomTestCase):
    def tearDown(self):
        reset_context()

    def _cases(self):
        for graph in (False, True):
            with self.subTest(graph=graph):
                yield graph

    def test_backend_verify_commit(self):
        """`HybridLinearAttnBackend.update_mamba_state_after_mtp_verify`, with and
        without pipeline-stable request rows."""
        for graph in self._cases():
            for stable_rows in (False, True):
                with self.subTest(stable_rows=stable_rows):
                    f = _Forward(mode=ForwardMode.TARGET_VERIFY, graph=graph)
                    f.reuse_rows()
                    steps = torch.tensor([1, 0])
                    with (
                        patch(
                            _HLA + ".pp_spec_stable_rows_enabled",
                            return_value=stable_rows,
                        ),
                        patch(_HLA + ".scatter_mamba_states_after_mtp_verify") as sc,
                        patch.object(
                            HybridLinearAttnBackend,
                            "_update_ple_state_after_mtp_verify",
                        ),
                    ):
                        f.hybrid.update_mamba_state_after_mtp_verify(
                            last_correct_step_indices=steps,
                            mamba_track_indices=f.track,
                            mamba_steps_to_track=steps,
                            model=None,
                            req_pool_indices=f.rows,
                        )
                    args = sc.call_args.args
                    self.assertEqual(args[1].tolist(), f.working_physical.tolist())
                    self.assertEqual(args[3].tolist(), f.track_physical.tolist())
                    rows = sc.call_args.kwargs["src_indices_raw"]
                    if stable_rows:
                        self.assertEqual(rows.tolist(), f.rows.tolist())
                    else:
                        self.assertIsNone(rows)

    def test_ascend_verify_commit(self):
        """`AscendHybridLinearAttnBackend.update_mamba_state_after_mtp_verify` writes
        the working and checkpoint slots the forward resolved, not the ids its
        caller passes, after the request table changed."""
        npu_ops = ModuleType("sgl_kernel_npu.mamba.mamba_state_update_triton")
        npu_ops.conv_state_rollback = None
        npu_ops.move_intermediate_cache = None
        with patch.dict("sys.modules", {npu_ops.__name__: npu_ops}):
            from sglang.srt.hardware_backend.npu.attention import (
                ascend_hybrid_linear_attn_backend as ascend,
            )
        for graph in self._cases():
            f = _Forward(mode=ForwardMode.TARGET_VERIFY, graph=graph)
            f.reuse_rows()
            hybrid = object.__new__(ascend.AscendHybridLinearAttnBackend)
            hybrid.linear_attn_backend = f.backend
            steps = torch.tensor([1, 0])
            with (
                patch.object(ascend, "move_intermediate_cache") as move,
                patch.object(ascend, "conv_state_rollback") as rollback,
            ):
                hybrid.update_mamba_state_after_mtp_verify(
                    last_correct_step_indices=steps,
                    mamba_track_indices=f.track,
                    mamba_steps_to_track=steps,
                    model=None,
                )
            (working, *_), (track, *_) = (c.args[2:] for c in move.call_args_list)
            self.assertEqual(working.tolist(), f.working_physical.tolist())
            self.assertEqual(track.tolist(), f.track_physical.tolist())
            self.assertEqual(
                rollback.call_args_list[1].args[1].tolist(), f.track_physical.tolist()
            )

    def test_verify_graph_replay_resolves_checkpoint_slots(self):
        """A verify graph replay resolves the batch's checkpoint slots through the
        runner's staging buffer, padded rows on the reserved slot 0."""
        from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
            DecodeCudaGraphRunner,
        )

        for bs in (2, 3):
            with self.subTest(bs=bs):
                f = _Forward(mode=ForwardMode.TARGET_VERIFY, graph=False)
                # A replayed batch carries virtual ids (the eager prep above
                # translated this one's in place).
                f.fb.mamba_track_indices = f.track.clone()
                backend = f.backend
                backend.init_cuda_graph_state(3, 3 * _DRAFT_TOKENS)
                rows = torch.cat([f.rows, f.rows.new_zeros(bs - 2)])
                backend._capture_metadata(
                    bs, rows, ForwardMode.TARGET_VERIFY, f.fb.spec_info
                )
                runner = object.__new__(DecodeCudaGraphRunner)
                runner.__dict__.update(
                    _verify_checkpoint_ids=torch.full((3,), 9, dtype=torch.int64),
                    _metadata_glue=None,
                    _early_verify_metadata=None,
                    enable_two_batch_overlap=False,
                    require_mlp_tp_gather=False,
                    require_attn_tp_gather=False,
                    model_runner=SimpleNamespace(lora_manager=None),
                    attn_backend=backend,
                    seq_len_fill_value=1,
                    capture_forward_mode=ForwardMode.TARGET_VERIFY,
                    is_encoder_decoder=False,
                    buffers=SimpleNamespace(
                        input_ids=torch.zeros(bs * _DRAFT_TOKENS, dtype=torch.int64),
                        positions=torch.zeros(bs * _DRAFT_TOKENS, dtype=torch.int64),
                        req_pool_indices=rows.clone(),
                        seq_lens=torch.full((bs,), 9),
                        seq_lens_cpu=torch.full((bs,), 9),
                        encoder_lens=None,
                        mamba_track_indices=None,
                    ),
                )
                runner._init_replay_attn_metadata(
                    f.fb,
                    raw_bs=2,
                    bs=bs,
                    num_tokens=bs * _DRAFT_TOKENS,
                    glue_key=None,
                    early_staged=False,
                    attention_variant=None,
                )
                f.reuse_rows()
                slots = f.hybrid.forward_state_slots(bs, f.track)
                self.assertEqual(
                    slots.working[:2].tolist(), f.working_physical.tolist()
                )
                self.assertEqual(slots.track[:2].tolist(), f.track_physical.tolist())
                self.assertEqual(slots.track[2:].tolist(), [0] * (bs - 2))
                # The batch's own ids stay virtual.
                self.assertEqual(f.fb.mamba_track_indices.tolist(), f.track.tolist())

    def _target_worker(self, f, mamba_pool):
        return SimpleNamespace(
            model_runner=SimpleNamespace(
                model_config=None,
                model=None,
                attn_backend=f.hybrid,
                req_to_token_pool=f.req,
                runner_ids=f.reader,
                token_to_kv_pool=SimpleNamespace(
                    layer_router=SimpleNamespace(part_view=lambda part: mamba_pool)
                ),
            )
        )

    def _verify_batch(self, f):
        return SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            req_pool_indices=f.rows.clone(),
            mamba_track_indices=f.track.clone(),
            seq_lens=torch.tensor([7, 9]),
            tree_cache=SimpleNamespace(page_size=4),
        )

    def test_replayssm_spec_commit(self):
        """`commit_mamba_states_after_verify`'s ReplaySSM branches: circular, fold
        and KDA fold."""
        accept_lens = torch.tensor([2, 1])
        accept_index = torch.tensor([[0, 1], [2, -1]])
        ring = torch.zeros((1, 8, 4, 2))
        spec_state = SimpleNamespace(
            temporal=ring,
            conv=[ring],
            intermediate_conv_window=[ring],
            **{
                f"replayssm_{name}": ring
                for name in ("d", "k", "g", "rawv", "rawk", "beta")
            },
        )
        branches = {
            "circular": (
                dict(replayssm_spec_fold=False, replayssm_is_kda=False),
                "sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode."
                "commit_gdn_replayssm_circular",
            ),
            "fold": (
                dict(replayssm_spec_fold=True, replayssm_is_kda=False),
                "sglang.kernels.ops.attention.linear.gdn_replayssm_spec_fold."
                "commit_gdn_replayssm_fold_after_verify",
            ),
            "kda": (
                dict(replayssm_spec_fold=True, replayssm_is_kda=True),
                "sglang.kernels.ops.attention.linear.kda_replayssm_spec_decode."
                "commit_kda_replayssm_after_verify",
            ),
        }
        for graph in self._cases():
            for name, (flags, kernel) in branches.items():
                with self.subTest(branch=name):
                    f = _Forward(mode=ForwardMode.TARGET_VERIFY, graph=graph)
                    mamba_pool = SimpleNamespace(
                        replayssm_cache_base=torch.zeros(8, dtype=torch.int32),
                        replayssm_spec_write_pos=torch.zeros(8, dtype=torch.int32),
                        replayssm_is_flush=torch.zeros(8, dtype=torch.int32),
                        replayssm_request_ring=False,
                        get_speculative_mamba2_params_all_layers=lambda: spec_state,
                        **flags,
                    )
                    f.reuse_rows()
                    with (
                        patch.object(spec_utils, "mambaish_config", return_value=True),
                        patch.object(spec_utils, "mamba_track_grid", return_value=8),
                        patch(kernel) as commit,
                        patch(
                            "sglang.kernels.ops.attention.linear."
                            "gdn_replayssm_spec_decode.commit_gdn_replayssm_spec"
                        ),
                        patch(
                            "sglang.kernels.ops.mamba.mamba_state_scatter_triton."
                            "fused_conv_window_scatter_with_mask"
                        ) as conv_scatter,
                        patch(_HLA + ".commit_ple_states_after_verify"),
                    ):
                        spec_utils.commit_mamba_states_after_verify(
                            self._target_worker(f, mamba_pool),
                            self._verify_batch(f),
                            accept_lens,
                            accept_index,
                            _DRAFT_TOKENS,
                        )
                    kwargs = commit.call_args.kwargs
                    self.assertEqual(
                        kwargs["state_batch_indices"].tolist(),
                        f.working_physical.tolist(),
                    )
                    self.assertEqual(
                        kwargs["mamba_track_indices"].tolist(),
                        f.track_physical.tolist(),
                    )
                    for call in conv_scatter.call_args_list:
                        self.assertIn(
                            call.args[2].tolist(),
                            (f.working_physical.tolist(), f.track_physical.tolist()),
                        )

    def test_post_verify_graph_commit(self):
        """The post-verify graphs' side-stream commit writes the slots staged
        with the round's inputs, not what the request table names at replay."""
        for graph in self._cases():
            f = _Forward(mode=ForwardMode.TARGET_VERIFY, graph=graph)
            target_worker = self._target_worker(
                f,
                SimpleNamespace(replayssm_spec_fold=False, replayssm_cache_base=None),
            )
            graphs = eagle_post_verify_graphs.EaglePostVerifyGraphs(
                target_worker=target_worker,
                simulated_acc_len_rng=torch.Generator(),
            )
            batch = self._verify_batch(f)
            batch.sampling_info = SimpleNamespace(is_all_greedy=True)
            verify_input = SimpleNamespace(
                draft_token=torch.zeros(4, dtype=torch.int64),
                retrieve_index=torch.zeros((2, 2), dtype=torch.int64),
                retrieve_next_token=torch.zeros((2, 2), dtype=torch.int64),
                retrieve_next_sibling=torch.zeros((2, 2), dtype=torch.int64),
            )
            with patch.object(
                eagle_post_verify_graphs, "mambaish_config", return_value=True
            ):
                inputs = eagle_post_verify_graphs._post_verify_inputs(
                    batch, verify_input, graphs._verify_state_slots(batch)
                )
            slot = SimpleNamespace(
                staged={name: t.clone() for name, t in inputs.items()},
                forward_mode=batch.forward_mode,
                tree_cache=batch.tree_cache,
                tokens_per_request=_DRAFT_TOKENS,
            )
            f.reuse_rows()
            steps = (torch.tensor([1, 0]), torch.tensor([0, -1]))
            outputs = (None, torch.tensor([2, 1]), torch.tensor([[0, 1], [2, -1]]))
            outputs += (None, None, None, steps)
            with (
                patch.object(spec_utils, "mambaish_config", return_value=True),
                patch(_HLA + ".scatter_mamba_states_after_mtp_verify") as sc,
                patch.object(
                    HybridLinearAttnBackend, "_update_ple_state_after_mtp_verify"
                ),
            ):
                graphs._commit(slot, outputs)
            args = sc.call_args.args
            self.assertEqual(args[1].tolist(), f.working_physical.tolist())
            self.assertEqual(args[3].tolist(), f.track_physical.tolist())

    def test_qwen4_ple_preparation(self):
        """`models/qwen4_exp.py::_prepare_ple_batch` runs inside the model
        forward; its N-gram history and state rows are the forward's."""
        for graph in self._cases():
            for fast in (False, True):
                with self.subTest(fast=fast):
                    self._check_ple(graph, fast)

    def _check_ple(self, graph, fast):
        f = _Forward(mode=ForwardMode.DECODE, graph=graph, ple=True)
        ngram = f.alloc.state.ngram_pool
        ngram.context.copy_(
            torch.arange(ngram.context.shape[0])[:, None] * 100
            + torch.tensor([11, 12, 13])
        )
        history = ngram.get_context(f.working_physical).clone()
        fresh = f.reuse_rows()
        fresh_physical = f.translate(fresh)
        before = ngram.context.clone()
        with (
            forward_context(
                ForwardContext(
                    attn_backend=SimpleNamespace(
                        req_to_token_pool=f.req,
                        token_to_kv_pool=f.pool,
                        runner_ids=f.reader,
                        linear_attn_backend=f.backend,
                        forward_state_slots=f.backend.forward_state_slots,
                    )
                )
            ),
            envs.SGLANG_ENABLE_QWEN4_PLE_FUSION.override(fast),
        ):
            batch = _prepare_ple_batch(
                f.fb.input_ids, f.fb, ngram_size=4, ngram_eos_token_id=17
            )
            self.assertEqual(batch.state_indices.tolist(), f.working_physical.tolist())
            torch.testing.assert_close(
                batch.ngram_context, torch.cat((history, f.fb.input_ids[:, None]), 1)
            )
            _commit_ple_batch(batch, f.fb)
        # The commit wrote the forward's working and checkpoint rows only.
        written = set(f.working_physical.tolist()) | set(f.track_physical.tolist())
        for row in fresh_physical.tolist():
            self.assertNotIn(row, written)
            torch.testing.assert_close(ngram.context[row], before[row])


class TestRelayedCommitSlots(CustomTestCase):
    def tearDown(self):
        reset_context()

    def test_relayed_commit_follows_a_move(self):
        """A PP stage's relayed commit resolves the batch's slots when it runs, so
        it follows a compaction that moved them after the forward."""
        f = _Forward(mode=ForwardMode.TARGET_VERIFY, graph=False)
        forward_steps_of(f.alloc).forward_processed(f.launch)
        slots = torch.cat([f.working, f.track])
        before = f.translate(slots)
        # Free a slot below the batch's and compact: one of the batch's moves.
        f.alloc.state_slots.free(f.spare)
        f.band._flush(urgent=True)
        self.assertNotEqual(f.translate(slots).tolist(), before.tolist())
        target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                model_config=None, req_to_token_pool=f.req, runner_ids=f.reader
            )
        )
        batch = SimpleNamespace(
            req_pool_indices=f.rows, mamba_track_indices=f.track.clone()
        )
        with patch.object(spec_utils, "mambaish_config", return_value=True):
            slots = spec_utils.relayed_commit_state_slots(target_worker, batch)
        self.assertIsInstance(slots, ForwardStateSlots)
        self.assertEqual(slots.working.tolist(), f.live_working().tolist())
        self.assertEqual(slots.track.tolist(), f.translate(f.track).tolist())


if __name__ == "__main__":
    unittest.main()
