"""Mixed chunked prefill must keep the prefill rows' Mamba tracking.

The prefill rows of a mixed batch donate ``mamba_ping_pong_track_buffer[
mamba_last_track_idx]`` to the radix tree after the forward (checkpoint_kv_cache).
prepare_for_extend points that slot at the forward's track destination, so the
mixed forward must still carry the track fields, or the donated checkpoint is
whatever the slot held before.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.attention.hybrid_linear_attn_backend import (  # noqa: E402
    MambaAttnBackendBase,
)
from sglang.srt.managers import schedule_batch as sb  # noqa: E402
from sglang.srt.managers.schedule_batch import ScheduleBatch  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

GRID = 128
STATE_CHUNK = 64


def _settings(lazy):
    return SimpleNamespace(
        mamba=SimpleNamespace(
            enable_mamba_extra_buffer=True, enable_mamba_extra_buffer_lazy=lazy
        )
    )


class _Pool:
    """Ping-pong slot bookkeeping of HybridReqToTokenPool, CPU only."""

    def __init__(self):
        self.freed = []
        self.mamba_allocator = SimpleNamespace(
            free=lambda slots: self.freed.extend(slots.tolist())
        )

    @staticmethod
    def set_mamba_ping_pong_slot(req, idx, value):
        req.kv.mamba_ping_pong_track_buffer[idx] = value

    @staticmethod
    def get_mamba_ping_pong_other_idx(idx):
        return 1 - idx


class _Req:
    def __init__(self, rid, origin_len, output_len, buf, next_idx, last_idx=None):
        self.rid = rid
        self.origin_input_ids = list(range(origin_len))
        self.output_ids = list(range(output_len))
        self.prefix_indices = []
        self.prefix_len = 0
        self.extend_end = origin_len
        self.mamba_branching_seqlen = None
        self.decode_batch_idx = 3
        self.kv = SimpleNamespace(
            mamba_ping_pong_track_buffer=torch.tensor(buf, dtype=torch.int64),
            mamba_next_track_idx=next_idx,
            mamba_last_track_idx=last_idx,
            mamba_last_track_seqlen=None,
            mamba_untracked_decode_batch_idx=None,
        )

    def refresh_fill_ids(self):
        pass

    def set_extend_range(self, start, end):
        self.prefix_len = start
        self.extend_end = end

    @property
    def extend_len(self):
        return self.extend_end - self.prefix_len


def _batch(reqs, pool, **fields):
    bs = len(reqs)
    batch = ScheduleBatch(
        reqs=reqs,
        req_to_token_pool=pool,
        model_config=SimpleNamespace(
            is_encoder_decoder=False,
            hf_text_config=SimpleNamespace(mamba_chunk_size=STATE_CHUNK),
        ),
        tree_cache=SimpleNamespace(page_size=GRID),
        sampling_info=MagicMock(),
        spec_algorithm=SpeculativeAlgorithm.NONE,
        req_pool_indices=torch.arange(bs),
        req_pool_indices_cpu=torch.arange(bs),
        orig_seq_lens=torch.zeros(bs, dtype=torch.int32),
    )
    for name, value in fields.items():
        setattr(batch, name, value)
    return batch


class TestMixWithRunningMambaTrack(unittest.TestCase):
    def setUp(self):
        for name, value in (
            ("get_parallel", lambda: SimpleNamespace(dcp_enabled=False)),
            ("mamba_cache_chunk_size", lambda: STATE_CHUNK),
            ("mamba_checkpoint_grid", lambda page_size: GRID),
        ):
            p = patch.object(sb, name, value)
            p.start()
            self.addCleanup(p.stop)

    def _mix(self, lazy):
        pool = _Pool()
        # Prefill rows: a 300-token extend after a 128-token prefix tracks at 384;
        # a 40-token extend tracks nothing.
        tracked = _Req("p0", 428, 0, [7, -1 if lazy else 8], next_idx=0)
        tracked.prefix_indices = list(range(128))
        tracked.set_extend_range(128, 428)
        short = _Req("p1", 40, 0, [9, -1 if lazy else 10], next_idx=0)
        short.set_extend_range(0, 40)
        extend = _batch(
            [tracked, short],
            pool,
            forward_mode=ForwardMode.EXTEND,
            out_cache_loc=torch.arange(340),
            seq_lens=torch.tensor([428, 40]),
            seq_lens_cpu=torch.tensor([428, 40]),
            prefix_lens=[128, 0],
            extend_lens=[300, 40],
            extend_num_tokens=340,
            extend_logprob_start_lens=[0, 0],
            return_logprob=False,
            is_prefill_only=True,
        )
        with patch.object(sb, "get_exec", return_value=_settings(lazy)):
            entries = [
                extend._mamba_radix_cache_v2_req_prepare_for_extend(r)
                for r in extend.reqs
            ]
        # The assembly at the end of prepare_for_extend.
        extend.mamba_prefill_track_mask_cpu = [e.track_mask for e in entries]
        extend.mamba_track_seqlens_cpu = [e.track_seqlen for e in entries]
        extend.mamba_track_indices = sb._gather_mamba_track_indices(
            torch.stack([r.kv.mamba_ping_pong_track_buffer for r in extend.reqs]),
            extend.req_pool_indices,
            torch.tensor([e.track_position for e in entries]),
        )
        extend.mamba_track_mask = torch.tensor([e.track_mask for e in entries])
        extend.mamba_track_seqlens = torch.tensor([e.track_seqlen for e in entries])

        # Decode tails after prepare_for_decode. d0 sits on a track boundary
        # (seq_len 640); in lazy mode the prealloc handed it slot 11 and moved
        # its next index there.
        if lazy:
            d0 = _Req("d0", 600, 40, [5, 11], next_idx=1, last_idx=0)
        else:
            d0 = _Req("d0", 600, 40, [5, 12], next_idx=1, last_idx=0)
        d1 = _Req("d1", 200, 3, [6, -1 if lazy else 13], next_idx=0)
        running = _batch(
            [d0, d1],
            pool,
            forward_mode=ForwardMode.DECODE,
            out_cache_loc=torch.tensor([1000, 1001]),
            seq_lens=torch.tensor([640, 203]),
            seq_lens_cpu=torch.tensor([640, 203]),
            return_logprob=False,
            mamba_track_indices=torch.tensor([11 if lazy else 12, 6]),
            mamba_track_buffer_indices=[1, 0],
            mamba_track_mask=torch.tensor([True, False]),
            mamba_track_mask_cpu=[True, False],
        )
        with patch.object(sb, "get_exec", return_value=_settings(lazy)):
            extend.mix_with_running(running)
        return extend, pool

    def test_mixed_batch_tracks_prefill_rows_into_the_donated_slot(self):
        for lazy in (False, True):
            with self.subTest(lazy=lazy):
                batch, _ = self._mix(lazy)
                self.assertEqual(batch.forward_mode, ForwardMode.MIXED)
                self.assertEqual(
                    batch.mamba_prefill_track_mask_cpu, [True, False, False, False]
                )
                self.assertEqual(batch.mamba_track_seqlens_cpu, [428, -1, -1, -1])
                self.assertEqual(
                    batch.mamba_track_mask.tolist(), [True, False, False, False]
                )
                self.assertEqual(batch.mamba_track_seqlens.tolist(), [428, -1, -1, -1])
                self.assertEqual(batch.mamba_track_indices.tolist()[:2], [7, 9])
                self.assertEqual(batch.mamba_track_indices.tolist()[2:], [-1, -1])

                # The slot checkpoint_kv_cache donates (donate_mamba_ping_pong_slot)
                # for the tracked prefill row ...
                tracked = batch.reqs[0]
                self.assertEqual(tracked.kv.mamba_last_track_seqlen, 384)
                donated = tracked.kv.mamba_ping_pong_track_buffer[
                    tracked.kv.mamba_last_track_idx
                ].item()
                # ... is the one the mixed forward writes (eager track plan).
                backend = object.__new__(MambaAttnBackendBase)
                backend.device = "cpu"
                backend._mamba_chunk_size = STATE_CHUNK
                view = SimpleNamespace(
                    batch_size=4,
                    forward_mode=batch.forward_mode,
                    mamba_prefill_track_mask_cpu=batch.mamba_prefill_track_mask_cpu,
                    mamba_track_seqlens_cpu=batch.mamba_track_seqlens_cpu,
                    extend_seq_lens_cpu=batch.extend_lens,
                    extend_prefix_lens_cpu=batch.prefix_lens,
                    mamba_track_indices=batch.mamba_track_indices,
                )
                self.assertTrue(backend._has_cpu_prefill_track_metadata(view))
                plan = backend._init_track_ssm_indices_from_cpu(
                    torch.arange(100, 104), view
                )
                h_dst, final_dst = plan[2], plan[5]
                self.assertEqual(sorted(h_dst.tolist() + final_dst.tolist()), [donated])

    def test_lazy_decode_tails_give_back_the_boundary_slot(self):
        batch, pool = self._mix(lazy=True)
        d0, d1 = batch.reqs[2], batch.reqs[3]
        self.assertEqual(pool.freed, [11])
        self.assertEqual(d0.kv.mamba_ping_pong_track_buffer.tolist(), [5, -1])
        self.assertEqual(d0.kv.mamba_next_track_idx, 0)
        self.assertEqual(d0.kv.mamba_last_track_idx, 0)
        self.assertEqual(d0.kv.mamba_untracked_decode_batch_idx, d0.decode_batch_idx)
        self.assertEqual(d1.kv.mamba_ping_pong_track_buffer.tolist(), [6, -1])
        self.assertEqual(d1.kv.mamba_next_track_idx, 0)
        self.assertIsNone(d1.kv.mamba_untracked_decode_batch_idx)

    def test_lazy_undo_keeps_the_slot_when_the_prealloc_failed(self):
        pool = _Pool()
        req = _Req("d0", 600, 40, [5, -1], next_idx=0, last_idx=0)
        running = _batch([req], pool, mamba_track_mask_cpu=[True])
        running.mamba_lazy_undo_prealloc_at_boundary()
        self.assertEqual(pool.freed, [])
        self.assertEqual(req.kv.mamba_ping_pong_track_buffer.tolist(), [5, -1])
        self.assertEqual(req.kv.mamba_next_track_idx, 0)
        self.assertEqual(req.kv.mamba_untracked_decode_batch_idx, req.decode_batch_idx)

    def test_non_lazy_decode_tails_keep_their_slots(self):
        batch, pool = self._mix(lazy=False)
        self.assertEqual(pool.freed, [])
        self.assertEqual(batch.reqs[2].kv.mamba_next_track_idx, 1)


class TestLazyFinishBeforeMixedBoundary(unittest.TestCase):
    """A request finishes at grid - 1 in step N while overlapped step N + 1, its
    boundary step, is mixed. The keep slot is untouched, so the radix insert must
    still happen; only an in-place boundary write may skip it."""

    def _finish(self, untracked):
        from sglang.srt.managers.scheduler_components import (
            batch_result_processor as brp,
        )

        req = SimpleNamespace(
            rid="r",
            decode_batch_idx=5,
            mamba_lazy_checkpoint=True,
            multimodal_inputs=None,
            finished=lambda: True,
            time_stats=MagicMock(),
            kv=SimpleNamespace(
                mamba_next_track_idx=0,
                mamba_last_track_idx=0,
                mamba_untracked_decode_batch_idx=5 if untracked else None,
            ),
        )
        batch = SimpleNamespace(
            mamba_track_mask_cpu=[False],
            mamba_track_mask_next_cpu=[True],
            mamba_decode_batch_idx_cpu=[4],
        )
        processor = SimpleNamespace(
            draft_worker=None,
            model_worker=None,
            tree_cache=object(),
            _mamba_can_cache_finished_req=lambda req, result, i: True,
            _prepare_kv_cache_release=lambda req: None,
            _maybe_collect_routed_experts=lambda req: None,
            _maybe_collect_indexer_topk=lambda req: None,
            _maybe_collect_customized_info=lambda i, req, out: None,
        )
        released = []
        with (
            patch.object(brp, "get_exec", return_value=_settings(lazy=True)),
            patch.object(
                brp,
                "get_disagg",
                return_value=SimpleNamespace(
                    disaggregation_decode_enable_offload_kvcache=False
                ),
            ),
            patch.object(
                brp, "get_memory", return_value=SimpleNamespace(enable_hisparse=False)
            ),
            patch.object(
                brp,
                "release_kv_cache",
                lambda req, cache, checkpoint: released.append(checkpoint),
            ),
        ):
            brp.SchedulerBatchResultProcessor._handle_finish_state_updated_req(
                processor, req, batch, None, 0, None
            )
        return released

    def test_mixed_boundary_step_keeps_the_radix_insert(self):
        self.assertEqual(self._finish(untracked=True), [True])

    def test_in_place_boundary_write_skips_the_radix_insert(self):
        self.assertEqual(self._finish(untracked=False), [False])


if __name__ == "__main__":
    unittest.main()
