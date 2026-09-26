"""A prefill co-batched with running decodes must still write its checkpoint.

prepare_for_extend claims a ping-pong slot and stamps mamba_last_track_seqlen
for every tracked extend row before the scheduler decides to mix. The merge
into the running decode batch dropped the batch's track tensors, so the MIXED
forward skipped the checkpoint write while cache_unfinished_req /
cache_finished_req still donated the claimed slot at the stamped depth. Every
later request sharing that prefix then restored whatever the slot held before
(#39342).

The GDN kernel is the only piece replaced here: the state it would produce is
stood in by distinguishable per-chunk / per-slot values, and the production
metadata, track copies, cache donation and prefix match run for real.
"""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.configs.mamba_utils import (  # noqa: E402
    Mamba2CacheParams,
    Mamba2StateShape,
)
from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend  # noqa: E402
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch  # noqa: E402
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator  # noqa: E402
from sglang.srt.mem_cache.cache_init_params import CacheInitParams  # noqa: E402
from sglang.srt.mem_cache.common import release_kv_cache  # noqa: E402
from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache  # noqa: E402
from sglang.srt.mem_cache.memory_pool import (  # noqa: E402
    HybridLinearKVPool,
    HybridReqToTokenPool,
)
from sglang.srt.model_executor.forward_batch_info import (  # noqa: E402
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.runtime_context import get_context  # noqa: E402
from sglang.srt.sampling.sampling_params import SamplingParams  # noqa: E402
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  # noqa: E402

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

# One grid for the chunk size, the tree page and the track interval, so every
# checkpoint depth under test is a multiple of CHUNK.
CHUNK = 4
VOCAB = 128
STALE = -7.0

MODEL_CONFIG = SimpleNamespace(
    is_encoder_decoder=False,
    vocab_size=VOCAB,
    hf_text_config=SimpleNamespace(mamba_chunk_size=CHUNK),
)

_real_torch_tensor = torch.tensor


def _tensor_without_pinning(*args, **kwargs):
    kwargs.pop("pin_memory", None)
    return _real_torch_tensor(*args, **kwargs)


class _Lifecycle:
    """Real pools, radix cache and GDN backend metadata on CPU."""

    def __init__(self, *, extra_buffer: bool, lazy: bool = False):
        shape = Mamba2StateShape.create(
            tp_world_size=1,
            intermediate_size=8,
            n_groups=1,
            num_heads=2,
            head_dim=4,
            state_size=4,
            conv_kernel=4,
        )
        with envs.SGLANG_MAMBA_SSM_DTYPE.override("float32"):
            cache_params = Mamba2CacheParams(shape=shape, layers=[0])
        self.req_to_token_pool = HybridReqToTokenPool(
            size=8,
            mamba_size=32,
            mamba_spec_state_size=8,
            max_context_len=64,
            device="cpu",
            enable_memory_saver=False,
            cache_params=cache_params,
            mamba_layer_ids=[0],
            enable_mamba_extra_buffer=extra_buffer,
            enable_mamba_extra_buffer_lazy=lazy,
            speculative_num_draft_tokens=3,
        )
        kv_pool = HybridLinearKVPool(
            size=128,
            dtype=torch.float32,
            page_size=1,
            head_num=1,
            head_dim=4,
            full_attention_layer_ids=[1],
            device="cpu",
            enable_memory_saver=False,
            mamba_pool=self.req_to_token_pool.mamba_pool,
        )
        self.allocator = TokenToKVPoolAllocator(
            size=128,
            dtype=torch.float32,
            device="cpu",
            kvcache=kv_pool,
            need_sort=False,
        )
        self.tree = MambaRadixCache(
            params=CacheInitParams(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.allocator,
                page_size=1,
                disable=False,
                enable_mamba_extra_buffer=extra_buffer,
                enable_mamba_extra_buffer_lazy=lazy,
            )
        )
        mamba_cache = self.req_to_token_pool.mamba_pool.mamba_cache
        self.ssm_states = mamba_cache.temporal[0]
        self.conv_states = mamba_cache.conv[0][0]
        self.ssm_states.fill_(STALE)
        self.conv_states.fill_(STALE)

        backend = object.__new__(GDNAttnBackend)
        backend.device = "cpu"
        backend._mamba_chunk_size = CHUNK
        backend.req_to_token_pool = self.req_to_token_pool
        backend.conv_states_shape = tuple(self.conv_states.shape[-2:])
        backend.kernel_dispatcher = SimpleNamespace(extend_uses_state_checkpoints=False)
        backend.enable_unified_memory = False
        backend.forward_metadata = None
        self.backend = backend

    def new_req(self, rid: str, tokens: list[int], *, max_new_tokens: int) -> Req:
        sampling_params = SamplingParams(max_new_tokens=max_new_tokens, temperature=0)
        sampling_params.normalize(None)
        req = Req(
            rid=rid,
            origin_input_text="",
            origin_input_ids=array("q", tokens),
            sampling_params=sampling_params,
            vocab_size=VOCAB,
        )
        req.init_next_round_input(self.tree)
        return req

    def prefill_batch(
        self, reqs: list[Req], extend_lens: list[int], *, chunked_req=None
    ) -> ScheduleBatch:
        for req, extend_len in zip(reqs, extend_lens, strict=True):
            prefix_len = len(req.prefix_indices)
            req.set_extend_range(prefix_len, prefix_len + extend_len)
        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.allocator,
            tree_cache=self.tree,
            model_config=MODEL_CONFIG,
            enable_overlap=True,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            chunked_req=chunked_req,
        )
        batch.prepare_for_extend()
        return batch

    def forward_batch(self, batch: ScheduleBatch) -> ForwardBatch:
        # ForwardBatch.init_new needs a ModelRunner; the fields the linear
        # backend reads are carried over the same way it does (track tensors
        # aliased, extend lens as int32 tensors, start locs as their cumsum).
        extend_seq_lens = torch.tensor(batch.extend_lens, dtype=torch.int32)
        extend_start_loc = torch.zeros_like(extend_seq_lens)
        extend_start_loc[1:] = torch.cumsum(extend_seq_lens[:-1], dim=0)
        return ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=len(batch.seq_lens),
            input_ids=None,
            req_pool_indices=batch.req_pool_indices,
            seq_lens=batch.seq_lens,
            out_cache_loc=batch.out_cache_loc,
            seq_lens_sum=int(batch.seq_lens_cpu.sum()),
            seq_lens_cpu=batch.seq_lens_cpu,
            mamba_track_indices=batch.mamba_track_indices,
            mamba_track_mask=batch.mamba_track_mask,
            mamba_track_seqlens=batch.mamba_track_seqlens,
            extend_seq_lens=extend_seq_lens,
            extend_prefix_lens=torch.tensor(batch.prefix_lens, dtype=torch.int32),
            extend_seq_lens_cpu=list(batch.extend_lens),
            extend_prefix_lens_cpu=list(batch.prefix_lens),
            extend_num_tokens=batch.extend_num_tokens,
            extend_start_loc=extend_start_loc,
        )

    def run_extend_forward(self, batch: ScheduleBatch, *, tag: int):
        """Production metadata and track copies around a stand-in kernel.

        Row r's final SSM state is ``final(tag, r)``, the packed chunk-grid
        state k is ``h(tag, k)``, and conv input column t holds t, so a
        checkpoint's contents name the position it was taken at.
        """
        fb = self.forward_batch(batch)
        self.backend.init_forward_metadata(fb)
        md = self.backend.forward_metadata
        for row, slot in enumerate(md.mamba_cache_indices.tolist()):
            self.ssm_states[slot] = final(tag, row)
        num_chunks = sum((n - 1) // CHUNK + 1 for n in batch.extend_lens)
        h = torch.empty((1, num_chunks, *self.ssm_states.shape[1:]))
        for k in range(num_chunks):
            h[0, k] = h_state(tag, k)
        if md.has_mamba_track_mask:
            # gdn_backend.forward_extend's conv-state snapshot.
            mixed_qkv = (
                torch.arange(fb.extend_num_tokens, dtype=torch.float32)
                .repeat(self.conv_states.shape[1], 1)
                .to(self.conv_states.dtype)
            )
            self.conv_states[md.conv_states_mask_indices] = mixed_qkv[
                :, md.track_conv_indices
            ].transpose(0, 1)
        self.backend._track_mamba_state_extend(fb, h, self.ssm_states, md)
        return md

    def finish_at_first_token(self, req: Req, token: int) -> None:
        req.output_ids.append(token)
        req.update_finish_state()
        assert req.finished()
        release_kv_cache(req, self.tree)

    def restored_state(self, tokens: list[int]):
        """What a new request over ``tokens`` resumes from: the matched
        prefix length and the SSM / conv checkpoint the tree hands it."""
        req = self.new_req(f"probe-{tokens[0]}", tokens, max_new_tokens=1)
        src = req.kv.mamba_cow_src_index
        if src is None:
            return len(req.prefix_indices), None, None
        slot = int(src.item())
        return (
            len(req.prefix_indices),
            self.ssm_states[slot].flatten()[0].item(),
            self.conv_states[slot][0].tolist(),
        )


def final(tag: int, row: int) -> float:
    return 1000.0 * (tag + 1) + row


def h_state(tag: int, k: int) -> float:
    return 100.0 * (tag + 1) + k


def slot_at(req: Req, idx: int) -> int:
    return req.kv.mamba_ping_pong_track_buffer[idx].item()


class TestMixedChunkMambaTracking(CustomTestCase):
    def setUp(self):
        # CPU CI has no pinned host memory; neutralize both spellings.
        for p in (
            patch.object(torch.Tensor, "pin_memory", lambda tensor: tensor),
            patch("torch.tensor", _tensor_without_pinning),
        ):
            p.start()
            self.addCleanup(p.stop)

    def _publish(self, strategy: str):
        override = get_context().override_server_args(
            mamba_radix_cache_strategy=strategy,
            mamba_track_interval=CHUNK,
            _mamba_cache_chunk_size=CHUNK,
            page_size=1,
            enable_mixed_chunk=True,
            disable_radix_cache=False,
            attention_backend="torch_native",
        )
        override.install()
        self.addCleanup(override.restore)

    def _running_decode_batch(self, lc: _Lifecycle, *, prompt: list[int]):
        """One request through prefill, its prefix insert, and the decode
        prepare the scheduler runs right before mixing."""
        req = lc.new_req(f"D{prompt[0]}", prompt, max_new_tokens=8)
        batch = lc.prefill_batch([req], [len(prompt)])
        lc.run_extend_forward(batch, tag=0)
        req.output_ids.append(60)
        lc.tree.cache_unfinished_req(req)
        # The prefill batch becomes the running batch; update_running_batch
        # filters it and prepares the decode step.
        batch.filter_batch()
        batch.prepare_for_decode()
        return req, batch

    def test_mixed_forward_checkpoints_extend_rows_for_later_prefix_hits(self):
        """Regression for #39342: a chunked prefill (intermediate-state
        checkpoint), a prefill that finishes on its first token (final-state
        checkpoint) and a too-short prefill share a MIXED batch with a running
        decode. Requests arriving later over the same prefixes must resume
        from the states this forward produced, not from what the ping-pong
        slots held before."""
        self._publish("extra_buffer")
        lc = _Lifecycle(extra_buffer=True)
        decoding, running = self._running_decode_batch(
            lc, prompt=[50, 51, 52, 53, 54, 55]
        )
        decoding_claim = (
            decoding.kv.mamba_next_track_idx,
            decoding.kv.mamba_last_track_idx,
            decoding.kv.mamba_last_track_seqlen,
        )

        chunked = lc.new_req("A", list(range(10, 22)), max_new_tokens=8)
        finishing = lc.new_req("B", list(range(30, 38)), max_new_tokens=1)
        short = lc.new_req("S", [70, 71], max_new_tokens=8)
        mixed = lc.prefill_batch(
            [chunked, finishing, short], [10, 8, 2], chunked_req=chunked
        )
        self.assertEqual(chunked.kv.mamba_last_track_seqlen, 8)
        self.assertEqual(finishing.kv.mamba_last_track_seqlen, 8)
        self.assertIsNone(short.kv.mamba_last_track_seqlen)

        mixed.mix_with_running(running)
        self.assertEqual(mixed.forward_mode, ForwardMode.MIXED)
        lc.run_extend_forward(mixed, tag=1)
        # process_batch_result_prefill: chunked -> cache_unfinished_req,
        # finished -> release_kv_cache, prefill-complete -> cache_unfinished_req.
        lc.tree.cache_unfinished_req(chunked, chunked=True)
        lc.finish_at_first_token(finishing, 99)
        lc.tree.cache_unfinished_req(short)

        # Row 0 (10 tokens) checkpoints depth 8 from the chunk grid: packed
        # chunk 2 of the batch, conv window over inputs 5..7. Row 1 (8 tokens)
        # is aligned: its final state, conv window over inputs 15..17.
        self.assertEqual(
            lc.restored_state(list(range(10, 18)) + [90, 91]),
            (8, h_state(1, 2), [5.0, 6.0, 7.0]),
        )
        self.assertEqual(
            lc.restored_state(list(range(30, 38)) + [92]),
            (8, final(1, 1), [15.0, 16.0, 17.0]),
        )
        self.assertEqual(lc.restored_state([70, 71, 72]), (0, None, None))
        # The decode tail neither claimed nor wrote anything on the mixed step.
        self.assertEqual(
            (
                decoding.kv.mamba_next_track_idx,
                decoding.kv.mamba_last_track_idx,
                decoding.kv.mamba_last_track_seqlen,
            ),
            decoding_claim,
        )
        for idx in range(2):
            self.assertEqual(lc.ssm_states[slot_at(decoding, idx)].flatten()[0], STALE)

    def test_decode_tail_at_a_track_boundary_is_not_a_writer(self):
        """A running decode whose new length sits on the track grid carries a
        True decode-boundary mask into the mix. Through the extend kernels
        that row has no chunk-grid state to snapshot, so the mixed batch must
        mask it off and its slots must stay untouched."""
        self._publish("extra_buffer")
        lc = _Lifecycle(extra_buffer=True)
        decoding, running = self._running_decode_batch(
            lc, prompt=[50, 51, 52, 53, 54, 55, 56]
        )
        self.assertEqual(running.seq_lens_cpu.tolist(), [8])
        self.assertEqual(running.mamba_track_mask.tolist(), [True])

        extend = lc.new_req("A", list(range(10, 18)), max_new_tokens=8)
        mixed = lc.prefill_batch([extend], [8])
        mixed.mix_with_running(running)
        self.assertIsNotNone(mixed.mamba_track_mask)
        self.assertEqual(mixed.mamba_track_mask.tolist(), [True, False])

        md = lc.run_extend_forward(mixed, tag=1)
        decode_slots = {slot_at(decoding, 0), slot_at(decoding, 1)}
        written = set(md.track_ssm_final_dst.tolist() + md.track_ssm_h_dst.tolist())
        written |= set(md.conv_states_mask_indices.tolist())
        self.assertEqual(written & decode_slots, set())
        for slot in decode_slots:
            self.assertEqual(lc.ssm_states[slot].flatten()[0], STALE)
            self.assertEqual(lc.conv_states[slot][0, 0], STALE)

    def test_running_batch_without_track_indices_gets_its_slots_from_reqs(self):
        """Spec decode prepares the running rows' track indices inside the
        forward, so the running batch reaches the mix without them. The mixed
        batch still needs a real slot id per decode row (index translation
        and graph track buffers read every row): the slot the decode step
        would target, masked off."""
        self._publish("extra_buffer")
        lc = _Lifecycle(extra_buffer=True)
        decoding, running = self._running_decode_batch(
            lc, prompt=[50, 51, 52, 53, 54, 55]
        )
        running.mamba_track_indices = None

        extend = lc.new_req("A", list(range(10, 18)), max_new_tokens=8)
        mixed = lc.prefill_batch([extend], [8])
        mixed.mix_with_running(running)
        self.assertIsNotNone(mixed.mamba_track_mask)
        self.assertEqual(mixed.mamba_track_mask.tolist(), [True, False])
        self.assertEqual(mixed.mamba_track_seqlens.tolist(), [8, -1])
        self.assertEqual(
            mixed.mamba_track_indices.tolist(),
            [
                slot_at(extend, extend.kv.mamba_last_track_idx),
                slot_at(decoding, decoding.kv.mamba_next_track_idx),
            ],
        )

    def test_lazy_strategy_mixes_with_a_preallocated_boundary_slot(self):
        """extra_buffer_lazy holds one slot per request and allocates the
        second only when a decode step lands on the track grid, before the
        mix. That freshly allocated slot must ride the mixed batch masked off
        (the decode tail takes no checkpoint here) and the extend row must
        still checkpoint into its single slot and donate it."""
        self._publish("extra_buffer_lazy")
        lc = _Lifecycle(extra_buffer=True, lazy=True)
        decoding, running = self._running_decode_batch(
            lc, prompt=[50, 51, 52, 53, 54, 55, 56]
        )
        self.assertEqual(running.seq_lens_cpu.tolist(), [8])
        self.assertEqual(running.mamba_track_mask.tolist(), [True])
        self.assertEqual(decoding.kv.mamba_next_track_idx, 1)
        boundary_slot = slot_at(decoding, 1)
        self.assertNotEqual(boundary_slot, -1)
        self.assertEqual(running.mamba_track_indices.tolist(), [boundary_slot])
        # Spec decode leaves the running indices unset; rebuild them under lazy.
        running.mamba_track_indices = None

        extend = lc.new_req("A", list(range(10, 20)), max_new_tokens=8)
        mixed = lc.prefill_batch([extend], [10])
        self.assertEqual(extend.kv.mamba_next_track_idx, extend.kv.mamba_last_track_idx)
        self.assertEqual(slot_at(extend, 1), -1)
        extend_slot = slot_at(extend, extend.kv.mamba_last_track_idx)
        mixed.mix_with_running(running)
        self.assertIsNotNone(mixed.mamba_track_mask)
        self.assertEqual(mixed.mamba_track_mask.tolist(), [True, False])
        self.assertEqual(mixed.mamba_track_seqlens.tolist(), [10, -1])
        self.assertEqual(
            mixed.mamba_track_indices.tolist(), [extend_slot, boundary_slot]
        )

        lc.run_extend_forward(mixed, tag=1)
        lc.tree.cache_unfinished_req(extend)
        self.assertEqual(
            lc.restored_state(list(range(10, 18)) + [90]),
            (8, h_state(1, 2), [5.0, 6.0, 7.0]),
        )
        for idx in range(2):
            self.assertEqual(lc.ssm_states[slot_at(decoding, idx)].flatten()[0], STALE)
        self.assertEqual(
            (decoding.kv.mamba_next_track_idx, decoding.kv.mamba_last_track_idx),
            (1, 0),
        )

    def test_mix_without_extra_buffer_carries_no_tracking(self):
        self._publish("no_buffer")
        lc = _Lifecycle(extra_buffer=False)
        decoding, running = self._running_decode_batch(
            lc, prompt=[50, 51, 52, 53, 54, 55]
        )
        self.assertIsNone(running.mamba_track_mask)

        extend = lc.new_req("A", list(range(10, 18)), max_new_tokens=8)
        mixed = lc.prefill_batch([extend], [8])
        mixed.mix_with_running(running)
        self.assertIsNone(mixed.mamba_track_mask)
        self.assertIsNone(mixed.mamba_track_indices)
        self.assertIsNone(mixed.mamba_track_seqlens)

    def test_post_forward_filter_and_decode_prepare_rebuild_tracking(self):
        """After the mixed forward the scheduler filters the finished and
        chunked rows out and prepares the survivors for decode. The extend
        rows' checkpoint tracking must not outlive that step: the next decode
        tracks by its own boundary mask and slot per surviving request."""
        self._publish("extra_buffer")
        lc = _Lifecycle(extra_buffer=True)
        decoding, running = self._running_decode_batch(
            lc, prompt=[50, 51, 52, 53, 54, 55]
        )

        chunked = lc.new_req("A", list(range(10, 22)), max_new_tokens=8)
        finishing = lc.new_req("B", list(range(30, 38)), max_new_tokens=1)
        short = lc.new_req("S", [70, 71], max_new_tokens=8)
        mixed = lc.prefill_batch(
            [chunked, finishing, short], [10, 8, 2], chunked_req=chunked
        )
        mixed.mix_with_running(running)
        lc.run_extend_forward(mixed, tag=1)
        lc.tree.cache_unfinished_req(chunked, chunked=True)
        lc.finish_at_first_token(finishing, 99)
        lc.tree.cache_unfinished_req(short)
        short.output_ids.append(61)
        decoding.output_ids.append(62)

        mixed.filter_batch(chunked_req_to_exclude=[chunked])
        self.assertEqual([r.rid for r in mixed.reqs], ["S", "D50"])
        self.assertIsNone(mixed.mamba_track_mask)
        self.assertIsNone(mixed.mamba_track_seqlens)

        mixed.prepare_for_decode()
        self.assertEqual(mixed.seq_lens_cpu.tolist(), [3, 8])
        self.assertEqual(mixed.mamba_track_mask.tolist(), [False, True])
        self.assertEqual(
            mixed.mamba_track_indices.tolist(),
            [
                slot_at(short, short.kv.mamba_next_track_idx),
                slot_at(decoding, decoding.kv.mamba_next_track_idx),
            ],
        )


if __name__ == "__main__":
    unittest.main()
