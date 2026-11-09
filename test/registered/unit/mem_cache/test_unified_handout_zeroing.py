from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.unified_allocator_fixtures import arena_of, stand_in_views

register_cuda_ci(est_time=11, stage="base-b", runner_config="1-gpu-small")
# Backend-specific: AMD lowers unified page zeroing through ROCm Triton,
# catching HIP-only codegen, launch, or stale-page failures.
register_amd_ci(est_time=9, suite="stage-b-test-1-gpu-small-amd")

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.kvcache.kv_read_table import build_kv_read_table
from sglang.srt.environ import envs
from sglang.srt.mem_cache.kv.alloc.band import Band
from sglang.srt.mem_cache.kv.alloc.chain import ChainAllocator, ChainRoles
from sglang.srt.mem_cache.kv.alloc.forward_steps import ForwardSteps
from sglang.srt.mem_cache.kv.layout.mha import MHASubPoolSpec, MHAView
from sglang.srt.mem_cache.kv.layout.mla import MLASubPoolSpec, MLAView
from sglang.srt.mem_cache.kv.layout.state import MambaSubPoolSpec
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.unified_allocator_fixtures import build_test_unified_pools

BF16_NAN = 0x7FC1  # LE bf16 NaN bit pattern, as SGLANG_DEBUG_POISON_POOL fills


def _build(device, page_size=1):
    """A tiny MLA+mamba unified pool + full-side allocator.

    Mirrors the unified pool builder's full + state stack just enough for the
    allocator hand-out path (the piece under test)."""
    layer_num = 2
    full_spec = MLASubPoolSpec(
        name="full",
        layer_num=layer_num,
        grow_direction="up",
        kv_lora_rank=16,
        qk_rope_head_dim=8,
        store_dtype=torch.bfloat16,
    )
    mamba_spec = MambaSubPoolSpec(
        name="mamba",
        layer_num=1,
        grow_direction="down",
        conv_state_shapes=((8, 3),),
        conv_dtype=torch.bfloat16,
        temporal_state_shape=(2, 4, 4),
        temporal_dtype=torch.float32,
    )
    total_bytes = 4096 * full_spec.entry_bytes()
    buf = arena_of(
        total_bytes=total_bytes,
        sub_pool_specs=[full_spec, mamba_spec],
        device=device,
        enable_memory_saver=False,
        page_size=page_size,
    )
    kvcache = MLAView(
        arena=buf,
        spec=buf.spec("full"),
        kv_cache_dtype=torch.bfloat16,
        page_size=page_size,
    )
    allocator = Band(
        arena=buf,
        sub_pool_name="full",
        device=device,
        is_id_owner=True,
        page_size=page_size,
    )
    return buf, kvcache, allocator


def _build_mha(device, page_size=8):
    """The same pool with an MHA full side, as a GDN hybrid (Qwen3.5) builds it:
    its mamba neighbour's fp32 state shares the bytes the KV pages reuse."""
    full_spec = MHASubPoolSpec(
        name="full",
        layer_num=2,
        grow_direction="up",
        head_num=2,
        head_dim=8,
        store_dtype=torch.bfloat16,
    )
    mamba_spec = MambaSubPoolSpec(
        name="mamba",
        layer_num=1,
        grow_direction="down",
        conv_state_shapes=((8, 3),),
        conv_dtype=torch.bfloat16,
        temporal_state_shape=(2, 4, 4),
        temporal_dtype=torch.float32,
    )
    buf = arena_of(
        total_bytes=4096 * full_spec.entry_bytes(),
        sub_pool_specs=[full_spec, mamba_spec],
        device=device,
        enable_memory_saver=False,
        page_size=page_size,
    )
    kvcache = MHAView(
        arena=buf,
        spec=buf.spec("full"),
        page_size=page_size,
        model_layer_ids=tuple(range(buf.spec("full").layer_num)),
    )
    allocator = Band(
        arena=buf,
        sub_pool_name="full",
        device=device,
        is_id_owner=True,
        page_size=page_size,
    )
    return buf, kvcache, allocator


def _build_tri(device, page_size=4):
    """A tri-pool, [mamba (up) | swa (float) | full (down)], with MHA pools on
    both KV sides, as a mamba + sliding-window hybrid builds it."""
    full_spec = MHASubPoolSpec(
        name="full",
        layer_num=2,
        grow_direction="down",
        head_num=2,
        head_dim=8,
        store_dtype=torch.bfloat16,
    )
    swa_spec = MHASubPoolSpec(
        name="swa",
        layer_num=2,
        grow_direction="float",
        head_num=2,
        head_dim=8,
        store_dtype=torch.bfloat16,
    )
    mamba_spec = MambaSubPoolSpec(
        name="mamba",
        layer_num=1,
        grow_direction="up",
        conv_state_shapes=((3, 8),),
        conv_dtype=torch.bfloat16,
        temporal_state_shape=(0, 0, 0),
        temporal_dtype=torch.float32,
    )
    n_full, n_swa = 256, 128
    buf = arena_of(
        total_bytes=n_full * full_spec.entry_bytes()
        + n_swa * swa_spec.entry_bytes()
        + 16 * mamba_spec.entry_bytes(),
        sub_pool_specs=[full_spec, swa_spec, mamba_spec],
        device=device,
        enable_memory_saver=False,
        page_size=page_size,
    )
    kvcache = SimpleNamespace(
        full_kv_pool=MHAView(
            arena=buf,
            spec=buf.spec("full"),
            page_size=page_size,
            model_layer_ids=tuple(range(buf.spec("full").layer_num)),
        ),
        swa_kv_pool=MHAView(
            arena=buf,
            spec=buf.spec("swa"),
            page_size=page_size,
            model_layer_ids=tuple(range(buf.spec("swa").layer_num)),
        ),
        attach_allocators=lambda **_: None,
    )
    allocator = ChainAllocator(
        roles=ChainRoles(owner="full", window="swa", state="mamba"),
        arena=buf,
        kvcache=kvcache,
        views=stand_in_views(
            ChainRoles(owner="full", window="swa", state="mamba"),
            kvcache,
            SimpleNamespace(move_kv_cache=lambda dst, src: None),
        ),
        device=device,
        full_max_total_num_tokens=n_full,
        swa_max_total_num_tokens=n_swa,
        page_size=page_size,
    )
    return buf, kvcache.swa_kv_pool, allocator


def _build_swa_poisoned(device, page_size=4):
    """A full + SWA unified pool built under SGLANG_DEBUG_POISON_POOL."""
    with envs.SGLANG_DEBUG_POISON_POOL.override(True):
        return build_test_unified_pools(
            device=device,
            kv_cache_dtype=torch.bfloat16,
            head_num=2,
            head_dim=8,
            v_head_dim=8,
            swa_head_num=2,
            swa_head_dim=8,
            swa_v_head_dim=8,
            page_size=page_size,
            start_layer=0,
            end_layer=2,
            swa_attention_layer_ids=[1],
            full_attention_layer_ids=[0],
            full_max_total_num_tokens=64,
            swa_max_total_num_tokens=32,
            enable_memory_saver=False,
            need_sort=False,
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required (fused alloc kernel)")
class TestUnifiedHandoutZeroing(unittest.TestCase):
    """Root-cause guard: pages must leave the allocator ZEROED.

    The trtllm MLA kernel arithmetically masks (NaN-unsafe) the unwritten
    tail rows of a request's last partial page, so recycled / fresh page
    bytes must never carry NaN bit patterns. Static pools get this from
    torch.zeros; the unified pool must re-establish it at every hand-out."""

    build = staticmethod(_build)

    def _poison(self, buf):
        buf.raw.view(torch.int16).fill_(BF16_NAN)

    def _env(self, buf, kvcache):
        return buf.raw[: kvcache.num_pages * kvcache.page_bytes].view(
            kvcache.num_pages, kvcache.page_bytes
        )

    def _phys_pages(self, allocator, virt_tokens):
        return (allocator.ids.translate(virt_tokens) // allocator.page_size).unique()

    def test_fresh_and_recycled_pages_zeroed(self):
        buf, kvcache, allocator = self.build("cuda")
        env = self._env(buf, kvcache)

        # Fresh hand-out over a poisoned pool (the deterministic form of
        # "freed GPU heap happened to contain NaN patterns").
        self._poison(buf)
        out = allocator.alloc(16)
        self.assertIsNotNone(out)
        pages = self._phys_pages(allocator, out)
        self.assertTrue((env[pages] == 0).all().item())
        # Untouched pages must still be poisoned, else the assert above is
        # vacuous (a whole-pool memset would also pass it).
        wm_page = int(pages.max().item()) + 2
        self.assertFalse((env[wm_page] == 0).all().item())

        # Recycle: free, re-poison the raw bytes (data only; v2p bookkeeping
        # is separate storage), re-alloc — recycled pages must be zeroed too.
        allocator.free(out)
        self._poison(buf)
        out2 = allocator.alloc(16)
        self.assertIsNotNone(out2)
        pages2 = self._phys_pages(allocator, out2)
        self.assertTrue((env[pages2] == 0).all().item())

    def test_zeroing_keys_on_pool_type(self):
        # The MLA view has NaN-unsafe partial-page reads whatever
        # its geometry; zeroing must key on the pool type, not on any
        # id-space scale.
        buf, kvcache, allocator = self.build("cuda")
        self.assertTrue(allocator._zero_pages_on_alloc)
        self._poison(buf)
        out = allocator.alloc(8)
        self.assertIsNotNone(out)
        env = self._env(buf, kvcache)
        pages = self._phys_pages(allocator, out)
        self.assertTrue((env[pages] == 0).all().item())


class TestUnifiedMHAHandoutZeroing(TestUnifiedHandoutZeroing):
    """The same contract for an MHA pool: trtllm_mha's fmha_v2 prefill loads a
    request's last partial page whole and multiplies its unwritten V rows by a
    zero probability, so one NaN there turns the request's output into NaN."""

    build = staticmethod(_build_mha)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required (fused alloc kernel)")
class TestTriPoolSWAHandoutZeroing(unittest.TestCase):
    """A tri-pool hands its sliding-window pages out through the float middle's
    own allocator; they must leave it zeroed like every other MHA page."""

    def setUp(self):
        publish(ServerArgs(model_path="dummy"), role="tokenizer")
        self.addCleanup(reset_context)

    def test_fresh_and_recycled_swa_pages_zeroed(self):
        buf, swa, allocator = _build_tri("cuda")
        sa = allocator.swa_attn_allocator
        self.assertTrue(sa._zero_pages_on_alloc)
        env = buf.raw[: swa.num_pages * swa.page_bytes].view(
            swa.num_pages, swa.page_bytes
        )
        for _ in range(2):  # fresh, then recycled
            buf.raw.view(torch.int16).fill_(BF16_NAN)
            out = allocator.alloc(16)
            self.assertIsNotNone(out)
            pages = (sa.ids.translate(out) // sa.page_size).unique()
            self.assertTrue((env[pages] == 0).all().item())
            allocator.free(out)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required (fused alloc kernel)")
class TestPoisonSparesTheSlotZeroSink(unittest.TestCase):
    """SGLANG_DEBUG_POISON_POOL marks never-written bytes. The slot-0 sink is
    not one: a freed window page's read-table entry resolves to it, kernels that
    load whole tiles read its rows (at zero weight), and in serving it holds
    zeros. So the poison must leave it zero, or every window model fails it."""

    def setUp(self):
        publish(ServerArgs(model_path="dummy"), role="tokenizer")
        self.addCleanup(reset_context)

    def test_freed_window_page_reads_zeros(self):
        ps = 4
        bundle = _build_swa_poisoned("cuda", page_size=ps)
        alloc = bundle.token_to_kv_pool_allocator
        ids = alloc.alloc(4 * ps)
        self.assertIsNotNone(ids)
        # The window moved past the request's first two pages.
        alloc.free_swa_segment(ids[: 2 * ps], start_pos=0)

        dev = ids.device
        table = torch.full((1, 4), -1, dtype=torch.int32, device=dev)
        build_kv_read_table(
            req_to_token=ids.to(torch.int32).view(1, -1),
            req_pool_indices=torch.tensor([0], device=dev),
            seq_lens=torch.tensor([4 * ps], device=dev),
            v2p=alloc.swa_attn_allocator.ids.virtual_to_physical,
            page_size=ps,
            max_pages=4,
            out=table,
        )
        self.assertEqual(table[0, :2].tolist(), [0, 0])  # freed: the sink
        v = bundle.token_to_kv_pool.layer_router.view("swa").v_buffer[0]
        self.assertFalse(torch.isnan(v[:ps].float()).any().item())


# An FP32 word whose low half, read as BF16, is a NaN: 1.0038987 is finite.
FP32_WITH_BF16_NAN = 0x3F807FC1


def _build_state_attn(device):
    """[mamba (up) | full (down)]: an FP32 state band beside a BF16 MHA band
    that zeroes its pages on hand-out, so the attention band can grow into
    bytes a state slot held."""
    full_spec = MHASubPoolSpec(
        name="full",
        layer_num=2,
        grow_direction="down",
        head_num=2,
        head_dim=8,
        store_dtype=torch.bfloat16,
    )
    mamba_spec = MambaSubPoolSpec(
        name="mamba",
        layer_num=1,
        grow_direction="up",
        conv_state_shapes=((3, 8),),
        conv_dtype=torch.bfloat16,
        temporal_state_shape=(2, 4, 4),
        temporal_dtype=torch.float32,
    )
    roles = ChainRoles(owner="full", state="mamba")
    buf = arena_of(
        total_bytes=16 * full_spec.entry_bytes() + 4 * mamba_spec.entry_bytes(),
        sub_pool_specs=[full_spec, mamba_spec],
        device=device,
        enable_memory_saver=False,
        page_size=1,
    )
    kvcache = SimpleNamespace(
        full_kv_pool=MHAView(
            arena=buf,
            spec=buf.spec("full"),
            page_size=1,
            model_layer_ids=tuple(range(full_spec.layer_num)),
        ),
        attach_allocators=lambda **_: None,
    )
    allocator = ChainAllocator(
        roles=roles,
        arena=buf,
        kvcache=kvcache,
        views=stand_in_views(
            roles, kvcache, SimpleNamespace(move_kv_cache=lambda dst, src: None)
        ),
        device=device,
        page_size=1,
        need_sort=False,
    )
    return buf, allocator


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required (streams)")
class TestZeroingWaitsALateStateWrite(unittest.TestCase):
    """Bytes of a state slot freed while a forward still writes it reach the
    attention band only once that forward is processed; otherwise the page's
    zeroing races the forward's late FP32 state write, leaving BF16 NaNs in
    the page's unwritten rows."""

    def setUp(self):
        publish(ServerArgs(model_path="dummy"), role="tokenizer")
        self.addCleanup(reset_context)

    def _free_slot_under_a_late_write(self, sleep_cycles):
        """Fill [state | attention], then free the top state slot while a
        forward on its own stream, ``sleep_cycles`` late, writes FP32 words
        into it; the attention band then grows into the slot's bytes. Returns
        the arena, the attention page bytes, the pages handed out over the
        slot, and whether the forward still ran at the hand-out."""
        buf, alloc = _build_state_attn("cuda")
        state, full = alloc.mamba_allocator, alloc.full_attn_allocator
        slots = alloc.alloc_state_slots(4)
        self.assertIsNotNone(slots)
        self.assertIsNotNone(alloc.alloc(full.available_size()))
        self.assertEqual(full.available_size(), 0)

        top = slots[-1:]
        slot = int(state.ids.virtual_to_physical[top][0])
        lo = slot * state.entry_bytes_per_page
        hi = lo + state.entry_bytes_per_page
        forward_stream = torch.cuda.Stream()
        forward_done = torch.cuda.Event()
        with torch.cuda.stream(forward_stream):
            torch.cuda._sleep(sleep_cycles)
            buf.raw[lo:hi].view(torch.int32).fill_(FP32_WITH_BF16_NAN)
            forward_done.record(forward_stream)
        alloc.set_latest_forward_done_event(forward_done)
        alloc.set_inflight_forward(forward_done, None, top)
        launch = alloc.forward_steps.forward_launched()
        alloc.forward_steps.forward_recorded(forward_done)
        state.free(top)
        alloc.flush_opportunistic()
        # Held while the forward runs: the slot's bytes stay the state band's.
        running = not forward_done.query()
        self.assertEqual(full.available_size(), 0)
        self.assertGreater(state._byte_high_frontier(), lo)

        torch.cuda.synchronize()
        alloc.forward_steps.forward_processed(launch)
        alloc.flush_opportunistic()
        self.assertLessEqual(state._byte_high_frontier(), lo)
        grown = alloc.alloc(full.available_size())
        self.assertIsNotNone(grown)
        epp = full.entry_bytes_per_page
        pages = [
            p
            for p in (full.ids.translate(grown) // full.page_size).unique().tolist()
            if p * epp < hi and lo < (p + 1) * epp
        ]
        self.assertTrue(pages, "the attention band did not reach the slot's bytes")
        return buf, epp, pages, running

    def test_unused_attention_rows_stay_finite(self):
        # Load every kernel the path launches first: a module load
        # synchronizes the device, which would hide the race.
        self._free_slot_under_a_late_write(0)
        torch.cuda.synchronize()

        buf, epp, pages, running = self._free_slot_under_a_late_write(2_000_000_000)
        self.assertTrue(running, "the forward ended before its pages went pending")
        torch.cuda.synchronize()
        # Nothing has written the new pages since their hand-out.
        env = torch.stack([buf.raw[p * epp : (p + 1) * epp] for p in pages])
        self.assertFalse(torch.isnan(env.view(torch.bfloat16).float()).any().item())
        self.assertTrue((env == 0).all().item())


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required (streams)")
class TestReuseWaitsAForwardThatReadsTheFreedPage(unittest.TestCase):
    """A forward launched before a release may still read the released pages
    until it ends: its attention page table was prepared with them. No later
    writer of those bytes -- the zeroing of a hand-out, or another band growing
    into them -- reaches them before it ends, even when it writes none of
    them."""

    def setUp(self):
        publish(ServerArgs(model_path="dummy"), role="tokenizer")
        self.addCleanup(reset_context)

    def _forward_reading(self, buf, lo, hi, sleep_cycles):
        """A forward on its own stream that, ``sleep_cycles`` late, reads bytes
        ``[lo, hi)``; returns its done event and what it read."""
        seen = torch.empty(hi - lo, dtype=torch.uint8, device="cuda")
        stream = torch.cuda.Stream()
        done = torch.cuda.Event()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            torch.cuda._sleep(sleep_cycles)
            seen.copy_(buf.raw[lo:hi])
            done.record(stream)
        return done, seen

    def _zero_a_page_the_forward_reads(self, sleep_cycles):
        buf, _kvcache, band = _build_mha("cuda")
        ps, epp = band.page_size, band.entry_bytes_per_page
        written = band.alloc(ps)
        freed = band.alloc(ps)
        self.assertIsNotNone(written)
        self.assertIsNotNone(freed)
        page = int(band.ids.translate(freed[:1])[0]) // ps
        lo, hi = page * epp, (page + 1) * epp
        buf.raw[lo:hi].fill_(0x11)
        torch.cuda.synchronize()
        done, seen = self._forward_reading(buf, lo, hi, sleep_cycles)
        band.set_latest_forward_done_event(done)
        band.set_inflight_forward(done, written)
        steps = ForwardSteps((band,))
        launch = steps.forward_launched()
        steps.forward_recorded(done)
        band.free(freed)
        running = not done.query()
        # Pending while the forward runs: the hand-out takes it, its zeroing
        # queued behind the forward.
        out = band.alloc(ps)
        self.assertIsNotNone(out)
        self.assertEqual(int(band.ids.translate(out[:1])[0]) // ps, page)
        torch.cuda.synchronize()
        steps.forward_processed(launch)
        return buf, lo, hi, seen, running

    def test_zeroing_waits_a_forward_that_only_reads_the_page(self):
        # Load every kernel the path launches first: a module load
        # synchronizes the device, which would hide the race.
        self._zero_a_page_the_forward_reads(0)
        torch.cuda.synchronize()

        buf, lo, hi, seen, running = self._zero_a_page_the_forward_reads(2_000_000_000)
        self.assertTrue(running, "the forward ended before its pages went pending")
        torch.cuda.synchronize()
        self.assertTrue(
            bool((seen == 0x11).all()), "the forward read a page zeroed under it"
        )
        self.assertTrue(bool((buf.raw[lo:hi] == 0).all()), "the page was not zeroed")

    def _state_into_bytes_the_forward_reads(self, sleep_cycles):
        """[state | attention]: free every attention page while a forward that
        writes none of them reads the bytes the next state slot lands on; the
        state band then grows into them and a state copy writes FP32 words."""
        buf, alloc = _build_state_attn("cuda")
        state, full = alloc.mamba_allocator, alloc.full_attn_allocator
        self.assertIsNotNone(alloc.alloc_state_slots(4))
        ids = alloc.alloc(full.available_size())
        self.assertIsNotNone(ids)
        lo = state._byte_high_frontier()
        hi = lo + state.entry_bytes_per_page
        buf.raw[lo:hi].fill_(0x11)
        torch.cuda.synchronize()
        done, seen = self._forward_reading(buf, lo, hi, sleep_cycles)
        alloc.set_latest_forward_done_event(done)
        alloc.set_inflight_forward(done, None, None)
        launch = alloc.forward_steps.forward_launched()
        alloc.forward_steps.forward_recorded(done)
        alloc.free(ids)
        running = not done.query()
        # Pending while the forward runs: an urgent flush settles them behind
        # a stream wait on it, so the state band grows into the bytes at once
        # and its copy queues after the forward.
        full.flush_for_allocation()
        slot = alloc.alloc_state_slots(1)
        self.assertIsNotNone(slot, "the state band did not grow into the bytes")
        page = int(state.ids.virtual_to_physical[slot][0])
        self.assertEqual(page * state.entry_bytes_per_page, lo)
        # A state copy into the new slot, on the schedule stream.
        buf.raw[lo:hi].view(torch.int32).fill_(FP32_WITH_BF16_NAN)
        torch.cuda.synchronize()
        alloc.forward_steps.forward_processed(launch)
        return seen, running

    def test_a_state_write_waits_a_forward_that_reads_the_bytes(self):
        self._state_into_bytes_the_forward_reads(0)
        torch.cuda.synchronize()

        seen, running = self._state_into_bytes_the_forward_reads(2_000_000_000)
        self.assertTrue(running, "the forward ended before its pages went pending")
        torch.cuda.synchronize()
        self.assertTrue(
            bool((seen == 0x11).all()),
            "the forward read another band's bytes as its attention page",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
