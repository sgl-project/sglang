"""Unit tests for ``cuda_graph_buffer_registry`` — CPU-only.

Covers:
  * ``GraphSlot`` shape / axis / device validation.
  * ``register_slot`` allocation + ``FILL_SENTINEL`` init.
  * ``fill_from`` D2D copy across all four ``PaddingPolicy`` modes.
  * ``extract_buffer`` returns a ``ForwardBatch`` view backed by slot
    buffers, non-slot fields carried from template.
  * ``post_fill`` hook runs after the grouped copy.
  * Missing FB attributes are silently skipped.

The registry is GPU-agnostic — tests run on CPU. The PaddingPolicy /
foreach_copy / view-slicing logic is fully exercised without any CUDA
context.
"""

import dataclasses
import unittest
from typing import Optional

import torch

from sglang.srt.model_executor.cuda_graph_buffer_registry import (
    CudaGraphBufferRegistry,
    GraphSlot,
    PaddingPolicy,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@dataclasses.dataclass
class _MiniForwardBatch:
    """Minimal FB stand-in: dataclass so ``dataclasses.replace`` works."""

    batch_size: int = 0
    input_ids: Optional[torch.Tensor] = None
    seq_lens: Optional[torch.Tensor] = None
    req_pool_indices: Optional[torch.Tensor] = None
    out_cache_loc: Optional[torch.Tensor] = None
    positions: Optional[torch.Tensor] = None
    seq_lens_cpu: Optional[torch.Tensor] = None
    encoder_lens: Optional[torch.Tensor] = None
    mrope_positions: Optional[torch.Tensor] = None
    num_token_non_padded: Optional[torch.Tensor] = None
    global_num_tokens_gpu: Optional[torch.Tensor] = None
    global_num_tokens_for_logprob_gpu: Optional[torch.Tensor] = None
    forward_mode: Optional[str] = None
    spec_info: Optional[object] = None


def _make_registry(max_bs: int = 8, max_num_tokens: int = 16):
    return CudaGraphBufferRegistry(
        device=torch.device("cpu"),
        max_bs=max_bs,
        max_num_tokens=max_num_tokens,
    )


class TestGraphSlot(unittest.TestCase):
    def test_axis_validation(self):
        with self.assertRaises(ValueError):
            GraphSlot(
                name="bad",
                shape_fn=lambda bs, mt: (bs,),
                dtype=torch.int32,
                axis="garbage",
            )

    def test_view_before_buffer_alloc_raises(self):
        slot = GraphSlot(
            name="x",
            shape_fn=lambda bs, mt: (bs,),
            dtype=torch.int32,
            axis="bs",
        )
        with self.assertRaises(RuntimeError):
            slot.view(padded_bs=1, padded_num_tokens=1)


class TestRegistryRegister(unittest.TestCase):
    def test_register_allocates_zero_buffer(self):
        r = _make_registry()
        slot = r.register_slot(
            GraphSlot(
                name="input_ids",
                shape_fn=lambda bs, mt: (mt,),
                dtype=torch.int64,
                axis="tokens",
            )
        )
        self.assertEqual(slot.buffer.shape, (16,))
        self.assertEqual(slot.buffer.dtype, torch.int64)
        self.assertTrue(torch.equal(slot.buffer, torch.zeros(16, dtype=torch.int64)))

    def test_register_fill_sentinel_init(self):
        r = _make_registry()
        slot = r.register_slot(
            GraphSlot(
                name="seq_lens",
                shape_fn=lambda bs, mt: (bs,),
                dtype=torch.int32,
                axis="bs",
                padding_policy=PaddingPolicy.FILL_SENTINEL,
                pad_value=7,
            )
        )
        self.assertTrue(
            torch.equal(slot.buffer, torch.full((8,), 7, dtype=torch.int32))
        )

    def test_register_duplicate_raises(self):
        r = _make_registry()
        r.register_slot(
            GraphSlot(
                name="dup",
                shape_fn=lambda bs, mt: (bs,),
                dtype=torch.int32,
                axis="bs",
            )
        )
        with self.assertRaises(ValueError):
            r.register_slot(
                GraphSlot(
                    name="dup",
                    shape_fn=lambda bs, mt: (bs,),
                    dtype=torch.int32,
                    axis="bs",
                )
            )

    def test_disabled_slot_not_allocated(self):
        r = _make_registry()
        slot = r.register_slot(
            GraphSlot(
                name="off",
                shape_fn=lambda bs, mt: (bs,),
                dtype=torch.int32,
                axis="bs",
                enabled=False,
            )
        )
        self.assertIsNone(slot.buffer)
        self.assertFalse(r.has_slot("off"))
        self.assertNotIn("off", r.slot_names())

    def test_cpu_device_override(self):
        r = _make_registry()
        slot = r.register_slot(
            GraphSlot(
                name="seq_lens_cpu",
                shape_fn=lambda bs, mt: (bs,),
                dtype=torch.int32,
                axis="bs",
                device=torch.device("cpu"),
                padding_policy=PaddingPolicy.FILL_SENTINEL,
                pad_value=11,
            )
        )
        self.assertEqual(slot.buffer.device.type, "cpu")
        self.assertEqual(int(slot.buffer[0].item()), 11)


class TestFillFromAndExtract(unittest.TestCase):
    """End-to-end exercise: register a representative slot set, fill from
    a mini FB, then extract a FB view and assert all field-views match."""

    def _build_registry(self):
        r = _make_registry(max_bs=4, max_num_tokens=8)
        r.register_slot(
            GraphSlot(
                name="input_ids",
                shape_fn=lambda bs, mt: (mt,),
                dtype=torch.int64,
                axis="tokens",
            )
        )
        r.register_slot(
            GraphSlot(
                name="req_pool_indices",
                shape_fn=lambda bs, mt: (bs,),
                dtype=torch.int64,
                axis="bs",
                padding_policy=PaddingPolicy.ZERO,
            )
        )
        r.register_slot(
            GraphSlot(
                name="seq_lens",
                shape_fn=lambda bs, mt: (bs,),
                dtype=torch.int32,
                axis="bs",
                padding_policy=PaddingPolicy.FILL_SENTINEL,
                pad_value=5,
            )
        )
        r.register_slot(
            GraphSlot(
                name="out_cache_loc",
                shape_fn=lambda bs, mt: (mt,),
                dtype=torch.int64,
                axis="tokens",
                padding_policy=PaddingPolicy.ZERO,
            )
        )
        r.register_slot(
            GraphSlot(
                name="positions",
                shape_fn=lambda bs, mt: (mt,),
                dtype=torch.int64,
                axis="tokens",
            )
        )
        r.register_slot(
            GraphSlot(
                name="seq_lens_cpu",
                shape_fn=lambda bs, mt: (bs,),
                dtype=torch.int32,
                axis="bs",
                device=torch.device("cpu"),
                padding_policy=PaddingPolicy.FILL_SENTINEL,
                pad_value=5,
            )
        )
        return r

    def test_basic_fill_no_padding(self):
        r = self._build_registry()
        fb = _MiniForwardBatch(
            batch_size=4,
            input_ids=torch.arange(8, dtype=torch.int64),
            req_pool_indices=torch.tensor([3, 1, 4, 2], dtype=torch.int64),
            seq_lens=torch.tensor([10, 11, 12, 13], dtype=torch.int32),
            out_cache_loc=torch.arange(8, dtype=torch.int64) + 100,
            positions=torch.arange(8, dtype=torch.int64),
            seq_lens_cpu=torch.tensor([10, 11, 12, 13], dtype=torch.int32),
        )
        r.fill_from(
            fb,
            raw_bs=4,
            padded_bs=4,
            raw_num_tokens=8,
            padded_num_tokens=8,
        )
        self.assertTrue(torch.equal(r.get_slot("input_ids").buffer, fb.input_ids))
        self.assertTrue(
            torch.equal(r.get_slot("req_pool_indices").buffer, fb.req_pool_indices)
        )
        self.assertTrue(torch.equal(r.get_slot("seq_lens").buffer, fb.seq_lens))
        self.assertTrue(
            torch.equal(r.get_slot("out_cache_loc").buffer, fb.out_cache_loc)
        )
        self.assertTrue(torch.equal(r.get_slot("positions").buffer, fb.positions))
        self.assertTrue(torch.equal(r.get_slot("seq_lens_cpu").buffer, fb.seq_lens_cpu))

    def test_fill_with_padding_resets_zero_and_sentinel(self):
        r = self._build_registry()
        # Pre-poison the padded tail to a non-zero value so we can prove
        # the reset_padding step ran.
        r.get_slot("req_pool_indices").buffer.fill_(99)
        r.get_slot("seq_lens").buffer.fill_(99)
        r.get_slot("out_cache_loc").buffer.fill_(99)
        # Raw 2 reqs, 4 tokens; padded 4 reqs, 8 tokens.
        fb = _MiniForwardBatch(
            batch_size=2,
            input_ids=torch.arange(4, dtype=torch.int64),
            req_pool_indices=torch.tensor([3, 1], dtype=torch.int64),
            seq_lens=torch.tensor([10, 11], dtype=torch.int32),
            out_cache_loc=torch.arange(4, dtype=torch.int64) + 100,
            positions=torch.arange(4, dtype=torch.int64),
            seq_lens_cpu=torch.tensor([10, 11], dtype=torch.int32),
        )
        r.fill_from(
            fb,
            raw_bs=2,
            padded_bs=4,
            raw_num_tokens=4,
            padded_num_tokens=8,
        )
        # Raw region copied.
        self.assertTrue(
            torch.equal(
                r.get_slot("req_pool_indices").buffer[:2],
                torch.tensor([3, 1], dtype=torch.int64),
            )
        )
        # ZERO padding tail.
        self.assertTrue(
            torch.equal(
                r.get_slot("req_pool_indices").buffer[2:],
                torch.zeros(2, dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                r.get_slot("out_cache_loc").buffer[4:],
                torch.zeros(4, dtype=torch.int64),
            )
        )
        # FILL_SENTINEL padding tail.
        self.assertTrue(
            torch.equal(
                r.get_slot("seq_lens").buffer[2:],
                torch.tensor([5, 5], dtype=torch.int32),
            )
        )
        # seq_lens_cpu lives on CPU device.
        self.assertEqual(r.get_slot("seq_lens_cpu").buffer.device.type, "cpu")
        self.assertTrue(
            torch.equal(
                r.get_slot("seq_lens_cpu").buffer[2:],
                torch.tensor([5, 5], dtype=torch.int32),
            )
        )

    def test_keep_pad_preserves_padded_tail(self):
        r = _make_registry(max_bs=4, max_num_tokens=8)
        r.register_slot(
            GraphSlot(
                name="positions",
                shape_fn=lambda bs, mt: (mt,),
                dtype=torch.int64,
                axis="tokens",
                padding_policy=PaddingPolicy.KEEP_PAD,
            )
        )
        # Poison padded tail.
        r.get_slot("positions").buffer.fill_(77)
        fb = _MiniForwardBatch(
            batch_size=1,
            positions=torch.arange(2, dtype=torch.int64),
        )
        r.fill_from(
            fb,
            raw_bs=1,
            padded_bs=4,
            raw_num_tokens=2,
            padded_num_tokens=8,
        )
        # Raw is copied; padded tail stays at the poison value.
        self.assertTrue(
            torch.equal(
                r.get_slot("positions").buffer[:2],
                torch.tensor([0, 1], dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                r.get_slot("positions").buffer[2:],
                torch.full((6,), 77, dtype=torch.int64),
            )
        )

    def test_extract_buffer_returns_fb_view(self):
        r = self._build_registry()
        fb = _MiniForwardBatch(
            batch_size=2,
            input_ids=torch.arange(4, dtype=torch.int64),
            req_pool_indices=torch.tensor([3, 1], dtype=torch.int64),
            seq_lens=torch.tensor([10, 11], dtype=torch.int32),
            out_cache_loc=torch.arange(4, dtype=torch.int64) + 100,
            positions=torch.arange(4, dtype=torch.int64),
            seq_lens_cpu=torch.tensor([10, 11], dtype=torch.int32),
            forward_mode="DECODE",
            spec_info="dummy_spec",
        )
        r.fill_from(
            fb,
            raw_bs=2,
            padded_bs=4,
            raw_num_tokens=4,
            padded_num_tokens=8,
        )
        fb_view = r.extract_buffer(
            padded_bs=4,
            padded_num_tokens=8,
            forward_batch_template=fb,
        )
        # batch_size is padded.
        self.assertEqual(fb_view.batch_size, 4)
        # Tensor fields are now buffer views — same data_ptr as the slot.
        self.assertEqual(
            fb_view.input_ids.data_ptr(),
            r.get_slot("input_ids").buffer.data_ptr(),
        )
        # Length is padded.
        self.assertEqual(len(fb_view.input_ids), 8)
        self.assertEqual(len(fb_view.req_pool_indices), 4)
        # Non-slot fields carried from template.
        self.assertEqual(fb_view.forward_mode, "DECODE")
        self.assertEqual(fb_view.spec_info, "dummy_spec")
        # Template itself NOT mutated (replace returns a new instance).
        self.assertEqual(fb.batch_size, 2)
        self.assertIsNot(fb_view, fb)


class TestMissingAndOptionalSlots(unittest.TestCase):
    def test_missing_fb_attr_is_skipped(self):
        r = _make_registry()
        r.register_slot(
            GraphSlot(
                name="encoder_lens",
                shape_fn=lambda bs, mt: (bs,),
                dtype=torch.int32,
                axis="bs",
                padding_policy=PaddingPolicy.FILL_SENTINEL,
                pad_value=0,
            )
        )
        fb = _MiniForwardBatch(
            batch_size=2,
            input_ids=torch.arange(4, dtype=torch.int64),
            encoder_lens=None,  # FB doesn't carry this for this request.
        )
        # Should NOT raise; encoder_lens buffer stays at the FILL_SENTINEL
        # init value.
        r.fill_from(
            fb,
            raw_bs=2,
            padded_bs=4,
            raw_num_tokens=4,
            padded_num_tokens=8,
        )
        self.assertTrue(
            torch.equal(
                r.get_slot("encoder_lens").buffer,
                torch.zeros(8, dtype=torch.int32),
            )
        )


class TestPostFillHook(unittest.TestCase):
    def test_post_fill_runs_after_copy(self):
        observed = {}

        def hook(buf, fb, ctx):
            # Multiply the raw region by 10 in-place.
            buf[: ctx.raw_num_tokens] *= 10
            observed["raw_n"] = ctx.raw_num_tokens
            observed["padded_n"] = ctx.padded_num_tokens

        r = _make_registry(max_bs=4, max_num_tokens=8)
        r.register_slot(
            GraphSlot(
                name="input_ids",
                shape_fn=lambda bs, mt: (mt,),
                dtype=torch.int64,
                axis="tokens",
                post_fill=hook,
            )
        )
        fb = _MiniForwardBatch(
            batch_size=2,
            input_ids=torch.arange(1, 5, dtype=torch.int64),  # [1,2,3,4]
        )
        r.fill_from(
            fb,
            raw_bs=2,
            padded_bs=4,
            raw_num_tokens=4,
            padded_num_tokens=8,
        )
        self.assertTrue(
            torch.equal(
                r.get_slot("input_ids").buffer[:4],
                torch.tensor([10, 20, 30, 40], dtype=torch.int64),
            )
        )
        self.assertEqual(observed, {"raw_n": 4, "padded_n": 8})


class TestSliceFnSlot(unittest.TestCase):
    """``mrope_positions`` has shape ``[3, T]`` — sliced on axis 1."""

    def test_slice_fn_handles_2d_tokens_axis(self):
        r = _make_registry(max_bs=4, max_num_tokens=8)
        r.register_slot(
            GraphSlot(
                name="mrope_positions",
                shape_fn=lambda bs, mt: (3, mt),
                dtype=torch.int64,
                axis="tokens",
                slice_fn=lambda buf, n: buf[:, :n],
            )
        )
        fb = _MiniForwardBatch(
            batch_size=2,
            mrope_positions=torch.arange(12, dtype=torch.int64).reshape(3, 4),
        )
        r.fill_from(
            fb,
            raw_bs=2,
            padded_bs=4,
            raw_num_tokens=4,
            padded_num_tokens=8,
        )
        # Raw 3x4 region copied.
        self.assertTrue(
            torch.equal(
                r.get_slot("mrope_positions").buffer[:, :4],
                fb.mrope_positions,
            )
        )
        # extract_buffer should hand back the [:, :padded_num_tokens] view.
        fb_view = r.extract_buffer(
            padded_bs=4,
            padded_num_tokens=8,
            forward_batch_template=fb,
        )
        self.assertEqual(fb_view.mrope_positions.shape, (3, 8))


class TestPoolBackedAlloc(unittest.TestCase):
    """``share_pool=True`` coalesces same-named slot buffers through the
    global ForwardInputBuffers pool (so a registry can share storage with
    the legacy DecodeInputBuffers during migration)."""

    def setUp(self):
        from sglang.srt.model_executor import input_buffers

        input_buffers._forward_input_buffer_pool.clear()

    def _reg(self, *, max_num_tokens=16, share_pool):
        return CudaGraphBufferRegistry(
            device=torch.device("cpu"),
            max_bs=8,
            max_num_tokens=max_num_tokens,
            share_pool=share_pool,
        )

    @staticmethod
    def _ids_slot(name):
        return GraphSlot(
            name=name,
            shape_fn=lambda bs, mt: (mt,),
            dtype=torch.int64,
            axis="tokens",
        )

    def test_share_pool_off_is_independent(self):
        r1, r2 = self._reg(share_pool=False), self._reg(share_pool=False)
        r1.register_slot(self._ids_slot("ids"))
        r2.register_slot(self._ids_slot("ids"))
        self.assertNotEqual(
            r1.get_slot("ids").buffer.data_ptr(),
            r2.get_slot("ids").buffer.data_ptr(),
        )

    def test_larger_pool_buffer_is_reused(self):
        big = self._reg(max_num_tokens=32, share_pool=True)
        small = self._reg(max_num_tokens=16, share_pool=True)
        big.register_slot(self._ids_slot("ids"))
        small.register_slot(self._ids_slot("ids"))
        # small asks for 16 but is aliased onto big's 32-elem storage.
        self.assertEqual(tuple(small.get_slot("ids").buffer.shape), (16,))
        self.assertEqual(
            small.get_slot("ids").buffer.data_ptr(),
            big.get_slot("ids").buffer.data_ptr(),
        )


class TestBuildDecodeRegistry(unittest.TestCase):
    """``build_decode_registry`` registers the always-on FB-shared decode
    slots with padding policies matching
    ``DecodeInputBuffers.populate_from_forward_batch``."""

    def setUp(self):
        from sglang.srt.model_executor import input_buffers

        input_buffers._forward_input_buffer_pool.clear()

    def test_factory_slot_set_and_padding(self):
        from sglang.srt.model_executor.cuda_graph_buffer_registry import (
            build_decode_registry,
        )

        FILL = 5
        reg = build_decode_registry(
            device=torch.device("cpu"),
            max_bs=4,
            max_num_token=8,
            seq_len_fill_value=FILL,
            cache_loc_dtype=torch.int64,
            share_pool=False,
        )
        for name in (
            "input_ids",
            "positions",
            "out_cache_loc",
            "req_pool_indices",
            "seq_lens",
            "seq_lens_cpu",
            "mrope_positions",
        ):
            self.assertTrue(reg.has_slot(name), name)
        self.assertFalse(reg.has_slot("mamba_track_indices"))

        raw_bs, padded_bs, raw_nt, padded_nt = 2, 4, 2, 4
        fb = _MiniForwardBatch(
            batch_size=raw_bs,
            input_ids=torch.tensor([10, 11], dtype=torch.int64),
            positions=torch.tensor([0, 1], dtype=torch.int64),
            out_cache_loc=torch.tensor([100, 101], dtype=torch.int64),
            req_pool_indices=torch.tensor([1, 2], dtype=torch.int64),
            seq_lens=torch.tensor([7, 8], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([7, 8], dtype=torch.int32),
            mrope_positions=torch.tensor(
                [[0, 1], [0, 1], [0, 1]], dtype=torch.int64
            ),
        )
        # Poison tails so resets are observable.
        for n in ("input_ids", "positions", "out_cache_loc", "req_pool_indices"):
            reg.get_slot(n).buffer.fill_(99)
        reg.fill_from(
            fb,
            raw_bs=raw_bs,
            padded_bs=padded_bs,
            raw_num_tokens=raw_nt,
            padded_num_tokens=padded_nt,
        )

        # FOREACH_COPY: head copied, tail kept (poison).
        ids = reg.get_slot("input_ids").buffer
        self.assertTrue(torch.equal(ids[:2], torch.tensor([10, 11])))
        self.assertTrue(torch.equal(ids[2:4], torch.tensor([99, 99])))
        # ZERO: head copied, tail zeroed.
        oc = reg.get_slot("out_cache_loc").buffer
        self.assertTrue(torch.equal(oc[:2], torch.tensor([100, 101])))
        self.assertTrue(torch.equal(oc[2:4], torch.tensor([0, 0])))
        rp = reg.get_slot("req_pool_indices").buffer
        self.assertTrue(torch.equal(rp[:2], torch.tensor([1, 2])))
        self.assertTrue(torch.equal(rp[2:4], torch.tensor([0, 0])))
        # FILL_SENTINEL: head copied, tail = seq_len_fill_value.
        sl = reg.get_slot("seq_lens").buffer
        self.assertTrue(torch.equal(sl[:2], torch.tensor([7, 8], dtype=torch.int32)))
        self.assertTrue(
            torch.equal(sl[2:4], torch.tensor([FILL, FILL], dtype=torch.int32))
        )
        slc = reg.get_slot("seq_lens_cpu").buffer
        self.assertEqual(slc.device.type, "cpu")
        self.assertTrue(
            torch.equal(slc[2:4], torch.tensor([FILL, FILL], dtype=torch.int32))
        )
        # 2D mrope via slice_fn.
        mr = reg.get_slot("mrope_positions").buffer
        self.assertTrue(torch.equal(mr[:, :2], fb.mrope_positions))

        fb_view = reg.extract_buffer(
            padded_bs=padded_bs,
            padded_num_tokens=padded_nt,
            forward_batch_template=fb,
        )
        self.assertEqual(fb_view.batch_size, padded_bs)
        self.assertEqual(fb_view.input_ids.shape[0], padded_nt)
        self.assertEqual(fb_view.seq_lens.shape[0], padded_bs)

    def test_source_adopts_buffers(self):
        from types import SimpleNamespace

        from sglang.srt.model_executor.cuda_graph_buffer_registry import (
            build_decode_registry,
        )

        src = SimpleNamespace(
            input_ids=torch.zeros(8, dtype=torch.int64),
            positions=torch.zeros(8, dtype=torch.int64),
            out_cache_loc=torch.zeros(8, dtype=torch.int64),
            req_pool_indices=torch.zeros(4, dtype=torch.int64),
            seq_lens=torch.full((4,), 5, dtype=torch.int32),
            seq_lens_cpu=torch.full((4,), 5, dtype=torch.int32),
            mrope_positions=torch.zeros((3, 8), dtype=torch.int64),
            global_num_tokens_gpu=torch.zeros(1, dtype=torch.int32),
            global_num_tokens_for_logprob_gpu=torch.zeros(1, dtype=torch.int32),
        )
        reg = build_decode_registry(
            device=torch.device("cpu"),
            max_bs=4,
            max_num_token=8,
            seq_len_fill_value=5,
            cache_loc_dtype=torch.int64,
            source=src,
        )
        # Registry slots share storage with the source's tensors.
        for name in ("input_ids", "seq_lens", "seq_lens_cpu", "mrope_positions"):
            self.assertEqual(
                reg.get_slot(name).buffer.data_ptr(),
                getattr(src, name).data_ptr(),
                name,
            )


class TestFillOncePolicy(unittest.TestCase):
    """FILL_ONCE initializes the whole buffer at alloc and never resets the
    padded tail per iter (unlike FILL_SENTINEL)."""

    def test_fill_once_inits_once_and_keeps_tail(self):
        reg = CudaGraphBufferRegistry(
            device=torch.device("cpu"), max_bs=4, max_num_tokens=8
        )
        reg.register_slot(
            GraphSlot(
                "encoder_lens",
                lambda bs, mt: (bs,),
                torch.int32,
                axis="bs",
                padding_policy=PaddingPolicy.FILL_ONCE,
                pad_value=9,
            )
        )
        buf = reg.get_slot("encoder_lens").buffer
        self.assertTrue(torch.equal(buf, torch.tensor([9, 9, 9, 9], dtype=torch.int32)))
        buf[2:].fill_(99)  # poison the tail
        fb = _MiniForwardBatch(
            batch_size=2, encoder_lens=torch.tensor([1, 2], dtype=torch.int32)
        )
        reg.fill_from(fb, raw_bs=2, padded_bs=4, raw_num_tokens=2, padded_num_tokens=4)
        # Head copied; tail NOT reset (FILL_ONCE skips the per-iter reset).
        self.assertTrue(torch.equal(buf[:2], torch.tensor([1, 2], dtype=torch.int32)))
        self.assertTrue(
            torch.equal(buf[2:], torch.tensor([99, 99], dtype=torch.int32))
        )


class TestComputedSlots(unittest.TestCase):
    """num_token_non_padded (copy_from_fb + post_fill) and global_num_tokens
    (copy_from_fb=False + post_fill fill)."""

    def test_num_token_non_padded_copy_path(self):
        from sglang.srt.model_executor.cuda_graph_buffer_registry import (
            build_decode_registry,
        )

        reg = build_decode_registry(
            device=torch.device("cpu"),
            max_bs=4,
            max_num_token=8,
            seq_len_fill_value=5,
            cache_loc_dtype=torch.int64,
            enable_num_token_non_padded=True,
            require_gathered_buffer=False,
        )
        self.assertTrue(reg.has_slot("num_token_non_padded"))
        fb = _MiniForwardBatch(
            batch_size=2,
            num_token_non_padded=torch.tensor([7], dtype=torch.int32),
        )
        reg.fill_from(fb, raw_bs=2, padded_bs=2, raw_num_tokens=2, padded_num_tokens=2)
        # Non-gathered: plain FB copy, post_fill is a no-op.
        self.assertTrue(
            torch.equal(
                reg.get_slot("num_token_non_padded").buffer,
                torch.tensor([7], dtype=torch.int32),
            )
        )

    def test_global_num_tokens_fill_path(self):
        from sglang.srt.model_executor.cuda_graph_buffer_registry import (
            build_decode_registry,
        )

        reg = build_decode_registry(
            device=torch.device("cpu"),
            max_bs=4,
            max_num_token=8,
            seq_len_fill_value=5,
            cache_loc_dtype=torch.int64,
            require_gathered_buffer=True,
        )
        self.assertTrue(reg.has_slot("global_num_tokens_gpu"))
        # FB carries a stale value; copy_from_fb=False means it's ignored and
        # the slot is filled with padded_num_tokens by post_fill.
        fb = _MiniForwardBatch(
            batch_size=2,
            global_num_tokens_gpu=torch.tensor([999], dtype=torch.int32),
        )
        reg.fill_from(fb, raw_bs=2, padded_bs=4, raw_num_tokens=2, padded_num_tokens=4)
        self.assertTrue(
            torch.equal(
                reg.get_slot("global_num_tokens_gpu").buffer,
                torch.tensor([4], dtype=torch.int32),
            )
        )


if __name__ == "__main__":
    unittest.main()
