# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""INT4, INT6 and FP16 wire codecs for quick-allreduce.

A rank-tile is ``block`` threads x one 16 B atom, quantized in packed fp16 with
a group-16 signed 8-bit scale (E5M2, bias 12). INT4 is one nibble plane; INT6 adds a dense 2-bit
plane, which keeps the nibble plane byte-identical to INT4's and every region
on the 64 B fabric sector grid. FP16 is a passthrough wire format -- the
thread's eight fp16 values verbatim, no quantization -- used to test the
reduce-scatter/all-gather transport in isolation from the codec.

Limitations
--------------------

The kernels convert the bf16 payload to fp16 on load and back on store, for
every codec, and all in-kernel arithmetic is packed fp16. Two limits follow.

* fp16 holds at most 65504. A larger magnitude saturates to 65504 (see
  :func:`_clamp_fp16_overflow`), and a magnitude below fp16's resolution loses
  mantissa bits that bf16 has. The FP16 wire format is therefore not lossless
  for bf16 data outside the fp16 range.
* The group scale is a signed E5M2 byte with bias 12, spanning
  ``[2**-12, 1.75 * 2**19]``, so every fp16 group extremum fits and nothing
  clips below the fp16 limit above. It has 2 mantissa bits, so the scale
  rounds to within 1/8 of the extremum. INT6 floors the scale at ``2**-10``
  (see :attr:`Codec.scale_min_e5`). NaN and Inf are not propagated: they
  become finite values.

``block`` is a build parameter: it sets the tile width, and through it how
many blocks a payload gets.

Imported by the mesh and ring kernels, which must agree on it byte for byte.
Depends on ``allreduce_shared`` for ``BLOCK``, ``WAVE`` and ``I32_BYTES``.
"""

import functools
import math
from dataclasses import dataclass

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import gpu, range_constexpr

from ..kernels_common import ceildiv
from .allreduce_shared import BLOCK, I32_BYTES, SUPPORTED_BLOCKS, WAVE

SUPER_TILES = (1, 8)
# Two threads (PAIR) share one scale byte; GROUP threads share the i32 slot.
GROUP = 8
PAIR = 2

# Every workgroup width in ``SUPPORTED_BLOCKS`` can build a codec.
#
# Every region of a rank-tile has to be a whole number of 64 B fabric sectors:
# the fanout moves one sector per quad, so a region ending mid-sector would
# ship the next region's bytes with it and overwrite them in the peer's inbox.
# The nibble and 2-bit planes are 4*block and 2*block bytes, whole sectors at
# every width here. The scale region is block/2 bytes, a whole sector only from
# block 128 up; at 64 it is padded, see :func:`codecs_for_block`.
# i32 per 64 B fabric sector.
SECTOR_I32 = 16


def validate_block(block: int) -> int:
    block = int(block)
    if block not in SUPPORTED_BLOCKS:
        raise ValueError(
            f"codec block must be one of {SUPPORTED_BLOCKS}, got {block!r}"
        )
    return block


def _round_up_to_sector(n_i32: int) -> int:
    return ceildiv(n_i32, SECTOR_I32) * SECTOR_I32


# Dequant bit-trick: code | 0x6400 then + (-(1024+bias)) as f16x2 reconstructs
# (q - bias). fp16 with exponent field 1024.0 holds the integer in its low
# mantissa bits, so this works for any field that fits below bit 10 -- 4 bits
# with bias 8, 6 bits with bias 32.
_K_MASK_000F = 0x000F000F
_K_MASK_0003 = 0x00030003
_K_HALF2_1024 = 0x64006400
_K_HALF2_1032 = 0xE408E408  # -1032.0 fp16x2 = -(1024 + 8)
_K_HALF2_1056 = 0xE420E420  # -1056.0 fp16x2 = -(1024 + 32)

# Largest finite fp16. The encode scale is materialised as fp16, so anything
# above this becomes Inf there; see _codec_quant.
_FP16_MAX = 65504.0
# Exponent bias of the E5M2 group scale.
_SCALE_BIAS = 12
# Bump whenever the group-scale codec changes. The codec helpers are only
# called from nested functions, so FlyDSL does not hash them into the disk
# cache key; the kernels put this tag in their launch names, which is what
# keeps a warm cache from loading binaries built with an older codec.
SCALE_CODEC_TAG = "s5m2b12"


@dataclass(frozen=True)
class Codec:
    """One wire format for a rank-tile: how it packs, and the geometry that implies.

    ``bias`` is both the zero point of the unsigned code and the magnitude of
    the most negative one, because the codec maps a group's signed extremum
    onto ``-bias`` -- that is what uses the asymmetric range fully. The
    decoding factor is therefore ``-1/bias``.

    INT6 is INT4's nibble plane plus a dense 2-bit plane, rather than a 48-bit
    field per thread. A 48-bit field straddles i32 boundaries and leaves the
    regions off the 64 B fabric sector grid; two planes keep every region
    sector-aligned (16 + 8 + 2 = 26) and leave the nibble plane byte-identical
    to INT4's, so ``_fanout_to_next`` needs nothing but a different sector
    count.
    """

    name: str
    bits: int
    bias: int | None
    #: fp16x2 constant added after the ``| 0x6400`` trick: -(1024 + bias).
    dequant_bias: int | None
    #: i32 offset of the dense 2-bit plane in the rank-tile; None when the
    #: codec has only a nibble plane.
    hi2_i32_off: int | None
    scale_i32_off: int | None
    rank_tile_i32: int
    #: Payload i32 a thread contributes per atom.
    n_words_per_thread: int
    #: Workgroup width this instance was built for.
    block: int = BLOCK

    @property
    def has_scale(self) -> bool:
        return self.scale_i32_off is not None

    @property
    def dec_step(self) -> float:
        return -1.0 / self.bias

    @property
    def scale_min_e5(self) -> int:
        """Smallest scale exponent field for this codec.

        The encode factor ``1 / (scale * |dec_step|)`` is splat to fp16, so the
        scale must stay at or above ``1 / (|dec_step| * 65504)``: field 0
        (``2**-12``) for INT4, field 2 (``2**-10``) for INT6. A group below the
        floor is quantized against the floor scale.
        """
        if not self.has_scale:
            return 0
        floor = 1.0 / (abs(self.dec_step) * _FP16_MAX)
        return max(0, math.ceil(math.log2(floor)) + _SCALE_BIAS)

    @property
    def scale_rounds_up(self) -> bool:
        """Round the group scale up to the next E5M2 value, not to nearest.

        Rounding to nearest can round the scale below the extremum, which then
        clips by up to 1/8. That is about one INT4 step, but four INT6 steps,
        so INT6 rounds up: no clip, at a step up to 1/4 coarser. On a normal
        fill that keeps INT6 at E4M3's SQNR (33.5 vs 33.4 dB), where rounding
        to nearest loses 2.4 dB; INT4 keeps rounding to nearest (21.9 dB, vs
        21.5 dB rounding up).
        """
        return self.bits > 4

    @property
    def qmin(self) -> float:
        return -float(self.bias)

    @property
    def qmax(self) -> float:
        return float(self.bias - 1)

    @property
    def rank_tile_bytes(self) -> int:
        return self.rank_tile_i32 * I32_BYTES

    @property
    def n_sectors(self) -> int:
        return self.rank_tile_bytes // 64

    def plane_slots(self, tid):
        """``[(i32 offset within the rank-tile, store predicate)]``, one per
        payload word, in the order :func:`_codec_quant` returns them.

        The offset is where this thread's word lives; the predicate says
        whether this thread is the one that stores it -- a Python ``True``
        when every thread owns its word (INT4's nibble plane, fp16's four
        dense planes), or ``hi2_leader`` when a lane pair shares one slot
        (INT6's 2-bit plane). A load ignores the predicate: every thread reads
        a plane's word regardless of who wrote it.
        """
        if self.n_words_per_thread == 1:
            return [(tid, True)]
        if self.hi2_i32_off is not None:
            hi2_leader, hi2_slot = hi2_slot_of(tid)
            return [(tid, True), (fx.Int32(self.hi2_i32_off) + hi2_slot, hi2_leader)]
        # Dense multi-word codec (fp16): every thread owns every word, at a
        # fixed stride of one rank-tile row (``block`` i32) per word.
        return [
            (fx.Int32(w * self.block) + tid, True)
            for w in range_constexpr(self.n_words_per_thread)
        ]


@functools.cache
def codecs_for_block(block: int = BLOCK) -> dict[str, "Codec"]:
    """The three wire formats at a given workgroup width."""

    block = validate_block(block)
    nibble = block  # one i32 of nibbles per thread
    hi2 = block // PAIR  # INT6's dense 2-bit plane, one i32 per lane pair
    # One i32 per GROUP threads (4 packed scale bytes), padded to a sector.
    scale = _round_up_to_sector(block // GROUP)
    return {
        c.name: c
        for c in (
            # block*4 B nibbles then block/2 B scale: a 4.5*block B rank-tile,
            # 1152 B and 18 sectors at block=256.
            Codec(
                name="int4",
                bits=4,
                bias=8,
                dequant_bias=_K_HALF2_1032,
                hi2_i32_off=None,
                scale_i32_off=nibble,
                rank_tile_i32=nibble + scale,
                n_words_per_thread=1,
                block=block,
            ),
            # Plus a block*2 B 2-bit plane between them: 6.5*block B, 1664 B
            # and 26 sectors at block=256.
            Codec(
                name="int6",
                bits=6,
                bias=32,
                dequant_bias=_K_HALF2_1056,
                hi2_i32_off=nibble,
                scale_i32_off=nibble + hi2,
                rank_tile_i32=nibble + hi2 + scale,
                n_words_per_thread=2,
                block=block,
            ),
            # Four dense fp16x2 planes, no scale region: 16*block B, 4096 B and
            # 64 sectors at block=256. Passthrough wire format.
            Codec(
                name="fp16",
                bits=16,
                bias=None,
                dequant_bias=None,
                hi2_i32_off=None,
                scale_i32_off=None,
                rank_tile_i32=block * 4,
                n_words_per_thread=4,
                block=block,
            ),
        )
    }


CODECS = codecs_for_block(BLOCK)
INT4, INT6, FP16 = CODECS["int4"], CODECS["int6"], CODECS["fp16"]


def thread_lane(tid, block: int = BLOCK):
    """(wave, lane) for *tid* in a ``(block // WAVE) x WAVE`` block.

    ``lane`` is the codec's own required argument to :func:`_codec_quant` /
    :func:`_codec_dequant` (the pairing and shuffle width both key off it);
    ``wave`` is only a byproduct callers use for their own fanout layouts.
    Shared by the mesh, ring and codec-test kernels.

    Every supported *block* is a whole number of waves, so there is no partial
    wave for the shuffles to fall off.
    """
    if block == WAVE:
        return fx.Int32(0), tid
    thread_layout = fx.make_layout((block // WAVE, WAVE), (WAVE, 1))
    return fx.idx2crd(tid, thread_layout).unpack()


def scale_slot_of(tid, block: int = BLOCK):
    """(scale_slot, pair_in_slot) -- this thread's group-16 scale slot,
    and which half of the ``PAIR`` sharing that slot this thread is.

    ``scale_slot`` addresses the i32 the byte lives in; ``pair_in_slot``
    selects which of the two packed bytes within it, via
    :func:`_scale_from_word`.
    """
    scale_own_layout = fx.make_layout(
        (block // GROUP, GROUP // PAIR, PAIR), (GROUP, PAIR, 1)
    )
    scale_slot, pair_in_slot, _lane_in_pair = fx.idx2crd(tid, scale_own_layout).unpack()
    return scale_slot, pair_in_slot


def hi2_slot_of(tid):
    """(hi2_leader, hi2_slot) for INT6's dense 2-bit plane.

    Threads ``2t`` and ``2t+1`` share one i32 of that plane (see
    :func:`_compact_hi2`); ``hi2_leader`` is whether this thread is the even
    one that stores it, ``hi2_slot`` is which i32. Meaningless for INT4
    (``codec.n_words_per_thread == 1``), so callers that only ever run INT4 need not
    call this at all.
    """
    hi2_leader = (tid & fx.Int32(1)) == fx.Int32(0)
    hi2_slot = tid.shrui(fx.Int32(1))
    return hi2_leader, hi2_slot


def _scale_from_word(codec, word, pair_in_slot):
    """This thread's decoding scale, out of a packed group-16 scale word.

    ``None`` for a codec with no scale plane (fp16 passthrough).
    """
    if not codec.has_scale:
        return None
    e = word.shrui(pair_in_slot * fx.Int32(8)) & fx.Int32(0xFF)
    return _e4m3_decoding_scale(codec, e)


def _f16x2(packed):
    return fx.Vector.from_elements([packed], fx.Int32).bitcast(fx.Float16)


def _i32(vec):
    return vec.bitcast(fx.Int32)[0]


def _splat_f16x2(x):
    return fx.Vector.filled(2, x, fx.Float16)


def _clamp_fp16_overflow():
    """Saturate packed fp16 overflow to ±65504 instead of Inf.

    Packed add/mul/FMA follow MODE bit 23 (FP16_OVFL). Unset, overflow
    becomes Inf and every later FMA in that tile is Inf. Set, it saturates
    to the max finite fp16. INT4 is already a saturating codec, so a rare
    overflow should not poison the all-reduce.

    FlyDSL has no MODE helper; ``llvm.amdgcn.s.setreg`` is the same
    intrinsic ``rocdl.disable_xdl_arb_stall`` uses for a different bit.
    """
    # hwreg(HW_REG_MODE=1, offset=23, size=2): id | (offset << 6) | ((size - 1) << 11)
    imm = fx.Int32(1 | (23 << 6) | ((2 - 1) << 11)).ir_value()
    val = fx.Int32(1).ir_value()
    llvm.call_intrinsic(None, "llvm.amdgcn.s.setreg", [imm, val], [], [])


def _shuffle_f16x2(vec, xor_off):
    return _f16x2(fx.Int32(gpu.shuffle_xor(_i32(vec), xor_off, WAVE)))


def _pair_signed_ext_f16(atom):
    """Signed extremum of 16 fp16 (this thread's 8 + xor-1 neighbor)."""
    p0, p1, p2, p3 = (
        _f16x2(atom[0]),
        _f16x2(atom[1]),
        _f16x2(atom[2]),
        _f16x2(atom[3]),
    )
    wmax = fx.maxnumf(fx.maxnumf(p0, p1), fx.maxnumf(p2, p3))
    wmin = fx.min(fx.min(p0, p1), fx.min(p2, p3))
    wmax = fx.maxnumf(wmax, _shuffle_f16x2(wmax, 1))
    wmin = fx.min(wmin, _shuffle_f16x2(wmin, 1))
    pk = (abs(wmax) > abs(wmin)).select(wmax, wmin)
    lo, hi = pk[0], pk[1]
    return fx.Float32((abs(lo) > abs(hi)).select(lo, hi))


def _atom_bf16_to_f16(atom):
    """Convert a bf16 atom to fp16. Magnitudes above 65504 saturate, and values
    that fp16 cannot represent as finely as bf16 lose mantissa bits."""
    return fx.Vector(atom).bitcast(fx.BFloat16).to(fx.Float16).bitcast(fx.Int32)


def _atom_f16_to_bf16(atom):
    return fx.Vector(atom).bitcast(fx.Float16).to(fx.BFloat16).bitcast(fx.Int32)


def _f32_to_e4m3(x, min_e5=0, round_up=False):
    """Pack f32 to a signed 8-bit scale: 1 sign, 5-bit exp (bias 12), 2-bit mantissa.

    The scale was E4M3 with bias 7, which spans only ``[2**-7, 480]``: a
    group whose extremum is above 480 clipped, and the ring, which requantizes
    the running partial sum at every hop, clipped sums of a few values near
    300. E5M2 with bias 12 spans ``[2**-12, 1.75 * 2**19]``, so every fp16
    extremum fits. The byte, the layout and the instruction count are
    unchanged; the function names keep "e4m3" for a small diff.

    Decode is ``±(1 + m/4) * 2**(e - 12)`` for every ``e``, including 0
    (no denorms). ``0x7F`` / ``0xFF`` are max finite, not NaN.

    *min_e5* floors the exponent field (see :attr:`Codec.scale_min_e5`), and
    *round_up* rounds the mantissa up instead of to nearest (see
    :attr:`Codec.scale_rounds_up`).
    ``0x00`` is +0 only. Any other value that would land there (positive
    underflow, or exactly ``2**-12``) is stored as ``0x01``. Negative
    underflow keeps ``0x80`` (``-2**-12``). Overflow clamps both exponent
    and mantissa.
    """
    is_z = x == fx.Float32(0.0)
    sign = (x < fx.Float32(0.0)).select(fx.Int32(0x80), fx.Int32(0))
    bits = abs(x).bitcast(fx.Int32)
    e = (bits.shrui(fx.Int32(23)) & fx.Int32(255)) - fx.Int32(127)
    mant = bits & fx.Int32(0x7FFFFF)
    m2 = (mant + fx.Int32((1 << 21) - 1 if round_up else 1 << 20)).shrui(fx.Int32(21))
    carry = m2 == fx.Int32(4)
    e = e + carry.select(fx.Int32(1), fx.Int32(0))
    m2 = carry.select(fx.Int32(0), m2)
    e5 = e + fx.Int32(_SCALE_BIAS)
    overflow = e5 > fx.Int32(31)
    m2 = overflow.select(fx.Int32(3), m2)
    e5 = (e5 < fx.Int32(min_e5)).select(
        fx.Int32(min_e5), overflow.select(fx.Int32(31), e5)
    )
    byte = sign | (e5 << fx.Int32(2)) | (m2 & fx.Int32(3))
    return is_z.select(fx.Int32(0), (byte == fx.Int32(0)).select(fx.Int32(1), byte))


def _e4m3_to_f32(b):
    is_z = b == fx.Int32(0)
    sign = (b & fx.Int32(0x80)) != fx.Int32(0)
    e5 = b.shrui(fx.Int32(2)) & fx.Int32(31)
    m2 = b & fx.Int32(3)
    mag_bits = ((e5 + fx.Int32(127 - _SCALE_BIAS)) << fx.Int32(23)) | (
        m2 << fx.Int32(21)
    )
    mag = mag_bits.bitcast(fx.Float32)
    signed = sign.select(-mag, mag)
    return is_z.select(fx.Float32(0.0), signed)


def _pack_fields(fields, width, mask=None):
    """``fields[i] << (i * width)`` OR-ed together, each masked first if *mask*."""
    out = fields[0] if mask is None else fields[0] & mask
    for i in range_constexpr(1, len(fields)):
        f = fields[i] if mask is None else fields[i] & mask
        out = out | (f << fx.Int32(i * width))
    return out


def _pack_e4m3_word(e, lane):
    """Four pair-scale bytes into the i32 scale slot (lanes 0,2,4,6 of GROUP)."""
    base = (lane // GROUP) * GROUP
    src = [base] + [base + fx.Int32(k) for k in range_constexpr(PAIR, GROUP, PAIR)]
    e_pair = [fx.Int32(gpu.shuffle_idx(e, s, WAVE)) for s in src]
    return _pack_fields(e_pair, 8, mask=fx.Int32(0xFF))


def _e4m3_decoding_scale(codec, e):
    return _splat_f16x2(_e4m3_to_f32(e) * fx.Float32(codec.dec_step))


def _clamp_f32(x, lo, hi):
    x = (x < fx.Float32(lo)).select(fx.Float32(lo), x)
    return (x > fx.Float32(hi)).select(fx.Float32(hi), x)


def _quant_atom_fp16(codec, atom, enc_pk):
    """Quantize 8 fp16 into this codec's planes, as packed i32 words."""
    q = []
    lo = _splat_f16x2(fx.Float16(codec.qmin))
    hi = _splat_f16x2(fx.Float16(codec.qmax))
    bias = fx.Vector.filled(2, fx.Int16(codec.bias), fx.Int16)
    for i in range_constexpr(4):
        w = fx.min(fx.maxnumf(_f16x2(atom[i]) * enc_pk, lo), hi)
        q.append(_i32(fx.roundeven(w).to(fx.Int16) + bias))
    if codec.n_words_per_thread == 1:
        return (_pack_fields(q, 4),)
    # Every code is masked here. In INT4 each field already fills its whole
    # nibble, so the shift-or cannot collide; a 6-bit code would overrun its
    # neighbour's slot if left whole.
    m4 = fx.Int32(_K_MASK_000F)
    m2 = fx.Int32(_K_MASK_0003)
    lo4 = [qi & m4 for qi in q]
    hi2 = [qi.shrui(fx.Int32(4)) & m2 for qi in q]
    return (_pack_fields(lo4, 4), _pack_fields(hi2, 2))


def _compact_hi2(packed_hi, lane):
    """Two threads' 2-bit planes into the single i32 they share on the wire.

    ``packed_hi`` carries its 8 live bits at [0..7] and [16..23] -- the f16x2
    pairing puts a thread's even elements in the low half of every i32 and its
    odd ones in the high half. Squeeze those to 16 dense bits, then merge with
    the xor-1 neighbour. Both lanes of the pair compute the same word; only the
    even one stores it, at ``hi2_i32_off + (tid >> 1)``.

    ``lane & 1 == tid & 1``, so this is the same pairing
    :func:`_pair_signed_ext_f16` already uses for the group-16 extremum -- no
    second convention is introduced.
    """
    c = (packed_hi & fx.Int32(0xFF)) | (packed_hi.shrui(fx.Int32(8)) & fx.Int32(0xFF00))
    other = c.shuffle_xor(1, WAVE)
    is_even = (lane & fx.Int32(1)) == fx.Int32(0)
    return is_even.select(c | (other << fx.Int32(16)), other | (c << fx.Int32(16)))


def _expand_hi2(word, tid):
    """Inverse of :func:`_compact_hi2`, for the calling thread's half."""
    c = word.shrui((tid & fx.Int32(1)) * fx.Int32(16)) & fx.Int32(0xFFFF)
    return (c & fx.Int32(0xFF)) | ((c & fx.Int32(0xFF00)) << fx.Int32(8))


def _codec_quant(codec, atom, lane, tid):
    """Quantize one atom. Returns ``(words, scale_word, is_leader)``.

    ``words`` is this codec's payload i32s in wire order, and is opaque to the
    caller: the ring restages received words verbatim on its all-gather lap and
    must not have to know how many there are.

    A codec with no scale plane (fp16 passthrough) skips quantization
    entirely.
    """
    if not codec.has_scale:
        return tuple(atom[i] for i in range_constexpr(4)), None, False
    ext = _pair_signed_ext_f16(atom)
    e = _f32_to_e4m3(ext, codec.scale_min_e5, codec.scale_rounds_up)
    d = _e4m3_to_f32(e) * fx.Float32(codec.dec_step)
    # Clamp before the fp16 splat. An all-zero group gives d == 0, so the
    # reciprocal is 1e7 -- Inf once narrowed to fp16, and 0 * Inf is NaN. Only
    # MODE.FP16_OVFL saturation has been keeping that from poisoning a tile,
    # and INT6 cuts the headroom fourfold: |d| is four times smaller for a
    # given extremum, so 1/|d| peaks near 4096 rather than 1024.
    enc = _clamp_f32(fx.Float32(1.0) / (d + fx.Float32(1e-7)), -_FP16_MAX, _FP16_MAX)
    words = _quant_atom_fp16(codec, atom, _splat_f16x2(enc))
    if codec.n_words_per_thread == 2:
        words = (words[0], _compact_hi2(words[1], lane))
    is_leader = (tid % GROUP) == 0
    return words, _pack_e4m3_word(e, lane), is_leader


def _codec_dequant(codec, words, scale, tid, acc=None):
    """Unpack four codes to f16x2, scale, optionally FMA into *acc*.

    ``a * b + c`` does not contract to ``v_pk_fma_f16``; ``fx.fma`` does.
    Two fp16 lanes are independent channels, not a dot into f32.

    *words* are as they sit on the wire, so the 2-bit plane is still compacted
    and is expanded here -- once, outside the loop.

    A codec with no scale plane (fp16 passthrough) has nothing to unpack:
    *words* are already the atom's four dwords, so this reduces to a packed
    fp16 add into *acc* (or a passthrough when *acc* is ``None``).
    """
    if not codec.has_scale:
        if acc is None:
            return fx.Vector.from_elements(list(words[:4]), fx.Int32)
        out = [_i32(_f16x2(words[i]) + _f16x2(acc[i])) for i in range_constexpr(4)]
        return fx.Vector.from_elements(out, fx.Int32)
    out = []
    mask = fx.Int32(_K_MASK_000F)
    bias_hi = fx.Int32(_K_HALF2_1024)
    bias_lo = _f16x2(fx.Int32(codec.dequant_bias))
    packed = words[0]
    if codec.n_words_per_thread == 2:
        hi = _expand_hi2(words[1], tid)
        m2 = fx.Int32(_K_MASK_0003)
    for i in range_constexpr(4):
        code = packed.shrui(fx.Int32(i * 4)) & mask
        if codec.n_words_per_thread == 2:
            code = code | ((hi.shrui(fx.Int32(i * 2)) & m2) << fx.Int32(4))
        dq = _f16x2(code | bias_hi) + bias_lo
        if acc is None:
            out.append(_i32(dq * scale))
        else:
            out.append(_i32(fx.fma(dq, scale, _f16x2(acc[i]))))
    return fx.Vector.from_elements(out, fx.Int32)


def _codec_load(codec, get, tid, scale_slot):
    """Read one packet through ``get(i32_off_in_tile) -> i32``.

    Returns ``(words, e4m3_word)`` exactly as they sit on the wire -- the 2-bit
    plane stays compacted -- so a forwarding path can restage them byte for
    byte without decoding. ``e4m3_word`` is ``None`` for a codec with no scale
    plane (fp16 passthrough).
    """
    words = tuple(get(off) for off, _pred in codec.plane_slots(tid))
    scale_word = (
        get(fx.Int32(codec.scale_i32_off) + scale_slot) if codec.has_scale else None
    )
    return words, scale_word
