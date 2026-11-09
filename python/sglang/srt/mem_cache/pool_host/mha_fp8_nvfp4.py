"""Host (L2) pool holding an NVFP4 copy of an FP8 MHA device pool
(``SGLANG_HICACHE_FP4_HOST_KV``, see ``sglang.srt.mem_cache.hicache_fp4_host``).

- Host payload: the regular MHA host buffers in byte geometry, ``[2, tokens, layers, heads,
  head_dim // 2]`` uint8 (``page_first``), element ``2j`` in the low nibble of byte ``j``.
- Host scales: ``[tokens, layers, heads * head_dim // 16]`` uint8 (E4M3) for K and for V, one per
  16 consecutive values of a token's row; a fixed global scale of 1/4.
- Backup: one persistent Triton launch per K and V over every layer reads the FP8 device rows,
  quantizes them (``scale = e4m3(amax / 6 / g)``, ``q = e2m1_rn(x * 1 / (scale * g))``, TRT-LLM's
  NVFP4 quantizer, rounding with ``cvt.rn.satfinite.e2m1x2.f32``) and writes the packed bytes and
  scales straight to pinned host memory.
- Load-back: one persistent launch per K and V per layer reads them from host and converts each
  byte with ``cvt.rn.f16x2.e2m1x2``, ``mul.rn.f16x2`` by the scale (exact) and
  ``cvt.rn.satfinite.e4m3x2.f16x2`` (nearest even) into the device FP8 rows. The hardware
  conversions keep the copy within the CTA quota's time.
- Both run on at most a quota of CTAs (``hicache_fp4_host``), and are built at init.
- ``SGLANG_HICACHE_FP4_HOST_FUSED_BACKUP=1``: the backup runs as one fused kernel per op
  (``sglang.kernels.ops.kvcache.hicache_nvfp4_backup``), the Triton backup's arithmetic with each
  token's host span written whole; a startup self-test against the Triton backup on every (block amax,
  value) pair and on random rows must match, else Triton stays.
- ``SGLANG_HICACHE_FP4_HOST_TMA_LOAD=1``: the load-back runs as one fused TMA kernel per layer
  (``sglang.kernels.ops.kvcache.hicache_nvfp4_load``) that dequantizes by a table this pool builds
  with the Triton load above and checks on all 65,536 (scale, byte) pairs; a startup self-test
  against the Triton load on this pool's own host buffers must also match, else Triton stays.
- Engagement: ``HICACHE_FP4_HOST load-back: N ops, T tokens ...`` and ``HICACHE_FP4_HOST backup:
  ...`` are logged at 1, 2, 4, 8, ... ops, so a run's log shows how much KV the NVFP4 copy carried.
- Packed MTP draft layers keep their FP8 geometry in an inner MHA host pool sharing this
  pool's host token slots (the controller's draft layer ``target + d`` is its layer ``d``). It
  takes the unstaged write-back: the controller hands this pool GPU host indices, since this
  pool has no staged write-back, and the staged kernel takes CPU indices only.
- L3 storage is not supported (the scales live outside ``kv_buffer``).
"""

from __future__ import annotations

import logging
from typing import Sequence

import msgspec
import torch
import triton
import triton.language as tl

from sglang.srt.mem_cache.device_pool_info import MHABufferInfo
from sglang.srt.mem_cache.hicache_fp4_host import (
    BACKUP_BLOCK_QUOTA,
    FUSED_BACKUP,
    FUSED_BACKUP_ENV,
    LOAD_BLOCK_QUOTA,
    SCALE_BLOCK,
    TMA_LOAD,
    TMA_LOAD_ENV,
    log_once,
)
from sglang.srt.mem_cache.pool_host.common import (
    ALLOC_MEMORY_FUNCS,
    _cuda_host_unregister,
)
from sglang.srt.mem_cache.pool_host.mha import (
    MHATokenToKVPoolHost,
    _is_cuda,
    _is_hip,
)

logger = logging.getLogger(__name__)

# Tokens per tile and warps: the fastest of 32-128 x 4-16 on VR200, load at quota 8
# (mb-h365h-perf-4), backup at quota 64 (mb-h530-backup).
LOAD_TOKENS_PER_TILE = 128
LOAD_WARPS = 16
BACKUP_TOKENS_PER_TILE = 32
BACKUP_WARPS = 16
# Fused TMA load (TMA_LOAD): shared-memory ring stages of one 64-token page (18 KB at 512 B rows)
# and dequant warps per CTA; the CTA count is LOAD_BLOCK_QUOTA. Tuned on VR200 (mb-h536).
TMA_LOAD_STAGES = 10
TMA_LOAD_STORE_WARPS = 4
# Fused backup (FUSED_BACKUP): warps per CTA and steps of two layer rows each warp loads before it
# quantizes; the CTA count is BACKUP_BLOCK_QUOTA. Tuned on VR200 (mb-h539).
FUSED_BACKUP_WARPS = 16
FUSED_BACKUP_GROUP = 3
# A power of two: exact, keeps small blocks' scales out of E4M3 subnormals, and 448 / 6 / g
# (FP8's largest value) stays below E4M3's 448.
GLOBAL_SCALE = 0.25


@triton.jit
def _e2m1x2_from_f32(low, high):
    """``cvt.rn.satfinite.e2m1x2.f32``: two FP32 values to one E2M1 pair (``low`` in the low
    nibble), round to nearest even, saturating at 6, the sign kept on a round to zero."""
    return tl.inline_asm_elementwise(
        "{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, $2, $1; cvt.u16.u8 $0, t; }",
        "=h,r,r", [low, high], dtype=tl.int16, is_pure=True, pack=1,
    )


@triton.jit
def _f16x2_from_e2m1x2(byte):
    """``cvt.rn.f16x2.e2m1x2``: one E2M1 pair (in a 16-bit lane) to two F16 values, exactly."""
    return tl.inline_asm_elementwise(
        "{ .reg .b8 t; cvt.u8.u16 t, $1; cvt.rn.f16x2.e2m1x2 $0, t; }",
        "=r,h", [byte], dtype=tl.int32, is_pure=True, pack=1,
    )


@triton.jit
def _e4m3x2_from_f16x2_scaled(values, scale):
    """``values * scale`` (F16x2; exact for an E2M1 times an E4M3 scale times a power of two)
    rounded to an E4M3 pair, nearest even, saturating."""
    return tl.inline_asm_elementwise(
        "{ .reg .b32 t; mul.rn.f16x2 t, $1, $2; cvt.rn.satfinite.e4m3x2.f16x2 $0, t; }",
        "=h,r,r", [values, scale], dtype=tl.int16, is_pure=True, pack=1,
    )


@triton.jit(do_not_specialize=["DEV_IDX", "HOST_IDX", "n", "layers"])
def _backup_fp8_to_nvfp4(
    SRC_PTRS, DEV_IDX, HOST_IDX, HOST_DATA, HOST_SCALE, n, layers,
    ROW: tl.constexpr, BT: tl.constexpr, GS: tl.constexpr,
):
    """Quantize device FP8 rows (``ROW`` values each) of every layer to host NVFP4, ``BT``
    tokens of one layer per work item, the items strided over the grid."""
    tiles = tl.cdiv(n, BT)
    for item in range(tl.program_id(0), tiles * layers, tl.num_programs(0)):
        _backup_tile(SRC_PTRS, DEV_IDX, HOST_IDX, HOST_DATA, HOST_SCALE, n, layers,
                     item // layers, item % layers, ROW, BT, GS)


@triton.jit
def _backup_tile(SRC_PTRS, DEV_IDX, HOST_IDX, HOST_DATA, HOST_SCALE, n, layers, tile, layer,
                 ROW: tl.constexpr, BT: tl.constexpr, GS: tl.constexpr):
    t = tile * BT + tl.arange(0, BT)
    live = (t < n)[:, None]
    dev = tl.load(DEV_IDX + t, mask=t < n, other=0).to(tl.int64)
    host = tl.load(HOST_IDX + t, mask=t < n, other=0).to(tl.int64)
    COLS: tl.constexpr = ROW // 2
    NB: tl.constexpr = ROW // 16
    # One 16-bit lane holds elements 2j (low byte) and 2j + 1 of the FP8 row.
    src = tl.load(SRC_PTRS + layer).to(tl.pointer_type(tl.int16))
    j = tl.arange(0, COLS)
    pair = tl.load(src + dev[:, None] * COLS + j[None, :], mask=live, other=0).to(tl.int32)
    even = (pair & 255).to(tl.uint8).to(tl.float8e4nv, bitcast=True).to(tl.float32)
    odd = ((pair >> 8) & 255).to(tl.uint8).to(tl.float8e4nv, bitcast=True).to(tl.float32)
    amax = tl.max(tl.reshape(tl.maximum(tl.abs(even), tl.abs(odd)), (BT, NB, 8)), axis=2)
    scale = (amax * (1.0 / 6.0) * (1.0 / GS)).to(tl.float8e4nv)
    scale32 = scale.to(tl.float32)
    inv = tl.where(scale32 == 0, 0.0, 1.0 / (scale32 * GS))
    inv = tl.reshape(tl.broadcast_to(inv[:, :, None], (BT, NB, 8)), (BT, COLS))
    byte = _e2m1x2_from_f32(even * inv, odd * inv).to(tl.uint8)
    row = host * layers + layer
    tl.store(HOST_DATA + row[:, None] * COLS + j[None, :], byte, mask=live)
    b = tl.arange(0, NB)
    tl.store(HOST_SCALE + row[:, None] * NB + b[None, :], scale.to(tl.uint8, bitcast=True), mask=live)


@triton.jit(do_not_specialize=["DEV_IDX", "HOST_IDX", "n", "layer", "layers"])
def _load_nvfp4_to_fp8(
    DST, DEV_IDX, HOST_IDX, HOST_DATA, HOST_SCALE, n, layer, layers,
    ROW: tl.constexpr, BT: tl.constexpr, GS: tl.constexpr,
):
    """Dequantize one layer's host NVFP4 rows into the device FP8 rows (``DST`` as 16-bit
    pairs), ``BT`` tokens per tile, the tiles strided over the grid."""
    for tile in range(tl.program_id(0), tl.cdiv(n, BT), tl.num_programs(0)):
        _load_tile(DST, DEV_IDX, HOST_IDX, HOST_DATA, HOST_SCALE, n, layer, layers, tile, ROW, BT, GS)


@triton.jit
def _load_tile(DST, DEV_IDX, HOST_IDX, HOST_DATA, HOST_SCALE, n, layer, layers, tile,
               ROW: tl.constexpr, BT: tl.constexpr, GS: tl.constexpr):
    t = tile * BT + tl.arange(0, BT)
    live = (t < n)[:, None]
    dev = tl.load(DEV_IDX + t, mask=t < n, other=0).to(tl.int64)
    host = tl.load(HOST_IDX + t, mask=t < n, other=0).to(tl.int64)
    COLS: tl.constexpr = ROW // 2
    NB: tl.constexpr = ROW // 16
    row = host * layers + layer
    j = tl.arange(0, COLS)
    byte = tl.load(HOST_DATA + row[:, None] * COLS + j[None, :], mask=live, other=0).to(tl.int16)
    b = tl.arange(0, NB)
    scale = tl.load(HOST_SCALE + row[:, None] * NB + b[None, :], mask=live, other=0)
    # E4M3 * GS is exact in F16 (GS a power of two); duplicate it into both F16x2 halves.
    half = (scale.to(tl.float8e4nv, bitcast=True).to(tl.float16) * GS).to(tl.int16, bitcast=True)
    half = half.to(tl.int32) & 0xFFFF
    scale2 = tl.reshape(tl.broadcast_to((half | (half << 16))[:, :, None], (BT, NB, 8)), (BT, COLS))
    pair = _e4m3x2_from_f16x2_scaled(_f16x2_from_e2m1x2(byte), scale2)
    tl.store(DST + dev[:, None] * COLS + j[None, :], pair, mask=live)


class _HostBytes:
    """A pinned host buffer as a Triton pointer argument: registered host memory is mapped
    under UVA, so kernels address it directly, as the JIT HiCache kernels do."""

    __slots__ = ("_tensor", "dtype")

    def __init__(self, tensor: torch.Tensor):
        self._tensor = tensor
        self.dtype = tensor.dtype

    def data_ptr(self) -> int:
        return self._tensor.data_ptr()


def _row_bytes_per_token(info: MHABufferInfo) -> int:
    return (info.head_dim + info.v_head_dim) * info.head_num * info.dtype.itemsize


class _NestedDraftHost(MHATokenToKVPoolHost):
    """The FP8 MTP draft rows inside the NVFP4 pool, on its token slots. The outer pool's
    allocation line and host-memory check already count these bytes, so this pool logs none."""

    def _allocate_host_buffer(self) -> None:
        self.kv_buffer = self.init_kv_buffer()
        self.fd = getattr(self.allocator, "fd", None)


class MHATokenToKVPoolFP8NVFP4Host(MHATokenToKVPoolHost):
    def __init__(
        self,
        buffer_info: MHABufferInfo,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
        *,
        device_capacity_tokens: int,
        start_layer: int,
        end_layer: int,
        draft_buffer_infos: Sequence[MHABufferInfo] = (),
        write_back_staging=None,
        pool_label: str = "kv",
        host_token_capacity: int | None = None,
    ):
        if layout != "page_first":
            raise NotImplementedError(
                f"NVFP4 host copy of FP8 KV supports only the page_first layout, got {layout!r}."
            )
        info = buffer_info
        if info.transfer_buffers is not None or info.scales is not None:
            raise NotImplementedError(f"{pool_label}: needs plain FP8 device rows.")
        if not (_is_cuda and pin_memory and info.device.type == "cuda"):
            # The kernels address the pinned host buffers through UVA.
            raise NotImplementedError(f"{pool_label}: NVFP4 host copy needs CUDA and pinned host memory.")
        half = info.head_dim // 2
        self._row = info.head_num * info.head_dim
        if self._row & (self._row - 1):
            raise NotImplementedError(f"{pool_label}: heads * head_dim must be a power of two.")
        self._device_k = tuple(t.view(torch.uint8) for t in info.buffers.k)
        self._device_v = tuple(t.view(torch.uint8) for t in info.buffers.v)
        for t in self._device_k + self._device_v:
            if not t.is_contiguous() or t.stride(0) != self._row:
                raise NotImplementedError(f"{pool_label}: device rows must be contiguous.")

        # The base class sizes and lays out the host payload in byte geometry; it sees the
        # device rows as strided half-rows and never transfers them (both paths are overridden).
        def half_rows(t: torch.Tensor) -> torch.Tensor:
            return t.as_strided((t.shape[0], info.head_num, half), (self._row, info.head_dim, 1))

        byte_info = msgspec.structs.replace(
            info,
            buffers=msgspec.structs.replace(
                info.buffers,
                k=tuple(half_rows(t) for t in self._device_k),
                v=tuple(half_rows(t) for t in self._device_v),
            ),
            head_dim=half,
            v_head_dim=half,
            dtype=torch.uint8,
        )
        self._scale_cols = self._row // SCALE_BLOCK
        self._draft_infos = tuple(draft_buffer_infos)
        self._draft_bytes_per_token = sum(map(_row_bytes_per_token, self._draft_infos))
        self.k_scale_host = None
        self.v_scale_host = None
        self._draft_host = None
        self._tma_load = None
        self._fused_backup = None
        self._fused_backup_draft = None  # (k table, v table, row stride) when the fused backup copies the draft rows
        # Engagement counters: [ops, tokens] per direction (_count).
        self._copied = {"backup": [0, 0], "load-back": [0, 0]}
        super().__init__(
            byte_info,
            host_to_device_ratio,
            host_size,
            page_size,
            layout,
            pin_memory,
            device,
            allocator_type,
            device_capacity_tokens=device_capacity_tokens,
            start_layer=start_layer,
            end_layer=end_layer,
            pool_label=pool_label,
            host_token_capacity=host_token_capacity,
        )
        self._init_scale_buffers()
        self._k_ptrs, self._v_ptrs = (
            torch.tensor([t.data_ptr() for t in ts], dtype=torch.int64, device=self._transfer_device)
            for ts in (self._device_k, self._device_v)
        )
        if self._draft_infos:
            self._draft_host = _NestedDraftHost(
                self._draft_infos[0],
                host_to_device_ratio,
                0,
                page_size,
                layout,
                pin_memory,
                device,
                allocator_type,
                device_capacity_tokens=device_capacity_tokens,
                start_layer=0,
                end_layer=len(self._draft_infos) - 1,
                draft_buffer_infos=self._draft_infos[1:],
                pool_label=f"{pool_label}_draft",
                host_token_capacity=self.size,
            )
            self._unstage(self._draft_host)
        self._k_dst_ptrs = tuple(t.data_ptr() for t in self._device_k)
        self._v_dst_ptrs = tuple(t.data_ptr() for t in self._device_v)
        if TMA_LOAD:
            self._tma_load = self._build_tma_load()
        if FUSED_BACKUP:
            self._fused_backup = self._build_fused_backup()
        self._warm_up()
        log_once(
            "host_pool",
            "NVFP4 host copy of FP8 KV: %d tokens, %d target layers of %d B packed rows plus "
            "%d B scales per token, %d FP8 draft layers (%d B/token), %.2f GB total; copy "
            "kernels built, load quota %d, backup quota %d CTAs",
            self.size,
            self.target_layer_num,
            self.token_stride_size,
            2 * self._scale_cols,
            len(self._draft_infos),
            self._draft_bytes_per_token,
            self.size * self.size_per_token / 1e9,
            LOAD_BLOCK_QUOTA,
            BACKUP_BLOCK_QUOTA,
        )

    def _count(self, kind: str, tokens: int) -> None:
        """Count one backup or load-back op; log the totals at 1, 2, 4, 8, ... ops."""
        totals = self._copied[kind]
        totals[0] += 1
        totals[1] += tokens
        if totals[0] & (totals[0] - 1) == 0:
            logger.info(
                "HICACHE_FP4_HOST %s: %d ops, %d tokens through the NVFP4 host copy",
                kind,
                totals[0],
                totals[1],
            )

    def _resolve_load_block_quota(self) -> None:
        """No stock JIT load runs on this pool's packed rows; its own load kernel takes
        hicache_fp4_host.LOAD_BLOCK_QUOTA. The inner FP8 draft pool keeps the stock resolver."""
        return None

    @staticmethod
    def _unstage(pool: MHATokenToKVPoolHost) -> None:
        """The draft pool's backups take the unstaged kernel (GPU host indices) and free its staging."""
        pool.can_use_write_back_jit = False
        pool.staging_k_buffer = pool.staging_v_buffer = pool.staging_buffer = None
        pool.staging_page_capacity = pool.staging_token_capacity = 0

    def _warm_up(self) -> None:
        """Build both kernels now rather than at the first eviction or host hit."""
        idx = torch.zeros(1, dtype=torch.int64, device=self._transfer_device)
        self._backup(idx, idx, 0)
        self._load(0, idx, idx, 0)
        torch.cuda.synchronize(self._transfer_device)

    def get_size_per_token(self):
        payload = super().get_size_per_token()
        scales = 2 * self._scale_cols * self.target_layer_num
        return payload + scales + self._draft_bytes_per_token

    def host_bytes_detail(self) -> str:
        """Per-token parts of this pool's host bytes, for hicache_fp4_host.check_host_tier."""
        scales = 2 * self._scale_cols * self.target_layer_num
        payload = self.size_per_token - scales - self._draft_bytes_per_token
        return (
            f" ({self.size} tokens x {self.size_per_token} B: NVFP4 {payload} + scales "
            f"{scales} + FP8 draft {self._draft_bytes_per_token})"
        )

    def _init_scale_buffers(self):
        alloc_func = ALLOC_MEMORY_FUNCS[self._transfer_device.type]
        shape = (self.size, self.target_layer_num, self._scale_cols)
        self.k_scale_host, self.v_scale_host = (
            alloc_func(
                shape,
                dtype=torch.uint8,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
            for _ in range(2)
        )

    def _transfer_args(self, host_indices, device_indices, io_backend):
        if io_backend != "kernel":
            raise NotImplementedError(
                f"NVFP4 host copy of FP8 KV supports only io_backend='kernel', got {io_backend!r}."
            )
        if host_indices.numel() != device_indices.numel():
            raise ValueError(
                f"{host_indices.numel()} host and {device_indices.numel()} device tokens."
            )
        # The controller hands GPU indices (move_indices); one int64 signature keeps the build.
        to_device = lambda x: x.to(self._transfer_device, torch.int64, non_blocking=True)
        return to_device(host_indices), to_device(device_indices), host_indices.numel()

    def _backup(self, host_idx, dev_idx, n) -> None:
        if self._fused_backup is not None:
            if n:
                draft = self._fused_backup_draft or (None, None, 0)
                self._fused_backup(
                    self._k_ptrs, self._v_ptrs, host_idx, dev_idx, self._row, BACKUP_BLOCK_QUOTA, *draft
                )
                log_once("fused_backup_first", "first backup through the fused kernel: %d tokens", n)
            return
        self._backup_triton(host_idx, dev_idx, n, self._k_ptrs, self._v_ptrs)

    def _backup_triton(self, host_idx, dev_idx, n, k_ptrs, v_ptrs) -> None:
        items = triton.cdiv(n, BACKUP_TOKENS_PER_TILE) * self.target_layer_num
        grid = (max(1, min(items, BACKUP_BLOCK_QUOTA)),)
        for ptrs, data, scale in (
            (k_ptrs, self.k_buffer, self.k_scale_host),
            (v_ptrs, self.v_buffer, self.v_scale_host),
        ):
            _backup_fp8_to_nvfp4[grid](
                ptrs, dev_idx, host_idx, _HostBytes(data), _HostBytes(scale), n,
                self.target_layer_num, ROW=self._row, BT=BACKUP_TOKENS_PER_TILE, GS=GLOBAL_SCALE,
                num_warps=BACKUP_WARPS,
            )

    def _load(self, layer_id, host_idx, dev_idx, n) -> None:
        if self._tma_load is not None:
            if n:
                self._tma_load(
                    host_idx,
                    dev_idx,
                    self._k_dst_ptrs[layer_id],
                    self._v_dst_ptrs[layer_id],
                    layer_id,
                    LOAD_BLOCK_QUOTA,
                )
                log_once("tma_load_first", "first load-back through the fused kernel: %d tokens, layer %d", n, layer_id)
            return
        self._load_triton(layer_id, host_idx, dev_idx, n)

    def _load_triton(self, layer_id, host_idx, dev_idx, n) -> None:
        grid = (max(1, min(triton.cdiv(n, LOAD_TOKENS_PER_TILE), LOAD_BLOCK_QUOTA)),)
        for dst, data, scale in (
            (self._device_k[layer_id].view(torch.int16), self.k_buffer, self.k_scale_host),
            (self._device_v[layer_id].view(torch.int16), self.v_buffer, self.v_scale_host),
        ):
            _load_nvfp4_to_fp8[grid](
                dst, dev_idx, host_idx, _HostBytes(data), _HostBytes(scale), n, layer_id,
                self.target_layer_num, ROW=self._row, BT=LOAD_TOKENS_PER_TILE, GS=GLOBAL_SCALE,
                num_warps=LOAD_WARPS,
            )

    def _reference_lut(self) -> torch.Tensor | None:
        """[256, 16] uint8: the FP8 byte the Triton load gives for each (E4M3 scale, E2M1 nibble),
        read off one launch over all 65,536 (scale, packed byte) pairs; None unless every pair's
        two nibbles match the table."""
        dev = self._transfer_device
        cols = self._row // 2
        pairs = 256 * 256
        rows = pairs // cols
        flat = torch.arange(pairs, dtype=torch.int64, device=dev)
        data = (flat % 256).to(torch.uint8).reshape(rows, cols)  # pair f: byte f % 256, scale f // 256
        scale = (flat[::8] // 256).to(torch.uint8).reshape(rows, self._scale_cols)
        out = torch.zeros(rows, self._row, dtype=torch.uint8, device=dev)
        idx = torch.arange(rows, dtype=torch.int64, device=dev)
        _load_nvfp4_to_fp8[(max(1, min(triton.cdiv(rows, LOAD_TOKENS_PER_TILE), LOAD_BLOCK_QUOTA)),)](
            out.view(torch.int16), idx, idx, data, scale, rows, 0, 1,
            ROW=self._row, BT=LOAD_TOKENS_PER_TILE, GS=GLOBAL_SCALE, num_warps=LOAD_WARPS,
        )
        out = out.reshape(pairs, 2)  # pair f -> (low nibble's FP8, high nibble's FP8)
        lut = out[:, 0].reshape(256, 256)[:, :16].contiguous()  # bytes 0..15: low nibble k, high 0
        byte, sc = flat % 256, flat // 256
        if not (torch.equal(out[:, 0], lut[sc, byte & 15]) and torch.equal(out[:, 1], lut[sc, byte >> 4])):
            return None
        return lut

    def _self_test_tma_load(self, loader) -> int | None:
        """Random bytes and scales in the first host tokens (free slots: a load-back reads only
        backed-up slots), loaded by both kernels for every layer as page runs, as single rows and as a
        partial page; returns the bytes compared, or None on any difference. The slots are zeroed after."""
        dev = self._transfer_device
        tokens = min(self.size // loader.rows_per_chunk, 2) * loader.rows_per_chunk
        if tokens == 0:
            return None
        gen = torch.Generator().manual_seed(536)
        bufs = (self.k_buffer, self.v_buffer, self.k_scale_host, self.v_scale_host)
        for buf in bufs:
            buf[:tokens].copy_(torch.randint(0, 256, buf[:tokens].shape, generator=gen, dtype=torch.uint8))
        run = torch.arange(tokens, dtype=torch.int64, device=dev)
        # Page boxes, one bulk copy per row, and both again ending in a partial page.
        cases = (run, run.flip(0), run[3 : tokens - 4], run[3 : tokens - 4].flip(0))
        compared = 0
        try:
            for host_idx in cases:
                n = host_idx.numel()
                dev_idx = torch.arange(n, dtype=torch.int64, device=dev)
                for layer in range(self.target_layer_num):
                    ref = [torch.zeros(n, self._row, dtype=torch.uint8, device=dev) for _ in range(2)]
                    new = [torch.full((n, self._row), 0xA5, dtype=torch.uint8, device=dev) for _ in range(2)]
                    for dst, data, scale in zip(ref, (self.k_buffer, self.v_buffer), (self.k_scale_host, self.v_scale_host)):
                        _load_nvfp4_to_fp8[(max(1, min(triton.cdiv(n, LOAD_TOKENS_PER_TILE), LOAD_BLOCK_QUOTA)),)](
                            dst.view(torch.int16), dev_idx, host_idx, _HostBytes(data), _HostBytes(scale), n, layer,
                            self.target_layer_num, ROW=self._row, BT=LOAD_TOKENS_PER_TILE, GS=GLOBAL_SCALE,
                            num_warps=LOAD_WARPS,
                        )
                    loader(host_idx, dev_idx, new[0].data_ptr(), new[1].data_ptr(), layer, LOAD_BLOCK_QUOTA)
                    torch.cuda.synchronize(dev)
                    if not all(torch.equal(a, b) for a, b in zip(ref, new)):
                        return None
                    compared += 2 * ref[0].numel()
        finally:
            for buf in bufs:
                buf[:tokens].zero_()
        return compared

    def _build_tma_load(self):
        """The fused TMA load when its table and self-test reproduce the Triton load, else None."""
        try:
            from sglang.kernels.ops.kvcache.hicache_nvfp4_load import Nvfp4HostLoad

            lut = self._reference_lut()
            if lut is None:
                raise RuntimeError("the dequant table does not reproduce the Triton load on every (scale, byte) pair")
            loader = Nvfp4HostLoad(
                row=self._row,
                k_data=self.k_buffer,
                v_data=self.v_buffer,
                k_scale=self.k_scale_host,
                v_scale=self.v_scale_host,
                tokens=self.size,
                layers=self.target_layer_num,
                page_size=self.page_size,
                lut=lut,
                dst_stride=self._row,
                device=self._transfer_device,
                stages=TMA_LOAD_STAGES,
                store_warps=TMA_LOAD_STORE_WARPS,
            )
            compared = self._self_test_tma_load(loader)
            if compared is None:
                raise RuntimeError("the startup self-test differs from the Triton load")
        except Exception as e:  # never serve a load that is not the Triton load's bytes
            logger.warning("HICACHE_FP4_HOST tma_load: %s=1 kept the Triton load-back: %s", TMA_LOAD_ENV, e)
            return None
        log_once(
            "tma_load",
            "fused TMA load-back on: %d-token chunks, %d stages, %d dequant warps, quota %d CTAs; "
            "dequant table equals the Triton load on 65536/65536 (scale, byte) pairs; startup self-test "
            "bitwise equal on %d bytes (%d layers, page runs, single rows, partial page)",
            loader.rows_per_chunk,
            loader.stages,
            loader.store_warps,
            LOAD_BLOCK_QUOTA,
            compared,
            self.target_layer_num,
        )
        return loader

    @staticmethod
    def _backup_domain_blocks() -> torch.Tensor:
        """[blocks, 16] FP8 bytes covering every (block amax, value) pair a backup can see: for each
        finite magnitude code a, blocks holding +a and -a with every value of magnitude <= a and both
        NaN codes (the amax ignores NaN); then all-NaN blocks."""
        blocks = []
        for a in range(0x7F):
            values = [c for c in range(256) if (c & 0x7F) <= a or (c & 0x7F) == 0x7F]
            for i in range(0, len(values), 14):
                part = values[i : i + 14]
                blocks.append([a, a | 0x80] + part + [a] * (14 - len(part)))
        blocks += [[0x7F] * 8 + [0xFF] * 8, [0x7F] * 16, [0xFF] * 16]
        return torch.tensor(blocks, dtype=torch.uint8)

    def _self_test_fused_backup(self, backup) -> int | None:
        """Both backups of scratch FP8 rows (every domain block, then random bytes) into the first host
        tokens (free slots: nothing reads a slot before a backup writes it), in shuffled host order, the
        fused one at the backup quota and at one CTA (each warp then takes several tokens); returns the
        host bytes compared, or None on any difference. The slots are zeroed after."""
        dev = self._transfer_device
        tokens = min(self.size, 128)
        layers = self.target_layer_num
        gen = torch.Generator().manual_seed(539)
        rows = [torch.randint(0, 256, (tokens, self._row), generator=gen, dtype=torch.uint8) for _ in range(2 * layers)]
        domain = self._backup_domain_blocks().reshape(-1)
        for buf in (rows[0], rows[2 * layers - 1]):  # K of the first layer, V of the last
            flat = buf.reshape(-1)
            flat[: min(domain.numel(), flat.numel())] = domain[: flat.numel()]
        draft_rows = [
            torch.randint(0, 256, (tokens, self._draft_row_bytes()), generator=gen, dtype=torch.uint8).to(dev)
            for _ in range(2 * backup.draft_layers)
        ]
        rows = [r.to(dev) for r in rows]
        k_ptrs, v_ptrs, dk_ptrs, dv_ptrs = (
            torch.tensor([r.data_ptr() for r in part], dtype=torch.int64, device=dev)
            for part in (rows[:layers], rows[layers:], draft_rows[: backup.draft_layers], draft_rows[backup.draft_layers :])
        )
        dev_idx = torch.arange(tokens, dtype=torch.int64, device=dev)
        host_idx = torch.randperm(tokens, generator=gen).to(dev)
        bufs = (self.k_buffer, self.v_buffer, self.k_scale_host, self.v_scale_host)
        draft_bufs = (self._draft_host.k_buffer, self._draft_host.v_buffer) if backup.draft_layers else ()
        draft_row_bytes = self._draft_row_bytes() if backup.draft_layers else 0
        order = host_idx.cpu()
        compared = 0
        try:
            for buf in bufs:
                buf[:tokens].zero_()
            self._backup_triton(host_idx, dev_idx, tokens, k_ptrs, v_ptrs)
            torch.cuda.synchronize(dev)
            ref = [buf[:tokens].clone() for buf in bufs]
            for quota in (BACKUP_BLOCK_QUOTA, 1):
                for buf in bufs + draft_bufs:
                    buf[:tokens].zero_()
                backup(k_ptrs, v_ptrs, host_idx, dev_idx, self._row, quota, dk_ptrs, dv_ptrs, draft_row_bytes)
                torch.cuda.synchronize(dev)
                if not all(torch.equal(buf[:tokens], r) for buf, r in zip(bufs, ref)):
                    return None
                # The draft rows are copied byte for byte (as the stock draft backup does).
                for buf, part in zip(draft_bufs, (draft_rows[: backup.draft_layers], draft_rows[backup.draft_layers :])):
                    want = torch.stack([r.cpu() for r in part], dim=1)  # [tokens (device order), draft layers, row]
                    if not torch.equal(buf[order].view(torch.uint8).reshape(want.shape), want):
                        return None
                compared += sum(r.numel() for r in ref) + sum(2 * r.numel() for r in draft_rows[: backup.draft_layers])
            return compared
        finally:
            for buf in bufs + draft_bufs:
                buf[:tokens].zero_()

    def _draft_row_bytes(self) -> int:
        return self._draft_host.element_dim * self._draft_host.dtype.itemsize

    def _fused_draft_tables(self):
        """(k table, v table, row stride) of the draft's device rows when the fused backup can copy them
        (contiguous host spans, rows of 16 B multiples), else None: the stock draft backup stays."""
        draft = self._draft_host
        if draft is None or self._draft_infos[0].transfer_buffers is not None:
            return None  # the device tables exist only for plain device rows
        if draft.device_k_data_ptrs.numel() != draft.layer_num:
            return None
        if draft.device_row_stride_bytes % 16 or self._draft_row_bytes() % 16:
            return None
        if not (draft.k_buffer.is_contiguous() and draft.v_buffer.is_contiguous()):
            return None
        return draft.device_k_data_ptrs, draft.device_v_data_ptrs, draft.device_row_stride_bytes

    def _misaligned_for_fused_backup(self, draft) -> list:
        """Buffers whose base is not 16 B aligned among those the fused backup reads (device layer rows,
        the draft's when it copies them) and writes (host spans) in 16 B units. The self-test runs on scratch
        device rows, so the served device bases are checked here."""
        bases = [(f"device K layer {i}", t) for i, t in enumerate(self._device_k)]
        bases += [(f"device V layer {i}", t) for i, t in enumerate(self._device_v)]
        bases += [("host K", self.k_buffer), ("host V", self.v_buffer)]
        bases += [("host K scales", self.k_scale_host), ("host V scales", self.v_scale_host)]
        if draft is not None:
            d = self._draft_host
            bases += [(f"device draft K layer {i}", t) for i, t in enumerate(d.device_k_buffers)]
            bases += [(f"device draft V layer {i}", t) for i, t in enumerate(d.device_v_buffers)]
            bases += [("host draft K", d.k_buffer), ("host draft V", d.v_buffer)]
        return [name for name, t in bases if t.data_ptr() % 16]

    def _build_fused_backup(self):
        """The fused backup when its buffers are 16 B aligned and its self-test reproduces the Triton
        backup, else None."""
        try:
            from sglang.kernels.ops.kvcache.hicache_nvfp4_backup import Nvfp4HostBackup

            draft = self._fused_draft_tables()
            misaligned = self._misaligned_for_fused_backup(draft)
            if misaligned:
                raise RuntimeError(f"bases not 16 B aligned: {', '.join(misaligned)}")
            backup = Nvfp4HostBackup(
                row=self._row,
                global_scale=GLOBAL_SCALE,
                k_data=self.k_buffer,
                v_data=self.v_buffer,
                k_scale=self.k_scale_host,
                v_scale=self.v_scale_host,
                tokens=self.size,
                layers=self.target_layer_num,
                warps=FUSED_BACKUP_WARPS,
                group=FUSED_BACKUP_GROUP,
                device=self._transfer_device,
                draft_k=self._draft_host.k_buffer if draft else None,
                draft_v=self._draft_host.v_buffer if draft else None,
                draft_layers=self._draft_host.layer_num if draft else 0,
                draft_row_bytes=self._draft_row_bytes() if draft else 0,
            )
            compared = self._self_test_fused_backup(backup)
            if compared is None:
                raise RuntimeError("the startup self-test differs from the Triton backup")
        except Exception as e:  # never write host bytes that are not the Triton backup's
            logger.warning("HICACHE_FP4_HOST fused_backup: %s=1 kept the Triton backup: %s", FUSED_BACKUP_ENV, e)
            return None
        self._fused_backup_draft = draft
        log_once(
            "fused_backup",
            "fused backup on: one warp per token, %d warps x %d CTAs, %d-step loads, %d FP8 draft layers copied "
            "in the same pass; startup self-test bitwise equal to the Triton backup and the draft bytes on %d "
            "host bytes (%d (block amax, value) blocks, random rows, %d layers, shuffled host order, quota and 1 CTA)",
            backup.warps,
            BACKUP_BLOCK_QUOTA,
            backup.group,
            backup.draft_layers,
            compared,
            self._backup_domain_blocks().shape[0],
            self.target_layer_num,
        )
        return backup

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        host_idx, dev_idx, n = self._transfer_args(host_indices, device_indices, io_backend)
        if n:
            self._backup(host_idx, dev_idx, n)
            self._count("backup", n)
        if self._draft_host is not None and (self._fused_backup is None or self._fused_backup_draft is None):
            self._draft_host.backup_from_device_all_layer(
                device_pool, host_indices, device_indices, io_backend
            )

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft: bool = False,
    ):
        if is_draft:
            if self._draft_host is None:
                raise RuntimeError("NVFP4 host copy has no packed draft layers.")
            self._draft_host.load_to_device_per_layer(
                device_pool,
                host_indices,
                device_indices,
                layer_id - self.target_layer_num,
                io_backend,
                is_draft=True,
            )
            return
        if not 0 <= layer_id < self.target_layer_num:
            return
        host_idx, dev_idx, n = self._transfer_args(host_indices, device_indices, io_backend)
        if n:
            self._load(layer_id, host_idx, dev_idx, n)
            if layer_id == 0:  # one op loads every target layer, layer 0 first
                self._count("load-back", n)

    def destroy(self):
        for buf in (self.k_scale_host, self.v_scale_host):
            if buf is not None and self.pin_memory and (_is_cuda or _is_hip):
                _cuda_host_unregister(buf)
        self.k_scale_host = None
        self.v_scale_host = None
        if self._draft_host is not None:
            self._draft_host.destroy()
        super().destroy()

    def _storage_pages_unsupported(self) -> NotImplementedError:
        return NotImplementedError(
            "NVFP4 host copy of FP8 KV does not expose flat storage (L3) pages: a page's "
            "scales and packed draft rows live outside kv_buffer."
        )

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        raise self._storage_pages_unsupported()

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        raise self._storage_pages_unsupported()

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        raise self._storage_pages_unsupported()

    def get_page_buffer_meta(self, indices):
        raise self._storage_pages_unsupported()

    def get_split_heads_page_buffer_meta(
        self, indices: torch.Tensor, split_factor: int
    ):
        raise self._storage_pages_unsupported()
