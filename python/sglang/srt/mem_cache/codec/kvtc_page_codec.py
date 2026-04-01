from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Any, Optional

import struct
import torch
import zlib

from sglang.srt.mem_cache.codec.bitpack import (
    pack_bits,
    pack_tensor_ints_twos_complement,
    unpack_bits,
    unpack_tensor_ints_twos_complement,
    warmup_bitpack_backend,
)
from sglang.srt.utils.common import get_bool_env_var, get_int_env_var

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KvtcParams:
    proj: Optional[torch.Tensor]
    mean: Optional[torch.Tensor]
    scale: Optional[torch.Tensor]
    bits: Optional[torch.Tensor]
    shape: dict[str, Any]
    output_dtype: str
    format_version: int
    entropy: str = "zlib"
    fp8_dtype: str = "e4m3fn"
    k_proj: Optional[torch.Tensor] = None
    k_mean: Optional[torch.Tensor] = None
    k_scale: Optional[torch.Tensor] = None
    k_bits: Optional[torch.Tensor] = None
    v_proj: Optional[torch.Tensor] = None
    v_mean: Optional[torch.Tensor] = None
    v_scale: Optional[torch.Tensor] = None
    v_bits: Optional[torch.Tensor] = None


def _load_kvtc_params(path: str) -> KvtcParams:
    params = torch.load(path, map_location="cpu")
    proj = params.get("proj")
    shape = params.get("shape", {})
    input_dim = shape.get("numel") or shape.get("input_dim")
    if input_dim is None:
        if proj is None:
            raise ValueError("kvtc params missing shape.numel/input_dim and proj")
        input_dim = proj.shape[0] if proj.shape[0] >= proj.shape[1] else proj.shape[1]

    mean = params.get("mean")
    if mean is None and proj is not None:
        mean = torch.zeros(input_dim, dtype=proj.dtype)
    scale = params.get("scale")
    bits = params.get("bits")
    output_dtype = params.get("output_dtype", "float16")
    format_version = int(params.get("format_version", 1))
    entropy = str(params.get("entropy", "zlib"))
    fp8_dtype = str(params.get("fp8_dtype", "e4m3fn"))
    k_proj = params.get("k_proj")
    k_mean = params.get("k_mean")
    k_scale = params.get("k_scale")
    k_bits = params.get("k_bits")
    v_proj = params.get("v_proj")
    v_mean = params.get("v_mean")
    v_scale = params.get("v_scale")
    v_bits = params.get("v_bits")
    return KvtcParams(
        proj=proj,
        mean=mean,
        scale=scale,
        bits=bits,
        shape=shape,
        output_dtype=output_dtype,
        format_version=format_version,
        entropy=entropy,
        fp8_dtype=fp8_dtype,
        k_proj=k_proj,
        k_mean=k_mean,
        k_scale=k_scale,
        k_bits=k_bits,
        v_proj=v_proj,
        v_mean=v_mean,
        v_scale=v_scale,
        v_bits=v_bits,
    )


def _dtype_from_name(name: str) -> torch.dtype:
    name = name.lower()
    if name in {"float16", "fp16"}:
        return torch.float16
    if name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if name in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported output_dtype: {name}")


class KvtcPageCodec:
    def __init__(self, params_path: str):
        self.params = _load_kvtc_params(params_path)
        self.proj = self.params.proj
        self.mean = self.params.mean
        self.scale = self.params.scale
        self.bits = self.params.bits
        self.output_dtype = _dtype_from_name(self.params.output_dtype)
        self._deflate_raw = self.params.entropy.lower() in {"deflate", "raw_deflate"} or int(self.params.format_version) >= 4
        self._deflate_level = 1
        self._fp8_dtype = self._resolve_fp8_dtype(self.params.fp8_dtype)
        self._perf_log = get_bool_env_var("SGLANG_KVTC_PERF_LOG", "false")
        self._perf_log_every = max(1, get_int_env_var("SGLANG_KVTC_PERF_LOG_EVERY", 128))
        self._gpu_project_enabled = get_bool_env_var("SGLANG_KVTC_GPU_PROJECT", "true")
        self._gpu_pack_enabled = get_bool_env_var("SGLANG_KVTC_GPU_PACK", "true")
        self._bitpack_backend = warmup_bitpack_backend()
        self._gpu_project_failed = False
        self._gpu_project_device = None
        self._k_proj_f32_gpu = None
        self._v_proj_f32_gpu = None
        self._k_mean_f32_gpu = None
        self._v_mean_f32_gpu = None
        self._k_scale_f32_gpu = None
        self._v_scale_f32_gpu = None
        self._k_fp8_mask_gpu = None
        self._v_fp8_mask_gpu = None
        self._k_bits_int_i16_gpu = None
        self._v_bits_int_i16_gpu = None
        self._perf_encode_count = 0
        self._perf_decode_count = 0
        self._perf_encode_ns = {
            "project": 0,
            "pack": 0,
            "compress": 0,
            "total": 0,
            "input_bytes": 0,
            "output_bytes": 0,
        }
        self._perf_decode_ns = {
            "decompress": 0,
            "unpack": 0,
            "inverse_project": 0,
            "total": 0,
            "input_bytes": 0,
            "output_bytes": 0,
        }
        self._split_kv = self.params.k_proj is not None and self.params.v_proj is not None
        if self._split_kv:
            self.k_proj = self.params.k_proj
            self.k_mean = self.params.k_mean
            self.k_scale = self.params.k_scale
            self.k_bits = self.params.k_bits
            self.v_proj = self.params.v_proj
            self.v_mean = self.params.v_mean
            self.v_scale = self.params.v_scale
            self.v_bits = self.params.v_bits

            for name, mat in [("k_proj", self.k_proj), ("v_proj", self.v_proj)]:
                if mat is None or mat.dim() != 2:
                    raise ValueError(f"{name} must be 2D")
            for name, vec in [("k_scale", self.k_scale), ("v_scale", self.v_scale)]:
                if vec is None or vec.dim() != 1:
                    raise ValueError(f"{name} must be 1D")

            self.k_proj_f32_cpu = self.k_proj.to(torch.float32).cpu().contiguous()
            self.v_proj_f32_cpu = self.v_proj.to(torch.float32).cpu().contiguous()
            self.k_proj_t_f32_cpu = self.k_proj_f32_cpu.t().contiguous()
            self.v_proj_t_f32_cpu = self.v_proj_f32_cpu.t().contiguous()

            self.k_bits = (
                self.k_bits.to(torch.int16).cpu() if self.k_bits is not None else None
            )
            self.v_bits = (
                self.v_bits.to(torch.int16).cpu() if self.v_bits is not None else None
            )
            if self.k_bits is None or self.v_bits is None:
                raise ValueError("kvtc split-kv requires k_bits and v_bits")
            if self.k_bits.numel() != self.k_scale.numel():
                raise ValueError("k_bits must align with k_scale length")
            if self.v_bits.numel() != self.v_scale.numel():
                raise ValueError("v_bits must align with v_scale length")

            shape = self.params.shape or {}
            self.k_dim = (
                int(shape["layer_num"])
                * int(shape["page_size"])
                * int(shape["head_num"])
                * int(shape["head_dim"])
            )
            self.v_dim = (
                int(shape["layer_num"])
                * int(shape["page_size"])
                * int(shape["head_num"])
                * int(shape["v_head_dim"])
            )
            self.k_mean_f32_cpu = self.k_mean.to(torch.float32).cpu()
            self.v_mean_f32_cpu = self.v_mean.to(torch.float32).cpu()
            self.k_scale_f32_cpu = self.k_scale.to(torch.float32).cpu()
            self.v_scale_f32_cpu = self.v_scale.to(torch.float32).cpu()
            self.k_bits_i16_cpu = self.k_bits
            self.v_bits_i16_cpu = self.v_bits
            self.k_fp8_mask_cpu = self.k_bits_i16_cpu == 8
            self.v_fp8_mask_cpu = self.v_bits_i16_cpu == 8
            self.k_bits_int_i16_cpu = torch.where(
                self.k_fp8_mask_cpu,
                torch.zeros_like(self.k_bits_i16_cpu),
                self.k_bits_i16_cpu,
            )
            self.v_bits_int_i16_cpu = torch.where(
                self.v_fp8_mask_cpu,
                torch.zeros_like(self.v_bits_i16_cpu),
                self.v_bits_i16_cpu,
            )
            self.k_scale_fp8_cpu = self.k_scale_f32_cpu[self.k_fp8_mask_cpu]
            self.v_scale_fp8_cpu = self.v_scale_f32_cpu[self.v_fp8_mask_cpu]
        else:
            if self.proj is None or self.proj.dim() != 2:
                raise ValueError("kvtc proj must be 2D")
            if self.scale is None or self.scale.dim() != 1:
                raise ValueError("kvtc scale must be 1D")
            self.proj_f32_cpu = self.proj.to(torch.float32).cpu().contiguous()
            if self.proj.shape[1] == self.scale.shape[0]:
                self._proj_t_f32_cpu = self.proj_f32_cpu.t().contiguous()
                self._proj_is_d_by_k = True
            elif self.proj.shape[0] == self.scale.shape[0]:
                self._proj_t_f32_cpu = self.proj_f32_cpu.contiguous()
                self._proj_is_d_by_k = False
            else:
                raise ValueError("kvtc proj shape must align with scale")

            if self.bits is not None:
                self.bits = self.bits.to(torch.int16).cpu()
                if self.bits.numel() != self.scale.numel():
                    raise ValueError("kvtc bits must align with scale length")
            if self.mean is not None:
                self.mean_f32_cpu = self.mean.to(torch.float32).cpu()
            if self.scale is not None:
                self.scale_f32_cpu = self.scale.to(torch.float32).cpu()
            if self.bits is not None:
                self.bits_i16_cpu = self.bits
                self.fp8_mask_cpu = self.bits_i16_cpu == 8
                self.bits_int_i16_cpu = torch.where(
                    self.fp8_mask_cpu,
                    torch.zeros_like(self.bits_i16_cpu),
                    self.bits_i16_cpu,
                )
                self.scale_fp8_cpu = self.scale_f32_cpu[self.fp8_mask_cpu]

    def _resolve_fp8_dtype(self, name: str):
        name = (name or "").lower()
        if hasattr(torch, "float8_e4m3fn") and name in {"e4m3fn", "float8_e4m3fn"}:
            return torch.float8_e4m3fn
        if hasattr(torch, "float8_e5m2") and name in {"e5m2", "float8_e5m2"}:
            return torch.float8_e5m2
        if hasattr(torch, "float8_e4m3fn"):
            return torch.float8_e4m3fn
        if hasattr(torch, "float8_e5m2"):
            return torch.float8_e5m2
        return None

    def _ensure_gpu_project_cache(self, device: torch.device) -> bool:
        if (
            not self._gpu_project_enabled
            or self._gpu_project_failed
            or not self._split_kv
            or device.type != "cuda"
        ):
            return False
        if self._gpu_project_device == device:
            return True
        try:
            self._k_proj_f32_gpu = self.k_proj_f32_cpu.to(device=device, non_blocking=True)
            self._v_proj_f32_gpu = self.v_proj_f32_cpu.to(device=device, non_blocking=True)
            self._k_mean_f32_gpu = self.k_mean_f32_cpu.to(device=device, non_blocking=True)
            self._v_mean_f32_gpu = self.v_mean_f32_cpu.to(device=device, non_blocking=True)
            self._k_scale_f32_gpu = self.k_scale_f32_cpu.to(device=device, non_blocking=True)
            self._v_scale_f32_gpu = self.v_scale_f32_cpu.to(device=device, non_blocking=True)
            self._k_fp8_mask_gpu = self.k_fp8_mask_cpu.to(device=device, non_blocking=True)
            self._v_fp8_mask_gpu = self.v_fp8_mask_cpu.to(device=device, non_blocking=True)
            self._k_bits_int_i16_gpu = self.k_bits_int_i16_cpu.to(
                device=device, non_blocking=True
            )
            self._v_bits_int_i16_gpu = self.v_bits_int_i16_cpu.to(
                device=device, non_blocking=True
            )
            self._gpu_project_device = device
            logger.info("KVTC GPU project cache enabled on device=%s", device)
            return True
        except Exception as e:
            self._gpu_project_failed = True
            self._k_proj_f32_gpu = None
            self._v_proj_f32_gpu = None
            self._k_mean_f32_gpu = None
            self._v_mean_f32_gpu = None
            self._k_scale_f32_gpu = None
            self._v_scale_f32_gpu = None
            self._k_fp8_mask_gpu = None
            self._v_fp8_mask_gpu = None
            self._k_bits_int_i16_gpu = None
            self._v_bits_int_i16_gpu = None
            self._gpu_project_device = None
            logger.warning("KVTC GPU project cache disabled due to initialization failure: %s", e)
            return False

    def _perf_maybe_log(self, mode: str) -> None:
        if not self._perf_log:
            return
        if mode == "encode":
            count = self._perf_encode_count
            stats = self._perf_encode_ns
        else:
            count = self._perf_decode_count
            stats = self._perf_decode_ns
        if count <= 0 or count % self._perf_log_every != 0:
            return
        logger.info(
            "KVTC %s perf: count=%d total_ms=%.3f stage_ms(project=%.3f,pack=%.3f,compress=%.3f,decompress=%.3f,unpack=%.3f,inverse_project=%.3f) avg_in_bytes=%.1f avg_out_bytes=%.1f",
            mode,
            count,
            stats.get("total", 0) / 1e6 / count,
            stats.get("project", 0) / 1e6 / count,
            stats.get("pack", 0) / 1e6 / count,
            stats.get("compress", 0) / 1e6 / count,
            stats.get("decompress", 0) / 1e6 / count,
            stats.get("unpack", 0) / 1e6 / count,
            stats.get("inverse_project", 0) / 1e6 / count,
            stats.get("input_bytes", 0) / count,
            stats.get("output_bytes", 0) / count,
        )

    def _check_shape(self, kv_page_meta: Optional[dict[str, Any]], numel: int) -> None:
        if kv_page_meta is None:
            return
        shape = self.params.shape or {}
        expected = shape.get("numel")
        if expected is not None and expected != numel:
            raise ValueError(
                f"kvtc numel mismatch: params {expected}, input {numel}"
            )
        page_size = shape.get("page_size")
        if page_size is not None and kv_page_meta.get("page_size") != page_size:
            raise ValueError(
                f"kvtc page_size mismatch: params {page_size}, meta {kv_page_meta.get('page_size')}"
            )

    def _pack_quantized(
        self,
        y: torch.Tensor,
        scale: torch.Tensor,
        bits: torch.Tensor,
        *,
        fp8_mask: Optional[torch.Tensor] = None,
        bits_int: Optional[torch.Tensor] = None,
    ) -> bytes:
        use_gpu_pack = self._gpu_pack_enabled and y.device.type == "cuda"
        if use_gpu_pack:
            scale_dev = scale if scale.device == y.device else scale.to(y.device, non_blocking=True)
            bits_dev = bits if bits.device == y.device else bits.to(y.device, non_blocking=True)
            fp8_mask_dev = (
                (bits_dev == 8)
                if fp8_mask is None
                else (fp8_mask if fp8_mask.device == y.device else fp8_mask.to(y.device, non_blocking=True))
            )
            bits_int_dev = (
                torch.where(fp8_mask_dev, torch.zeros_like(bits_dev), bits_dev)
                if bits_int is None
                else (bits_int if bits_int.device == y.device else bits_int.to(y.device, non_blocking=True))
            )
            active_dev = bits_dev > 0
            int_mask_dev = active_dev & (~fp8_mask_dev)
            bits_int_clamped = torch.clamp(bits_int_dev, min=1).to(torch.int32)
            pow2 = torch.pow(
                torch.tensor(2, dtype=torch.int32, device=y.device),
                bits_int_clamped - 1,
            )
            qmax = (pow2 - 1).to(torch.float32)
            qmin = (-pow2).to(torch.float32)
            q = torch.round(y.to(torch.float32) / scale_dev)
            q = torch.where(int_mask_dev, torch.clamp(q, qmin, qmax), torch.zeros_like(q))
            q_i32_cpu = q.to(torch.int32).cpu()
            bits_int_cpu = (
                bits_int if bits_int is not None and bits_int.device.type == "cpu" else bits_int_dev.cpu()
            )
            int_packed = pack_tensor_ints_twos_complement(q_i32_cpu, bits_int_cpu)

            fp8_mask_cpu = (
                fp8_mask if fp8_mask is not None and fp8_mask.device.type == "cpu" else fp8_mask_dev.cpu()
            )
            fp8_bytes = b""
            if fp8_mask_dev.any().item():
                if self._fp8_dtype is None:
                    raise ValueError("fp8 requested but torch float8 is not available")
                fp_full = (y.to(torch.float32) / scale_dev).to(self._fp8_dtype).contiguous()
                fp_u8 = fp_full.view(torch.uint8)
                fp = fp_u8[fp8_mask_dev].cpu()
                try:
                    fp8_bytes = fp.numpy().tobytes()
                except Exception:
                    fp8_bytes = bytes(fp.tolist())

            mask_bytes = pack_bits(fp8_mask_cpu)
        else:
            y = y.to(torch.float32).cpu()
            scale = scale.to(torch.float32).cpu()
            bits = bits.to(torch.int16).cpu()
            active = bits > 0
            fp8_mask = (
                (bits == 8)
                if fp8_mask is None
                else fp8_mask.to(torch.bool).cpu()
            )
            int_mask = active & (~fp8_mask)
            bits_int = (
                torch.where(fp8_mask, torch.zeros_like(bits), bits)
                if bits_int is None
                else bits_int.to(torch.int16).cpu()
            )
            bits_int_clamped = torch.clamp(bits_int, min=1).to(torch.int32)
            pow2 = torch.pow(torch.tensor(2, dtype=torch.int32), bits_int_clamped - 1)
            qmax = (pow2 - 1).to(torch.float32)
            qmin = (-pow2).to(torch.float32)
            q = torch.round(y / scale)
            q = torch.where(int_mask, torch.clamp(q, qmin, qmax), torch.zeros_like(q))
            q_i32 = q.to(torch.int32)
            int_packed = pack_tensor_ints_twos_complement(q_i32, bits_int)

            fp8_bytes = b""
            if fp8_mask.any().item():
                if self._fp8_dtype is None:
                    raise ValueError("fp8 requested but torch float8 is not available")
                fp_full = (y / scale).to(self._fp8_dtype).contiguous()
                fp_u8 = fp_full.view(torch.uint8)
                fp = fp_u8[fp8_mask]
                try:
                    fp8_bytes = fp.cpu().numpy().tobytes()
                except Exception:
                    fp8_bytes = bytes(fp.cpu().tolist())

            mask_bytes = pack_bits(fp8_mask)
        header = struct.pack("<III", len(mask_bytes), len(fp8_bytes), len(int_packed))
        return header + mask_bytes + fp8_bytes + int_packed

    def _quantize_pack_deflate(
        self,
        y: torch.Tensor,
        scale: torch.Tensor,
        bits: torch.Tensor,
        *,
        fp8_mask: Optional[torch.Tensor] = None,
        bits_int: Optional[torch.Tensor] = None,
    ) -> bytes:
        t0 = time.perf_counter_ns()
        packed = self._pack_quantized(
            y, scale, bits, fp8_mask=fp8_mask, bits_int=bits_int
        )
        t1 = time.perf_counter_ns()
        blob = self._compress(packed)
        t2 = time.perf_counter_ns()
        if self._perf_log:
            self._perf_encode_ns["pack"] += t1 - t0
            self._perf_encode_ns["compress"] += t2 - t1
            self._perf_encode_ns["output_bytes"] += len(blob)
        return blob

    def _inflate_unpack_dequant(
        self,
        blob: bytes,
        scale: torch.Tensor,
        bits: torch.Tensor,
        *,
        raw_deflate: bool | None = None,
    ) -> torch.Tensor:
        bits = bits.to(torch.int16).cpu()
        t0 = time.perf_counter_ns()
        packed = self._decompress(blob, raw_deflate=raw_deflate)
        t1 = time.perf_counter_ns()
        if len(packed) >= 12:
            mask_len, fp8_len, int_len = struct.unpack("<III", packed[:12])
            off = 12
            if off + mask_len + fp8_len + int_len == len(packed):
                mask_bytes = packed[off : off + mask_len]
                off += mask_len
                fp8_bytes = packed[off : off + fp8_len]
                off += fp8_len
                int_bytes = packed[off : off + int_len]

                fp8_mask = unpack_bits(mask_bytes, bits.numel())
                fp8_mask = fp8_mask.to(torch.bool)
                bits_int = torch.where(fp8_mask, torch.zeros_like(bits), bits)
                q_int = unpack_tensor_ints_twos_complement(int_bytes, bits_int).to(
                    torch.float32
                )
                y = q_int * scale.to(torch.float32).cpu()

                if fp8_mask.any().item():
                    if self._fp8_dtype is None:
                        raise ValueError("fp8 blob but torch float8 is not available")
                    u8 = torch.frombuffer(memoryview(fp8_bytes), dtype=torch.uint8)
                    fp = u8.view(self._fp8_dtype).to(torch.float32)
                    y_fp8 = fp * scale.to(torch.float32).cpu()[fp8_mask]
                    y[fp8_mask] = y_fp8
                if self._perf_log:
                    self._perf_decode_ns["decompress"] += t1 - t0
                    self._perf_decode_ns["unpack"] += time.perf_counter_ns() - t1
                    self._perf_decode_ns["input_bytes"] += len(blob)
                return y

        q = unpack_tensor_ints_twos_complement(packed, bits).to(torch.float32)
        if self._perf_log:
            self._perf_decode_ns["decompress"] += t1 - t0
            self._perf_decode_ns["unpack"] += time.perf_counter_ns() - t1
            self._perf_decode_ns["input_bytes"] += len(blob)
        return q * scale.to(torch.float32).cpu()

    def _compress(self, data: bytes, *, raw_deflate: bool | None = None) -> bytes:
        raw = self._deflate_raw if raw_deflate is None else raw_deflate
        if raw:
            c = zlib.compressobj(self._deflate_level, zlib.DEFLATED, -15)
            return c.compress(data) + c.flush()
        return zlib.compress(data, level=self._deflate_level)

    def _decompress(self, blob: bytes, *, raw_deflate: bool | None = None) -> bytes:
        raw = self._deflate_raw if raw_deflate is None else raw_deflate
        if raw:
            d = zlib.decompressobj(-15)
            return d.decompress(blob) + d.flush()
        return zlib.decompress(blob)

    def encode_vector(self, x: torch.Tensor, kv_page_meta: Optional[dict[str, Any]] = None) -> bytes:
        t_total0 = time.perf_counter_ns()
        x = x.flatten().to(torch.float32)
        self._check_shape(kv_page_meta, x.numel())
        if self._perf_log:
            self._perf_encode_ns["input_bytes"] += x.numel() * x.element_size()
        if self._split_kv and int(self.params.format_version) >= 3:
            xk = x[: self.k_dim]
            xv = x[self.k_dim : self.k_dim + self.v_dim]
            if kv_page_meta and kv_page_meta.get("bypass"):
                k_blob = xk.to(torch.float16).contiguous().cpu().numpy().tobytes()
                v_blob = xv.to(torch.float16).contiguous().cpu().numpy().tobytes()
                flags = 0x1 | 0x4
                header = struct.pack(
                    "<4sBBII",
                    b"KVTC",
                    int(self.params.format_version),
                    flags,
                    len(k_blob),
                    len(v_blob),
                )
                blob = header + k_blob + v_blob
                if self._perf_log:
                    self._perf_encode_count += 1
                    self._perf_encode_ns["total"] += time.perf_counter_ns() - t_total0
                    self._perf_encode_ns["output_bytes"] += len(blob)
                    self._perf_maybe_log("encode")
                return blob

            t_proj0 = time.perf_counter_ns()
            use_gpu_project = x.device.type == "cuda" and self._ensure_gpu_project_cache(
                x.device
            )
            if use_gpu_project:
                yk = (xk - self._k_mean_f32_gpu) @ self._k_proj_f32_gpu
                yv = (xv - self._v_mean_f32_gpu) @ self._v_proj_f32_gpu
            else:
                yk = (xk - self.k_mean_f32_cpu) @ self.k_proj_f32_cpu
                yv = (xv - self.v_mean_f32_cpu) @ self.v_proj_f32_cpu
            t_proj1 = time.perf_counter_ns()
            k_blob = self._quantize_pack_deflate(
                yk,
                self._k_scale_f32_gpu if use_gpu_project else self.k_scale_f32_cpu,
                self.k_bits_i16_cpu,
                fp8_mask=self._k_fp8_mask_gpu if use_gpu_project else self.k_fp8_mask_cpu,
                bits_int=(
                    self._k_bits_int_i16_gpu
                    if use_gpu_project
                    else self.k_bits_int_i16_cpu
                ),
            )
            v_blob = self._quantize_pack_deflate(
                yv,
                self._v_scale_f32_gpu if use_gpu_project else self.v_scale_f32_cpu,
                self.v_bits_i16_cpu,
                fp8_mask=self._v_fp8_mask_gpu if use_gpu_project else self.v_fp8_mask_cpu,
                bits_int=(
                    self._v_bits_int_i16_gpu
                    if use_gpu_project
                    else self.v_bits_int_i16_cpu
                ),
            )
            flags = 0x1 | 0x2
            header = struct.pack(
                "<4sBBII",
                b"KVTC",
                int(self.params.format_version),
                flags,
                len(k_blob),
                len(v_blob),
            )
            blob = header + k_blob + v_blob
            if self._perf_log:
                self._perf_encode_count += 1
                self._perf_encode_ns["project"] += t_proj1 - t_proj0
                self._perf_encode_ns["total"] += time.perf_counter_ns() - t_total0
                self._perf_maybe_log("encode")
            return blob

        if self.proj is None or self.mean is None or self.scale is None:
            raise ValueError("kvtc params are incomplete")
        t_proj0 = time.perf_counter_ns()
        if self._proj_is_d_by_k:
            y = (x - self.mean_f32_cpu) @ self.proj_f32_cpu
        else:
            y = (x - self.mean_f32_cpu) @ self._proj_t_f32_cpu.t()
        t_proj1 = time.perf_counter_ns()
        if self.bits is None or int(self.params.format_version) <= 1:
            q = torch.round(y / self.scale_f32_cpu).to(torch.int16).cpu()
            blob = q.numpy().tobytes()
            if self._perf_log:
                self._perf_encode_count += 1
                self._perf_encode_ns["project"] += t_proj1 - t_proj0
                self._perf_encode_ns["total"] += time.perf_counter_ns() - t_total0
                self._perf_encode_ns["output_bytes"] += len(blob)
                self._perf_maybe_log("encode")
            return blob
        blob = self._quantize_pack_deflate(
            y,
            self.scale_f32_cpu,
            self.bits_i16_cpu,
            fp8_mask=self.fp8_mask_cpu,
            bits_int=self.bits_int_i16_cpu,
        )
        if self._perf_log:
            self._perf_encode_count += 1
            self._perf_encode_ns["project"] += t_proj1 - t_proj0
            self._perf_encode_ns["total"] += time.perf_counter_ns() - t_total0
            self._perf_maybe_log("encode")
        return blob

    def decode_vector(self, blob: bytes, kv_page_meta: Optional[dict[str, Any]] = None) -> torch.Tensor:
        t_total0 = time.perf_counter_ns()
        if len(blob) >= 14 and blob[:4] == b"KVTC":
            magic, ver, flags, k_len, v_len = struct.unpack("<4sBBII", blob[:14])
            if ver not in (3, 4):
                raise ValueError(f"Unsupported KVTC blob version: {ver}")
            raw_deflate = ver >= 4
            off = 14
            k_blob = blob[off : off + k_len]
            off += k_len
            v_blob = blob[off : off + v_len]
            if (flags & 0x1) == 0:
                raise ValueError("KVTC header indicates non-split, unsupported here")
            if (flags & 0x4) != 0:
                xk = torch.frombuffer(memoryview(k_blob), dtype=torch.float16).to(
                    torch.float32
                )
                xv = torch.frombuffer(memoryview(v_blob), dtype=torch.float16).to(
                    torch.float32
                )
            else:
                yk = self._inflate_unpack_dequant(
                    k_blob,
                    self.k_scale_f32_cpu,
                    self.k_bits_i16_cpu,
                    raw_deflate=raw_deflate,
                )
                yv = self._inflate_unpack_dequant(
                    v_blob,
                    self.v_scale_f32_cpu,
                    self.v_bits_i16_cpu,
                    raw_deflate=raw_deflate,
                )
                t_inv0 = time.perf_counter_ns()
                xk = yk @ self.k_proj_t_f32_cpu + self.k_mean_f32_cpu
                xv = yv @ self.v_proj_t_f32_cpu + self.v_mean_f32_cpu
                t_inv1 = time.perf_counter_ns()
                if self._perf_log:
                    self._perf_decode_ns["inverse_project"] += t_inv1 - t_inv0
            x = torch.cat([xk, xv], dim=0)
            self._check_shape(kv_page_meta, x.numel())
            out = x.to(self.output_dtype)
            if self._perf_log:
                self._perf_decode_count += 1
                self._perf_decode_ns["total"] += time.perf_counter_ns() - t_total0
                self._perf_decode_ns["output_bytes"] += out.numel() * out.element_size()
                self._perf_maybe_log("decode")
            return out

        if self.bits is None or int(self.params.format_version) <= 1:
            q = torch.frombuffer(memoryview(blob), dtype=torch.int16)
            y = q.to(torch.float32) * self.scale_f32_cpu
            t_inv0 = time.perf_counter_ns()
            if self._proj_is_d_by_k:
                x = y @ self._proj_t_f32_cpu + self.mean_f32_cpu
            else:
                x = y @ self._proj_t_f32_cpu + self.mean_f32_cpu
            t_inv1 = time.perf_counter_ns()
            self._check_shape(kv_page_meta, x.numel())
            out = x.to(self.output_dtype)
            if self._perf_log:
                self._perf_decode_count += 1
                self._perf_decode_ns["inverse_project"] += t_inv1 - t_inv0
                self._perf_decode_ns["input_bytes"] += len(blob)
                self._perf_decode_ns["total"] += time.perf_counter_ns() - t_total0
                self._perf_decode_ns["output_bytes"] += out.numel() * out.element_size()
                self._perf_maybe_log("decode")
            return out

        y = self._inflate_unpack_dequant(blob, self.scale_f32_cpu, self.bits_i16_cpu)
        t_inv0 = time.perf_counter_ns()
        if self._proj_is_d_by_k:
            x = y @ self._proj_t_f32_cpu + self.mean_f32_cpu
        else:
            x = y @ self._proj_t_f32_cpu + self.mean_f32_cpu
        t_inv1 = time.perf_counter_ns()
        self._check_shape(kv_page_meta, x.numel())
        out = x.to(self.output_dtype)
        if self._perf_log:
            self._perf_decode_count += 1
            self._perf_decode_ns["inverse_project"] += t_inv1 - t_inv0
            self._perf_decode_ns["total"] += time.perf_counter_ns() - t_total0
            self._perf_decode_ns["output_bytes"] += out.numel() * out.element_size()
            self._perf_maybe_log("decode")
        return out
