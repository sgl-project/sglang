"""NPU interleaved RoPE cos/sin cache for DeepSeek-V4 on Ascend.

One Dsv4NpuRoPE per freqs_cis (singleton by id). Tables are built once at
init and registered as buffers on the shared rotary_emb, so model.to() moves
them and a captured aclgraph sees stable tensors; decode only does index_select.

mscale: cos/sin stored in freqs_cis must already be pre-multiplied by the YARN
mscale at precompute time (see precompute_freqs_cis). We just read what's stored.
"""

from typing import Optional

import torch


class Dsv4NpuRoPE:
    """Interleaved cos/sin tables, layout [c0,c0,c1,c1,...] / [s0,s0,s1,s1,...]."""

    # id(freqs_cis) -> instance. freqs_cis is a module buffer, lives with the model.
    _instances: dict[int, "Dsv4NpuRoPE"] = {}

    def __init__(
        self, freqs_cis: torch.Tensor, rotary_emb: Optional[object] = None
    ) -> None:
        self.freqs_cis = freqs_cis
        # cos/sin registered as buffers on this module (None -> fall back to _tables).
        self.rotary_emb = rotary_emb
        # contiguous real/imag halves of complex freqs_cis [max_pos, rope_dim/2];
        # .real/.imag are strided views, materialize once to avoid per-call
        # StridedSlice from aclnnIndex over the strided views.
        self._real_imag: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        self._tables: dict[
            tuple[torch.dtype, torch.device], tuple[torch.Tensor, torch.Tensor]
        ] = {}

    @classmethod
    def for_freqs(
        cls, freqs_cis: torch.Tensor, rotary_emb: Optional[object] = None
    ) -> "Dsv4NpuRoPE":
        # rotary_emb is only used at creation; callers sharing a warmed-up freqs_cis may omit it.
        inst = cls._instances.get(id(freqs_cis))
        if inst is None or inst.freqs_cis is not freqs_cis:
            inst = cls(freqs_cis, rotary_emb)
            cls._instances[id(freqs_cis)] = inst
        return inst

    def _contig_real_imag(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._real_imag is None:
            self._real_imag = (
                self.freqs_cis.real.contiguous(),
                self.freqs_cis.imag.contiguous(),
            )
        return self._real_imag

    @staticmethod
    def _buffer_names(dtype: torch.dtype) -> tuple[str, str]:
        suffix = str(dtype).replace("torch.", "").replace(".", "_")
        return (
            f"_npu_interleaved_rope_cos_cache_{suffix}",
            f"_npu_interleaved_rope_sin_cache_{suffix}",
        )

    def _register_or_set_buffer(self, name: str, tensor: torch.Tensor) -> None:
        owner = self.rotary_emb
        if hasattr(owner, "register_buffer"):
            if name in getattr(owner, "_buffers", {}):
                setattr(owner, name, tensor)
            else:
                owner.register_buffer(name, tensor, persistent=False)
        else:
            setattr(owner, name, tensor)

    def ensure_tables(
        self, dtype: torch.dtype, *, allow_build: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Returns [max_pos, rope_dim] tables. Call once at init (allow_build=True);
        # decode uses allow_build=False (no repeat_interleave inside the captured graph).
        expected_shape = (self.freqs_cis.shape[0], self.freqs_cis.shape[1] * 2)

        if self.rotary_emb is not None:
            cos_name, sin_name = self._buffer_names(dtype)
            cos = getattr(self.rotary_emb, cos_name, None)
            sin = getattr(self.rotary_emb, sin_name, None)
            if (
                cos is not None
                and sin is not None
                and tuple(cos.shape) == expected_shape
                and tuple(sin.shape) == expected_shape
                and cos.dtype == dtype
                and sin.dtype == dtype
                and cos.device == self.freqs_cis.device
                and sin.device == self.freqs_cis.device
            ):
                return cos, sin
        else:
            cached = self._tables.get((dtype, self.freqs_cis.device))
            if cached is not None:
                cos, sin = cached
                if (
                    tuple(cos.shape) == expected_shape
                    and tuple(sin.shape) == expected_shape
                ):
                    return cached

        if not allow_build:
            raise RuntimeError(
                "NPU interleaved RoPE cache is missing in a no-build path. "
                "Initialize it before forward to keep decode free of repeat_interleave."
            )

        real_contig, imag_contig = self._contig_real_imag()
        cos = real_contig.repeat_interleave(2, dim=-1).to(dtype=dtype).contiguous()
        sin = imag_contig.repeat_interleave(2, dim=-1).to(dtype=dtype).contiguous()

        if self.rotary_emb is not None:
            cos_name, sin_name = self._buffer_names(dtype)
            self._register_or_set_buffer(cos_name, cos)
            self._register_or_set_buffer(sin_name, sin)
        else:
            self._tables[(dtype, self.freqs_cis.device)] = (cos, sin)
        return cos, sin

    def get_cos_sin(
        self,
        positions: torch.Tensor,
        dtype: torch.dtype,
        *,
        view_4d: bool = False,
        inverse: bool = False,
        allow_build: bool = True,
        cache_dtype: Optional[torch.dtype] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # positions: [T]. Returns [T, rope_dim], or [T, 1, 1, rope_dim] if view_4d.
        # Position-gathered tensors are forward-local; do not cache across forwards
        # or MTP decode reuses the previous step's RoPE when only positions change.
        cache_dtype = dtype if cache_dtype is None else cache_dtype
        cos_cache, sin_cache = self.ensure_tables(cache_dtype, allow_build=allow_build)
        cos = cos_cache.index_select(0, positions)
        sin = sin_cache.index_select(0, positions)
        if inverse:
            sin = -sin
        if cos.dtype != dtype:
            cos = cos.to(dtype)
            sin = sin.to(dtype)
        if view_4d:
            rope_dim = cos.shape[-1]
            cos = cos.view(-1, 1, 1, rope_dim)
            sin = sin.view(-1, 1, 1, rope_dim)
        return cos, sin

    @staticmethod
    def apply_rotary_mul_inplace(
        q_rope: torch.Tensor,
        kv_rope: Optional[torch.Tensor],
        cos4: torch.Tensor,
        sin4: torch.Tensor,
        qk_nope_dim: int = 0,
    ) -> None:
        # q_rope: [T, n_heads, head_dim]; cos4/sin4: [T, 1, 1, rope_dim];
        # kv_rope: [T, 1, head_dim] or None. Prefer the NPU kernel: torch accumulates
        # bf16 muls in bf16 while the kernel uses fp32; drift compounds and flips argmax.
        rope_dim = cos4.shape[-1]
        torch.ops.custom.inplace_partial_rotary_mul(
            q_rope.unsqueeze(1),
            cos4,
            sin4,
            rotary_mode="interleave",
            partial_slice=[qk_nope_dim, qk_nope_dim + rope_dim],
        )
        if kv_rope is not None:
            if kv_rope.dim() == 3:
                kv_view = kv_rope.unsqueeze(1)
            else:
                kv_view = kv_rope.view(-1, 1, 1, rope_dim)
            torch.ops.custom.inplace_partial_rotary_mul(
                kv_view,
                cos4,
                sin4,
                rotary_mode="interleave",
                partial_slice=[qk_nope_dim, qk_nope_dim + rope_dim],
            )
