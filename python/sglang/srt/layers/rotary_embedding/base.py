"""RotaryEmbedding base class + LinearScalingRotaryEmbedding."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch

from sglang.srt.layers.rotary_embedding.utils import apply_rotary_emb
from sglang.srt.layers.utils import MultiPlatformOp
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    is_cpu,
    is_cuda,
    is_hip,
    is_musa,
    is_npu,
    is_xpu,
)

if TYPE_CHECKING:
    from sglang.jit_kernel.rope import FusedSetKVBufferArg  # For type check-only

_is_cuda = is_cuda()
_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_is_npu = is_npu()
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
_is_xpu = is_xpu()
_is_musa = is_musa()

if _is_cuda:
    from sglang.jit_kernel.rope import apply_rope_with_cos_sin_cache_inplace

if _is_npu:
    import torch_npu


class RotaryEmbedding(MultiPlatformOp):
    """Original rotary positional embedding."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype

        cache = self._compute_cos_sin_cache()
        # NOTE(ByronHsu): cache needs to be in FP32 for numerical stability
        if not _is_cuda:
            cache = cache.to(dtype)

        if (
            (not (_is_cuda) or self.head_size not in [64, 128, 256, 512])
            and not (_is_cpu)
            and not (_is_xpu)
            and not (_is_npu)
            and not (_is_musa)
        ):
            # rotary_embedding from sglang.jit_kernel.pos_enc and vllm._custom_ops has the same implementation.
            # TODO: Test on different devices and remove this conditional.
            if _is_cuda:
                from sglang.jit_kernel.pos_enc import rotary_embedding
            elif _is_hip:
                from sgl_kernel import rotary_embedding
            else:
                from vllm._custom_ops import rotary_embedding

            self.use_fallback_kernel = True
            self.fallback_rotary_embedding = rotary_embedding
        else:
            self.use_fallback_kernel = False

        self.cos_sin_cache: torch.Tensor
        self.register_buffer("cos_sin_cache", cache, persistent=False)

        self._apply_rotary_emb_wrapped = apply_rotary_emb

        if get_global_server_args().rl_on_policy_target is not None:
            self._forward_method = self.forward_native
            self._apply_rotary_emb_wrapped = torch.compile(dynamic=True)(
                apply_rotary_emb
            )
        self.position_cos, self.position_sin = None, None

    def _compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:
        """Compute the inverse frequency."""
        # NOTE(woosuk): To exactly match the HF implementation, we need to
        # use CPU to compute the cache and then move it to GPU. However, we
        # create the cache on GPU for faster initialization. This may cause
        # a slight numerical difference between the HF implementation and ours.
        init_device = (
            "cpu" if get_global_server_args().rl_on_policy_target is not None else None
        )
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(
                    0, self.rotary_dim, 2, dtype=torch.float, device=init_device
                )
                / self.rotary_dim
            )
        )
        if get_global_server_args().rl_on_policy_target is not None:
            inv_freq = inv_freq.cuda()
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute the cos and sin cache."""
        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def _ensure_cos_sin_cache_length(self, needed_max_pos: int):
        """Ensure cos_sin_cache length > needed_max_pos."""
        from sglang.srt.environ import envs

        cur_len = int(self.cos_sin_cache.shape[0])
        if needed_max_pos < cur_len:
            return

        # Align to reduce realloc frequency
        align = envs.SGLANG_ROPE_CACHE_ALIGN.get()
        new_len = ((needed_max_pos + align) // align) * align
        device = self.cos_sin_cache.device
        dtype = self.cos_sin_cache.dtype

        # Compute inv_freq on same device
        inv_freq = self._compute_inv_freq(self.base).to(device=device)

        # Incremental computation for new positions only
        start = cur_len
        t_new = torch.arange(start, new_len, dtype=inv_freq.dtype, device=device)
        if t_new.numel() == 0:
            return

        freqs_new = torch.einsum("i,j->ij", t_new, inv_freq)
        cos_new = freqs_new.cos()
        sin_new = freqs_new.sin()
        new_rows = torch.cat((cos_new, sin_new), dim=-1).to(dtype=dtype)

        # Update cache with new rows
        self.cos_sin_cache = torch.cat((self.cos_sin_cache, new_rows), dim=0).to(
            device=device, dtype=dtype
        )

    def get_cos_sin_with_position(self, positions):
        assert positions.ndim == 1, (
            "2D positions (multimodal RoPE) are not supported by the base "
            "RotaryEmbedding. Override this method in a subclass (e.g. MRotaryEmbedding)."
        )
        cos_sin = self.cos_sin_cache.index_select(0, positions.flatten())
        last_dim = cos_sin.size()[-1]
        cos, sin = (
            cos_sin.reshape(-1, 2, last_dim // 2).repeat(1, 1, 2).chunk(2, dim=-2)
        )
        # BSNH
        self.position_cos, self.position_sin = (
            cos.view(-1, 1, 1, last_dim).contiguous(),
            sin.view(-1, 1, 1, last_dim).contiguous(),
        )

    def get_cos_sin(self, seqlen: int) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[:seqlen]
        cos, sin = cos_sin.chunk(2, dim=-1)
        return cos, sin

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
        fused_set_kv_buffer_arg: Optional[FusedSetKVBufferArg] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A PyTorch-native implementation of forward()."""
        assert (
            fused_set_kv_buffer_arg is None
        ), "fused_set_kv_buffer_arg is not supported for native implementation"

        if offsets is not None:
            positions = positions + offsets
        positions = positions.flatten()
        num_tokens = positions.shape[0]
        cos_sin = self.cos_sin_cache.index_select(0, positions)
        cos, sin = cos_sin.chunk(2, dim=-1)

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query_rot = query[..., : self.rotary_dim]
        query_pass = query[..., self.rotary_dim :]
        query_rot = self._apply_rotary_emb_wrapped(
            query_rot, cos, sin, self.is_neox_style
        )
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key_rot = key[..., : self.rotary_dim]
        key_pass = key[..., self.rotary_dim :]
        key_rot = self._apply_rotary_emb_wrapped(key_rot, cos, sin, self.is_neox_style)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key

    def forward_npu(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
        fused_set_kv_buffer_arg: Optional[FusedSetKVBufferArg] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A PyTorch-npu implementation of forward()."""
        assert (
            fused_set_kv_buffer_arg is None
        ), "fused_set_kv_buffer_arg is not supported for npu implementation"
        if query.dtype == torch.bfloat16 and self.cos_sin_cache.dtype == torch.float:
            return self.forward_native(positions, query, key, offsets)
        if self.is_neox_style:
            rotary_mode = "half"
        else:
            rotary_mode = "interleave"
        mrope_section = [0, 0, 0]
        query_out, key_out = torch_npu.npu_mrope(
            positions,
            query,
            key,
            self.cos_sin_cache,
            self.head_size,
            mrope_section=mrope_section,
            rotary_mode=rotary_mode,
        )
        return query_out, key_out

    def forward_cpu(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
        fused_set_kv_buffer_arg: Optional[FusedSetKVBufferArg] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert (
            fused_set_kv_buffer_arg is None
        ), "fused_set_kv_buffer_arg is not supported for cpu implementation"

        positions = torch.add(positions, offsets) if offsets is not None else positions
        if _is_cpu_amx_available:
            return torch.ops.sgl_kernel.rotary_embedding_cpu(
                positions,
                query,
                key,
                self.head_size,
                self.cos_sin_cache,
                self.is_neox_style,
            )
        else:
            return self.forward_native(
                positions, query, key, offsets, fused_set_kv_buffer_arg
            )

    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
        fused_set_kv_buffer_arg: Optional[FusedSetKVBufferArg] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.use_fallback_kernel:
            batch_size = positions.size(0)
            q_rope = query.view(batch_size, -1, self.head_size)
            k_rope = key.view(batch_size, -1, self.head_size)
            if self.head_size != self.rotary_dim:
                q_rope = q_rope[..., : self.rotary_dim]
                k_rope = k_rope[..., : self.rotary_dim]
            apply_rope_with_cos_sin_cache_inplace(
                positions=positions,
                q=q_rope,
                k=k_rope,
                cos_sin_cache=self.cos_sin_cache,
                is_neox=self.is_neox_style,
                fused_args=fused_set_kv_buffer_arg,
            )
        else:
            assert (
                fused_set_kv_buffer_arg is None
            ), "save kv cache is not supported for fallback_rotary_embedding."
            self.cos_sin_cache = self.cos_sin_cache.to(query.device, dtype=query.dtype)
            self.fallback_rotary_embedding(
                positions,
                query,
                key,
                self.head_size,
                self.cos_sin_cache,
                self.is_neox_style,
            )
        return query, key

    def extra_repr(self) -> str:
        s = f"head_size={self.head_size}, rotary_dim={self.rotary_dim}"
        s += f", max_position_embeddings={self.max_position_embeddings}"
        s += f", base={self.base}, is_neox_style={self.is_neox_style}"
        return s

    def forward_xpu(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
        fused_set_kv_buffer_arg: Optional[FusedSetKVBufferArg] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert (
            fused_set_kv_buffer_arg is None
        ), "fused_set_kv_buffer_arg is not supported for xpu implementation"
        positions = torch.add(positions, offsets) if offsets is not None else positions

        return torch.ops.sgl_kernel.rotary_embedding(
            positions,
            query,
            key,
            self.head_size,
            self.cos_sin_cache,
            self.is_neox_style,
        )


class LinearScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with linear scaling.

    It supports multiple scaling factors. Since multiple LoRA adapters may have
    different scaling factors, we need multiple cos/sin caches. In this way,
    instead of running rotary embedding kernel per lora, we can run multiple
    lora in a batched way.

    In addition to that, we also keep the cos/sin cache for the scaling factor
    of 1 (default) at all times.

    Exemplary for two scaling factors x=1, y and z with embeddings
    [[x11, x12, ... x1m], ..., [xn1, xn2, ..., xnm]] and
    [[y11, y12, ... y1o], ..., [yn1, yn2, ..., yno]], and
    [[z11, z12, ... z1p], ..., [zn1, zn2, ..., znp]],

    we construct the cos/sin cache as follows:
    [[x11, x12, ... x1m, y11, y12, ... y1o, z11, z12, ... z1p],
        ...
     [xn1, xn2, ... xnm, yn1, yn2, ... yno, zn1, zn2, ... znp]]

    We then use offsets to index into the cos/sin cache for
    the respective scaling factors.

    The offset to cache can be accessed via `scaling_factor_to_offset` API.

    Credits to the Reddit user /u/kaiokendev
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factors: Union[List[float], float],
        dtype: torch.dtype,
    ) -> None:
        if isinstance(scaling_factors, float):
            scaling_factors = [scaling_factors]
        self.scaling_factors: List[float] = scaling_factors  # noqa
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )
        # Lazy initialized.
        self._scaling_factor_to_offset: Dict[float, int]

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.base)
        cache_list: List[torch.Tensor] = []
        # offsets to the next cache in a tensor.
        # Each offset corresponds to the same index in scaling_factors.
        offsets: List[int] = []
        for scaling_factor in self.scaling_factors:
            # NOTE(woosuk): self.max_position_embeddings is the original
            # maximum length before applying the rope scaling.
            # Thus, the maximum length after applying the rope scaling is
            # self.max_position_embeddings * self.scaling_factor.
            max_len = self.max_position_embeddings * scaling_factor
            t = torch.arange(max_len, dtype=torch.float)
            t = t / scaling_factor

            freqs = torch.einsum("i,j -> ij", t, inv_freq)
            cos = freqs.cos()
            sin = freqs.sin()
            cache = torch.cat((cos, sin), dim=-1)
            if not cache_list:
                offset = 0
            else:
                last_offset = offsets[-1]
                next_max_len = cache_list[-1].shape[0]
                offset = last_offset + next_max_len
            offsets.append(offset)
            cache_list.append(cache)
        self._scaling_factor_to_offset = {
            float(scaling_factor): offsets[i]
            for i, scaling_factor in enumerate(self.scaling_factors)
        }
        assert len(self.scaling_factors) == len(offsets)
        return torch.cat(cache_list, dim=0)

    @property
    def scaling_factor_to_offset(self) -> Dict[float, int]:
        return self._scaling_factor_to_offset
