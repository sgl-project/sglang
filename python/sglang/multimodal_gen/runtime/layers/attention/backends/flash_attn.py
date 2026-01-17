# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
# SPDX-License-Identifier: Apache-2.0
import inspect
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, List, Optional, Tuple

import torch
from torch.library import Library

from sglang.multimodal_gen.runtime.managers.forward_context import get_forward_context
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)

try:
    from sgl_kernel.flash_attn import flash_attn_varlen_func

    # flash_attn 3 no longer have a different API, see following commit:
    # https://github.com/Dao-AILab/flash-attention/commit/ed209409acedbb2379f870bbd03abce31a7a51b7
    flash_attn_func = flash_attn_varlen_func
except ImportError as e:
    raise e

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def maybe_contiguous(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


# -----------------------------
# Fake implementations for schema / tracing
# custom op schema requires FIXED return structure.
# We provide TWO ops:
# 1) out-only op: always returns Tensor
# 2) out+lse op: always returns Tuple[Tensor, Tensor]
# -----------------------------
def flash_attn_varlen_func_fake_out(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    qv: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    window_size: Optional[List[int]] = None,
    attention_chunk: int = 0,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    sm_margin: int = 0,
    return_softmax_lse: bool = False,
    sinks: Optional[torch.Tensor] = None,
    ver: int = 4,
) -> torch.Tensor:
    assert ver == 4, "only support flash attention v4"
    q, k, v = [maybe_contiguous(t) for t in (q, k, v)]
    num_head, head_dim = q.shape[-2:]
    if cu_seqlens_q is None:
        batch_size, seqlen_q = q.shape[:2]
    else:
        batch_size = cu_seqlens_q.shape[0] - 1
        seqlen_q = None
    head_dim_v = v.shape[-1]

    if cu_seqlens_q is not None:
        assert cu_seqlens_q.shape == (
            batch_size + 1,
        ), "cu_seqlens_q must have shape (batch_size + 1,)"
        assert cu_seqlens_q.dtype == torch.int32, "cu_seqlens_q must be int32"
        assert cu_seqlens_q.stride(0) == 1, "cu_seqlens_q must be contiguous"

    assert q.dtype in [
        torch.float16,
        torch.bfloat16,
    ], "inputs must be float16 or bfloat16"
    assert q.dtype == k.dtype == v.dtype, "inputs must have the same dtype"
    assert head_dim <= 256, "head_dim must be less than or equal to 256"
    alignment = 16 // q.element_size()
    assert head_dim_v % alignment == 0, f"head_dim_v must be divisible by {alignment}"

    q_batch_seqlen_shape = (
        (batch_size, seqlen_q) if cu_seqlens_q is None else (q.shape[0],)
    )
    out = q.new_empty(*q_batch_seqlen_shape, num_head, head_dim_v)
    return out


def flash_attn_varlen_func_fake_out_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    qv: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    window_size: Optional[List[int]] = None,
    attention_chunk: int = 0,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    sm_margin: int = 0,
    return_softmax_lse: bool = True,
    sinks: Optional[torch.Tensor] = None,
    ver: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert ver == 4, "only support flash attention v4"
    q, k, v = [maybe_contiguous(t) for t in (q, k, v)]
    num_head, head_dim = q.shape[-2:]
    if cu_seqlens_q is None:
        batch_size, seqlen_q = q.shape[:2]
        total_q = batch_size * seqlen_q
    else:
        batch_size = cu_seqlens_q.shape[0] - 1
        seqlen_q = None
        total_q = q.shape[0]
    head_dim_v = v.shape[-1]

    if cu_seqlens_q is not None:
        assert cu_seqlens_q.shape == (
            batch_size + 1,
        ), "cu_seqlens_q must have shape (batch_size + 1,)"
        assert cu_seqlens_q.dtype == torch.int32, "cu_seqlens_q must be int32"
        assert cu_seqlens_q.stride(0) == 1, "cu_seqlens_q must be contiguous"

    assert q.dtype in [
        torch.float16,
        torch.bfloat16,
    ], "inputs must be float16 or bfloat16"
    assert q.dtype == k.dtype == v.dtype, "inputs must have the same dtype"
    assert head_dim <= 256, "head_dim must be less than or equal to 256"
    alignment = 16 // q.element_size()
    assert head_dim_v % alignment == 0, f"head_dim_v must be divisible by {alignment}"

    q_batch_seqlen_shape = (
        (batch_size, seqlen_q) if cu_seqlens_q is None else (total_q,)
    )
    lse_shape = (
        (batch_size, num_head, seqlen_q)
        if cu_seqlens_q is None
        else (num_head, total_q)
    )

    out = q.new_empty(*q_batch_seqlen_shape, num_head, head_dim_v)
    lse = q.new_empty(lse_shape, dtype=torch.float32)
    return out, lse


sglang_lib = Library("sglang", "FRAGMENT")  # noqa


def direct_register_custom_op(
    op_name: str,
    op_func: Callable,
    mutates_args: List[str],
    fake_impl: Optional[Callable] = None,
    target_lib: Optional[Library] = None,
):
    """
    `torch.library.custom_op` can have significant overhead because it
    needs to consider complicated dispatching logic. This function
    directly registers a custom op and dispatches it to the CUDA backend.
    See https://gist.github.com/youkaichao/ecbea9ec9fc79a45d2adce1784d7a9a5
    for more details.

    By default, the custom op is registered to the vLLM library. If you
    want to register it to a different library, you can pass the library
    object to the `target_lib` argument.

    IMPORTANT: the lifetime of the operator is tied to the lifetime of the
    library object. If you want to bind the operator to a different library,
    make sure the library object is alive when the operator is used.

    Note: This function will silently skip registration if the operator
    with the same name is already registered to avoid RuntimeError in
    multi-engine scenarios (e.g., VERL framework).
    """
    import torch.library

    my_lib = target_lib or sglang_lib

    # Check if operator is already registered to avoid duplicate registration
    # This is important for scenarios where multiple SGLang engines run in the same process
    try:
        # Try to access the operator to see if it's already registered
        lib_name = my_lib.m.name if hasattr(my_lib.m, "name") else "sglang"
        if hasattr(torch.ops, lib_name) and hasattr(
            getattr(torch.ops, lib_name), op_name
        ):
            # Operator already exists, skip registration
            return
    except (AttributeError, RuntimeError):
        # Operator doesn't exist, proceed with registration
        pass

    if hasattr(torch.library, "infer_schema"):
        schema_str = torch.library.infer_schema(op_func, mutates_args=mutates_args)
    else:
        # for pytorch 2.4
        import torch._custom_op.impl

        schema_str = torch._custom_op.impl.infer_schema(op_func, mutates_args)

    try:
        my_lib.define(op_name + schema_str)
        my_lib.impl(
            op_name, op_func, "CUDA" if not current_platform.is_npu() else "PrivateUse1"
        )
        if fake_impl is not None:
            my_lib._register_fake(op_name, fake_impl)
    except RuntimeError as error:
        if "Tried to register an operator" in str(error) and "multiple times" in str(
            error
        ):
            # Silently ignore duplicate registration errors
            # This can happen in multi-engine scenarios
            pass
        else:
            # Re-raise other RuntimeErrors
            raise error
    except AttributeError as error:
        # Always re-raise AttributeError as it indicates missing dependencies
        raise error


class CustomOpWrapper:
    def __init__(
        self,
        op_name: str,
        op_func: Callable,
        mutates_args: List[str],
        **extra_kwargs,
    ):
        self.op_name = op_name
        self.op_func = op_func
        self.mutates_args = mutates_args
        self.extra_kwargs = extra_kwargs
        self._impl: Optional[Callable] = None

    def __call__(self, *args, **kwargs):
        return self.real_impl(*args, **kwargs)

    @property
    def real_impl(self) -> Callable:
        if self._impl is None:
            if not hasattr(torch.ops.sglang, self.op_name):

                # NOTE(dark): if torch compile fail here, mark the decorator as eager
                # lazy registration does not work with torch compile
                direct_register_custom_op(
                    op_name=self.op_name,
                    op_func=self.op_func,
                    mutates_args=self.mutates_args,
                    fake_impl=self.fake_impl,
                )
            self._impl = getattr(torch.ops.sglang, self.op_name)
            assert self._impl is not None
        return self._impl

    @property
    def fake_impl(self) -> Callable:
        if "fake_impl" in self.extra_kwargs:
            return self.extra_kwargs["fake_impl"]
        assert "out_shape" in self.extra_kwargs
        signature = inspect.signature(self.op_func)
        out_shape = self.extra_kwargs["out_shape"]

        # check out_shape in signature

        def fake_impl(*args, **kwargs):
            if out_shape is None:
                return None
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            try:
                return torch.empty_like(
                    bound.args[out_shape]
                    if isinstance(out_shape, int)
                    else bound.arguments[out_shape]
                )
            except (IndexError, KeyError):
                raise RuntimeError(
                    f"Cannot find output argument at position `{out_shape}` for "
                    f"custom operator `{self.op_name}` with signature `{signature}`."
                )

        return fake_impl


# Real implementation
def register_custom_op(
    fn: Optional[Callable] = None,
    *,
    op_name: Optional[str] = None,
    mutates_args: Optional[List[str]] = None,
    eager: bool = True,
    **extra_kwargs,
) -> Any:
    """
    A decorator to register a custom operator.

    Example usage:
    ```python
    # inplace operator, out_shape is None by default
    @register_custom_op(mutates_args=["x"])
    def add_1_(x: torch.Tensor) -> None:
        x.add_(1)

    # operator with output, out_shape indicates the position of output
    @register_custom_op(mutates_args=["x"], out_shape=0)
    def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x.add_(y)
    ```

    :param fn: The function to be registered as a custom operator.
               If None, return a decorator.
    :type fn: Callable
    :param op_name: The name of the operator. If None, use the function name
    :type op_name: Optional[str]
    :param mutates_args: A list of argument names that are mutated in-place.
    :type mutates_args: List[str]
    :param out_shape: The position (int for positional, str for keyword) of the output-shape tensor.
                      It is used to generate a fake implementation for torch.compile compatibility.
                      If the operator is inplace and has no output, set to None.
    :type out_shape: Optional[List[Union[int, str]]]
    :param fake_impl: A fake implementation for the operator.
                      Only one of `out_shape` or `fake_impl` should be provided.
    :type fake_impl: Optional[Callable]
    :param eager: Whether to register the operator eagerly.
                  If False, the registration will be deferred until the first call.
                  If you met any issue with torch.compile, try to set eager=True.
                  Currently, to avoid misuse, we set eager=True by default.
    :type eager: bool
    :return: The registered JIT custom operator, or a decorator.
             NOTE: the real register will occur at the first call of the function.
    :rtype: Callable
    """
    extra_kwarg_keys = set(extra_kwargs.keys())
    expected_kwarg_keys = set({"out_shape", "fake_impl"})
    assert (
        expected_kwarg_keys >= extra_kwarg_keys
    ), f"Unexpected extra kwargs: {extra_kwarg_keys - expected_kwarg_keys}"

    has_out_shape = "out_shape" in extra_kwargs
    has_fake_impl = "fake_impl" in extra_kwargs
    assert not (
        has_out_shape and has_fake_impl
    ), "Only one of `out_shape` or `fake_impl` should be provided."
    # Assume inplace if neither out_shape nor fake_impl is provided
    if not (has_out_shape or has_fake_impl):
        extra_kwargs["out_shape"] = None

    def decorator(op_func: Callable) -> Callable:
        wrapper = CustomOpWrapper(
            op_name=op_name or op_func.__name__,
            op_func=op_func,
            mutates_args=mutates_args or [],
            **extra_kwargs,
        )
        return wrapper.real_impl if eager else wrapper

    if fn is not None:
        return decorator(fn)
    return decorator


# -----------------------------
# Registered custom ops
# NOTE: fixed return schemas to avoid:
# "Object of type 'Tensor' is not an instance of 'sequence'"
# -----------------------------
@register_custom_op(fake_impl=flash_attn_varlen_func_fake_out)
def flash_attn_varlen_func_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    qv: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    window_size: Optional[List[int]] = None,
    attention_chunk: int = 0,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    sm_margin: int = 0,
    return_softmax_lse: bool = False,
    sinks: Optional[torch.Tensor] = None,
    ver: int = 4,
) -> torch.Tensor:
    if window_size is None:
        window_size = [-1, -1]
    if return_softmax_lse:
        raise ValueError(
            "flash_attn_varlen_func_op is out-only op; return_softmax_lse must be False. "
            "Use flash_attn_varlen_func_op_lse for (out, lse)."
        )
    return flash_attn_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        page_table=page_table,
        softmax_scale=softmax_scale,
        causal=causal,
        qv=qv,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        window_size=tuple(window_size),
        attention_chunk=attention_chunk,
        softcap=softcap,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        sm_margin=sm_margin,
        return_softmax_lse=False,
        sinks=sinks,
        ver=ver,
    )


@register_custom_op(fake_impl=flash_attn_varlen_func_fake_out_lse)
def flash_attn_varlen_func_op_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    qv: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    window_size: Optional[List[int]] = None,
    attention_chunk: int = 0,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    sm_margin: int = 0,
    return_softmax_lse: bool = True,
    sinks: Optional[torch.Tensor] = None,
    ver: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if window_size is None:
        window_size = [-1, -1]
    if not return_softmax_lse:
        raise ValueError(
            "flash_attn_varlen_func_op_lse is out+lse op; return_softmax_lse must be True. "
            "Use flash_attn_varlen_func_op for out-only."
        )
    return flash_attn_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        page_table=page_table,
        softmax_scale=softmax_scale,
        causal=causal,
        qv=qv,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        window_size=tuple(window_size),
        attention_chunk=attention_chunk,
        softcap=softcap,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        sm_margin=sm_margin,
        return_softmax_lse=True,
        sinks=sinks,
        ver=ver,
    )


try:
    if current_platform.is_hopper():
        from flash_attn_interface import (
            flash_attn_varlen_func as flash_attn_varlen_func_upstream,
        )
    else:
        flash_attn_varlen_func_upstream = None

except Exception:
    flash_attn_varlen_func_upstream = None
    logger.warning(
        "flash_attn 3 package is not installed. It's recommended to install flash_attn3 on hopper, otherwise performance is sub-optimal"
    )

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)

fa_ver = 3


@lru_cache(maxsize=128)
def _get_cu_seqlens(device_index: int, bsz: int, seqlen: int) -> torch.Tensor:
    return torch.arange(
        0,
        (bsz + 1) * seqlen,
        step=seqlen,
        device=torch.device("cuda", device_index),
        dtype=torch.int32,
    )


@lru_cache(maxsize=256)
def _should_use_upstream_flash_attention(
    upstream_available: bool,
    upstream_heads_ok: bool,
    q_shape: tuple[int, ...],
    k_shape: tuple[int, ...],
    v_shape: tuple[int, ...],
) -> bool:
    if not upstream_available or not upstream_heads_ok:
        return False

    if len(q_shape) != 4 or len(k_shape) != 4 or len(v_shape) != 4:
        return False

    bsz, seqlen, nheads_q, d = q_shape
    bsz_k, seqlen_k, nheads_k, d_k = k_shape
    bsz_v, seqlen_v, nheads_v, d_v = v_shape

    if (
        bsz != bsz_k
        or bsz != bsz_v
        or seqlen != seqlen_k
        or seqlen != seqlen_v
        or d != d_k
        or d != d_v
    ):
        return False
    if nheads_k != nheads_v:
        return False
    if nheads_k == 0 or (nheads_q % nheads_k) != 0:
        return False
    return True


def set_fa_ver(ver: int) -> None:
    global fa_ver
    fa_ver = ver


@dataclass
class FlashAttentionMetadata:
    # Sequence lengths for the forward batch
    # Maximum sequence length for query
    max_seqlen_q: int = 1
    # Maximum sequence length for key
    max_seqlen_k: int = 0
    # Cumulative sequence lengths for query
    cu_seqlens_q: torch.Tensor = None
    # Cumulative sequence lengths for key
    cu_seqlens_k: torch.Tensor = None


class FlashAttentionMetadataBuilder(AttentionMetadataBuilder):
    def __init__(self) -> None:
        pass

    def prepare(self) -> None:
        pass

    def build(  # type: ignore
        self,
        raw_latent_shape=list,
        **kwargs: dict[str, Any],
    ) -> FlashAttentionMetadata:
        # TODO: put empty values here to be set at first-run, since the q_len calculation can be complicated
        return FlashAttentionMetadata(max_seqlen_q=None, max_seqlen_k=None)


class FlashAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.FA

    @staticmethod
    def get_impl_cls() -> type["FlashAttentionImpl"]:
        return FlashAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        raise NotImplementedError

    @staticmethod
    def get_builder_cls() -> type["AttentionMetadataBuilder"]:
        return FlashAttentionMetadataBuilder


class FlashAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.attention_metadata = FlashAttentionMetadata()
        if self.num_kv_heads is None:
            self._upstream_heads_ok = True
        else:
            # For gqa, the num_heads must be a multiple of num_kv_heads
            self._upstream_heads_ok = (
                self.num_kv_heads > 0 and (self.num_heads % self.num_kv_heads) == 0
            )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
        *,
        return_softmax_lse: bool = False,
    ):
        attn_metadata: FlashAttentionMetadata = get_forward_context().attn_metadata
        if attn_metadata is not None and attn_metadata.max_seqlen_q is None:
            attn_metadata.max_seqlen_q = query.shape[1]
            attn_metadata.max_seqlen_k = key.shape[1]
            max_seqlen_q = attn_metadata.max_seqlen_q
            max_seqlen_k = attn_metadata.max_seqlen_k
        else:
            max_seqlen_q = query.shape[1]
            max_seqlen_k = key.shape[1]

        q_shape = tuple(query.shape)
        k_shape = tuple(key.shape)
        v_shape = tuple(value.shape)

        use_upstream = _should_use_upstream_flash_attention(
            flash_attn_varlen_func_upstream is not None,
            self._upstream_heads_ok,
            q_shape,
            k_shape,
            v_shape,
        )

        if use_upstream:
            bsz, seqlen, nheads_q, d = q_shape
            q_ = query.contiguous()
            k_ = key.contiguous()
            v_ = value.contiguous()
            out = flash_attn_varlen_func_upstream(
                q_,
                k_,
                v_,
                None,
                None,
                seqlen,
                seqlen,
                softmax_scale=self.softmax_scale,
                causal=self.causal,
                return_attn_probs=return_softmax_lse,
            )
            if return_softmax_lse:
                out_tensor, softmax_lse = out
                return out_tensor.reshape(bsz, seqlen, nheads_q, -1), softmax_lse
            return out.reshape(bsz, seqlen, nheads_q, d)

        # FA version selection:
        # - fa_ver == 3: call python function (can return Tensor or (Tensor, Tensor) depending on flag)
        # - fa_ver == 4: call custom ops with FIXED return schema
        if fa_ver == 3:
            flash_attn_op = flash_attn_func
            output = flash_attn_op(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=None,
                cu_seqlens_k=None,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=self.softmax_scale,
                causal=self.causal,
                return_softmax_lse=return_softmax_lse,
                ver=fa_ver,
            )
            return output

        if fa_ver == 4:
            if return_softmax_lse:
                out_tensor, softmax_lse = flash_attn_varlen_func_op_lse(
                    q=query,
                    k=key,
                    v=value,
                    cu_seqlens_q=None,
                    cu_seqlens_k=None,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    softmax_scale=self.softmax_scale,
                    causal=self.causal,
                    return_softmax_lse=True,
                    ver=fa_ver,
                )
                return out_tensor, softmax_lse
            out_tensor = flash_attn_varlen_func_op(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=None,
                cu_seqlens_k=None,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=self.softmax_scale,
                causal=self.causal,
                return_softmax_lse=False,
                ver=fa_ver,
            )
            return out_tensor

        raise ValueError(f"flash attention version {fa_ver} is not supported.")
