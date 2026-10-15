from typing import Collection, Iterable, Iterator, NamedTuple, Optional, Tuple

import torch

from sglang.srt.layers.quantization.fp8 import (
    Fp8LinearMethod,
    Fp8MoEMethod,
    unshuffle_fp8_weight,
)
from sglang.srt.layers.quantization.fp8_utils import (
    block_quant_dequant,
    inverse_transform_scale_ue8m0,
)
from sglang.srt.layers.quantization.marlin_utils import (
    get_scale_perms,
    get_weight_perm,
    marlin_permute_weights,
)
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4LinearMethod,
    ModelOptNvFp4FusedMoEMethod,
)
from sglang.srt.layers.quantization.muse_spark_nvfp4_moe import (
    MuseSparkNvFp4FusedMoEMethod,
)
from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod

# chunk to avoid too high GPU memory peak
CHUNK_NUMEL = 64 * 1024 * 1024


class CompareResult(NamedTuple):
    equal: bool
    max_abs_err: float
    mean_abs_err: float
    num_exceed: int  # elements past the combined per-side tolerance


class QuantizedNames(NamedTuple):
    """Parameter names of one quantized weight on its module."""

    weight: str
    scale: str
    global_scale: Optional[str] = None
    # Backend repacks computed from this weight; neither compared nor hashed.
    derived: Tuple[str, ...] = ()


class ComparableWeight:
    """Base comparable-weight class; one subclass per precision or raw tensor."""

    # `<x>weight` pairs with `<x>SCALE_SUFFIX` on the same module
    SCALE_SUFFIX = "weight_scale_inv"

    @classmethod
    def match_quantized_weights(
        cls, param_names: Collection[str]
    ) -> Iterator[QuantizedNames]:
        for name in param_names:
            scale = name.replace("weight", cls.SCALE_SUFFIX)
            if name.endswith("weight") and scale in param_names:
                yield QuantizedNames(name, scale)

    @staticmethod
    def _quant_ulp(w_q: torch.Tensor) -> torch.Tensor:
        """Per-element ULP of w_q in its own dtype."""
        finfo = torch.finfo(w_q.dtype)
        x = w_q.to(torch.float32).abs()
        # frexp: x = m * 2^e, m in [0.5, 1), so 2^(e-1) is x's binade base.
        _, exponent = torch.frexp(x)
        binade = torch.exp2((exponent - 1).to(torch.float32))
        # Zeros and subnormals share the spacing of the smallest normal binade.
        binade = binade.masked_fill(x < finfo.smallest_normal, finfo.smallest_normal)
        return binade * finfo.eps

    def iter_chunks(self) -> Iterable[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        raise NotImplementedError

    def dequantize(self, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        raise NotImplementedError


class Fp8BlockComparable(ComparableWeight):
    """Deepseek-style FP8 quantization."""

    def __init__(
        self,
        w_q: torch.Tensor,
        w_s: torch.Tensor,
        is_shuffled: bool = False,
    ):
        self.w_q = w_q
        self.w_s = w_s
        self.is_shuffled = is_shuffled

    def __repr__(self) -> str:
        return (
            f"fp8_block(shape={tuple(self.w_q.shape)} dtype={self.w_q.dtype} "
            f"is_shuffled={self.is_shuffled})"
        )

    @staticmethod
    def _normalize_scale(w_q: torch.Tensor, w_s: torch.Tensor) -> torch.Tensor:
        if w_s.dtype == torch.int32:
            w_s = inverse_transform_scale_ue8m0(w_s, mn=w_q.shape[-2])
            # ue8m0 packing aligns k to a multiple of 4; drop the padding blocks.
            w_s = w_s[..., : -(-w_q.shape[-1] // 128)]
        return w_s.to(torch.float32)

    @staticmethod
    def _infer_block_size(w_q: torch.Tensor, w_s: torch.Tensor) -> list:
        k, s_k = w_q.shape[-1], w_s.shape[-1]
        assert k % s_k == 0, f"cannot infer block size from {w_q.shape=} {w_s.shape=}"
        block = k // s_k
        return [block, block]

    @staticmethod
    def _iter_quant_chunks(w_q: torch.Tensor, w_s: torch.Tensor, block_n: int):
        """Yields block-row-aligned (q_slice, s_slice) pairs of bounded size."""
        q3 = w_q.reshape(-1, *w_q.shape[-2:])
        s3 = w_s.reshape(-1, *w_s.shape[-2:])
        n, k = q3.shape[-2:]
        rows = max(block_n, CHUNK_NUMEL // k // block_n * block_n)
        for b in range(q3.shape[0]):
            for r0 in range(0, n, rows):
                r1 = min(r0 + rows, n)
                yield q3[b, r0:r1], s3[b, r0 // block_n : -(-r1 // block_n)]

    def _scale_and_block_size(self):
        s = self._normalize_scale(self.w_q, self.w_s)
        return s, self._infer_block_size(self.w_q, s)

    def iter_chunks(self):
        s, block_size = self._scale_and_block_size()
        for q, s_chunk in self._iter_quant_chunks(self.w_q, s, block_size[0]):
            q, s_chunk = q.cuda(), s_chunk.cuda()
            if self.is_shuffled:
                q = unshuffle_fp8_weight(q)
            yield (
                block_quant_dequant(q, s_chunk, block_size, dtype=torch.bfloat16),
                block_quant_dequant(
                    self._quant_ulp(q), s_chunk, block_size, dtype=torch.float32
                ),
            )

    def dequantize(self, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        s, block_size = self._scale_and_block_size()
        w_q = self.w_q
        if self.is_shuffled:
            w_q = unshuffle_fp8_weight(w_q)
        return block_quant_dequant(w_q, s, block_size, dtype=dtype)


_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_SPACING = (0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 2.0, 2.0)
_MXFP4_GROUP_SIZE = 32


class Mxfp4MarlinComparable(ComparableWeight):
    """MXFP4 MoE experts after `prepare_moe_mxfp4_layer_for_marlin`: e2m1 nibbles in
    Marlin-repacked int32 `(E, K/16, 2N)`, e8m0 scales in Marlin order `(E, K/32, N)`.

    Compared in dequantized `(E, N, K)` space, so a re-quantization that picks a different
    but equivalent (scale, nibble) encoding still compares equal."""

    SCALE_SUFFIX = "weight_scale"

    # (size_k, size_n) -> (nibble source index per (k, n), scale source index per (g, n))
    _layout_cache: dict = {}

    def __init__(self, w_q: torch.Tensor, w_s: torch.Tensor, is_shuffled: bool = False):
        # the checker passes every quantized weight's AITER shuffle flag; Marlin never shuffles
        assert not is_shuffled, "Marlin MXFP4 experts are never AITER-shuffled"
        assert w_q.dtype == torch.int32, (
            f"expected Marlin int32 weights, got {w_q.dtype}"
        )
        self.w_q = w_q
        self.w_s = w_s
        self.size_k = w_q.shape[-2] * 16
        self.size_n = w_q.shape[-1] // 2
        assert tuple(w_s.shape[-2:]) == (
            self.size_k // _MXFP4_GROUP_SIZE,
            self.size_n,
        ), (
            f"scale shape {tuple(w_s.shape)} does not match weight shape {tuple(w_q.shape)}"
        )

    def __repr__(self) -> str:
        return f"mxfp4_marlin(shape={tuple(self.w_q.shape)} dtype={self.w_q.dtype})"

    @classmethod
    def _layout(cls, size_k: int, size_n: int, device: torch.device):
        """Index maps that undo the Marlin repack, built from the Python reference permutation."""
        key = (size_k, size_n, str(device))
        if key not in cls._layout_cache:
            nibble_source = marlin_permute_weights(
                torch.arange(size_k * size_n).reshape(size_k, size_n),
                size_k,
                size_n,
                get_weight_perm(num_bits=4),
            ).reshape(-1)
            # marlin_weights packs consecutive columns into one int32, low nibble first, so the
            # flattened permuted matrix already lists nibbles in packed order
            nibble_position = torch.empty_like(nibble_source)
            nibble_position[nibble_source] = torch.arange(nibble_source.numel())

            scale_perm, _ = get_scale_perms()
            num_groups = size_k // _MXFP4_GROUP_SIZE
            scale_source = torch.arange(num_groups * size_n)
            scale_source = scale_source.reshape(-1, len(scale_perm))[:, scale_perm]
            # mxfp4_marlin_process_scales swaps the middle pair of every four scales
            scale_source = scale_source.reshape(-1, 4)[:, [0, 2, 1, 3]].reshape(-1)
            scale_position = torch.empty_like(scale_source)
            scale_position[scale_source] = torch.arange(scale_source.numel())

            cls._layout_cache[key] = (
                nibble_position.to(device),
                scale_position.to(device),
            )
        return cls._layout_cache[key]

    def _dequantize_experts(self, w_q: torch.Tensor, w_s: torch.Tensor):
        """`(e, K/16, 2N)` int32 + `(e, K/32, N)` e8m0 -> dequantized and ULP tensors `(e, N, K)`."""
        num_experts = w_q.shape[0]
        nibble_position, scale_position = self._layout(
            self.size_k, self.size_n, w_q.device
        )
        shifts = torch.arange(
            0, 32, 4, device=w_q.device, dtype=torch.int32
        )  # nibble i sits at bits [4i, 4i+4)
        nibbles = (w_q.reshape(num_experts, -1, 1) >> shifts) & 0xF
        nibbles = nibbles.reshape(num_experts, -1)[:, nibble_position]
        nibbles = nibbles.reshape(num_experts, self.size_k, self.size_n)

        exponents = w_s.view(torch.uint8).reshape(num_experts, -1)[:, scale_position]
        scale = torch.exp2(exponents.to(torch.float32) - 127.0).reshape(
            num_experts, self.size_k // _MXFP4_GROUP_SIZE, 1, self.size_n
        )

        magnitude = (nibbles & 0x7).long()
        sign = 1.0 - 2.0 * ((nibbles >> 3) & 0x1).to(torch.float32)
        values = torch.tensor(_E2M1_VALUES, device=w_q.device)[magnitude] * sign
        spacing = torch.tensor(_E2M1_SPACING, device=w_q.device)[magnitude]
        grouped = (
            num_experts,
            self.size_k // _MXFP4_GROUP_SIZE,
            _MXFP4_GROUP_SIZE,
            self.size_n,
        )
        dequantized = (values.reshape(grouped) * scale).reshape(
            num_experts, self.size_k, self.size_n
        )
        ulp = (spacing.reshape(grouped) * scale).reshape(
            num_experts, self.size_k, self.size_n
        )
        # the checkpoint stores experts as (N, K)
        return dequantized.transpose(-1, -2), ulp.transpose(-1, -2)

    def _experts_per_chunk(self) -> int:
        # the unpack holds several int32/float32 copies of every nibble
        return max(1, CHUNK_NUMEL // (4 * self.size_k * self.size_n))

    def iter_chunks(self):
        q = self.w_q.reshape(-1, *self.w_q.shape[-2:])
        s = self.w_s.reshape(-1, *self.w_s.shape[-2:])
        step = self._experts_per_chunk()
        for e0 in range(0, q.shape[0], step):
            dequantized, ulp = self._dequantize_experts(
                q[e0 : e0 + step].cuda(), s[e0 : e0 + step].cuda()
            )
            yield dequantized.to(torch.bfloat16), ulp

    def dequantize(self, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        q = self.w_q.reshape(-1, *self.w_q.shape[-2:])
        s = self.w_s.reshape(-1, *self.w_s.shape[-2:])
        step = self._experts_per_chunk()
        chunks = [
            self._dequantize_experts(
                q[e0 : e0 + step].cuda(), s[e0 : e0 + step].cuda()
            )[0].to(dtype)
            for e0 in range(0, q.shape[0], step)
        ]
        return torch.cat(chunks).reshape(*self.w_q.shape[:-2], self.size_n, self.size_k)


_NVFP4_GROUP_SIZE = 16


class NvFp4Comparable(ComparableWeight):
    """ModelOpt NVFP4 experts in checkpoint layout: e2m1 pairs `(E, N, K/2)` uint8, low
    nibble first; e4m3 block scales `(E, N, K/16)`; fp32 global scales `(E,)`, or
    `(E, S)` with one column per equal row block (gate/up halves of a fused w13).

    Compared in dequantized space, so a re-quantization with different but equivalent
    scales compares equal, and one that rounds differently stays within ULP tolerance.
    The Marlin repack derived from these tensors is excluded from compare and checksum."""

    SCALE_SUFFIX = "_weight_scale"
    GLOBAL_SCALE_SUFFIX = "_weight_scale_2"
    # Marlin repack (prepare_moe_nvfp4_layer_for_marlin) of each checkpoint-layout stack
    _MARLIN_REPACK_PREFIX = {"gate_up_proj": "w13", "down_proj": "w2"}

    def __init__(
        self,
        w_q: torch.Tensor,
        w_s: torch.Tensor,
        w_s2: torch.Tensor,
        is_shuffled: bool = False,
    ):
        # the checker passes every quantized weight's AITER shuffle flag; NVFP4 never shuffles
        assert not is_shuffled, "NVFP4 checkpoint-layout experts are never shuffled"
        assert w_q.dtype == torch.uint8, f"expected packed e2m1 bytes, got {w_q.dtype}"
        self.w_q = w_q
        self.w_s = w_s
        self.w_s2 = w_s2
        self.size_n = w_q.shape[-2]
        self.size_k = w_q.shape[-1] * 2
        assert tuple(w_s.shape) == (
            *w_q.shape[:-1],
            self.size_k // _NVFP4_GROUP_SIZE,
        ), (
            f"scale shape {tuple(w_s.shape)} does not match weight shape {tuple(w_q.shape)}"
        )
        lead = tuple(w_q.shape[:-2])
        self.num_shards = 1 if tuple(w_s2.shape) == lead else w_s2.shape[-1]
        assert tuple(w_s2.shape) in (lead, (*lead, self.num_shards)) and (
            self.size_n % self.num_shards == 0
        ), (
            f"global scale shape {tuple(w_s2.shape)} does not match weight shape "
            f"{tuple(w_q.shape)}"
        )

    def __repr__(self) -> str:
        return (
            f"nvfp4(shape={tuple(self.w_q.shape)} dtype={self.w_q.dtype} "
            f"shards={self.num_shards})"
        )

    @classmethod
    def match_quantized_weights(
        cls, param_names: Collection[str]
    ) -> Iterator[QuantizedNames]:
        for name in param_names:
            scale = name + cls.SCALE_SUFFIX
            global_scale = name + cls.GLOBAL_SCALE_SUFFIX
            if scale not in param_names or global_scale not in param_names:
                continue
            prefix = cls._MARLIN_REPACK_PREFIX.get(name)
            derived = (
                ()
                if prefix is None
                else (
                    f"{prefix}_weight",
                    f"{prefix}_weight_scale",
                    f"{prefix}_weight_scale_2",
                )
            )
            yield QuantizedNames(
                name,
                scale,
                global_scale,
                tuple(n for n in derived if n in param_names),
            )

    def _dequantize_experts(self, w_q, w_s, w_s2):
        """`(e, N, K/2)` uint8 + `(e, N, K/16)` e4m3 + `(e[, S])` fp32 -> dequantized and
        ULP tensors `(e, N, K)` in fp32."""
        num_experts = w_q.shape[0]
        nibbles = torch.stack((w_q & 0xF, w_q >> 4), dim=-1)
        magnitude = (nibbles & 0x7).long()
        sign = 1.0 - 2.0 * ((nibbles >> 3) & 0x1).to(torch.float32)
        values = torch.tensor(_E2M1_VALUES, device=w_q.device)[magnitude] * sign
        spacing = torch.tensor(_E2M1_SPACING, device=w_q.device)[magnitude]
        grouped = (
            num_experts,
            self.num_shards,
            self.size_n // self.num_shards,
            self.size_k // _NVFP4_GROUP_SIZE,
            _NVFP4_GROUP_SIZE,
        )
        scale = w_s.to(torch.float32).reshape(*grouped[:-1], 1) * w_s2.to(
            torch.float32
        ).reshape(num_experts, self.num_shards, 1, 1, 1)
        shape = (num_experts, self.size_n, self.size_k)
        # +0.0 folds the -0 nibble into +0 so equal values hash equal
        dequantized = (values.reshape(grouped) * scale).reshape(shape) + 0.0
        ulp = (spacing.reshape(grouped) * scale.abs()).reshape(shape)
        return dequantized, ulp

    def _flat_experts(self):
        return (
            self.w_q.reshape(-1, *self.w_q.shape[-2:]),
            self.w_s.reshape(-1, *self.w_s.shape[-2:]),
            self.w_s2.reshape(-1, self.num_shards),
        )

    def _experts_per_chunk(self) -> int:
        # the unpack holds several int64/float32 copies of every element
        return max(1, CHUNK_NUMEL // (4 * self.size_n * self.size_k))

    def iter_chunks(self):
        q, s, s2 = self._flat_experts()
        step = self._experts_per_chunk()
        for e0 in range(0, q.shape[0], step):
            yield self._dequantize_experts(
                q[e0 : e0 + step].cuda(),
                s[e0 : e0 + step].cuda(),
                s2[e0 : e0 + step].cuda(),
            )

    def dequantize(self, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        q, s, s2 = self._flat_experts()
        step = self._experts_per_chunk()
        chunks = [
            self._dequantize_experts(
                q[e0 : e0 + step], s[e0 : e0 + step], s2[e0 : e0 + step]
            )[0].to(dtype)
            for e0 in range(0, q.shape[0], step)
        ]
        return torch.cat(chunks).reshape(*self.w_q.shape[:-1], self.size_k)


class RawComparable(ComparableWeight):
    """Bitwise equal compare on raw tensor."""

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def __repr__(self) -> str:
        return f"raw(shape={tuple(self.tensor.shape)} dtype={self.tensor.dtype})"

    def iter_chunks(self):
        flat = self.tensor.reshape(-1)
        for start in range(0, flat.numel(), CHUNK_NUMEL):
            yield flat[start : start + CHUNK_NUMEL].cuda(), None

    def dequantize(self, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        return self.tensor


def compare_weights(
    expect: ComparableWeight, actual: ComparableWeight
) -> CompareResult:
    """Chunked element-wise compare in ComparableWeight space."""
    equal = True
    max_abs_err = torch.zeros((), dtype=torch.float32)
    sum_abs_err = 0.0
    num_exceed = 0
    numel = 0
    for (expect_dq, expect_tol), (actual_dq, actual_tol) in zip(
        expect.iter_chunks(), actual.iter_chunks(), strict=True
    ):
        assert expect_dq.shape == actual_dq.shape, (
            f"{expect_dq.shape=} {actual_dq.shape=}"
        )
        numel += expect_dq.numel()
        abs_diff = (actual_dq.float() - expect_dq.float()).abs()
        if torch.all(abs_diff == 0):
            continue
        equal = False
        # |actual_dq - expect_dq| ≤ |actual_dq - w| + |expect_dq - w| ≤ actual_tol + expect_tol
        tol = (
            0.0 if expect_tol is None or actual_tol is None else expect_tol + actual_tol
        )
        max_abs_err = torch.maximum(max_abs_err, abs_diff.max().cpu())
        sum_abs_err += abs_diff.sum().item()
        # `~(diff <= tol)` instead of `diff > tol` so NaN counts as exceeding.
        num_exceed += int((~(abs_diff <= tol)).sum())
    return CompareResult(
        equal, max_abs_err.item(), sum_abs_err / max(numel, 1), num_exceed
    )


def select_comparable_weight(quant_method) -> Optional[type]:
    """Map a module's quant_method to its ComparableWeight. None means raw (bitwise equal) compare."""
    if (
        isinstance(quant_method, (Fp8LinearMethod, Fp8MoEMethod))
        and quant_method.block_quant
        and not quant_method.use_mxfp8
    ):
        return Fp8BlockComparable
    if (
        isinstance(quant_method, MuseSparkNvFp4FusedMoEMethod)
        and quant_method.uses_marlin()
    ):
        if quant_method.keep_checkpoint_layout:
            return NvFp4Comparable
        raise NotImplementedError(
            "weight checker needs the checkpoint-layout Muse Spark NVFP4 experts on "
            "Marlin; restart with SGLANG_ENABLE_MUSE_SPARK_NVFP4_WEIGHT_UPDATE=1"
        )
    if isinstance(quant_method, ModelOptNvFp4FusedMoEMethod):
        if getattr(quant_method, "enable_flashinfer_trtllm_moe", False):
            return None
        raise NotImplementedError(
            f"weight checker has no ComparableWeight for {type(quant_method).__name__}"
        )
    if isinstance(quant_method, ModelOptFp4LinearMethod):
        raise NotImplementedError(
            f"weight checker has no ComparableWeight for {type(quant_method).__name__}"
        )
    if isinstance(quant_method, Mxfp4MoEMethod):
        if quant_method.use_marlin:
            return Mxfp4MarlinComparable
        raise NotImplementedError(
            "weight checker has no ComparableWeight for Mxfp4MoEMethod outside the Marlin backend"
        )
    return None
