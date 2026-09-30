from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

import torch


logger = logging.getLogger(__name__)


KVTC_FILE_VERSION = "v3-worker-quant"


KVTC_QUANT_STORAGE_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "int4": torch.int32,
}
KVTC_QUANT_PRECISION_BITS = {
    "float32": 32,
    "bfloat16": 16,
    "int8": 8,
    "int4": 4,
}
KVTC_QUANT_METADATA_DTYPE = torch.float16
KVTC_QUANTIZED_DTYPES = frozenset(("int8", "int4"))


@dataclass(frozen=True)
class KVTCQuantGroup:
    feature_start: int
    feature_end: int
    dtype_name: str
    payload_start: int
    payload_end: int
    metadata_index: int | None


@dataclass(frozen=True)
class KVTCQuantGroupedLayout:
    direct_storage_groups: dict[str, tuple[KVTCQuantGroup, ...]]
    integer_quant_groups: dict[str, tuple[KVTCQuantGroup, ...]]
    feature_count: int
    payload_elements: dict[str, int]
    metadata_count: int
    bytes_per_token: int

    @property
    def group_count(self) -> int:
        return sum(
            len(groups)
            for groups_by_dtype in (
                self.direct_storage_groups,
                self.integer_quant_groups,
            )
            for groups in groups_by_dtype.values()
        )


@dataclass(frozen=True)
class KVTCLoadedSide:
    mu: torch.Tensor
    basis: torch.Tensor
    schema: object | None
    source_basis_rank: int


def quant_group_bits(group_size: int, dtype_name: str) -> int:
    """Return per-token storage, including one FP16 scale/offset pair."""
    bits = group_size * KVTC_QUANT_PRECISION_BITS[dtype_name]
    if dtype_name in KVTC_QUANTIZED_DTYPES:
        bits += 2 * KVTC_QUANT_METADATA_DTYPE.itemsize * 8
    return bits


@dataclass(frozen=True)
class _KVTCQuantSide:
    layout: KVTCQuantGroupedLayout
    staging: dict[str, torch.Tensor]
    dequant_indices: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


class KVTCQuantizer:
    """Quantize projected K/V pages into caller-owned host buffers.

    The K/V schemas and staging capacity are fixed at construction. Calls that
    share one side's staging must be serialized on the caller's device stream.
    """

    def __init__(
        self,
        *,
        keys_schema: object | None,
        values_schema: object | None,
        keys_basis_rank: int | None,
        values_basis_rank: int | None,
        artifact_path: str,
        page_size: int,
        device: torch.device | str,
        cache_dtype: torch.dtype,
        staging_capacity_pages: int,
    ) -> None:
        if keys_schema is None and values_schema is None:
            raise ValueError("KVTC quantizer requires a keys or values schema")
        if page_size <= 0 or staging_capacity_pages <= 0:
            raise ValueError("KVTC page size and staging capacity must be positive")

        self.page_size = page_size
        self.device = device
        self.cache_dtype = cache_dtype
        self.staging_capacity_pages = staging_capacity_pages
        self._npu_ops = None

        self._keys = self._initialize_side("keys", keys_schema, keys_basis_rank)
        self._values = self._initialize_side("values", values_schema, values_basis_rank)

        logger.info(
            "KVTC quantizer artifact=%s staging_capacity_pages=%d enabled=%s",
            artifact_path,
            staging_capacity_pages,
            [
                name
                for name, side in (("keys", self._keys), ("values", self._values))
                if side is not None
            ],
        )
        for name, side, schema, rank in (
            ("keys", self._keys, keys_schema, keys_basis_rank),
            ("values", self._values, values_schema, values_basis_rank),
        ):
            if side is None:
                continue
            layout = side.layout
            direct_group_counts = {
                dtype: len(groups)
                for dtype, groups in layout.direct_storage_groups.items()
            }
            integer_group_counts = {
                dtype: len(groups)
                for dtype, groups in layout.integer_quant_groups.items()
            }
            logger.info(
                "KVTC quantizer %s basis_rank=%d retained=%d groups=%d "
                "direct_storage_groups=%s integer_quant_groups=%s "
                "bytes_per_token=%d",
                name,
                rank,
                layout.feature_count,
                sum(direct_group_counts.values()) + sum(integer_group_counts.values()),
                direct_group_counts,
                integer_group_counts,
                layout.bytes_per_token,
            )
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "KVTC quantizer %s schema=%s layout=%s",
                    name,
                    schema,
                    layout,
                )

    @staticmethod
    def build_layout(
        schema: object, *, page_size: int, basis_rank: int, matrix_name: str
    ) -> KVTCQuantGroupedLayout:
        """Validate a schema and group storage by dtype without allocating buffers."""
        if not isinstance(schema, (list, tuple)) or not schema:
            raise ValueError(
                f"{matrix_name} KVTC quantization schema must be a non-empty list"
            )

        direct_storage_groups = {
            name: []
            for name in KVTC_QUANT_STORAGE_DTYPES
            if name not in KVTC_QUANTIZED_DTYPES
        }
        integer_quant_groups = {
            name: []
            for name in KVTC_QUANT_STORAGE_DTYPES
            if name in KVTC_QUANTIZED_DTYPES
        }
        feature_offset = 0
        metadata_count = 0
        used_bits = 0
        payload_offsets = {name: 0 for name in KVTC_QUANT_STORAGE_DTYPES}
        for group_index, entry in enumerate(schema):
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                raise ValueError(
                    f"{matrix_name} KVTC quantization entry {group_index} "
                    "must be (group_size, dtype)"
                )

            group_size, dtype_name = entry
            if (
                isinstance(group_size, bool)
                or not isinstance(group_size, int)
                or group_size <= 0
            ):
                raise ValueError(
                    f"{matrix_name} KVTC quantization group {group_index} "
                    f"has invalid size {group_size!r}"
                )
            if dtype_name not in KVTC_QUANT_STORAGE_DTYPES:
                raise ValueError(
                    f"{matrix_name} KVTC quantization group {group_index} "
                    f"has unsupported dtype {dtype_name!r}"
                )
            if dtype_name == "int4":
                if group_size < 8 or group_size % 8 != 0:
                    raise ValueError(
                        f"{matrix_name} KVTC int4 group {group_index} "
                        f"has size {group_size}; "
                        "packed int4 requires a group size of at least 8 "
                        "and a multiple of 8"
                    )
                payload_elements = page_size * group_size // 8
            else:
                payload_elements = page_size * group_size

            metadata_index = None
            if dtype_name in KVTC_QUANTIZED_DTYPES:
                metadata_index = metadata_count
                metadata_count += 1

            payload_start = payload_offsets[dtype_name]
            payload_end = payload_start + payload_elements
            groups = (
                integer_quant_groups
                if dtype_name in KVTC_QUANTIZED_DTYPES
                else direct_storage_groups
            )
            groups[dtype_name].append(
                KVTCQuantGroup(
                    feature_start=feature_offset,
                    feature_end=feature_offset + group_size,
                    dtype_name=dtype_name,
                    payload_start=payload_start,
                    payload_end=payload_end,
                    metadata_index=metadata_index,
                )
            )
            feature_offset += group_size
            payload_offsets[dtype_name] = payload_end
            used_bits += quant_group_bits(group_size, dtype_name)

        if feature_offset > basis_rank:
            raise ValueError(
                f"{matrix_name} KVTC quantization schema retains "
                f"{feature_offset} features, "
                f"but basis rank is only {basis_rank}"
            )
        if used_bits % 8:
            raise ValueError("KVTC quantized token size is not byte-aligned")

        return KVTCQuantGroupedLayout(
            direct_storage_groups={
                name: tuple(groups)
                for name, groups in direct_storage_groups.items()
                if groups
            },
            integer_quant_groups={
                name: tuple(groups)
                for name, groups in integer_quant_groups.items()
                if groups
            },
            feature_count=feature_offset,
            payload_elements={
                name: count for name, count in payload_offsets.items() if count
            },
            metadata_count=metadata_count,
            bytes_per_token=used_bits // 8,
        )

    def _ensure_npu_ops(self) -> None:
        if self._npu_ops is not None:
            return
        if self.cache_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                "KVTC integer quantization requires an FP16 or BF16 cache, "
                f"got {self.cache_dtype}"
            )
        try:
            npu_ops = import_module("torch_npu")
        except ImportError as error:
            raise RuntimeError(
                "KVTC integer quantization requires torch_npu"
            ) from error
        missing = [
            name
            for name in ("npu_dynamic_quant_asymmetric", "npu_anti_quant")
            if not hasattr(npu_ops, name)
        ]
        if missing:
            raise RuntimeError(
                "KVTC integer quantization requires torch_npu APIs: "
                + ", ".join(missing)
            )
        self._npu_ops = npu_ops

    def _initialize_side(
        self, name: str, schema: object | None, basis_rank: int | None
    ) -> _KVTCQuantSide | None:
        if schema is None:
            if basis_rank is not None:
                raise ValueError(f"KVTC {name} basis rank requires a schema")
            return None
        if basis_rank is None:
            raise ValueError(f"KVTC {name} schema requires a basis rank")
        layout = self.build_layout(
            schema, page_size=self.page_size, basis_rank=basis_rank, matrix_name=name
        )
        if layout.metadata_count:
            self._ensure_npu_ops()
        staging = {
            dtype_name: torch.empty(
                (self.staging_capacity_pages, element_count),
                dtype=KVTC_QUANT_STORAGE_DTYPES[dtype_name],
                device=self.device,
            )
            for dtype_name, element_count in layout.payload_elements.items()
        }
        dequant_indices = self._build_dequant_indices(layout)
        return _KVTCQuantSide(
            layout=layout,
            staging=staging,
            dequant_indices=dequant_indices,
        )

    def _build_dequant_indices(
        self, layout: KVTCQuantGroupedLayout
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        indices_by_dtype = {}
        for dtype_name, groups in layout.integer_quant_groups.items():
            payload_indices = []
            for token_index in range(self.page_size):
                for group in groups:
                    packed_width = (
                        group.payload_end - group.payload_start
                    ) // self.page_size
                    token_start = group.payload_start + token_index * packed_width
                    payload_indices.extend(
                        range(token_start, token_start + packed_width)
                    )
            feature_indices = []
            metadata_indices = []
            for group in groups:
                group_size = group.feature_end - group.feature_start
                feature_indices.extend(range(group.feature_start, group.feature_end))
                metadata_indices.extend([group.metadata_index] * group_size)
            indices_by_dtype[dtype_name] = (
                torch.tensor(payload_indices, dtype=torch.long, device=self.device),
                torch.tensor(metadata_indices, dtype=torch.long, device=self.device),
                torch.tensor(feature_indices, dtype=torch.long, device=self.device),
            )
        return indices_by_dtype

    @staticmethod
    def _require_side(side: _KVTCQuantSide | None, name: str) -> _KVTCQuantSide:
        if side is None:
            raise RuntimeError(f"KVTC {name} quantization is not initialized")
        return side

    def key_bytes_per_token(self) -> int:
        return self._require_side(self._keys, "keys").layout.bytes_per_token

    def value_bytes_per_token(self) -> int:
        return self._require_side(self._values, "values").layout.bytes_per_token

    def key_layout(self) -> KVTCQuantGroupedLayout:
        return self._require_side(self._keys, "keys").layout

    def value_layout(self) -> KVTCQuantGroupedLayout:
        return self._require_side(self._values, "values").layout

    def quantize_pages_keys(
        self,
        pages: torch.Tensor,
        host_pages: torch.Tensor,
        payload_buffers: dict[str, torch.Tensor],
        scales: torch.Tensor,
        offsets: torch.Tensor,
    ) -> None:
        self._quantize_pages(
            self._keys, "keys", pages, host_pages, payload_buffers, scales, offsets
        )

    def quantize_pages_values(
        self,
        pages: torch.Tensor,
        host_pages: torch.Tensor,
        payload_buffers: dict[str, torch.Tensor],
        scales: torch.Tensor,
        offsets: torch.Tensor,
    ) -> None:
        self._quantize_pages(
            self._values, "values", pages, host_pages, payload_buffers, scales, offsets
        )

    def _quantize_pages(
        self,
        side: _KVTCQuantSide | None,
        name: str,
        pages: torch.Tensor,
        host_pages: torch.Tensor,
        payload_buffers: dict[str, torch.Tensor],
        scales: torch.Tensor,
        offsets: torch.Tensor,
    ) -> None:
        side = self._require_side(side, name)
        self._validate_pages(pages, host_pages, side.layout)
        num_pages = pages.shape[0]
        if num_pages == 0:
            return
        layout = side.layout
        flat = pages.flatten(0, 1)

        for dtype_name, groups in layout.direct_storage_groups.items():
            chunks = [
                flat[:, group.feature_start : group.feature_end]
                .reshape(num_pages, -1)
                .to(KVTC_QUANT_STORAGE_DTYPES[dtype_name])
                for group in groups
            ]
            host_payload = torch.cat(chunks, dim=1).to(device="cpu")
            payload_buffers[dtype_name].index_copy_(0, host_pages, host_payload)

        if layout.integer_quant_groups:
            assert self._npu_ops is not None
            cast = flat.to(self.cache_dtype)
            group_scales = [None] * layout.metadata_count
            group_offsets = [None] * layout.metadata_count
            for dtype_name, groups in layout.integer_quant_groups.items():
                chunks = []
                dst_type = torch.quint4x2 if dtype_name == "int4" else torch.int8
                for group in groups:
                    values = cast[:, group.feature_start : group.feature_end]
                    quantized, scale, quant_offset = (
                        self._npu_ops.npu_dynamic_quant_asymmetric(
                            values, dst_type=dst_type
                        )
                    )
                    chunks.append(quantized.reshape(num_pages, -1))
                    metadata_index = group.metadata_index
                    assert metadata_index is not None
                    group_scales[metadata_index] = scale.reshape(
                        num_pages, self.page_size
                    )
                    # npu_anti_quant reconstructs (q + offset) * scale.
                    group_offsets[metadata_index] = (-quant_offset).reshape(
                        num_pages, self.page_size
                    )
                host_payload = torch.cat(chunks, dim=1).to(device="cpu")
                payload_buffers[dtype_name].index_copy_(0, host_pages, host_payload)

            host_scales = torch.stack(group_scales, dim=2).to(
                device="cpu", dtype=KVTC_QUANT_METADATA_DTYPE
            )
            host_offsets = torch.stack(group_offsets, dim=2).to(
                device="cpu", dtype=KVTC_QUANT_METADATA_DTYPE
            )
            scales.index_copy_(0, host_pages, host_scales)
            offsets.index_copy_(0, host_pages, host_offsets)
        self._log_batch("quantized", name, host_pages, layout.feature_count)

    def dequantize_pages_keys(
        self,
        host_pages: torch.Tensor,
        payload_buffers: dict[str, torch.Tensor],
        scales: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        return self._dequantize_pages(
            self._keys, "keys", host_pages, payload_buffers, scales, offsets
        )

    def dequantize_pages_values(
        self,
        host_pages: torch.Tensor,
        payload_buffers: dict[str, torch.Tensor],
        scales: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        return self._dequantize_pages(
            self._values, "values", host_pages, payload_buffers, scales, offsets
        )

    def _dequantize_pages(
        self,
        side: _KVTCQuantSide | None,
        name: str,
        host_pages: torch.Tensor,
        payload_buffers: dict[str, torch.Tensor],
        scales: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        side = self._require_side(side, name)
        self._validate_host_pages(host_pages)
        num_pages = host_pages.numel()
        layout = side.layout
        if num_pages > self.staging_capacity_pages:
            raise ValueError(
                f"KVTC {name} dequantization received {num_pages} pages; "
                f"staging capacity is {self.staging_capacity_pages}"
            )
        if num_pages == 0:
            return torch.empty(
                (0, self.page_size, layout.feature_count),
                dtype=torch.float32,
                device=self.device,
            )

        device_payloads = {}
        host_page_list = host_pages.tolist()
        for dtype_name, host_payload in payload_buffers.items():
            staged = side.staging[dtype_name][:num_pages]
            for dst_page, src_page in enumerate(host_page_list):
                staged[dst_page].copy_(host_payload[src_page], non_blocking=True)
            device_payloads[dtype_name] = staged

        output = torch.empty(
            (num_pages, self.page_size, layout.feature_count),
            dtype=torch.float32,
            device=self.device,
        )
        for dtype_name, groups in layout.direct_storage_groups.items():
            for group in groups:
                payload = device_payloads[dtype_name][
                    :, group.payload_start : group.payload_end
                ]
                output[:, :, group.feature_start : group.feature_end] = payload.reshape(
                    num_pages,
                    self.page_size,
                    group.feature_end - group.feature_start,
                )

        if layout.metadata_count:
            host_scales = scales.index_select(0, host_pages)
            host_offsets = offsets.index_select(0, host_pages)
            device_scales = host_scales.to(
                device=self.device, dtype=torch.float32, non_blocking=True
            )
            device_offsets = host_offsets.to(
                device=self.device, dtype=torch.float32, non_blocking=True
            )
            for dtype_name, (
                payload_indices,
                metadata_indices,
                feature_indices,
            ) in side.dequant_indices.items():
                payload = device_payloads[dtype_name].index_select(
                    1, payload_indices
                ).reshape(1, -1)
                # Ascend packed-INT4 anti-quantization corrupts otherwise contiguous
                # views with nonzero storage offsets. Keep this guard after packing so
                # the exact tensor passed to npu_anti_quant is materialized if needed.
                if dtype_name == "int4" and payload.storage_offset() != 0:
                    payload = payload.clone()
                expanded_scales = device_scales.index_select(
                    2, metadata_indices
                ).reshape(-1)
                expanded_offsets = device_offsets.index_select(
                    2, metadata_indices
                ).reshape(-1)
                kwargs = {"offset": expanded_offsets, "dst_dtype": self.cache_dtype}
                if dtype_name == "int4" and hasattr(torch, "int4"):
                    kwargs["src_dtype"] = torch.quint4x2
                assert self._npu_ops is not None
                dequantized = self._npu_ops.npu_anti_quant(
                    payload, expanded_scales, **kwargs
                )
                output[:, :, feature_indices] = dequantized.to(
                    dtype=output.dtype
                ).reshape(num_pages, self.page_size, feature_indices.numel())

        self._log_batch("dequantized", name, host_pages, layout.feature_count)
        return output

    def _validate_host_pages(self, host_pages: torch.Tensor) -> None:
        if (
            host_pages.ndim != 1
            or host_pages.dtype != torch.int64
            or host_pages.device.type != "cpu"
        ):
            raise ValueError("KVTC host page IDs must be a CPU int64 vector")

    def _validate_pages(
        self,
        pages: torch.Tensor,
        host_pages: torch.Tensor,
        layout: KVTCQuantGroupedLayout,
    ) -> None:
        self._validate_host_pages(host_pages)
        if (
            pages.ndim != 3
            or pages.shape[0] != host_pages.numel()
            or pages.shape[1] != self.page_size
            or pages.shape[2] < layout.feature_count
        ):
            raise ValueError(
                "KVTC projected pages must have shape "
                f"[len(host_pages), {self.page_size}, >= {layout.feature_count}]"
            )

    @staticmethod
    def _log_batch(
        operation: str,
        name: str,
        host_pages: torch.Tensor,
        feature_count: int,
    ) -> None:
        if not logger.isEnabledFor(logging.DEBUG):
            return
        page_sample = host_pages[:8].tolist()
        logger.debug(
            "KVTC %s %s pages=%d host_page_ids=%s%s retained=%d",
            operation,
            name,
            host_pages.numel(),
            page_sample,
            "..." if host_pages.numel() > len(page_sample) else "",
            feature_count,
        )


# Calibration still imports this name; keep it as an alias, without a second
# implementation of schema validation or feature counting.
build_quant_layout = KVTCQuantizer.build_layout


class KVTCArtifactLoader:
    """Load and validate one worker's K/V calibration data on the CPU."""

    def __init__(
        self,
        artifact_path: str | Path,
        *,
        worker_key: str,
        p: int,
        page_size: int,
        k_cr: int | float,
        v_cr: int | float,
        quant_disable: bool,
    ) -> None:
        if isinstance(p, bool) or not isinstance(p, int) or p <= 0:
            raise ValueError(
                f"KVTC feature count p must be a positive integer, got {p!r}"
            )
        if (
            isinstance(page_size, bool)
            or not isinstance(page_size, int)
            or page_size <= 0
        ):
            raise ValueError(
                f"KVTC page size must be a positive integer, got {page_size!r}"
            )
        if not isinstance(worker_key, str) or not worker_key:
            raise ValueError(
                f"KVTC worker key must be a non-empty string, got {worker_key!r}"
            )

        k_ratio = self._validate_ratio(k_cr, "K")
        v_ratio = self._validate_ratio(v_cr, "V")
        path = Path(artifact_path)
        if not path.is_file():
            raise FileNotFoundError(f"KVTC artifact does not exist: {path}")
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(artifact, dict):
            raise ValueError(f"KVTC artifact must contain a dictionary: {path}")
        if artifact.get("version") != KVTC_FILE_VERSION:
            raise ValueError(
                f"KVTC artifact version mismatch in {path}: expected "
                f"{KVTC_FILE_VERSION!r}, found {artifact.get('version', '<missing>')!r}"
            )

        self.keys = self._load_side(
            artifact, "keys", "K", worker_key, p, page_size, k_ratio, quant_disable
        )
        self.values = self._load_side(
            artifact, "values", "V", worker_key, p, page_size, v_ratio, quant_disable
        )

    @staticmethod
    def _validate_ratio(ratio: int | float, side: str) -> int:
        if (
            isinstance(ratio, bool)
            or not isinstance(ratio, (int, float))
            or ratio < 0
            or (
                isinstance(ratio, float)
                and (not math.isfinite(ratio) or not ratio.is_integer())
            )
        ):
            raise ValueError(
                f"{side} KVTC compression ratio must be a nonnegative integer, "
                f"got {ratio!r}"
            )
        return int(ratio)

    @staticmethod
    def _load_side(
        artifact: dict,
        entry_name: str,
        side: str,
        worker_key: str,
        p: int,
        page_size: int,
        ratio: int,
        quant_disable: bool,
    ) -> KVTCLoadedSide | None:
        if ratio == 0:
            return None

        workers = artifact.get(entry_name)
        if not isinstance(workers, dict) or worker_key not in workers:
            raise ValueError(f"KVTC artifact is missing {entry_name}/{worker_key}")
        params = workers[worker_key]
        if not isinstance(params, dict):
            raise ValueError(
                f"KVTC artifact {entry_name}/{worker_key} must be a dictionary"
            )

        mu = params.get("mu")
        basis = params.get("basis")
        context = f"{side}/{worker_key}"
        if (
            not isinstance(mu, torch.Tensor)
            or mu.dtype != torch.float32
            or mu.shape != (p,)
        ):
            raise ValueError(
                f"KVTC artifact {context}/mu must be an FP32 tensor of shape [{p}]"
            )
        if (
            not isinstance(basis, torch.Tensor)
            or basis.dtype != torch.float32
            or basis.ndim != 2
            or basis.shape[0] != p
            or basis.shape[1] == 0
        ):
            raise ValueError(
                f"KVTC artifact {context}/basis must be an FP32 tensor "
                f"of shape [{p}, rank>0]"
            )

        basis_rank = basis.shape[1]
        if quant_disable:
            schema = None
            retained_rank = p // ratio
            if retained_rank == 0:
                raise ValueError(
                    f"KVTC artifact {context} compression ratio {ratio} "
                    "retains no features"
                )
        else:
            quant = params.get("quant")
            ratio_key = str(ratio)
            if not isinstance(quant, dict) or ratio_key not in quant:
                raise ValueError(
                    f"KVTC artifact {context} is missing quantization schema "
                    f"quant[{ratio_key!r}]"
                )
            schema = quant[ratio_key]
            retained_rank = KVTCQuantizer.build_layout(
                schema,
                page_size=page_size,
                basis_rank=basis_rank,
                matrix_name=context,
            ).feature_count
        if retained_rank > basis_rank:
            raise ValueError(
                f"KVTC artifact {context} basis rank {basis_rank} is shorter than "
                f"the retained rank {retained_rank}"
            )

        return KVTCLoadedSide(
            mu=mu,
            basis=basis[:, :retained_rank],
            schema=schema,
            source_basis_rank=basis_rank,
        )
