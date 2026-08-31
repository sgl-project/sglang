"""Multimodal support for the embedded Rust server."""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, Optional, Tuple

import msgspec

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_mm, get_parallel, get_serving

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class RustMmSpec(msgspec.Struct, frozen=True, kw_only=True):
    """Resolved parameters of the Rust MM pipeline for one model,
    consumed by the Rust worker pool (as the typed extension ``MmSpec``, see
    :meth:`RustServer._build_mm_spec`), the ``_multimodal`` parity API
    (:meth:`rust_json`) and the drain adapter
    (:meth:`RustMmProcessor.build_output`)."""

    family: str
    feature_shm: bool
    image_token_id: int
    patch_size: int
    merge_size: int
    temporal_patch_size: int
    min_pixels: int
    max_pixels: int
    image_mean: Tuple[float, ...]
    image_std: Tuple[float, ...]
    # Which HF processor the Rust resize must reproduce bit-exactly.
    resample: str
    vision_start_token_id: Optional[int]
    vision_end_token_id: Optional[int]
    video_token_id: Optional[int]

    # Used by the drain adapter only; every other field goes to Rust.
    DRAIN_ONLY = ("vision_start_token_id", "vision_end_token_id", "video_token_id")

    @property
    def feature_dim(self) -> int:
        return 3 * self.temporal_patch_size * self.patch_size * self.patch_size

    def rust_json(self) -> str:
        """The subset `sglang_mm::registry::pipeline_from_spec` parses — the
        JSON form the ``_multimodal`` parity API takes; the server itself is
        handed the typed ``MmSpec`` instead."""
        fields = (f for f in self.__struct_fields__ if f not in self.DRAIN_ONLY)
        return msgspec.json.encode({f: getattr(self, f) for f in fields}).decode()


class RustMmFamily(msgspec.Struct, frozen=True, kw_only=True):
    """The Python half of one Rust MM family (an arm of
    `sglang_mm::registry::pipeline_from_spec`): which models it serves.
    Supporting a new model family = one entry in :data:`RUST_MM_FAMILIES`
    plus its Rust arm — the launch gate is data-driven."""

    name: str
    # The registered Python MM processor the Rust pipeline replaces, as
    # "module:Class". Compared by identity, so an
    # SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE override still disables the Rust path.
    mm_processor: str
    # Model types whose image-only M-RoPE matches the family's fast path.
    model_types: FrozenSet[str]
    # HF image processors the Rust resize reproduces bit-exactly, each mapped
    # to the `resample` the Rust pipeline must use (see `RustMmSpec.resample`).
    image_processors: Dict[str, str]

    def serves(self, mm_processor_cls: Any, model_type: Optional[str]) -> bool:
        module_name, _, class_name = self.mm_processor.partition(":")
        cls = getattr(importlib.import_module(module_name), class_name)
        return mm_processor_cls is cls and model_type in self.model_types


RUST_MM_FAMILIES: Tuple[RustMmFamily, ...] = (
    RustMmFamily(
        name="qwen_vl",
        mm_processor="sglang.srt.multimodal.processors.qwen_vl:QwenVLImageProcessor",
        model_types=frozenset(
            (
                "qwen2_vl",
                "qwen2_5_vl",
                "qwen3_vl",
                "qwen3_vl_moe",
                "qwen3_5",
                "qwen3_5_moe",
            )
        ),
        image_processors={
            "Qwen2VLImageProcessor": "aten_u8",
            "Qwen2VLImageProcessorFast": "aten_u8",
            "Qwen2VLImageProcessorPil": "pil",
        },
    ),
)


def rust_mm_family_for(
    mm_processor_cls: Any, model_type: Optional[str]
) -> Optional[RustMmFamily]:
    """The declared family serving this model, or ``None`` — which
    :meth:`RustServer.launch` turns into a hard error (no Python fallback)."""
    return next(
        (f for f in RUST_MM_FAMILIES if f.serves(mm_processor_cls, model_type)), None
    )


class RustMmProcessor:
    """Builds and validates the Rust MM pipeline for one model.

    Construction registers the same ``mm_processor`` mapping the Python
    TokenizerManager would build — not to process requests (the Rust worker pool
    does that, GIL-free) but as the source of truth
    :meth:`resolve_spec` resolves the pipeline parameters from. At drain
    time :meth:`build_output` wraps the Rust-produced buffers into the
    scheduler's ``MultimodalProcessorOutput``.

    There is no Python fallback: a model without a Rust MM spec fails at launch,
    and inputs outside the pipeline's scope are rejected per request.
    """

    # Rust mm-worker threads when --mm-processor-worker-num is 0. They are
    # GIL-free, so unlike the Python processor pool more than one always helps.
    AUTO_MM_WORKERS = 8

    def __init__(
        self,
        *,
        server_args: ServerArgs,
        model_config: ModelConfig,
        processor: Any = None,
    ):
        # Lazy: this class exists only for multimodal models under
        # SGLANG_RUST_SERVER.
        from sglang.srt.managers.multimodal_processor import import_processors
        from sglang.srt.managers.tokenizer_manager import get_processor_wrapper

        self.server_args = server_args
        self.model_config = model_config
        # Worker threads == max concurrently-processed mm requests.
        self.mm_workers = get_mm().mm_processor_worker_num or self.AUTO_MM_WORKERS

        # The mapping the Python TokenizerManager builds in
        # init_tokenizer_and_processor. The caller's already-loaded HF
        # AutoProcessor is reused when available (identical construction args).
        import_processors("sglang.srt.multimodal.processors")
        if mm_process_pkg := envs.SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE.get():
            import_processors(mm_process_pkg, overwrite=True)
        self._processor = processor or get_processor_wrapper()

    def resolve_spec(self) -> Optional[RustMmSpec]:
        """The :class:`RustMmSpec` for this model, or ``None`` when it has no
        Rust pipeline (the launch gate turns that into a hard error).

        Carries only resolved settings — patch geometry, pixel limits,
        normalization, token ids — never the HF config, and is conservative by
        design: an unrecognized knob disables the Rust path rather than being
        approximated."""
        from sglang.srt.managers.multimodal_processor import get_mm_processor_cls

        hf_config = self.model_config.hf_config
        mm_processor_cls = get_mm_processor_cls(
            hf_config, self.server_args, model_config=self.model_config
        )
        family = rust_mm_family_for(
            mm_processor_cls, getattr(hf_config, "model_type", None)
        )
        if family is None:
            return None
        ip = getattr(self._processor, "image_processor", None)
        resample = family.image_processors.get(type(ip).__name__)
        if resample is None:
            return None
        # The Rust pipeline always resizes, rescales by 1/255 and normalizes;
        # Rust's fused normalize constants assume that factor. Anything else
        # would silently produce different features.
        stages = ("do_resize", "do_rescale", "do_normalize")
        if not all(getattr(ip, stage, True) for stage in stages):
            return None
        if getattr(ip, "rescale_factor", None) != 1 / 255:
            return None

        # `--mm-process-config {"image": {...}}`: only pixel-limit overrides are
        # mirrored by Rust; anything else disables the pipeline.
        image_overrides = dict((get_mm().mm_process_config or {}).get("image", {}))
        if not set(image_overrides) <= {"min_pixels", "max_pixels"}:
            return None

        size = getattr(ip, "size", None) or {}
        min_pixels = image_overrides.get(
            "min_pixels", getattr(ip, "min_pixels", None) or size.get("shortest_edge")
        )
        max_pixels = image_overrides.get(
            "max_pixels", getattr(ip, "max_pixels", None) or size.get("longest_edge")
        )
        try:
            spec = RustMmSpec(
                family=family.name,
                feature_shm=self._use_feature_shm(),
                image_token_id=hf_config.image_token_id,
                patch_size=ip.patch_size,
                merge_size=ip.merge_size,
                temporal_patch_size=ip.temporal_patch_size,
                min_pixels=int(min_pixels),
                max_pixels=int(max_pixels),
                image_mean=tuple(float(x) for x in ip.image_mean),
                image_std=tuple(float(x) for x in ip.image_std),
                resample=resample,
                vision_start_token_id=getattr(hf_config, "vision_start_token_id", None),
                vision_end_token_id=getattr(hf_config, "vision_end_token_id", None),
                video_token_id=getattr(hf_config, "video_token_id", None),
            )
        except (AttributeError, TypeError):  # missing/odd processor attrs
            return None
        logger.info("rust server: Rust MM pipeline enabled (family=%s)", family.name)
        return spec

    def _use_feature_shm(self) -> bool:
        """Whether to park feature buffers in POSIX shm rather than inline.

        On exactly when the drained request is broadcast across TP ranks *and*
        the receiver's ``unwrap_shm_features`` will materialize the stubs (its
        gates: non-default tensor transport, no ``skip_tokenizer_init``).

        Inline, the whole ~20 MB/image buffer rides ``broadcast_pyobj`` serially
        on the scheduler loop, so ranks 1..n start the TP-sharded ViT ~30 ms
        after rank 0 and every rank then stalls that long at the first
        collective. With shm the broadcast carries a ~100-byte stub and all ranks
        map in parallel — the transport the Python TokenizerManager already uses.
        Single-rank serving stays inline, where shm would only add a copy.
        """
        from sglang.srt.multimodal.transport import (
            determine_tensor_transport_mode,
        )

        return (
            get_parallel().tp_size > 1
            and determine_tensor_transport_mode() != "default"
            and not get_serving().skip_tokenizer_init
        )

    @staticmethod
    def build_output(spec: RustMmSpec, entry):
        """Drain-time adapter: wrap the Rust-produced buffers of one ``MmEncodeResult``
        into the scheduler's ``MultimodalProcessorOutput``. Wrapping only — load,
        resize, patchify, token expansion and M-RoPE all ran in Rust.

        Runs on the scheduler loop, so it must stay copy-free *and* hash-free:
        ``take_mm_result``'s numpy arrays own the Rust buffers, ``torch.from_numpy`` just
        views them, and each item's ``hash`` is worker-precomputed so
        ``set_pad_value`` skips ``hash_feature``. Any per-byte work here — memcpy,
        sha256, tens of MB per image-heavy request — measurably inflates every
        running request's inter-token latency."""
        import torch

        from sglang.srt.managers.mm_utils import ShmPointerMMData
        from sglang.srt.managers.schedule_batch import (
            Modality,
            MultimodalDataItem,
            MultimodalProcessorOutput,
        )

        shm_names = entry.shm_names
        if shm_names is None:
            features = torch.from_numpy(entry.features.reshape(-1, spec.feature_dim))
        items = []
        row = 0
        for index, ((t, h, w), item_hash, offset) in enumerate(
            zip(entry.grids, entry.hashes, entry.offsets)
        ):
            n = t * h * w
            if shm_names is None:
                feature = features[row : row + n]
            else:
                # The worker parked this item's buffer in a named POSIX
                # segment (see `_use_feature_shm`). Build the stub in its
                # post-`__setstate__` form: rank 0 never pickle-roundtrips its
                # own copy, and `materialize()` needs the mapped view.
                # Ownership of the unlink moved here with `take_mm_result`.
                feature = ShmPointerMMData.__new__(ShmPointerMMData)
                feature.__setstate__(
                    {
                        "shm_name": shm_names[index],
                        "shape": (n, spec.feature_dim),
                        "dtype": torch.float32,
                        "precomputed_hash": item_hash,
                    }
                )
            items.append(
                MultimodalDataItem(
                    modality=Modality.IMAGE,
                    feature=feature,
                    hash=item_hash,
                    offsets=[tuple(offset)],
                    model_specific_data={
                        "image_grid_thw": torch.tensor([[t, h, w]], dtype=torch.long)
                    },
                )
            )
            row += n
        if envs.SGLANG_MM_PRECOMPUTE_HASH.get():
            for item in items:
                item.set_pad_value()
        return MultimodalProcessorOutput(
            mm_items=items,
            im_token_id=spec.image_token_id,
            im_start_id=spec.vision_start_token_id,
            im_end_id=spec.vision_end_token_id,
            video_token_id=spec.video_token_id,
            mrope_positions=torch.from_numpy(entry.mrope.reshape(3, -1)),
            mrope_position_delta=torch.tensor([[entry.mrope_delta]], dtype=torch.long),
        )
