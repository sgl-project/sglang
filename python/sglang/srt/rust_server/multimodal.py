"""Multimodal support for the embedded Rust server."""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, FrozenSet, Optional, Tuple

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
    (:meth:`RustMmProcessor.wrap_encoded`)."""

    family: str
    feature_shm: bool
    # The family's fields of Dynamo's `ProcessorSpec`.
    pipeline: Dict[str, Any]
    # The rest are used by the drain adapter only.
    image_token_id: int
    vision_start_token_id: Optional[int] = None
    vision_end_token_id: Optional[int] = None
    video_token_id: Optional[int] = None

    def rust_json(self) -> str:
        """Dynamo's `ProcessorSpec` JSON, tagged by `family`."""
        return msgspec.json.encode({"family": self.family, **self.pipeline}).decode()


class RustMmFamily(msgspec.Struct, frozen=True, kw_only=True):
    """The Python half of one Rust MM family (an arm of Dynamo's
    `ProcessorSpec`): which models it serves and how their spec resolves.
    Supporting a new model family = one entry in :data:`RUST_MM_FAMILIES`
    plus its Rust arm — the launch gate is data-driven."""

    name: str
    # The registered Python MM processor the Rust pipeline replaces, as
    # "module:Class". Compared by identity, so an
    # SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE override still disables the Rust path.
    mm_processor: str
    model_types: FrozenSet[str]
    # `(processor, hf_config, feature_shm) -> RustMmSpec | None`; `None` when
    # the Rust pipeline does not reproduce this processor's settings.
    resolve: Callable[..., Optional[RustMmSpec]]

    def serves(self, mm_processor_cls: Any, model_type: Optional[str]) -> bool:
        module_name, _, class_name = self.mm_processor.partition(":")
        cls = getattr(importlib.import_module(module_name), class_name)
        return mm_processor_cls is cls and model_type in self.model_types


# HF image processors the Qwen-VL resize reproduces bit-exactly, each mapped
# to the `resample` the Rust pipeline must use.
_QWEN_VL_RESAMPLE = {
    "Qwen2VLImageProcessor": "aten_u8",
    "Qwen2VLImageProcessorFast": "aten_u8",
    "Qwen2VLImageProcessorPil": "pil",
}


def _rust_resample(image_processor, resamples: Dict[str, str]) -> Optional[str]:
    """The `resample` reproducing this HF image processor, or ``None``."""
    resample = resamples.get(type(image_processor).__name__)
    # The Rust pipeline always resizes, rescales by 1/255 and normalizes;
    # Rust's fused normalize constants assume that factor. Anything else
    # would silently produce different features.
    stages = ("do_resize", "do_rescale", "do_normalize")
    if not all(getattr(image_processor, stage, True) for stage in stages):
        return None
    if getattr(image_processor, "rescale_factor", None) != 1 / 255:
        return None
    return resample


def _image_overrides(allowed: FrozenSet[str]) -> Optional[Dict[str, Any]]:
    """`--mm-process-config {"image": {...}}`, or ``None`` when it sets a knob
    Rust does not mirror."""
    image_overrides = dict((get_mm().mm_process_config or {}).get("image", {}))
    return image_overrides if set(image_overrides) <= allowed else None


def _resolve_qwen_vl(*, processor, hf_config, feature_shm) -> Optional[RustMmSpec]:
    image_processor = getattr(processor, "image_processor", None)
    resample = _rust_resample(image_processor, _QWEN_VL_RESAMPLE)
    image_overrides = _image_overrides(frozenset(("min_pixels", "max_pixels")))
    if resample is None or image_overrides is None:
        return None

    size = getattr(image_processor, "size", None) or {}
    min_pixels = image_overrides.get(
        "min_pixels",
        getattr(image_processor, "min_pixels", None) or size.get("shortest_edge"),
    )
    max_pixels = image_overrides.get(
        "max_pixels",
        getattr(image_processor, "max_pixels", None) or size.get("longest_edge"),
    )
    try:
        pipeline = {
            "image_token_id": hf_config.image_token_id,
            "patch_size": image_processor.patch_size,
            "merge_size": image_processor.merge_size,
            "temporal_patch_size": image_processor.temporal_patch_size,
            "min_pixels": int(min_pixels),
            "max_pixels": int(max_pixels),
            "image_mean": [float(x) for x in image_processor.image_mean],
            "image_std": [float(x) for x in image_processor.image_std],
            "resample": resample,
        }
    except (AttributeError, TypeError):  # missing/odd processor attrs
        return None
    return RustMmSpec(
        family="qwen_vl",
        feature_shm=feature_shm,
        pipeline=pipeline,
        image_token_id=hf_config.image_token_id,
        vision_start_token_id=getattr(hf_config, "vision_start_token_id", None),
        vision_end_token_id=getattr(hf_config, "vision_end_token_id", None),
        video_token_id=getattr(hf_config, "video_token_id", None),
    )


_GLM5_NEXT_RESAMPLE = {
    "Glm5NextImageProcessor": "aten_u8",
    "Glm5NextImageProcessorPil": "pil",
}


def _resolve_glm5_next(*, processor, hf_config, feature_shm) -> Optional[RustMmSpec]:
    image_processor = getattr(processor, "image_processor", None)
    resample = _rust_resample(image_processor, _GLM5_NEXT_RESAMPLE)
    image_overrides = _image_overrides(
        frozenset(("min_image_tokens", "max_image_tokens"))
    )
    if resample is None or image_overrides is None:
        return None
    knobs = ("patch_size", "merge_size", "temporal_patch_size", "patch_expand_factor")
    tokens = ("min_image_tokens", "max_image_tokens")
    pipeline = {knob: getattr(image_processor, knob) for knob in knobs}
    pipeline.update(
        {
            name: image_overrides.get(name, getattr(image_processor, name))
            for name in tokens
        }
    )
    pipeline.update(
        image_token_id=hf_config.image_token_id,
        video_token_id=hf_config.video_token_id,
        image_mean=[float(x) for x in image_processor.image_mean],
        image_std=[float(x) for x in image_processor.image_std],
        resample=resample,
    )
    return RustMmSpec(
        family="glm5_next",
        feature_shm=feature_shm,
        pipeline=pipeline,
        image_token_id=hf_config.image_token_id,
        # `Glm4vImageProcessor` expands video with the image token too.
        video_token_id=hf_config.image_token_id,
    )


# deepseek-recipe-image resizes to `ImageTokenSpec::v41()` only, so Rust serves
# no other geometry; `deepseek_image.rs` checks the same values.
_RECIPE_V41_GEOMETRY = {
    "vision_patch_size": 14,
    "vision_downsample_ratio": 3,
    "vision_min_pixels": 544 * 544,
    "vision_max_n_token": 1024,
    "vision_max_wh_ratio": None,
}


def _resolve_deepseek_v41(*, processor, hf_config, feature_shm) -> Optional[RustMmSpec]:
    geometry = {name: getattr(hf_config, name, None) for name in _RECIPE_V41_GEOMETRY}
    if geometry != _RECIPE_V41_GEOMETRY:
        return None
    return RustMmSpec(
        family="deepseek_v41",
        feature_shm=feature_shm,
        pipeline={"image_token_id": hf_config.image_token_id, **geometry},
        image_token_id=hf_config.image_token_id,
    )


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
        resolve=_resolve_qwen_vl,
    ),
    RustMmFamily(
        name="glm5_next",
        mm_processor="sglang.srt.multimodal.processors.glm4v:Glm4vImageProcessor",
        model_types=frozenset(("glm5_next",)),
        resolve=_resolve_glm5_next,
    ),
    RustMmFamily(
        name="deepseek_v41",
        mm_processor="sglang.srt.multimodal.processors.deepseek_v41:DeepseekV41ImageProcessor",
        model_types=frozenset(("deepseek_v41",)),
        resolve=_resolve_deepseek_v41,
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


class _ModalityTable(dict):
    """``mm.meta`` modality names -> scheduler ``Modality``, resolved lazily so
    this module keeps its light import footprint."""

    def __missing__(self, name):
        from sglang.srt.managers.schedule_batch import Modality

        self.update(
            {"image": Modality.IMAGE, "video": Modality.VIDEO, "audio": Modality.AUDIO}
        )
        # An unknown name must raise here, not recurse.
        if name not in self:
            raise KeyError(name)
        return dict.__getitem__(self, name)


_MODALITIES = _ModalityTable()


def settle_shm_buffers(buffers, *, admitted: bool) -> None:
    """Decide the unlink duty for every segment a drained request parked in
    POSIX shm (any buffer that is not an inline numpy array is the Rust
    extension's ``ShmBuffer``, whatever its name). Admitted: Rust hands the
    duty to the ``ShmPointerMMData`` stubs, which unlink after the
    post-broadcast open on every rank. Rejected: unlink now. Both are
    idempotent on the Rust side. This is the drain loop's job, never a
    wrapper's, so a package's own ``_wrap_mm_result`` needs no knowledge of it."""
    import numpy as np

    for value in buffers.values():
        if isinstance(value, np.ndarray):
            continue
        if admitted:
            value.release()
        else:
            value.discard()


class MmTransportStats:
    """Zero-copy accounting at the Rust-to-Python multimodal hand-off.

    A feature tensor crosses in one of two shapes: ``inline`` -- a numpy array
    that owns the Rust vector, viewed by ``torch.from_numpy`` -- or ``shm`` -- a
    POSIX segment the worker wrote, named in the request so every TP rank maps
    it after the broadcast."""

    __slots__ = (
        "inline_features",
        "inline_bytes",
        "shm_features",
        "shm_bytes",
        "copies",
    )

    def __init__(self) -> None:
        self.inline_features = 0
        self.inline_bytes = 0
        self.shm_features = 0
        self.shm_bytes = 0
        self.copies = 0

    def record_inline(self, array, tensor) -> None:
        self.inline_features += 1
        self.inline_bytes += array.nbytes
        if array.flags["OWNDATA"] or tensor.data_ptr() != array.ctypes.data:
            self.copies += 1

    def record_shm(self, stub) -> None:
        import numpy as np

        self.shm_features += 1
        self.shm_bytes += int(np.prod(stub.shape)) * np.dtype(stub.dtype).itemsize

    def snapshot(self) -> Dict[str, int]:
        return {name: getattr(self, name) for name in self.__slots__}


MM_TRANSPORT_STATS = MmTransportStats()


class RustMmProcessor:
    """Builds and validates the Rust MM pipeline for one model.

    Construction registers the same ``mm_processor`` mapping the Python
    TokenizerManager would build — not to process requests (the Rust worker pool
    does that, GIL-free) but as the source of truth
    :meth:`resolve_spec` resolves the pipeline parameters from. At drain
    time :meth:`wrap_encoded` wraps the Rust-produced buffers into the
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

        Carries only resolved settings, never the HF config, and is
        conservative by design: an unrecognized knob disables the Rust path
        rather than being approximated."""
        from sglang.srt.managers.multimodal_processor import get_mm_processor_cls

        hf_config = self.model_config.hf_config
        mm_processor_cls = get_mm_processor_cls(
            hf_config, model_config=self.model_config
        )
        family = rust_mm_family_for(
            mm_processor_cls, getattr(hf_config, "model_type", None)
        )
        if family is None:
            return None
        spec = family.resolve(
            processor=self._processor,
            hf_config=hf_config,
            feature_shm=self._use_feature_shm(),
        )
        if spec is not None:
            logger.info(
                "rust server: Rust MM pipeline enabled (family=%s)", family.name
            )
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
    def wrap_encoded(spec: RustMmSpec, buffers):
        """Drain-time adapter: wrap one request's ``mm.*`` buffers (a
        ``{name: numpy array | ShmBuffer}`` mapping from ``recv_requests``) into
        the scheduler's ``MultimodalProcessorOutput``. Wrapping only -- load,
        resize, patchify, token expansion and M-RoPE all ran in Rust.

        The buffers are ``mm.feature.{i}`` per item (f32, inline or shm),
        ``mm.mrope`` (``[3, seq_len]`` int64, M-RoPE families only), and ``mm.meta``, a msgpack sidecar decoding to ``{"items":
        [{"modality", "hash", "offsets", "model_specific_data"}, ...],
        "token_ids", "mrope_delta"}`` with the items in buffer order. External
        model packages receive the same layout and wrap it their own way.

        Runs on the scheduler loop, so it must stay copy-free *and* hash-free:
        the inline numpy arrays own the Rust buffers, ``torch.from_numpy`` just
        views them, and each item's ``hash`` is worker-precomputed so
        ``set_pad_value`` skips ``hash_feature``. Any per-byte work here — memcpy,
        sha256, tens of MB per image-heavy request — measurably inflates every
        running request's inter-token latency."""
        import msgspec
        import torch

        from sglang.srt.managers.schedule_batch import (
            MultimodalProcessorOutput,
        )

        # Ownership of the segments is not decided here: the drain loop
        # releases or discards every `ShmBuffer` once it knows whether the
        # request is admitted (`settle_shm_buffers`). A failure part-way only
        # has to close the stubs built so far, which hold an open mapping each.
        stubs: list = []
        try:
            # Decoded straight from the numpy buffer (a memoryview over the
            # Rust-owned bytes), not from a `.tobytes()` copy of it.
            meta = msgspec.msgpack.decode(memoryview(buffers["mm.meta"]))
            items = [
                RustMmProcessor._wrap_item(
                    index=index, item=item, buffers=buffers, stubs=stubs
                )
                for index, item in enumerate(meta["items"])
            ]
            if envs.SGLANG_MM_PRECOMPUTE_HASH.get():
                for item in items:
                    item.set_pad_value()
            mrope = {}
            if "mm.mrope" in buffers:
                mrope = dict(
                    mrope_positions=torch.from_numpy(buffers["mm.mrope"]),
                    mrope_position_delta=torch.tensor(
                        [[meta["mrope_delta"]]], dtype=torch.long
                    ),
                )
            output = MultimodalProcessorOutput(
                mm_items=items,
                im_token_id=spec.image_token_id,
                im_start_id=spec.vision_start_token_id,
                im_end_id=spec.vision_end_token_id,
                video_token_id=spec.video_token_id,
                **mrope,
            )
        except BaseException:
            for stub in stubs:
                stub.close_and_unlink()
            raise
        return output

    @staticmethod
    def _wrap_item(*, index: int, item: dict, buffers, stubs: list):
        import numpy as np
        import torch

        from sglang.srt.managers.mm_utils import ShmPointerMMData
        from sglang.srt.managers.schedule_batch import MultimodalDataItem

        feature = buffers[f"mm.feature.{index}"]
        if isinstance(feature, np.ndarray):
            array = feature
            feature = torch.from_numpy(array)
            MM_TRANSPORT_STATS.record_inline(array=array, tensor=feature)
        else:
            MM_TRANSPORT_STATS.record_shm(stub=feature)
            # The worker placed this item's buffer in a named POSIX segment
            # (see `_use_feature_shm`). Build the stub in its
            # post-`__setstate__` form: rank 0 never pickle-roundtrips its
            # own copy, and `materialize()` needs the mapped view.
            stub = ShmPointerMMData.__new__(ShmPointerMMData)
            stub.__setstate__(
                {
                    "shm_name": feature.name,
                    "shape": tuple(feature.shape),
                    "dtype": torch.float32,
                    "precomputed_hash": item["hash"],
                }
            )
            stubs.append(stub)
            feature = stub
        # A list (Qwen's `image_grid_thw`) is one item's row; ints stay ints.
        model_specific_data = {
            name: (
                torch.tensor([value], dtype=torch.long)
                if isinstance(value, list)
                else value
            )
            for name, value in item["model_specific_data"].items()
        }
        return MultimodalDataItem(
            modality=_MODALITIES[item["modality"]],
            feature=feature,
            hash=item["hash"],
            offsets=[tuple(span) for span in item["offsets"]],
            model_specific_data=model_specific_data,
        )
