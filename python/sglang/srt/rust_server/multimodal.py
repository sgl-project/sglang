"""Multimodal support for the embedded Rust server."""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, Optional, Tuple, Union

import msgspec

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_mm, get_model, get_parallel, get_serving

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class RustMmVideoSampling(msgspec.Struct, frozen=True, kw_only=True):
    """``qwen_vl_media_processing.py``'s frame sampling and pre-resize inputs,
    read at launch because ``VIDEO_TOTAL_PIXELS`` comes from the environment.
    Mirrors ``sglang_processor::VideoSampling``."""

    config: Dict[str, float]
    fps: float
    min_frames: float
    max_frames: float
    frame_factor: float
    factor: float
    min_pixels: float
    max_pixels: float
    total_pixels: float


class RustMmVideoSpec(msgspec.Struct, frozen=True, kw_only=True):
    """Qwen3-VL video: the token ids ``Qwen3VLProcessor`` inserts, its
    ``Qwen3VLVideoProcessor``, and SGLang's sampling ahead of it."""

    video_token_id: int
    vision_start_token_id: int
    vision_end_token_id: int
    patch_size: int
    merge_size: int
    temporal_patch_size: int
    # The video processor's size["shortest_edge"] and size["longest_edge"].
    min_pixels: int
    max_pixels: int
    image_mean: Tuple[float, ...]
    image_std: Tuple[float, ...]
    sampling: RustMmVideoSampling

    def rust_json(self) -> Dict[str, Any]:
        """``dynamo_multimodal``'s ``QwenVlVideoSpec``."""
        return {
            "video_token_id": self.video_token_id,
            "vision_start_token_id": self.vision_start_token_id,
            "vision_end_token_id": self.vision_end_token_id,
            "processor": {
                "video_processor_type": "Qwen3VLVideoProcessor",
                "patch_size": self.patch_size,
                "merge_size": self.merge_size,
                "temporal_patch_size": self.temporal_patch_size,
                "size": {
                    "shortest_edge": self.min_pixels,
                    "longest_edge": self.max_pixels,
                },
            },
            "image_mean": self.image_mean,
            "image_std": self.image_std,
        }


class RustMmSpec(msgspec.Struct, frozen=True, kw_only=True):
    """Resolved parameters of the Rust MM pipeline for one model,
    consumed by the Rust worker pool (as the typed extension ``MmSpec``, see
    :meth:`RustServer._build_mm_spec`), the ``_multimodal`` parity API
    (:meth:`rust_json`) and the drain adapter
    (:meth:`RustMmProcessor.wrap_encoded`)."""

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
    # None: the model or its config takes no video on the Rust path.
    video: Optional[RustMmVideoSpec] = None

    # Used by the drain adapter only; every other field goes to Rust.
    DRAIN_ONLY = ("vision_start_token_id", "vision_end_token_id", "video_token_id")

    @property
    def feature_dim(self) -> int:
        return 3 * self.temporal_patch_size * self.patch_size * self.patch_size

    def rust_json(self) -> str:
        """The subset `sglang_mm::registry::pipeline_from_spec` parses — the
        JSON form the ``_multimodal`` parity API takes; the server itself is
        handed the typed ``MmSpec`` instead."""
        fields = (
            f for f in self.__struct_fields__ if f not in self.DRAIN_ONLY + ("video",)
        )
        spec = {f: getattr(self, f) for f in fields}
        if self.video is not None:
            spec["video"] = self.video.rust_json()
            spec["video_sampling"] = self.video.sampling
        return msgspec.json.encode(spec).decode()


class RustMmNemotronSpec(msgspec.Struct, frozen=True, kw_only=True):
    """What ``NanoNemotronVLImageProcessor`` reads from the server and
    ``hf_config`` beyond the checkpoint files: the first four go to Rust, the
    rest select the model kwargs the drain adds (``_nemotron_model_kwargs``)."""

    max_model_len: int
    video_target_num_patches: int
    video_maintain_aspect_ratio: bool
    # ``--json-model-override-args`` can set it; above 0, EVS adds ``thw_grids``
    # and ``pre_chunked_input_ids``.
    video_pruning_rate: float
    patch_size: int
    downsample_factor: int


class RustMmModelDirSpec(msgspec.Struct, frozen=True, kw_only=True):
    """A family ``dynamo_multimodal`` resolves from the checkpoint directory
    (``MmSpec.from_model_dir``). The token ids are the ones the family's
    Python processor hands the scheduler; the drain adapter reads them."""

    family: str
    feature_shm: bool
    model_path: str
    revision: Optional[str]
    # The resampler of the class SGLang loaded: --image-processor-backend can
    # pick another class than the checkpoint declares (see RustMmSpec.resample).
    resample: str
    image_token_id: int
    video_token_id: Optional[int]
    vision_start_token_id: Optional[int] = None
    vision_end_token_id: Optional[int] = None
    nemotron: Optional[RustMmNemotronSpec] = None

    def rust_json(self) -> str:
        return msgspec.json.encode(self).decode()


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
    # Resolved by ``dynamo_multimodal`` from the checkpoint directory
    # (:class:`RustMmModelDirSpec`) instead of from the HF processor's attributes.
    from_model_dir: bool = False
    # The ``hf_config`` attribute whose id the family's Python processor reports
    # as ``video_token_id`` (model-dir families; ``None`` reports none).
    video_token_attr: Optional[str] = None
    # ``hf_config`` keys the family's Python processor reads and Rust re-reads
    # from config.json, so --json-model-override-args must not change them.
    config_keys: FrozenSet[str] = frozenset()

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
                "qwen4_exp",
            )
        ),
        image_processors={
            "Qwen2VLImageProcessor": "aten_u8",
            "Qwen2VLImageProcessorFast": "aten_u8",
            "Qwen2VLImageProcessorPil": "pil",
        },
    ),
    RustMmFamily(
        name="glm5_next",
        mm_processor="sglang.srt.multimodal.processors.glm4v:Glm4vImageProcessor",
        model_types=frozenset(("glm5_next",)),
        image_processors={"Glm5NextImageProcessor": "aten_u8"},
        from_model_dir=True,
        # Glm4vImageProcessor expands a video into image tokens.
        video_token_attr="image_token_id",
        config_keys=frozenset(
            (
                "image_token_id",
                "video_token_id",
                "image_start_token_id",
                "image_end_token_id",
                "video_start_token_id",
                "video_end_token_id",
                "vision_config",
            )
        ),
    ),
    RustMmFamily(
        name="nemotron_vl",
        mm_processor="sglang.srt.multimodal.processors.nano_nemotron_vl:NanoNemotronVLImageProcessor",
        model_types=frozenset(("nemotron_h_omni",)),
        # The Python processor resizes with PIL itself; the HF one goes unused.
        image_processors={"NemotronH_Omni_Reasoning_V3ImageProcessor": "pil"},
        from_model_dir=True,
        # Rust takes the video frame size and the EVS pruning rate from Python
        # (RustMmNemotronSpec), not from this list.
        config_keys=frozenset(
            (
                "image_size",
                "patch_size",
                "downsample_ratio",
                "norm_mean",
                "norm_std",
                "min_num_patches",
                "max_num_patches",
                "video_temporal_patch_size",
                "img_context_token",
                "img_start_token",
                "img_end_token",
                "video_context_token",
            )
        ),
    ),
)


# Qwen3-VL's video rule, which Qwen3.5 and qwen4_exp inherit: Qwen2-VL and
# Qwen2.5-VL size video and place it in M-RoPE differently.
RUST_MM_VIDEO_MODEL_TYPES = frozenset(
    ("qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe", "qwen4_exp")
)

# The `--mm-process-config` video keys `preprocess_video` consumes; any other key
# would reach the HF video processor, which Rust does not mirror.
RUST_MM_VIDEO_CONFIG_KEYS = frozenset(
    (
        "fps",
        "nframes",
        "min_frames",
        "max_frames",
        "min_pixels",
        "max_pixels",
        "total_pixels",
    )
)


def _video_processor_mismatch(video_processor: Any) -> Optional[str]:
    """Why the Rust video stages cannot reproduce this HF video processor, or
    ``None`` when they can."""
    from transformers.image_utils import PILImageResampling

    name = type(video_processor).__name__
    if name != "Qwen3VLVideoProcessor":
        return f"video processor {name}"
    vp = video_processor
    if not (vp.do_resize and vp.do_rescale and vp.do_normalize):
        return "a video processor stage is off"
    if vp.resample != PILImageResampling.BICUBIC:
        return f"video resample {vp.resample}"
    if vp.rescale_factor != 1 / 255:
        return "video rescale_factor is not 1/255"
    if vp.cap_pixels_per_frame:
        return "video cap_pixels_per_frame is on"
    return None


def _video_spec(
    hf_config: Any, video_processor: Any, config: Dict[str, float]
) -> RustMmVideoSpec:
    from sglang.srt.multimodal import qwen_vl_media_processing as media

    sampling = RustMmVideoSampling(
        config={key: float(value) for key, value in config.items()},
        fps=float(media.FPS),
        min_frames=float(media.FPS_MIN_FRAMES),
        max_frames=float(media.FPS_MAX_FRAMES),
        frame_factor=float(media.FRAME_FACTOR),
        factor=float(media.IMAGE_FACTOR),
        min_pixels=float(media.VIDEO_MIN_PIXELS),
        max_pixels=float(media.VIDEO_MAX_PIXELS),
        total_pixels=float(media.VIDEO_TOTAL_PIXELS),
    )
    return RustMmVideoSpec(
        video_token_id=hf_config.video_token_id,
        vision_start_token_id=hf_config.vision_start_token_id,
        vision_end_token_id=hf_config.vision_end_token_id,
        patch_size=video_processor.patch_size,
        merge_size=video_processor.merge_size,
        temporal_patch_size=video_processor.temporal_patch_size,
        min_pixels=int(video_processor.size["shortest_edge"]),
        max_pixels=int(video_processor.size["longest_edge"]),
        image_mean=tuple(float(x) for x in video_processor.image_mean),
        image_std=tuple(float(x) for x in video_processor.image_std),
        sampling=sampling,
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
        from sglang.srt.multimodal.modality import Modality

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
    extension's ``ShmBuffer``, whatever its name, or a processor's external
    buffer, which settles the same way). Admitted: Rust hands the duty to the
    ``ShmPointerMMData`` stubs, which unlink after the post-broadcast open on
    every rank. Rejected: unlink now. Both are idempotent on the Rust side.
    This is the drain loop's job, never a wrapper's, so a package's own
    ``_wrap_mm_result`` needs no knowledge of it."""
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

    A feature tensor crosses in one of three shapes: ``inline`` -- a numpy
    array that owns the Rust vector, viewed by ``torch.from_numpy`` -- ``shm``
    -- a POSIX segment the worker wrote, named in the request so every TP rank
    maps it after the broadcast -- or ``external`` -- a buffer the processor
    published in its own transport (``MmProcessor::publish_features``), which
    the package that wraps it records."""

    __slots__ = (
        "inline_features",
        "inline_bytes",
        "shm_features",
        "shm_bytes",
        "external_features",
        "external_bytes",
        "copies",
    )

    def __init__(self) -> None:
        self.inline_features = 0
        self.inline_bytes = 0
        self.shm_features = 0
        self.shm_bytes = 0
        self.external_features = 0
        self.external_bytes = 0
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

    def record_external(self, nbytes: int) -> None:
        self.external_features += 1
        self.external_bytes += nbytes

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
        from sglang.srt.multimodal.token_space_mm_process.registry import (
            register_token_space_models,
        )

        self.server_args = server_args
        self.model_config = model_config
        # Worker threads == max concurrently-processed mm requests.
        self.mm_workers = get_mm().mm_processor_worker_num or self.AUTO_MM_WORKERS

        # The mapping the Python TokenizerManager builds in
        # init_tokenizer_and_processor. The caller's already-loaded HF
        # AutoProcessor is reused when available (identical construction args).
        import_processors("sglang.srt.multimodal.processors")
        register_token_space_models()
        if mm_process_pkg := envs.SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE.get():
            import_processors(mm_process_pkg, overwrite=True)
        self._processor = processor or get_processor_wrapper()

    def resolve_spec(self) -> Optional[Union[RustMmSpec, RustMmModelDirSpec]]:
        """The :class:`RustMmSpec` for this model, or ``None`` when it has no
        Rust pipeline (the launch gate turns that into a hard error).

        Carries only resolved settings — patch geometry, pixel limits,
        normalization, token ids — never the HF config, and is conservative by
        design: an unrecognized knob disables the Rust path rather than being
        approximated."""
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
        image_processor = getattr(self._processor, "image_processor", None)
        resample = family.image_processors.get(type(image_processor).__name__)
        if resample is None:
            return None
        if family.from_model_dir:
            return self._resolve_model_dir_spec(family, hf_config, resample)
        # The Rust pipeline always resizes, rescales by 1/255 and normalizes;
        # Rust's fused normalize constants assume that factor. Anything else
        # would silently produce different features.
        stages = ("do_resize", "do_rescale", "do_normalize")
        if not all(getattr(image_processor, stage, True) for stage in stages):
            return None
        if getattr(image_processor, "rescale_factor", None) != 1 / 255:
            return None

        # `--mm-process-config {"image": {...}}`: only pixel-limit overrides are
        # mirrored by Rust; anything else disables the pipeline.
        image_overrides = dict((get_mm().mm_process_config or {}).get("image", {}))
        if not set(image_overrides) <= {"min_pixels", "max_pixels"}:
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
            spec = RustMmSpec(
                family=family.name,
                feature_shm=self._use_feature_shm(),
                image_token_id=hf_config.image_token_id,
                patch_size=image_processor.patch_size,
                merge_size=image_processor.merge_size,
                temporal_patch_size=image_processor.temporal_patch_size,
                min_pixels=int(min_pixels),
                max_pixels=int(max_pixels),
                image_mean=tuple(float(x) for x in image_processor.image_mean),
                image_std=tuple(float(x) for x in image_processor.image_std),
                resample=resample,
                vision_start_token_id=getattr(hf_config, "vision_start_token_id", None),
                vision_end_token_id=getattr(hf_config, "vision_end_token_id", None),
                video_token_id=getattr(hf_config, "video_token_id", None),
                video=self._resolve_video_spec(hf_config),
            )
        except (AttributeError, TypeError):  # missing/odd processor attrs
            return None
        logger.info("rust server: Rust MM pipeline enabled (family=%s)", family.name)
        return spec

    def _resolve_model_dir_spec(
        self, family: RustMmFamily, hf_config, resample: str
    ) -> Optional[RustMmModelDirSpec]:
        """Rust reads the processor settings from the checkpoint, so a server
        override of them would not reach it."""
        # Python loads its processor from --tokenizer-path; Rust reads --model-path.
        if get_serving().tokenizer_path != get_model().model_path:
            logger.warning(
                "rust server: %s takes no --tokenizer-path other than --model-path",
                family.name,
            )
            return None
        overrides = msgspec.json.decode(get_model().json_model_override_args or "{}")
        if overridden := sorted(family.config_keys & set(overrides)):
            logger.warning(
                "rust server: %s takes no --json-model-override-args %s",
                family.name,
                overridden,
            )
            return None
        mm_process_config = get_mm().mm_process_config or {}
        if mm_process_config.get("image") or mm_process_config.get("video"):
            logger.warning(
                "rust server: %s takes no --mm-process-config image or video keys",
                family.name,
            )
            return None
        if family.name == "nemotron_vl" and not hf_config.dynamic_resolution:
            # ``--json-model-override-args`` can select Python's tile path; the
            # checkpoint files Rust reads still say dynamic resolution.
            logger.warning("rust server: nemotron_vl serves dynamic resolution only")
            return None
        logger.info("rust server: Rust MM pipeline enabled (family=%s)", family.name)
        if family.name == "nemotron_vl":
            return self._resolve_nemotron_spec(hf_config, resample)
        return RustMmModelDirSpec(
            family=family.name,
            feature_shm=self._use_feature_shm(),
            model_path=self.model_config.model_path,
            revision=self.server_args.revision,
            resample=resample,
            image_token_id=hf_config.image_token_id,
            video_token_id=(
                getattr(hf_config, family.video_token_attr)
                if family.video_token_attr is not None
                else None
            ),
        )

    def _resolve_nemotron_spec(self, hf_config, resample: str) -> RustMmModelDirSpec:
        """The values and token ids ``NanoNemotronVLImageProcessor.__init__`` and
        ``process_mm_data_async`` read; a video expands into image tokens."""
        tokenizer = self._processor.tokenizer
        image_token_id = tokenizer.convert_tokens_to_ids(hf_config.img_context_token)
        return RustMmModelDirSpec(
            family="nemotron_vl",
            feature_shm=self._use_feature_shm(),
            model_path=self.model_config.model_path,
            revision=self.server_args.revision,
            resample=resample,
            image_token_id=image_token_id,
            video_token_id=image_token_id,
            vision_start_token_id=tokenizer.convert_tokens_to_ids(
                hf_config.img_start_token
            ),
            vision_end_token_id=tokenizer.convert_tokens_to_ids(
                hf_config.img_end_token
            ),
            nemotron=RustMmNemotronSpec(
                max_model_len=get_model().context_length or 8192,
                video_target_num_patches=hf_config.video_target_num_patches,
                video_maintain_aspect_ratio=hf_config.video_maintain_aspect_ratio,
                video_pruning_rate=hf_config.video_pruning_rate,
                patch_size=hf_config.patch_size,
                downsample_factor=int(1 / hf_config.downsample_ratio),
            ),
        )

    def _resolve_video_spec(self, hf_config) -> Optional[RustMmVideoSpec]:
        """The video half of the spec, or ``None`` when Rust cannot mirror
        this model's or this server's video processing; images still run."""
        if hf_config.model_type not in RUST_MM_VIDEO_MODEL_TYPES:
            return None
        config = dict((get_mm().mm_process_config or {}).get("video", {}))
        try:
            video_processor = self._processor.video_processor
            reason = _video_processor_mismatch(video_processor)
            if reason is None and not set(config) <= RUST_MM_VIDEO_CONFIG_KEYS:
                reason = f"--mm-process-config video keys {sorted(config)}"
            if reason is None and {"fps", "nframes"} <= set(config):
                reason = "--mm-process-config video sets both fps and nframes"
            if reason is None and any(
                type(v) not in (int, float) for v in config.values()
            ):
                reason = f"--mm-process-config video values {config}"
            if reason is None:
                return _video_spec(hf_config, video_processor, config)
        except (AttributeError, KeyError, TypeError) as error:
            reason = f"unreadable video processor: {error!r}"
        logger.warning("rust server: video input disabled: %s", reason)
        return None

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
    def wrap_encoded(spec: Union[RustMmSpec, RustMmModelDirSpec], buffers):
        """Drain-time adapter: wrap one request's ``mm.*`` buffers (a
        ``{name: numpy array | ShmBuffer}`` mapping from ``recv_requests``) into
        the scheduler's ``MultimodalProcessorOutput``. Wrapping only -- load,
        resize, patchify, token expansion and M-RoPE all ran in Rust.

        The buffers are ``mm.feature.{i}`` per item (shaped ``[rows,
        feature_dim]`` f32, inline or shm), ``mm.mrope`` (``[3, seq_len]``
        int64), and ``mm.meta``, a msgpack sidecar decoding to ``{"items":
        [{"modality", "hash", "identity", "offsets", "model_specific_data"}, ...],
        "token_ids", "mrope_delta"}`` with the items in buffer order. External
        model packages receive the same layout and wrap it their own way.

        Runs on the scheduler loop, so it must stay copy-free *and* hash-free:
        the inline numpy arrays own the Rust buffers, ``torch.from_numpy`` just
        views them, and each item's full ``identity`` is worker-precomputed so
        ``set_pad_value`` never rereads feature bytes. The drain boundary binds
        these short digests to the serving configuration. Any feature-sized work here — memcpy,
        sha256, tens of MB per image-heavy request — measurably inflates every
        running request's inter-token latency."""
        import msgspec
        import torch

        from sglang.srt.managers.req_types import MultimodalProcessorOutput

        # Ownership of the segments is not decided here: the drain loop
        # releases or discards every `ShmBuffer` once it knows whether the
        # request is admitted (`settle_shm_buffers`). A failure part-way only
        # has to close the stubs built so far, which hold an open mapping each.
        stubs: list = []
        try:
            # Decoded straight from the numpy buffer (a memoryview over the
            # Rust-owned bytes), not from a `.tobytes()` copy of it.
            meta = msgspec.msgpack.decode(memoryview(buffers["mm.meta"]))
            nemotron = spec.nemotron if isinstance(spec, RustMmModelDirSpec) else None
            items = [
                RustMmProcessor._wrap_item(
                    index=index,
                    item=item,
                    buffers=buffers,
                    stubs=stubs,
                    has_grid=nemotron is None,
                )
                for index, item in enumerate(meta["items"])
            ]
            if nemotron is not None:
                for index, (item, item_meta) in enumerate(zip(items, meta["items"])):
                    item.model_specific_data.update(
                        _nemotron_model_kwargs(
                            spec=nemotron,
                            video=item_meta["modality"] == "video",
                            counts=item_meta["model_specific_data"],
                            feature_shape=tuple(buffers[f"mm.feature.{index}"].shape),
                            input_ids=buffers["input_ids"],
                        )
                    )
            if envs.SGLANG_MM_PRECOMPUTE_HASH.get():
                for item in items:
                    item.set_pad_value()
            # Nemotron has 1-D positions and no grid; every other family sends both.
            mrope = buffers["mm.mrope"] if nemotron is None else None
            output = MultimodalProcessorOutput(
                mm_items=items,
                im_token_id=spec.image_token_id,
                im_start_id=spec.vision_start_token_id,
                im_end_id=spec.vision_end_token_id,
                video_token_id=spec.video_token_id,
                mrope_positions=None if mrope is None else torch.from_numpy(mrope),
                mrope_position_delta=(
                    None
                    if mrope is None
                    else torch.tensor([[meta["mrope_delta"]]], dtype=torch.long)
                ),
            )
        except BaseException:
            for stub in stubs:
                stub.close_and_unlink()
            raise
        return output

    @staticmethod
    def _wrap_item(*, index: int, item: dict, buffers, stubs: list, has_grid: bool):
        import numpy as np
        import torch

        from sglang.srt.managers.mm_utils import ShmPointerMMData
        from sglang.srt.managers.req_types import MultimodalDataItem

        feature = buffers[f"mm.feature.{index}"]
        if isinstance(feature, np.ndarray):
            array = feature
            feature = torch.from_numpy(array)
            if array.dtype == np.uint16:
                feature = feature.view(torch.bfloat16)
            MM_TRANSPORT_STATS.record_inline(array=array, tensor=feature)
        else:
            MM_TRANSPORT_STATS.record_shm(stub=feature)
            # ``ShmBuffer.dtype`` is a numpy name; uint16 holds bfloat16 bits.
            dtype = {"float32": torch.float32, "uint16": torch.bfloat16}[feature.dtype]
            # The worker placed this item's buffer in a named POSIX segment
            # (see `_use_feature_shm`). Build the stub in its
            # post-`__setstate__` form: rank 0 never pickle-roundtrips its
            # own copy, and `materialize()` needs the mapped view.
            stub = ShmPointerMMData.__new__(ShmPointerMMData)
            stub.__setstate__(
                {
                    "shm_name": feature.name,
                    "shape": tuple(feature.shape),
                    "dtype": dtype,
                    "precomputed_hash": item["hash"],
                }
            )
            stubs.append(stub)
            feature = stub
        model_specific_data = {}
        if has_grid:
            grid = "video_grid_thw" if item["modality"] == "video" else "image_grid_thw"
            t, h, w = item["model_specific_data"][grid]
            model_specific_data = {grid: torch.tensor([[t, h, w]], dtype=torch.long)}
        return MultimodalDataItem(
            modality=_MODALITIES[item["modality"]],
            feature=feature,
            hash=item["hash"],
            identity=bytes(item["identity"]),
            offsets=[tuple(span) for span in item["offsets"]],
            model_specific_data=model_specific_data,
        )


def _nemotron_model_kwargs(
    *,
    spec: RustMmNemotronSpec,
    video: bool,
    counts: Dict[str, Any],
    feature_shape: Tuple[int, ...],
    input_ids,
) -> Dict[str, Any]:
    """The ``model_specific_data`` of ``nano_nemotron_vl.py::_create_visual_items``
    and ``EVSProcessor.static_size_data_items`` for one item of one request."""
    if not video:
        # The Rust family refuses SGLang's tile path, so every image is dynamic.
        return {"num_tokens": counts["feature_tokens"], "is_dynamic": True}
    # [frames, 3, height, width]; one video per request.
    frames, _, height, width = feature_shape
    kwargs: Dict[str, Any] = {"frames_per_video": [frames]}
    if spec.video_pruning_rate > 0:
        side = spec.patch_size * spec.downsample_factor
        kwargs["thw_grids"] = [(counts["frame_groups"], height // side, width // side)]
        # ``pad_input_ids`` replaces it with the padded ids.
        kwargs["pre_chunked_input_ids"] = input_ids.tolist()
    return kwargs
