# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 sampling parameters.

A single ``SamplingParams`` class serves T2V, I2V, V2V, T2I, and
action-conditioned variants.  Per-request mode is dispatched in the pipeline
from ``num_frames`` (``== 1`` → T2I), ``image_path`` (set → I2V),
``video_path`` (set → V2V), and ``action_mode`` (set → action-conditioned).
For ``num_frames == 1`` the output ``data_type`` flips to ``IMAGE``
so the file extension and decode path agree.
"""

import json
from dataclasses import dataclass, field
from typing import Any, ClassVar

from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)

COSMOS3_DEFAULT_GUIDANCE_SCALE = 4.0
COSMOS3_EDGE_T2I_GUIDANCE_SCALE = 7.0
COSMOS3_EDGE_T2V_GUIDANCE_SCALE = 5.0
COSMOS3_EDGE_T2V_WIDTH = 832
COSMOS3_EDGE_T2V_HEIGHT = 480
COSMOS3_EDGE_T2I_SIZE = 640

# Image generation applies guidance only over a high-noise window; guiding the
# low-noise steps degrades sample quality (Kynkaeaenniemi et al. 2024). Video
# modes guide at every step.
COSMOS3_T2I_GUIDANCE_INTERVAL = (400.0, 1000.0)

# Edge is trained at 256p/480p only; larger frames push the spatial mRoPE grid
# past its trained range and shatter the output.
COSMOS3_EDGE_SUPPORTED_RESOLUTIONS = [
    (832, 480),
    (480, 832),
    (640, 480),
    (480, 640),
    (480, 480),
    (640, 640),
    (448, 256),
    (256, 448),
    (256, 256),
]


def _parse_request_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError, ValueError):
        return value


def _optional_int_list(value: Any) -> list[int] | None:
    value = _parse_request_value(value)
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(value)]


@dataclass
class Cosmos3SamplingParams(SamplingParams):
    """Cosmos3 sampling parameters (T2V defaults; also used for I2V / V2V / T2I).

    ``height``/``width`` default to ``None`` so the variant (Edge vs. base) can
    pick the right resolution at request time in
    :meth:`_resolve_variant_defaults`.
    """

    height: int | None = None
    width: int | None = None
    num_frames: int = 81
    fps: int = 24

    guidance_scale: float = COSMOS3_DEFAULT_GUIDANCE_SCALE
    num_inference_steps: int = 35

    negative_prompt: str = ""

    use_duration_template: bool | None = None
    use_resolution_template: bool | None = None
    use_system_prompt: bool | None = None
    use_guardrails: bool | None = None
    sound_duration: float = 0.0

    # Optional CFG window — T2I requests typically pass e.g. ``(400, 1000)`` to
    # skip guidance at low noise levels. T2V / I2V / V2V leave it unset.
    guidance_interval: tuple[float, float] | None = None

    # V2V conditioning: which latent-frame indices stay locked to the input
    # video. ``None`` resolves to ``[0]`` for I2V (single frame) and ``[0, 1]``
    # for V2V. ``condition_video_keep`` controls whether the first or last
    # source frames are used when the input video is longer than needed.
    condition_frame_indexes: list[int] | None = None
    condition_video_keep: str = "first"

    # Transfer (control-video) conditioning. ``control_path`` points to one or
    # more pre-computed control videos (e.g. edge / blur / depth / seg / wsm
    # maps). When set, each control clip is VAE-encoded and packed as clean
    # vision tokens that prefix the target clip in the GEN sequence; multiple
    # paths drive multi-hint transfer (e.g. edge + depth). Control clips reuse
    # ``proj_in``, so every Cosmos3 checkpoint supports transfer.
    control_path: str | list[str] | None = None

    # Optional hint type(s) parallel to ``control_path`` (one of
    # ``edge`` / ``blur`` / ``depth`` / ``seg`` / ``wsm``). Used only to apply
    # tuned per-hint defaults (``guidance`` / ``control_guidance`` / ``shift``)
    # when exactly one control input is given and the user left those unset.
    control_hint: str | list[str] | None = None

    # Control-CFG scale for transfer. ``1.0`` (default) disables the extra
    # control-dropped forward; values > 1.0 amplify the control map's influence
    # by blending the with-control and without-control predictions on the
    # generated span: ``cond_nc + control_guidance * (cond_full - cond_nc)``.
    control_guidance: float = 1.0

    # Optional timestep window ``(lo, hi)`` restricting where control-CFG is
    # applied (analogous to ``guidance_interval`` for text CFG). ``None`` applies
    # it at every step.
    control_guidance_interval: tuple[float, float] | None = None

    # Long-video transfer controls. Chunks overlap by
    # ``num_conditional_frames`` pixel frames; overlap frames from the previous
    # decoded chunk are kept clean in the next chunk.
    num_video_frames_per_chunk: int = 93
    num_conditional_frames: int = 1
    num_first_chunk_conditional_frames: int = 0
    max_frames: int = 5000
    show_control_condition: bool = False
    show_input: bool = False
    share_vision_temporal_positions: bool = True

    # Tuned per-hint defaults applied when exactly one control input is given
    # and the corresponding field was not set explicitly (mirrors the
    # cosmos-framework ``_TRANSFER_DEFAULTS`` table). ``shift`` maps to
    # ``flow_shift``. Multi-hint transfer keeps the request's own values.
    _TRANSFER_DEFAULTS: ClassVar[dict[str, dict[str, float | int]]] = {
        "edge": {"guidance": 3.0, "control_guidance": 1.5, "shift": 10.0},
        "blur": {"guidance": 3.0, "control_guidance": 1.5, "shift": 10.0},
        "depth": {"guidance": 3.0, "control_guidance": 1.5, "shift": 10.0},
        "seg": {"guidance": 3.0, "control_guidance": 2.0, "shift": 10.0},
        "wsm": {
            "guidance": 1.0,
            "control_guidance": 3.0,
            "shift": 10.0,
            "num_frames": 101,
            "fps": 10,
            "num_video_frames_per_chunk": 101,
        },
    }

    supported_resolutions: list[tuple[int, int]] | None = field(
        default_factory=lambda: [
            (1280, 720),
            (720, 1280),
            (832, 480),
            (480, 832),
            (1024, 1024),
            (640, 640),
        ]
    )

    # Action modality (requires action_gen=True in the model checkpoint)
    # action_mode: "forward_dynamics" | "policy" | "inverse_dynamics"
    action_mode: str | None = None
    domain_id: int | None = None
    domain_name: str | None = None
    raw_action_dim: int | None = None
    action_fps: float | None = None
    # Action data for forward_dynamics: [T, D] nested list (API) or JSON string
    # (CLI via --action). Ignored by the other action modes.
    action: Any = None
    # Viewpoint phrasing for the structured action caption.
    action_view_point: str = "ego_view"
    # Optional dataset-derived action stats (JSON) for (de)normalization. When
    # set, input actions are normalized and predicted actions de-normalized
    # into physical units with ``action_normalization``.
    action_stats_path: str | None = None
    action_normalization: str = "quantile"

    @classmethod
    def image_request_extra_fields(cls) -> frozenset[str]:
        return frozenset(
            {
                "guidance_interval",
                "use_duration_template",
                "use_guardrails",
                "use_resolution_template",
                "use_system_prompt",
            }
        )

    @classmethod
    def default_image_output_format(cls) -> str:
        return "png"

    @classmethod
    def default_image_response_format(cls) -> str:
        return "b64_json"

    @classmethod
    def video_request_extra_fields(cls) -> frozenset[str]:
        return cls.image_request_extra_fields() | frozenset(
            {
                "action",
                "action_fps",
                "action_mode",
                "action_normalization",
                "action_view_point",
                "condition_frame_indexes",
                "condition_frame_indexes_vision",
                "condition_video_keep",
                "control_guidance",
                "control_guidance_interval",
                "control_hint",
                "control_path",
                "domain_id",
                "domain_name",
                "generate_sound",
                "guardrails",
                "max_frames",
                "num_conditional_frames",
                "num_first_chunk_conditional_frames",
                "num_video_frames_per_chunk",
                "raw_action_dim",
                "share_vision_temporal_positions",
                "show_control_condition",
                "show_input",
                "sound_duration",
            }
        )

    def _resolve_control_paths(self) -> list[str]:
        cp = self.control_path
        if cp is None:
            return []
        if isinstance(cp, str):
            return [cp] if cp else []
        return [p for p in cp if isinstance(p, str) and p]

    def _resolve_control_hints(self) -> list[str]:
        hint = self.control_hint
        if hint is None:
            return []
        hints = [hint] if isinstance(hint, str) else list(hint)
        hints = [h for h in hints if h]
        for h in hints:
            if h not in self._TRANSFER_DEFAULTS:
                raise ValueError(
                    f"Unknown control_hint {h!r}; expected one of "
                    f"{sorted(self._TRANSFER_DEFAULTS)}"
                )
        return hints

    def _apply_transfer_hint_defaults(self) -> None:
        """Fill tuned per-hint defaults for a single, typed control input.

        Mirrors cosmos-framework: defaults apply only when there is exactly one
        control input with a known hint type, and only to fields the user did
        not pass explicitly (tracked via ``_explicit_fields``). Multi-hint
        transfer keeps the request's own ``guidance`` / ``control_guidance`` /
        ``flow_shift``.
        """
        if len(self._resolve_control_paths()) != 1:
            return
        hints = self._resolve_control_hints()
        if len(hints) != 1:
            return
        defaults = self._TRANSFER_DEFAULTS.get(hints[0])
        if defaults is None:
            return
        explicit = getattr(self, "_explicit_fields", None) or set()
        if "control_guidance" not in explicit:
            self.control_guidance = defaults["control_guidance"]
        if "guidance_scale" not in explicit:
            self.guidance_scale = defaults["guidance"]
        if "flow_shift" not in explicit and self.flow_shift is None:
            self.flow_shift = defaults["shift"]
        for field_name in (
            "num_frames",
            "fps",
            "num_video_frames_per_chunk",
        ):
            if field_name in defaults and field_name not in explicit:
                setattr(self, field_name, defaults[field_name])

    @classmethod
    def lower_video_request_kwargs(
        cls, request: Any, kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        kwargs = super().lower_video_request_kwargs(request, dict(kwargs))
        extras = getattr(request, "model_extra", None) or {}

        if "use_guardrails" not in kwargs and extras.get("guardrails") is not None:
            kwargs["use_guardrails"] = _parse_request_value(extras["guardrails"])

        condition_indexes = kwargs.get("condition_frame_indexes")
        if condition_indexes is None:
            condition_indexes = extras.get("condition_frame_indexes_vision")
        condition_indexes = _optional_int_list(condition_indexes)
        if condition_indexes is not None:
            kwargs["condition_frame_indexes"] = condition_indexes

        if "sound_duration" in kwargs:
            kwargs["sound_duration"] = float(
                _parse_request_value(kwargs["sound_duration"])
            )
        generate_sound = _parse_request_value(extras.get("generate_sound"))
        if generate_sound is False:
            kwargs["sound_duration"] = 0.0
        elif generate_sound is True and "sound_duration" not in kwargs:
            kwargs["sound_duration"] = float(kwargs["num_frames"]) / float(
                kwargs["fps"]
            )

        for name in ("control_path", "control_hint"):
            value = _parse_request_value(kwargs.get(name))
            if isinstance(value, (list, tuple)):
                value = [str(item) for item in value if str(item).strip()]
            elif value is not None and not isinstance(value, str):
                value = str(value)
            if isinstance(value, str):
                value = value if value.strip() else None
            if value:
                kwargs[name] = value
            else:
                kwargs.pop(name, None)

        if "control_guidance" in kwargs:
            kwargs["control_guidance"] = float(
                _parse_request_value(kwargs["control_guidance"])
            )
        if "control_guidance_interval" in kwargs:
            interval = _parse_request_value(kwargs["control_guidance_interval"])
            if interval is None or (isinstance(interval, str) and not interval.strip()):
                kwargs.pop("control_guidance_interval")
            else:
                if not isinstance(interval, (list, tuple)):
                    interval = [interval]
                kwargs["control_guidance_interval"] = tuple(
                    float(item) for item in interval
                )

        for name in (
            "num_video_frames_per_chunk",
            "num_conditional_frames",
            "num_first_chunk_conditional_frames",
            "max_frames",
        ):
            value = _parse_request_value(kwargs.get(name))
            if value is not None and value != "":
                kwargs[name] = int(value)

        for name in (
            "show_control_condition",
            "show_input",
            "share_vision_temporal_positions",
        ):
            value = _parse_request_value(kwargs.get(name))
            if value is None or (isinstance(value, str) and not value.strip()):
                kwargs.pop(name, None)
            elif isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"1", "true", "yes", "on"}:
                    kwargs[name] = True
                elif normalized in {"0", "false", "no", "off"}:
                    kwargs[name] = False
                else:
                    raise ValueError(f"Invalid boolean value: {value!r}")
            else:
                kwargs[name] = bool(value)

        for name in (
            "condition_video_keep",
            "action_mode",
            "domain_id",
            "domain_name",
            "raw_action_dim",
            "action_fps",
            "action",
            "action_view_point",
            "action_normalization",
        ):
            value = _parse_request_value(kwargs.get(name))
            if isinstance(value, str) and not value.strip():
                kwargs.pop(name, None)
            elif value is not None:
                kwargs[name] = value

        hint = kwargs.get("control_hint")
        paths = kwargs.get("control_path")
        hints = [hint] if isinstance(hint, str) else list(hint or [])
        control_paths = [paths] if isinstance(paths, str) else list(paths or [])
        if len(control_paths) == 1 and hints == ["wsm"]:
            defaults = cls._TRANSFER_DEFAULTS["wsm"]
            if request.num_frames is None:
                kwargs["num_frames"] = defaults["num_frames"]
            if request.fps is None:
                kwargs["fps"] = defaults["fps"]
        return kwargs

    def _adjust(self, server_args) -> None:
        # adjust distil and edge args — read from the pre-computed config fields
        # so no checkpoint download happens at request time.
        pipeline_config = server_args.pipeline_config
        if self.action_stats_path is None:
            self.action_stats_path = getattr(pipeline_config, "action_stats_path", None)
        distilled_sigmas = pipeline_config.distilled_sigmas
        if distilled_sigmas is not None:
            self.num_inference_steps = len(distilled_sigmas)
        self._resolve_variant_defaults(
            bool(pipeline_config.is_edge),
            is_distilled=distilled_sigmas is not None,
        )

        # adjust action args
        action_output = False
        if self.action_mode is not None:
            self.action_mode = str(self.action_mode).strip().lower()
            if self.action_mode not in (
                "policy",
                "forward_dynamics",
                "inverse_dynamics",
            ):
                raise ValueError(
                    f"Unsupported action_mode={self.action_mode!r}; expected "
                    "'policy', 'forward_dynamics', or 'inverse_dynamics'."
                )
            action_output = self.action_mode != "forward_dynamics"

        # Apply transfer per-hint defaults before the base resolves remaining
        # fields (e.g. flow_shift per mode), so an unset flow_shift can pick up
        # the hint's tuned shift.
        self._apply_transfer_hint_defaults()
        control_paths = self._resolve_control_paths()
        if control_paths:
            if pipeline_config.distilled_sigmas is not None:
                raise ValueError(
                    "Cosmos3 distilled checkpoints do not support transfer inference"
                )
            if pipeline_config.is_edge:
                raise ValueError(
                    "Cosmos3 Edge checkpoints do not support transfer inference"
                )
            if self.num_frames == 1:
                raise ValueError(
                    "Cosmos3 transfer inference is supported only for video outputs"
                )
            if self.image_path is not None:
                raise ValueError(
                    "Cosmos3 transfer accepts control videos and an optional source "
                    "video, not an image input"
                )
            if self.action_mode is not None:
                raise ValueError(
                    "Cosmos3 transfer cannot be combined with action generation"
                )
            if float(self.sound_duration or 0.0) > 0.0:
                raise ValueError(
                    "Cosmos3 transfer cannot be combined with sound generation"
                )
        super()._adjust(server_args)

        # Policy and inverse dynamics produce actions. Forward dynamics consumes
        # actions to produce video and therefore remains a visual request.
        if action_output:
            self.data_type = DataType.ACTION
            self.save_output = False
            self.return_file_paths_only = False
            self.return_frames = False
            self.output_file_name = None
            self.output_compression = 0

    def _validate(self) -> None:
        super()._validate()
        paths = self._resolve_control_paths()
        hints = self._resolve_control_hints()
        if hints and len(hints) != len(paths):
            raise ValueError(
                "control_hint must contain exactly one entry per control_path "
                f"(got {len(hints)} hint(s) for {len(paths)} path(s))"
            )
        if self.control_guidance_interval is not None:
            if len(self.control_guidance_interval) != 2:
                raise ValueError(
                    "control_guidance_interval must contain exactly two values"
                )
            lo, hi = self.control_guidance_interval
            if float(lo) > float(hi):
                raise ValueError(
                    "control_guidance_interval must be ordered as (low, high)"
                )
        if self.num_video_frames_per_chunk <= 0:
            raise ValueError("num_video_frames_per_chunk must be positive")
        if self.num_conditional_frames < 0:
            raise ValueError("num_conditional_frames must be non-negative")
        if self.num_conditional_frames >= self.num_video_frames_per_chunk:
            raise ValueError(
                "num_conditional_frames must be smaller than "
                "num_video_frames_per_chunk"
            )
        if self.num_first_chunk_conditional_frames < 0:
            raise ValueError("num_first_chunk_conditional_frames must be non-negative")
        if self.max_frames <= 0:
            raise ValueError("max_frames must be positive")

    def _guidance_is_explicit(self) -> bool:
        explicit = getattr(self, "_explicit_fields", None)
        return explicit is not None and "guidance_scale" in explicit

    def _resolve_variant_defaults(
        self, is_edge: bool, is_distilled: bool = False
    ) -> None:
        """Fill unset resolution/guidance with the variant's defaults.

        Base resolution defaulting (``supported_resolutions[0]``) covers the
        non-Edge path; only Edge and guidance need explicit handling here.
        """
        is_t2i = self.num_frames == 1
        if is_distilled:
            # Guidance is distilled into the model; run a single forward.
            self.guidance_scale = 1.0
        elif is_edge and not self._guidance_is_explicit():
            self.guidance_scale = (
                COSMOS3_EDGE_T2I_GUIDANCE_SCALE
                if is_t2i
                else COSMOS3_EDGE_T2V_GUIDANCE_SCALE
            )
        if is_t2i and not is_distilled and self.guidance_interval is None:
            self.guidance_interval = COSMOS3_T2I_GUIDANCE_INTERVAL
        if is_edge:
            self.supported_resolutions = COSMOS3_EDGE_SUPPORTED_RESOLUTIONS
            if self.height is None and self.width is None:
                if is_t2i:
                    self.width = self.height = COSMOS3_EDGE_T2I_SIZE
                else:
                    self.width, self.height = (
                        COSMOS3_EDGE_T2V_WIDTH,
                        COSMOS3_EDGE_T2V_HEIGHT,
                    )

    def _set_output_file_name(self) -> None:
        # Action outputs never need a visual filename. This also avoids hashing
        # in-memory observation images while base visual adjustment is running.
        if self.action_mode in ("policy", "inverse_dynamics"):
            return
        # The pipeline config's ``task_type=TI2V`` drives ``data_type`` to
        # VIDEO, but a single-frame request is a T2I and must pick the IMAGE
        # extension. Flip before the base derives the file name.
        if self.num_frames == 1:
            self.data_type = DataType.IMAGE
        super()._set_output_file_name()
