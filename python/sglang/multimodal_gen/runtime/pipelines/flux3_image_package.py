# SPDX-License-Identifier: Apache-2.0
"""Resolver for FLUX.3 image (Klein) packages, local or on the Hugging Face Hub."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import msgspec
from huggingface_hub.utils import validate_repo_id
from safetensors import safe_open

from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import snapshot_download
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

FORMAT_VERSION = 1
# Both release repos use flux-3-image-klein in their config and weight metadata.
# Sampling mode comes from the weights, not this shared identifier.
MODEL_NAMES = ("flux-3-image-klein", "flux-3-image-klein-base")
CONFIG_NAME = "config.json"
MANIFEST_NAME = "manifest.json"
WEIGHTS_NAME = "model.safetensors"
QUANT_CONFIG_NAME = "hf_quant_config.json"
# hf://<namespace>/<repo>/<path>: an asset in another Hub repo (split checkpoints).
HF_ASSET_PREFIX = "hf://"
# The BFL reference tells a flow-map DiT from its weights, never from config.json.
FLOW_MAP_KEY = "ref_time_in.in_layer.weight"
FP8R_VARIANT = "fp8r"
NVFP4_VARIANT = "nvfp4"
QUANTIZATIONS = (FP8R_VARIANT, NVFP4_VARIANT)
# The NVFP4 decode the runtime implements; a manifest must state every clause.
_NVFP4_SCALE_CLAUSES = (
    "e2m1(weight) * weight_scale[:, k // 16] * weight_scale_2",
    "weight_scale_2 = amax(weight) / (448 * 6)",
    "input_scale = p99.9 of per-call input amax / (448 * 6)",
)
# Variant directory -> the quantization its config must declare. ``sd`` is the
# step-distilled flow-map DiT; ``sd-*`` are its quantized exports.
VARIANT_QUANTIZATION = {
    FP8R_VARIANT: FP8R_VARIANT,
    NVFP4_VARIANT: NVFP4_VARIANT,
    "sd": None,
    f"sd-{FP8R_VARIANT}": FP8R_VARIANT,
    f"sd-{NVFP4_VARIANT}": NVFP4_VARIANT,
}
TORCH_DTYPES = ("bfloat16", "float16", "float32")


class Flux3ImagePackageError(ValueError):
    pass


@dataclass(frozen=True)
class Flux3ImageDiTPackageConfig:
    hidden: int
    heads: int
    depth: int
    depth_single: int
    mlp_ratio: float
    axes_dim: tuple[int, ...]
    theta: int
    in_channels: dict[str, int]
    out_modes: tuple[str, ...]
    vec_in_dim: int | None = None
    image_rope_scale: float = 0.5
    # A flow-map DiT also embeds the time each step jumps to (``ref_time_in``).
    # ``None`` until resolution reads it from the weights.
    flow_map: bool | None = None


@dataclass(frozen=True)
class Flux3ImageTextEncoderPackageConfig:
    path: str
    tokenizer: str
    layers: tuple[int, ...]
    pad_multiple: int
    max_tokens: int
    dtype: str = "bfloat16"


@dataclass(frozen=True)
class Flux3ImageVAEPackageConfig:
    path: str
    dtype: str = "float32"


@dataclass(frozen=True)
class Flux3ImageSamplerPackageConfig:
    steps: int
    alpha: float
    size: tuple[int, int]
    # CFG scale; ``None`` for a flow map, which samples without CFG.
    guidance: float | None = None


@dataclass(frozen=True)
class Flux3ImagePackageConfig:
    format_version: int
    model: str
    quantization: str | None
    torch_dtype: str
    dit: Flux3ImageDiTPackageConfig
    text_encoder: Flux3ImageTextEncoderPackageConfig
    vae: Flux3ImageVAEPackageConfig
    sampler: Flux3ImageSamplerPackageConfig

    @classmethod
    def from_dict(cls, raw: dict) -> Flux3ImagePackageConfig:
        try:
            config = msgspec.convert(raw, type=cls)
        except msgspec.ValidationError as exc:
            raise Flux3ImagePackageError(str(exc)) from exc
        _validate_config(config)
        return config


@dataclass(frozen=True)
class Flux3ImagePackage:
    root: Path
    config: Flux3ImagePackageConfig
    weights: Path
    vae: Path
    text_encoder: Path
    tokenizer: Path
    # Optional for a BF16 DiT; a quantized DiT needs it for its scale convention.
    manifest: Path | None
    directory: Path | None = None
    variant: str | None = None
    quant_config: Path | None = None


def load_flux3_image_package_config(path: str | Path) -> Flux3ImagePackageConfig:
    path = Path(path)
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError:
        raise Flux3ImagePackageError(
            f"FLUX.3 package config not found: {path}"
        ) from None
    except json.JSONDecodeError as exc:
        raise Flux3ImagePackageError(f"{path}: invalid JSON ({exc})") from None
    try:
        return Flux3ImagePackageConfig.from_dict(raw)
    except Flux3ImagePackageError as exc:
        raise Flux3ImagePackageError(f"{path}: {exc}") from None


def resolve_flux3_image_package(
    source: str | Path, variant: str | None = None, revision: str | None = None
) -> Flux3ImagePackage:
    """Resolve a package root, one of its variants, or a Hub repo id; a Hub
    package downloads only the files the selected DiT needs."""
    directory = Path(source).expanduser()
    if not directory.exists() and _is_hub_repo_id(str(source)):
        directory = _download_package(str(source), variant=variant, revision=revision)
    if directory.is_file():
        if directory.name not in {
            CONFIG_NAME,
            WEIGHTS_NAME,
            MANIFEST_NAME,
            QUANT_CONFIG_NAME,
        }:
            raise Flux3ImagePackageError(
                f"expected a FLUX.3 package directory, {CONFIG_NAME}, or "
                f"{WEIGHTS_NAME}; got {directory}"
            )
        directory = directory.parent
    if not directory.is_dir():
        raise Flux3ImagePackageError(
            f"FLUX.3 package not found: {source} is neither a local directory "
            "nor a Hugging Face repo id"
        )

    direct_variant = directory.parent.name == "variants"
    if direct_variant:
        detected_variant = directory.name
        root = directory.parent.parent
        if variant is not None and variant != detected_variant:
            raise Flux3ImagePackageError(
                f"variant path selects {detected_variant!r}, but variant={variant!r}"
            )
        variant = detected_variant
    else:
        root = directory
        if variant is not None:
            variant = _existing_variant(root, variant)
            directory = root / "variants" / variant
    if not directory.is_dir():
        available = sorted(
            path.parent.name for path in root.glob(f"variants/*/{CONFIG_NAME}")
        )
        raise Flux3ImagePackageError(
            f"unsupported FLUX.3 package variant {variant!r}: {directory} does not "
            f"exist; this package has {available or 'no variants'}"
        )

    config = load_flux3_image_package_config(directory / CONFIG_NAME)
    if config.model not in MODEL_NAMES:
        logger.debug(
            "FLUX.3 package model %r is not one of %s; loading it as FLUX.3 image "
            "since its config has the image DiT layout",
            config.model,
            ", ".join(MODEL_NAMES),
        )
    _check_quantization(config, variant)
    if config.torch_dtype != "bfloat16":
        raise Flux3ImagePackageError(
            f"FLUX.3 package expected torch_dtype='bfloat16', "
            f"got {config.torch_dtype!r}"
        )

    weights = directory / WEIGHTS_NAME
    vae = _resolve_asset(root, config.vae.path, "vae.path")
    text_encoder = _resolve_asset(root, config.text_encoder.path, "text_encoder.path")
    tokenizer = _resolve_asset(
        root, config.text_encoder.tokenizer, "text_encoder.tokenizer"
    )
    manifest = directory / MANIFEST_NAME
    quant_config = (
        directory / QUANT_CONFIG_NAME if config.quantization is not None else None
    )
    required = {
        "DiT weights": weights,
        "image VAE": vae,
        "text encoder": text_encoder,
        "tokenizer": tokenizer,
    }
    if quant_config is not None:
        required["manifest"] = manifest
        required[f"{config.quantization.upper()} quantization config"] = quant_config
    missing = [
        f"{name}: {path}" for name, path in required.items() if not path.exists()
    ]
    if missing:
        raise Flux3ImagePackageError(
            "FLUX.3 package is incomplete; missing " + ", ".join(missing)
        )
    if manifest.exists():
        _validate_manifest(manifest, config)
    return Flux3ImagePackage(
        root=root,
        config=_resolve_flow_map(config, weights),
        weights=weights,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        manifest=manifest if manifest.exists() else None,
        directory=directory,
        variant=variant,
        quant_config=quant_config,
    )


def _existing_variant(root: Path, variant: str) -> str:
    # The distilled repo may keep the combined package's sd-fp8r/sd-nvfp4 names;
    # fp8r/nvfp4 then select them. Never the reverse: sd-* must stay distilled.
    alias = f"sd-{variant}"
    if (root / "variants" / variant).is_dir() or alias not in VARIANT_QUANTIZATION:
        return variant
    if (root / "variants" / alias).is_dir():
        logger.info("FLUX.3 package variant %r not found; using %r", variant, alias)
        return alias
    return variant


def _check_quantization(config: Flux3ImagePackageConfig, variant: str | None) -> None:
    _check(
        config.quantization is None or config.quantization in QUANTIZATIONS,
        f"unsupported FLUX.3 package quantization {config.quantization!r}",
    )
    # A known variant name pins its quantization; a package root or another
    # name (e.g. a standalone quantized repo) takes the config's.
    if variant in VARIANT_QUANTIZATION:
        expected = VARIANT_QUANTIZATION[variant]
        _check(
            config.quantization == expected,
            f"FLUX.3 package variant {variant!r} requires "
            f"quantization={expected!r}, got {config.quantization!r}",
        )


def _resolve_flow_map(
    config: Flux3ImagePackageConfig, weights: Path
) -> Flux3ImagePackageConfig:
    flow_map = config.dit.flow_map
    if flow_map is None:
        flow_map = _has_flow_map_weights(weights)
    sampler = config.sampler
    if flow_map and sampler.guidance is not None:
        logger.warning(
            "FLUX.3 package %s is a flow map, which samples without CFG; "
            "ignoring sampler.guidance=%s",
            weights,
            sampler.guidance,
        )
        sampler = dataclasses.replace(sampler, guidance=None)
    _check(
        flow_map or _is_num(sampler.guidance),
        "sampler.guidance must be numeric for a CFG (non-flow-map) DiT",
    )
    return dataclasses.replace(
        config, dit=dataclasses.replace(config.dit, flow_map=flow_map), sampler=sampler
    )


def _has_flow_map_weights(weights: Path) -> bool:
    try:
        with safe_open(str(weights), framework="pt") as f:
            return FLOW_MAP_KEY in f.keys()
    except Exception as exc:
        raise Flux3ImagePackageError(
            f"{weights}: cannot read the safetensors header ({exc})"
        ) from None


def _is_hub_repo_id(source: str) -> bool:
    if source.startswith(("/", ".", "~")) or source.count("/") != 1:
        return False
    try:
        validate_repo_id(source)
    except ValueError:
        return False
    return True


def _hub_snapshot(
    repo_id: str, *, patterns: list[str], revision: str | None = None
) -> Path:
    try:
        return Path(
            snapshot_download(
                repo_id=repo_id, allow_patterns=patterns, revision=revision
            )
        )
    except Exception as exc:
        raise Flux3ImagePackageError(
            f"could not fetch {patterns} from Hugging Face repo {repo_id!r}: {exc}"
        ) from exc


def _asset_patterns(path: str) -> list[str]:
    # Hub patterns are fnmatch globs, where ``*`` also crosses ``/``.
    return [path, f"{path}/*"]


def _download_package(
    repo_id: str, *, variant: str | None, revision: str | None
) -> Path:
    prefix = f"variants/{variant}/" if variant else ""
    root = _hub_snapshot(
        repo_id,
        patterns=[
            prefix + name for name in (CONFIG_NAME, MANIFEST_NAME, QUANT_CONFIG_NAME)
        ],
        revision=revision,
    )
    config = load_flux3_image_package_config(root / prefix / CONFIG_NAME)
    patterns = [prefix + WEIGHTS_NAME]
    for reference in (
        config.vae.path,
        config.text_encoder.path,
        config.text_encoder.tokenizer,
    ):
        if not reference.startswith(HF_ASSET_PREFIX):
            patterns += _asset_patterns(reference)
    _hub_snapshot(repo_id, patterns=patterns, revision=revision)
    return root


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise Flux3ImagePackageError(message)


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_num(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _relative(path: str, what: str) -> str:
    value = PurePosixPath(path)
    _check(
        isinstance(path, str)
        and bool(path)
        and not value.is_absolute()
        and ".." not in value.parts,
        f"{what} must be a relative path inside the package, got {path!r}",
    )
    return path


def _hub_asset(path: str, what: str) -> tuple[str, str] | None:
    if not isinstance(path, str) or not path.startswith(HF_ASSET_PREFIX):
        return None
    raw = path.removeprefix(HF_ASSET_PREFIX)
    value = PurePosixPath(raw)
    _check(
        not value.is_absolute() and ".." not in value.parts and len(value.parts) >= 3,
        f"{what} must use hf://<namespace>/<repo>/<path>, got {path!r}",
    )
    repo_id = "/".join(value.parts[:2])
    _check(_is_hub_repo_id(repo_id), f"{what} has an invalid Hub repo in {path!r}")
    return repo_id, "/".join(value.parts[2:])


def _asset_reference(path: str, what: str) -> str:
    if _hub_asset(path, what) is None:
        _relative(path, what)
    return path


def _resolve_asset(root: Path, reference: str, what: str) -> Path:
    hub_asset = _hub_asset(reference, what)
    if hub_asset is None:
        return root / _relative(reference, what)
    repo_id, asset_path = hub_asset
    # Prefer a sibling checkout laid out as <namespace>/<repo>, as the Hub names it.
    namespace, repo_name = repo_id.split("/")
    sibling = root.parent / repo_name / asset_path
    if root.parent.name == namespace and sibling.exists():
        return sibling
    return _hub_snapshot(repo_id, patterns=_asset_patterns(asset_path)) / asset_path


def _validate_config(config: Flux3ImagePackageConfig) -> None:
    _check(
        config.format_version == FORMAT_VERSION,
        f"format_version must be {FORMAT_VERSION}, got {config.format_version!r}",
    )
    _check(
        config.torch_dtype in TORCH_DTYPES,
        f"unsupported torch_dtype {config.torch_dtype!r}",
    )
    dit = config.dit
    for key, value in (
        ("hidden", dit.hidden),
        ("heads", dit.heads),
        ("depth", dit.depth),
        ("depth_single", dit.depth_single),
        ("theta", dit.theta),
    ):
        _check(
            value > 0,
            f"dit.{key} must be a positive integer",
        )
    _check(
        dit.hidden % dit.heads == 0,
        f"dit.hidden {dit.hidden} must be divisible by dit.heads {dit.heads}",
    )
    _check(
        all(_is_int(axis) and axis > 0 and axis % 2 == 0 for axis in dit.axes_dim),
        "dit.axes_dim must contain positive even integers",
    )
    _check(
        sum(dit.axes_dim) == dit.hidden // dit.heads,
        "sum(dit.axes_dim) must equal the attention head dimension",
    )
    _check(
        dit.in_channels == {"image": 128, "image_cond": 128, "text": 20480},
        f"unsupported FLUX.3 Klein channels: {dit.in_channels}",
    )
    _check(
        dit.out_modes == ("image",),
        f"unsupported FLUX.3 Klein output modes: {dit.out_modes}",
    )
    _check(dit.vec_in_dim is None, "FLUX.3 Klein must not define vec_in_dim")
    _check(
        _is_num(dit.image_rope_scale) and dit.image_rope_scale > 0,
        "dit.image_rope_scale must be positive",
    )
    _check(
        dit.flow_map is None or isinstance(dit.flow_map, bool),
        "dit.flow_map must be a boolean",
    )
    text = config.text_encoder
    _asset_reference(text.path, "text_encoder.path")
    _asset_reference(text.tokenizer, "text_encoder.tokenizer")
    _check(bool(text.layers), "text_encoder.layers must not be empty")
    _check(
        _is_int(text.pad_multiple) and text.pad_multiple > 0,
        "text_encoder.pad_multiple must be positive",
    )
    _check(
        _is_int(text.max_tokens) and text.max_tokens >= text.pad_multiple,
        "text_encoder.max_tokens must be >= pad_multiple",
    )
    _asset_reference(config.vae.path, "vae.path")
    _check(
        config.vae.dtype in TORCH_DTYPES,
        f"unsupported VAE dtype {config.vae.dtype!r}",
    )
    sampler = config.sampler
    _check(
        _is_int(sampler.steps) and sampler.steps > 0, "sampler.steps must be positive"
    )
    _check(
        sampler.guidance is None or _is_num(sampler.guidance),
        "sampler.guidance must be numeric or null",
    )
    _check(
        _is_num(sampler.alpha) and sampler.alpha > 0,
        "sampler.alpha must be positive",
    )
    _check(
        len(sampler.size) == 2
        and all(
            _is_int(value) and value > 0 and value % 16 == 0 for value in sampler.size
        ),
        "sampler.size must be positive multiples of 16",
    )


def _validate_manifest(path: Path, config: Flux3ImagePackageConfig) -> None:
    try:
        manifest = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise Flux3ImagePackageError(f"{path}: invalid JSON ({exc})") from None
    _check(
        manifest.get("format_version") == 1,
        f"{path}: unsupported manifest format_version",
    )
    if manifest.get("model", config.model) != config.model:
        logger.warning(
            "%s: manifest model %r differs from config model %r",
            path,
            manifest.get("model"),
            config.model,
        )
    _check(
        manifest.get("quantization", config.quantization) == config.quantization,
        f"{path}: manifest quantization does not match config",
    )
    if config.quantization == FP8R_VARIANT:
        expected = {
            "dtype": "float8_e4m3fn",
            "scale_convention": "weight.float() * weight_scale[:, None]",
        }
        for key, value in expected.items():
            _check(
                manifest.get(key) == value,
                f"{path}: expected {key}={value!r}, got {manifest.get(key)!r}",
            )
        # Exports may keep different layers in BF16, so counts are only self-consistent.
        matrices = manifest.get("fp8_matrices")
        _check(
            _is_int(matrices)
            and matrices > 0
            and manifest.get("weight_scales") == matrices,
            f"{path}: expected one weight scale per FP8 matrix, got "
            f"fp8_matrices={matrices!r}, weight_scales={manifest.get('weight_scales')!r}",
        )
    elif config.quantization == NVFP4_VARIANT:
        _check(
            str(manifest.get("dtype", "")).startswith("nvfp4: weight uint8"),
            f"{path}: manifest dtype does not describe packed NVFP4 weights",
        )
        _check(
            manifest.get("group_size") == 16,
            f"{path}: expected group_size=16, got {manifest.get('group_size')!r}",
        )
        # Extra clauses only describe how scales were calibrated, not how they decode.
        clauses = str(manifest.get("scale_convention", "")).split("; ")
        missing = [c for c in _NVFP4_SCALE_CLAUSES if c not in clauses]
        _check(not missing, f"{path}: scale_convention lacks {missing!r}")
        # Exports may keep different layers in BF16, so counts are only self-consistent.
        # Older exports store the per-block scale count under weight_scales.
        matrices = manifest.get("nvfp4_matrices")
        counts = {
            key: manifest.get(key)
            for key in ("weight_scale_tensors", "weight_scale_2", "input_scales")
        }
        if counts["weight_scale_tensors"] is None:
            counts["weight_scale_tensors"] = manifest.get("weight_scales")
        _check(
            _is_int(matrices)
            and matrices > 0
            and all(count == matrices for count in counts.values()),
            f"{path}: expected {matrices!r} NVFP4 matrices with one scale set "
            f"each, got {counts!r}",
        )
