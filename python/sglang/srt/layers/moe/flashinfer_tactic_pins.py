"""Pinned FlashInfer TRT-LLM-gen FP4 MoE tactics, read from a tuning file.

``SGLANG_FLASHINFER_TACTIC_PIN_FILE`` names a JSON file such as::

    {
      "_metadata": {
        "flashinfer_version": "0.7.2",
        "cuda_version": "13.0",
        "cublas_version": "13.1.0",
        "gpu": "NVIDIA B200",
        "sm": "10.0"
      },
      "trtllm_fp4_block_scale_moe": {
        "hidden_size": 2560,
        "intermediate_size": 640,
        "num_experts": 512,
        "top_k": 10,
        "split_k_tactics": [[4, 11], [4, 12]],
        "prefill": {"1024": [64, 7], "16384": [256, 3], "8192": -1},
        "decode": {"4": [16, 5], "128": [64, 2]}
      }
    }

A tactic is FlashInfer's ``[tile_n, config]`` pair, or -1 for its default. Tactic
indices hold for one software stack, device and MoE shape only, so a file applies
only when every ``_metadata`` field equals this process's (FlashInfer's
``_collect_metadata()`` fields plus the device's SM) and a table only where the
shape matches; everywhere else the tuner's behavior is kept, with a warning. Pins
never keep a split-K tactic: those reduce in a different order.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
from typing import Optional, Union

import msgspec

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

MOE_OP = "flashinfer::trtllm_fp4_block_scale_moe"
DEFAULT_TACTIC = -1
Tactic = Union[int, tuple[int, int]]
MoeShape = tuple[int, int, int, int]  # hidden_size, intermediate_size, num_experts, top_k

_METADATA_KEY = "_metadata"
# Every field must match exactly; all but "sm" are FlashInfer's own cache metadata.
_METADATA_FIELDS = ("flashinfer_version", "cuda_version", "cublas_version", "gpu", "sm")
# Values FlashInfer records when it cannot tell; such a file, or such a process,
# cannot be matched (FlashInfer's own cache gate refuses them too).
_INDETERMINATE_METADATA = ("unknown", "None")
# FlashInfer builds whose AutoTuner cache-key layout cache_key_shape and
# valid_tactics_by_bucket below read.
_SUPPORTED_FLASHINFER = ("0.7.2",)
# Index of hidden_states among MoeRunnerInputs in those builds.
_HIDDEN_STATES_INPUT = 4
_MAX_DECODE_BUCKET = 512


class MoeTacticPins(msgspec.Struct, frozen=True):
    flashinfer_version: str
    shape: MoeShape
    split_k: frozenset[tuple[int, int]]
    # Token bucket -> tactic; None when the file has no table for that phase.
    prefill: Optional[dict[int, Tactic]] = None
    decode: Optional[dict[int, Tactic]] = None


def _parse_tactic(value, where: str) -> Tactic:
    if value == DEFAULT_TACTIC:
        return DEFAULT_TACTIC
    if (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(v, int) and not isinstance(v, bool) for v in value)
        and value[0] > 0
        and value[1] >= 0
    ):
        return (value[0], value[1])
    raise ValueError(f"{where}: a tactic is [tile_n, config] or -1, got {value!r}")


def _parse_table(table, where: str) -> dict[int, Tactic]:
    if not isinstance(table, dict):
        raise ValueError(f"{where} must map token buckets to tactics")
    pins = {}
    for bucket, tactic in table.items():
        try:
            tokens = int(bucket)
        except ValueError:
            raise ValueError(f"{where}: bad token bucket {bucket!r}") from None
        if tokens <= 0:
            raise ValueError(f"{where}: bad token bucket {bucket!r}")
        pins[tokens] = _parse_tactic(tactic, f"{where}[{bucket}]")
    return pins


def parse_metadata(document: dict) -> dict[str, str]:
    metadata = document.get(_METADATA_KEY)
    if not isinstance(metadata, dict) or not all(
        isinstance(metadata.get(field), str) and metadata[field]
        for field in _METADATA_FIELDS
    ):
        raise ValueError(
            f"tactic pin file: '{_METADATA_KEY}' must give {', '.join(_METADATA_FIELDS)}"
        )
    return {field: metadata[field] for field in _METADATA_FIELDS}


def parse_moe_tactic_pins(document: dict) -> Optional[MoeTacticPins]:
    """The MoE section of a tuning file, validated; None if the file has none."""
    version = parse_metadata(document)["flashinfer_version"]
    section = document.get("trtllm_fp4_block_scale_moe")
    if section is None:
        return None
    where = "tactic pin file: trtllm_fp4_block_scale_moe"
    try:
        shape = tuple(
            int(section[name])
            for name in ("hidden_size", "intermediate_size", "num_experts", "top_k")
        )
    except (KeyError, TypeError, ValueError):
        raise ValueError(
            f"{where} needs integer hidden_size, intermediate_size, num_experts and top_k"
        ) from None
    split_k = frozenset(
        _parse_tactic(t, f"{where}.split_k_tactics")
        for t in section.get("split_k_tactics", [])
    )
    if DEFAULT_TACTIC in split_k:
        raise ValueError(f"{where}.split_k_tactics cannot list the default tactic")
    prefill = section.get("prefill")
    decode = section.get("decode")
    if decode is not None:
        decode = _parse_table(decode, f"{where}.decode")
        for bucket in decode:
            # The decode-shaped tune stores power-of-two buckets up to 512 tokens.
            if bucket > _MAX_DECODE_BUCKET or bucket & (bucket - 1):
                raise ValueError(
                    f"{where}.decode: bucket {bucket} is not a power of two up to "
                    f"{_MAX_DECODE_BUCKET}"
                )
    return MoeTacticPins(
        flashinfer_version=version,
        shape=shape,
        split_k=split_k,
        prefill=None if prefill is None else _parse_table(prefill, f"{where}.prefill"),
        decode=decode,
    )


@functools.lru_cache(maxsize=1)
def read_tuning_file(path: str) -> dict:
    with open(path) as f:
        document = json.load(f)
    if not isinstance(document, dict):
        raise ValueError(f"tactic pin file {path}: expected a JSON object")
    parse_metadata(document)
    return document


def tuning_file_digest(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


@functools.lru_cache(maxsize=1)
def _moe_pins_of(path: str) -> Optional[MoeTacticPins]:
    return parse_moe_tactic_pins(read_tuning_file(path))


@functools.lru_cache(maxsize=None)
def _warn_inert_once(reason: str) -> None:
    logger.warning("FlashInfer tactic pins not applied: %s", reason)


def _current_metadata() -> dict[str, str]:
    import torch
    from flashinfer.autotuner import _collect_metadata

    collected = _collect_metadata()
    current = {
        field: str(collected.get(field, "unknown"))
        for field in _METADATA_FIELDS
        if field != "sm"
    }
    major, minor = torch.cuda.get_device_capability()
    current["sm"] = f"{major}.{minor}"
    return current


def tuning_file_applies(path: str, *, need_cache_key_layout: bool) -> bool:
    """Whether the tuning file at ``path`` was written for exactly this stack and
    device (and, if asked, a FlashInfer whose tuner cache-key layout is read here)."""
    pinned = parse_metadata(read_tuning_file(path))
    current = _current_metadata()
    indeterminate = [
        field
        for field in _METADATA_FIELDS
        if pinned[field] in _INDETERMINATE_METADATA
        or current[field] in _INDETERMINATE_METADATA
    ]
    if indeterminate:
        _warn_inert_once(f"{path}: indeterminate metadata ({', '.join(indeterminate)})")
        return False
    mismatched = [
        f"{field}: file={pinned[field]!r} here={current[field]!r}"
        for field in _METADATA_FIELDS
        if pinned[field] != current[field]
    ]
    if mismatched:
        _warn_inert_once(f"{path} is for another stack ({'; '.join(mismatched)})")
        return False
    if need_cache_key_layout and current["flashinfer_version"] not in _SUPPORTED_FLASHINFER:
        _warn_inert_once(
            f"FlashInfer {current['flashinfer_version']}'s tuner cache layout is not known"
        )
        return False
    return True


def active_moe_tactic_pins() -> Optional[MoeTacticPins]:
    """The MoE pins that apply to this process, or None.

    A malformed file raises, so a typo fails the boot instead of silently serving
    the tuner's picks; a file for another stack or device is ignored with a warning.
    """
    path = envs.SGLANG_FLASHINFER_TACTIC_PIN_FILE.get()
    if not path:
        return None
    pins = _moe_pins_of(path)
    if pins is None or not tuning_file_applies(path, need_cache_key_layout=True):
        return None
    return pins


def normalize_tactic(tactic) -> Tactic:
    if tactic is None or isinstance(tactic, int):
        return DEFAULT_TACTIC
    tile, config = (int(v) for v in tactic)
    return DEFAULT_TACTIC if DEFAULT_TACTIC in (tile, config) else (tile, config)


def cache_key_bucket(key) -> int:
    return int(key.nearest_profile[_HIDDEN_STATES_INPUT][0])


def cache_key_shape(key) -> Optional[MoeShape]:
    """(hidden, intermediate, experts, top_k) of an AutoTuner key's extras, or None."""
    extras = key.extras
    try:
        # No fused shared experts: they would offset the expert and top-k counts.
        if len(extras) != 16 or int(extras[4]) != 0:
            return None
        return int(extras[8]), int(extras[10]), int(extras[2]), int(extras[1])
    except (TypeError, ValueError):
        return None


def valid_tactics_by_bucket(valid_tactics_dict, shape: MoeShape) -> dict[int, set]:
    """{token bucket: tactics} that FlashInfer listed as valid for ``shape``."""
    hidden, intermediate, experts, top_k = shape
    valid = {}
    for (_, query), tactics in list(valid_tactics_dict.items()):
        # (dtype_act, dtype_w, fp8_quant, top_k, hidden, hidden_out, intermediate,
        #  experts, activation, shuffled, layout, per_token_scaling, num_tokens, lora)
        if len(query) != 14 or (query[3], query[4], query[6], query[7]) != (
            top_k,
            hidden,
            intermediate,
            experts,
        ):
            continue
        valid.setdefault(int(query[12]), set()).update(
            normalize_tactic(t) for t in tactics
        )
    return valid


def settle_prefill_tactics(
    winners: dict,
    new_keys,
    *,
    pins: Optional[MoeTacticPins],
    valid: dict[int, set],
) -> list[tuple[int, Tactic, Tactic, str]]:
    """Apply the prefill table to the tuner entries a prefill tuning pass added.

    With a table, each such bucket takes its pinned tactic; a bucket the table does
    not list, or whose pin FlashInfer did not offer there, falls back to the default
    tactic, which is what an untuned prefill bucket runs. Without a table the
    tuner's pick stands unless it is a split-K tactic. Edits ``winners`` in place
    and returns (bucket, final, tuned, source) per entry.
    """
    rows = []
    for key in sorted(new_keys, key=cache_key_bucket):
        tactic, profile = winners[key]
        bucket = cache_key_bucket(key)
        tuned = normalize_tactic(tactic)
        final, source = tuned, "tuner"
        pinned_shape = pins is not None and cache_key_shape(key) == pins.shape
        if pinned_shape and pins.prefill is not None:
            final, source = pins.prefill.get(bucket), "pin"
            if final is None:
                final, source = DEFAULT_TACTIC, "pin_absent"
            elif final == DEFAULT_TACTIC:
                source = "pin_default"
            elif final not in valid.get(bucket, ()):
                final, source = DEFAULT_TACTIC, "pin_invalid"
        if pinned_shape and final in pins.split_k:
            final, source = DEFAULT_TACTIC, source + "_split_k"
        if final != tuned:
            winners[key] = (
                DEFAULT_TACTIC if final == DEFAULT_TACTIC else list(final),
                profile,
            )
        rows.append((bucket, final, tuned, source))
    return rows


def settle_decode_tactics(
    winners: dict,
    *,
    pins: MoeTacticPins,
    valid: dict[int, set],
) -> list[tuple[int, Tactic, Tactic, str]]:
    """Apply the decode table to every MoE entry of the pinned shape at a listed
    bucket. Unlisted buckets, other shapes, split-K pins and pins FlashInfer did not
    offer at that bucket keep the tuner's pick. Edits ``winners`` in place and
    returns (bucket, final, current, source) per listed entry."""
    rows = []
    entries = [key for key in winners if key.custom_op == MOE_OP]
    for key in sorted(entries, key=cache_key_bucket):
        bucket = cache_key_bucket(key)
        if bucket not in pins.decode:
            continue
        tactic, profile = winners[key]
        current = normalize_tactic(tactic)
        want = pins.decode[bucket]
        if cache_key_shape(key) != pins.shape:
            rows.append((bucket, current, current, "pin_shape_skip"))
        elif want in pins.split_k:
            rows.append((bucket, current, current, "pin_split_k_refused"))
        elif want != DEFAULT_TACTIC and want not in valid.get(bucket, ()):
            rows.append((bucket, current, current, "pin_invalid"))
        else:
            if want != current:
                winners[key] = (
                    DEFAULT_TACTIC if want == DEFAULT_TACTIC else list(want),
                    profile,
                )
            rows.append(
                (bucket, want, current, "pin" if want != DEFAULT_TACTIC else "pin_default")
            )
    return rows


def pin_decode_moe_tactics() -> int:
    """Apply the decode table after a decode-shaped autotune pass, before any CUDA
    graph captures the tactics. Returns the number of pinned entries."""
    pins = active_moe_tactic_pins()
    if pins is None or pins.decode is None:
        return 0
    from flashinfer.autotuner import AutoTuner
    from flashinfer.fused_moe.backends.trtllm.sm100_runner import MoERunner

    tuner = AutoTuner.get()
    with tuner._lock:
        winners = tuner._winner_cache()
        valid = valid_tactics_by_bucket(MoERunner.valid_tactics_dict, pins.shape)
        rows = settle_decode_tactics(winners, pins=pins, valid=valid)
    for bucket, final, current, source in rows:
        logger.debug(
            "MoE decode tactic pin: bucket=%d tactic=%s tuned=%s source=%s",
            bucket,
            final,
            current,
            source,
        )
    refused = [
        f"{bucket}:{source}"
        for bucket, _, _, source in rows
        if source not in ("pin", "pin_default")
    ]
    if refused:
        _warn_inert_once("decode pins refused, tuner picks kept: " + ", ".join(refused))
    pinned = len(rows) - len(refused)
    logger.info(
        "MoE decode tactic pins: %d entries set from the tuning file, %d refused, "
        "%d listed buckets not tuned this pass (served from the autotune cache or unused)",
        pinned,
        len(refused),
        len(set(pins.decode) - {row[0] for row in rows}),
    )
    return pinned
