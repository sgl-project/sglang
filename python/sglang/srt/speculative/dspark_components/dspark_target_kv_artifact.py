"""Offline binding of a target-KV checkpoint to its fixed-input parity evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Annotated, Literal

import msgspec
from sglang.srt.speculative.dspark_components.dspark_target_kv_contract import (
    read_target_kv_draft_contract,
)
from sglang.srt.training_capture.protocol import (
    ContractError,
    Digest,
    Nonnegative,
    Positive,
    digest_bytes,
)

_ARTIFACTS = ("config.json", "model.safetensors", "validation/inputs.safetensors")
_REPORT = "validation/parity.json"


class _Stage(msgspec.Struct, forbid_unknown_fields=True):
    passed: bool
    elements: Positive
    mismatched: Nonnegative
    nonfinite: Nonnegative
    max_abs: Annotated[float, msgspec.Meta(ge=0)]
    rms: Annotated[float, msgspec.Meta(ge=0)]


class _ParityReport(msgspec.Struct):
    # Runtime/profiler metadata may grow without changing the numerical proof.
    status: Literal["passed"]
    artifact_sha256: dict[str, Digest]
    fixture_sha256: Digest
    weights_sha256: Digest
    rtol: Annotated[float, msgspec.Meta(ge=0)]
    atol: Annotated[float, msgspec.Meta(ge=0)]
    stages: dict[str, _Stage]
    anchors: list[Nonnegative]
    valid_labels: Positive
    loss: Annotated[float, msgspec.Meta(ge=0)]
    gradient_l1: dict[str, float]
    dtype: Literal["bfloat16", "float16", "float32"]
    reference_adapter: Literal["specforge_backbone_with_shared_kv_encoder"]
    reference_attention: Literal["eager", "sdpa", "flex_attention"]
    serving_attention: Literal["triton"]
    specforge_sources: dict[str, Digest]


def _stamp(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _read_metadata(path, stamps, limit):
    before = _stamp(path)
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ContractError(f"checkpoint metadata is too large: {path.name}")
    if _stamp(path) != before:
        raise ContractError(f"checkpoint artifact changed during audit: {path.name}")
    stamps[path] = before
    return data


def _hash_artifact(path, stamps):
    before = _stamp(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    if _stamp(path) != before:
        raise ContractError(f"checkpoint artifact changed during audit: {path.name}")
    stamps[path] = before
    return digest.hexdigest()


def _check_weight_layout(root):
    index = root / "model.safetensors.index.json"
    if (
        {path.name for path in root.glob("*.safetensors")} != {"model.safetensors"}
        or index.exists()
        or index.is_symlink()
    ):
        raise ContractError("parity evidence requires a single model.safetensors")


def _check_parity(config, contract, report):
    layers = config.get("num_hidden_layers")
    if type(layers) is not int or not 1 <= layers <= 4096:
        raise ContractError("checkpoint num_hidden_layers must be in [1, 4096]")
    for name, expected in (
        ("block_size", contract.sequence.prediction_count),
        ("mask_token_id", contract.sequence.mask_token_id),
        ("vocab_size", contract.teacher.vocab_size),
        ("hidden_size", contract.encoder.hidden_size),
    ):
        if type(config.get(name)) is not int or config[name] != expected:
            raise ContractError(f"checkpoint {name} differs from its contract")
    for key in ("dtype", "torch_dtype"):
        if config.get(key) is not None and config[key] != report.dtype:
            raise ContractError("parity dtype differs from the checkpoint")
    for name in ("rtol", "atol"):
        value = getattr(report, name)
        if not math.isfinite(value) or value > getattr(
            contract.validation, f"parity_{name}"
        ):
            raise ContractError("parity tolerances exceed the checkpoint contract")
    if not report.anchors or report.valid_labels > (
        len(report.anchors) * contract.sequence.prediction_count
    ):
        raise ContractError("parity report has invalid window/label coverage")
    rows = len(report.anchors) * contract.sequence.prediction_count
    expected = {
        **{f"layer.{index}": rows * config["hidden_size"] for index in range(layers)},
        "hidden": rows * config["hidden_size"],
        "base": rows * config["vocab_size"],
        "corrected": rows * config["vocab_size"],
    }
    if report.stages.keys() != expected.keys():
        raise ContractError("parity report must cover every layer and logits stage")
    for name, stage in report.stages.items():
        if (
            not stage.passed
            or stage.elements != expected[name]
            or stage.mismatched
            or stage.nonfinite
            or not math.isfinite(stage.max_abs)
            or not math.isfinite(stage.rms)
        ):
            raise ContractError(f"parity stage is incomplete or failed: {name}")
    if not math.isfinite(report.loss) or set(report.gradient_l1) != {
        "kv_encoder",
        "layers",
        "norm",
        "markov_head",
    }:
        raise ContractError("parity report is missing the finite training check")
    if any(not math.isfinite(x) or x <= 0 for x in report.gradient_l1.values()):
        raise ContractError("parity gradients must be finite and nonzero")
    if set(report.specforge_sources) != {
        "specforge.modeling.draft.dflash",
        "specforge.modeling.draft.dflash_kernels",
        "specforge.modeling.draft.dspark",
    }:
        raise ContractError("parity report is missing its training source identity")


def audit_target_kv_checkpoint(directory, *, require_acceptance=False):
    """Verify existing evidence without importing a trainer or running inference.

    This receipt binds local bytes to a trusted parity report. It neither reruns
    numerical parity nor certifies model quality, current-runtime compatibility
    or serving SLOs. Keep the audited deployment directory immutable.
    """
    root = Path(directory)
    stamps = {}
    try:
        _check_weight_layout(root)
        config_bytes = _read_metadata(root / "config.json", stamps, 1 << 20)
        config = msgspec.json.decode(config_bytes, type=dict)
        contract = read_target_kv_draft_contract(config)
        if contract is None:
            raise ContractError("artifact audit requires a target-KV checkpoint")
        report_bytes = _read_metadata(root / _REPORT, stamps, 8 << 20)
        report = msgspec.json.decode(report_bytes, type=_ParityReport)
        _check_parity(config, contract, report)
        artifacts = {"config.json": digest_bytes(config_bytes)}
        for name in _ARTIFACTS[1:]:
            artifacts[name] = _hash_artifact(root / name, stamps)
        if report.artifact_sha256 != artifacts:
            raise ContractError("parity report does not bind the current artifacts")
        if (
            report.fixture_sha256 != artifacts[_ARTIFACTS[2]]
            or report.fixture_sha256 != contract.validation.golden_fixture_sha256
            or report.weights_sha256 != artifacts["model.safetensors"]
        ):
            raise ContractError("checkpoint and parity fixture/weight digests disagree")
        acceptance = contract.validation.acceptance_report_sha256
        if require_acceptance and acceptance is None:
            raise ContractError("checkpoint does not pin an acceptance report")
        if acceptance is not None:
            name = "validation/acceptance.json"
            if _hash_artifact(root / name, stamps) != acceptance:
                raise ContractError("acceptance report differs from its pinned digest")
            artifacts[name] = acceptance
        # Detect replacement of an earlier file while later artifacts were read.
        for path, stamp in stamps.items():
            if _stamp(path) != stamp:
                raise ContractError(
                    f"checkpoint artifact changed during audit: {path.name}"
                )
        _check_weight_layout(root)
    except (OSError, msgspec.DecodeError, TypeError, ValueError) as error:
        raise ContractError(
            f"invalid target-KV checkpoint evidence: {error}"
        ) from error
    return {
        "scope": "fixed_input_parity_artifact_binding_v1",
        "status": "verified",
        "contract_sha256": contract.fingerprint,
        "artifact_sha256": artifacts,
        "parity_report_sha256": digest_bytes(report_bytes),
        "acceptance_report_bound": acceptance is not None,
        "dtype": report.dtype,
        "reference_attention": report.reference_attention,
        "stages": list(report.stages),
        "anchors": report.anchors,
        "valid_labels": report.valid_labels,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--require-acceptance", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        receipt = audit_target_kv_checkpoint(
            arguments.checkpoint, require_acceptance=arguments.require_acceptance
        )
    except ContractError as error:
        print(json.dumps({"status": "failed", "error": str(error)}))
        return 1
    print(json.dumps(receipt, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
