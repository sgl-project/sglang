"""从结构化质量报告生成趋势；不读取回答、环境或异常原文。"""

import os
import re
from pathlib import Path

from .trend_contract import (
    ACCURACY,
    ANSWER,
    CONTRACT,
    PACKAGES,
    PROVENANCE,
    QUALITY_REASONS,
    SHA1,
    SHA256,
    canonical,
    digest,
    finite,
    require,
    utc,
    validate_components,
    validate_provenance,
    validate_row,
)


def _provenance(source):
    result = {key: source.get(key) for key in PROVENANCE}
    # 可选诊断数据未知时留空；身份与配置摘要不能用推测值替代。
    for key, pattern in (
        ("source_revision", SHA1),
        ("checkpoint_config_sha256", SHA256),
    ):
        value = result[key]
        if not isinstance(value, str) or not pattern.fullmatch(value):
            result[key] = None
    image = result["base_image_digest"]
    if not isinstance(image, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        result["base_image_digest"] = None
    result["github_run_url"] = None
    versions = source.get("package_versions") or {}
    result["package_versions"] = {
        key: (
            value
            if isinstance(value, str)
            and re.fullmatch(r"[0-9][A-Za-z0-9.+_-]{0,159}", value)
            else None
        )
        for key, value in versions.items()
        if key in PACKAGES
    }
    validate_provenance(result)
    return result


def _shared(report, *, schema, test_id, generated_at, config_digest):
    return {
        "schema_version": schema,
        "test_id": test_id,
        "config_digest": config_digest,
        "generated_at": utc(generated_at).isoformat(),
        "provenance": _provenance(report.get("provenance") or {}),
    }


def _point(
    shared, *, measurement_id, metrics, reason=None, quality="not_evaluated", reasons=()
):
    return {
        **shared,
        "measurement_id": measurement_id,
        "status": "measured" if metrics is not None else "unmeasured",
        "metrics": metrics,
        "reason_code": reason,
        "quality_verdict": quality,
        "quality_reason_codes": sorted(set(reasons)),
    }


def _accuracy_missing(record):
    if not finite(record.get("score")):
        return "primary_metric_missing"
    if (
        not record.get("primary_metric")
        or record.get("metric_name") != record["primary_metric"]
    ):
        return "primary_metric_mismatch"
    execution = record.get("execution")
    if not isinstance(execution, dict):
        return "execution_failed"
    if execution.get("incomplete") or execution.get("errored"):
        return "incomplete_samples"
    if (
        type(record.get("samples")) is not int
        or record["samples"] <= 0
        or record["samples"] != record.get("expected_samples")
    ):
        return "incomplete_samples"
    if record.get("reason_code") not in (None, "below_baseline", "above_baseline"):
        return "execution_failed"
    return None


def accuracy_points(report):
    shared = _shared(
        report,
        schema=ACCURACY,
        test_id=report["test_id"],
        generated_at=report["generated_at"],
        config_digest=report["config_digest"],
    )
    rows = []
    for record in report["measurements"]:
        missing = _accuracy_missing(record)
        metrics = None if missing else {"score": record["score"]}
        quality, reasons = "not_evaluated", []
        if not missing and record.get("baseline") is not None:
            require(finite(record.get("ratio")), "基线比较缺少有效ratio")
            metrics["ratio"] = record["ratio"]
            reason = record.get("reason_code")
            quality = (
                "failed" if reason in {"below_baseline", "above_baseline"} else "passed"
            )
            reasons = [reason] if quality == "failed" else []
        row = _point(
            shared,
            measurement_id=record["id"],
            metrics=metrics,
            reason=missing,
            quality=quality,
            reasons=reasons,
        )
        row.update(
            metric_name=record.get("metric_name"),
            samples=record.get("samples"),
            expected_samples=record.get("expected_samples"),
        )
        validate_row(row)
        rows.append(row)
    require(bool(rows), "没有Accuracy测量记录")
    return rows


def _answer_missing(case, expected_model):
    if case.get("failure_class") in {"server_error", "runner_error"}:
        return "request_error"
    if not case.get("returned_model"):
        return "missing_response"
    if not expected_model or case["returned_model"] != expected_model:
        return "unexpected_model_name"
    if case.get("verdict") not in {"passed", "failed"}:
        return "missing_response"
    if any(f.get("reason_code") == "unexpected_model_name" for f in case["findings"]):
        return "unexpected_model_name"
    return None


def answer_points(report):
    parts = {
        "contract_version": CONTRACT,
        "test_config_sha256": report.get("provenance", {}).get("test_config_sha256"),
        "dataset_sha256": report.get("dataset", {}).get("sha256"),
        "quality_profile_sha256": report.get("quality_profile", {}).get("sha256"),
    }
    validate_components(parts)
    require(report.get("judge", {}).get("mode") == "disabled", "趋势契约尚未支持judge")
    shared = _shared(
        report,
        schema=ANSWER,
        test_id=report["provenance"]["test_config_id"],
        generated_at=report["created_at"],
        config_digest=digest(parts),
    )
    shared["config_components"] = parts
    total = report["summary"]["total"]
    require(
        type(total) is int and total > 0 and len(report["cases"]) == total,
        "Answer案例清单不完整",
    )
    require(len({case["case_id"] for case in report["cases"]}) == total, "重复案例身份")
    rows = []
    for case in report["cases"]:
        missing = _answer_missing(case, report["provenance"].get("served_model_name"))
        passed = case["verdict"] == "passed"
        reasons = []
        if not missing and not passed:
            reasons = [
                (
                    f["reason_code"]
                    if f.get("reason_code") in QUALITY_REASONS
                    else "candidate_quality_failed"
                )
                for f in case["findings"]
                if f.get("action") == "hard_fail"
            ]
            reasons = reasons or ["candidate_quality_failed"]
        rows.append(
            _point(
                shared,
                measurement_id="case:" + case["case_id"],
                metrics=None if missing else {"passed": int(passed)},
                reason=missing,
                quality=(
                    "not_evaluated" if missing else "passed" if passed else "failed"
                ),
                reasons=reasons,
            )
        )
    measured = sum(row["status"] == "measured" for row in rows)
    passed = sum(row["quality_verdict"] == "passed" for row in rows)
    suite = _point(
        shared,
        measurement_id="suite",
        metrics={"pass_rate": passed / total} if measured == total else None,
        reason=None if measured == total else "incomplete_suite",
        quality=(
            "not_evaluated"
            if measured != total
            else "passed" if passed == total else "failed"
        ),
        reasons=(
            ["candidate_quality_failed"] if measured == total and passed < total else []
        ),
    )
    suite["counts"] = {
        "total": total,
        "measured": measured,
        "unmeasured": total - measured,
    }
    rows.append(suite)
    for row in rows:
        validate_row(row)
    return rows


def write_quality_trend(report, output_dir: Path):
    """先落盘主报告；趋势错误固定编码并由独立发布job处理。"""
    try:
        converter = {
            "ppu-answer-result/v1": answer_points,
            "ppu-accuracy-result/v1": accuracy_points,
        }[report["schema_version"]]
        content = "".join(canonical(row) + "\n" for row in converter(report))
        temporary = output_dir / "trend.jsonl.tmp"
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, output_dir / "trend.jsonl")
        (output_dir / "trend-error.json").unlink(missing_ok=True)
    except Exception:
        # 不让转换或诊断写入失败改变既有评测门禁，不记录异常字符串。
        try:
            (output_dir / "trend-error.json").write_text(
                '{"reason_code":"trend_conversion_failed"}\n', encoding="utf-8"
            )
            (output_dir / "trend.jsonl").unlink(missing_ok=True)
            (output_dir / "trend.jsonl.tmp").unlink(missing_ok=True)
        except OSError:
            print("::warning::trend_conversion_failed")
