"""PPU 趋势的纯标准库线协议；生成端、归档端与派生端共用。"""

import hashlib
import json
import math
import re
from datetime import datetime, timezone

PERF = "ppu-perf-trend-point/v1"
ACCURACY = "ppu-accuracy-trend-point/v1"
ANSWER = "ppu-answer-trend-point/v1"
SCHEMAS = {PERF, ACCURACY, ANSWER}
ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}\Z")
MEASUREMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,239}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
SHA1 = re.compile(r"[0-9a-f]{40}\Z")
NUMBER = re.compile(r"[1-9][0-9]{0,19}\Z")
PACKAGES = {"sglang", "sglang-kernel", "torch", "transformers"}
PROVENANCE = {
    "source_revision",
    "github_run_id",
    "github_run_attempt",
    "github_run_url",
    "base_image_digest",
    "checkpoint_config_sha256",
    "package_versions",
}
QUALITY_REASONS = {
    "below_baseline",
    "above_baseline",
    "candidate_quality_failed",
    "fact_rule_failed",
    "periodic_fragment_repeat",
    "repeated_4gram_coverage",
    "finish_reason_length",
    "finish_reason_unexpected",
    "cross_case_duplicate",
    "empty_answer",
}
MISSING_REASONS = {
    "primary_metric_missing",
    "primary_metric_mismatch",
    "incomplete_samples",
    "execution_failed",
    "request_error",
    "unexpected_model_name",
    "missing_response",
    "incomplete_suite",
}
COMMON = {
    "schema_version",
    "test_id",
    "measurement_id",
    "config_digest",
    "generated_at",
    "status",
    "reason_code",
    "metrics",
    "quality_verdict",
    "quality_reason_codes",
    "provenance",
}
PERF_METRICS = {
    "ttft_mean_ms",
    "ttft_median_ms",
    "ttft_std_ms",
    "ttft_p99_ms",
    "e2e_latency_mean_ms",
    "e2e_latency_median_ms",
    "e2e_latency_p90_ms",
    "e2e_latency_p99_ms",
    "request_throughput_req_s",
    "input_token_throughput_tok_s",
    "output_token_throughput_tok_s",
    "output_token_throughput_peak_tok_s",
    "total_token_throughput_tok_s",
    "duration_s",
    "completed",
    "total_input_tokens",
    "total_output_tokens",
    "concurrency",
    "max_concurrent_requests",
    "tpot_mean_ms",
    "tpot_median_ms",
    "tpot_std_ms",
    "tpot_p99_ms",
    "itl_mean_ms",
    "itl_median_ms",
    "itl_std_ms",
    "itl_p95_ms",
    "itl_p99_ms",
}
CONTRACT = "ppu-answer-trend-contract/v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def positive_id(value):
    return isinstance(value, str) and NUMBER.fullmatch(value) is not None


def utc(value):
    require(isinstance(value, str) and "T" in value, "无效测量时间")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None, "测量时间缺少时区")
    return parsed.astimezone(timezone.utc)


def series_key(row):
    return tuple(
        row[field]
        for field in ("schema_version", "test_id", "measurement_id", "config_digest")
    )


def validate_components(parts):
    require(isinstance(parts, dict), "缺少组合摘要")
    require(
        set(parts)
        == {
            "contract_version",
            "test_config_sha256",
            "dataset_sha256",
            "quality_profile_sha256",
        },
        "组合摘要字段错误",
    )
    require(parts["contract_version"] == CONTRACT, "未知评测契约")
    for key in ("test_config_sha256", "dataset_sha256", "quality_profile_sha256"):
        require(
            isinstance(parts[key], str) and SHA256.fullmatch(parts[key]), "缺少内容摘要"
        )


def validate_provenance(prov):
    require(isinstance(prov, dict) and set(prov) <= PROVENANCE, "非白名单provenance")
    require(positive_id(prov.get("github_run_id")), "无效run身份")
    require(positive_id(prov.get("github_run_attempt")), "无效attempt身份")
    for key, pattern in (
        ("source_revision", SHA1),
        ("checkpoint_config_sha256", SHA256),
    ):
        value = prov.get(key)
        require(
            value is None or (isinstance(value, str) and pattern.fullmatch(value)),
            "无效版本摘要",
        )
    image = prov.get("base_image_digest")
    require(
        image is None
        or (isinstance(image, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", image)),
        "无效镜像摘要",
    )
    url = prov.get("github_run_url")
    require(
        url is None
        or url
        == "https://github.com/flytiger-eco/sglang-for-sail/actions/runs/"
        + prov["github_run_id"],
        "非当前仓库run链接",
    )
    packages = prov.get("package_versions")
    require(
        packages is None or (isinstance(packages, dict) and set(packages) <= PACKAGES),
        "非白名单包版本",
    )
    for value in (packages or {}).values():
        require(
            value is None
            or (
                isinstance(value, str)
                and re.fullmatch(r"[0-9][A-Za-z0-9.+_-]{0,159}", value)
            ),
            "无效包版本",
        )


def validate_row(row):
    require(
        isinstance(row, dict) and row.get("schema_version") in SCHEMAS, "未知趋势schema"
    )
    for field, pattern in (
        ("test_id", ID),
        ("measurement_id", MEASUREMENT),
        ("config_digest", SHA256),
    ):
        require(
            isinstance(row.get(field), str) and pattern.fullmatch(row[field]),
            "无效趋势标识",
        )
    utc(row.get("generated_at"))
    schema = row["schema_version"]
    measured = row.get("status") == "measured"
    metrics = row.get("metrics")
    allowed = (
        PERF_METRICS
        if schema == PERF
        else (
            {"score", "ratio"}
            if schema == ACCURACY
            else ({"pass_rate"} if row["measurement_id"] == "suite" else {"passed"})
        )
    )
    if measured:
        require(
            isinstance(metrics, dict) and bool(metrics) and set(metrics) <= allowed,
            "无效指标集合",
        )
        require(all(finite(value) for value in metrics.values()), "非有限数值指标")
        require(row.get("reason_code") is None, "有效测量带缺测原因")
    else:
        # 旧Perf部分请求失败时仍附带诊断数值；保留原行但不计入统计。
        if schema == PERF and metrics is not None:
            require(
                isinstance(metrics, dict)
                and bool(metrics)
                and set(metrics) <= allowed
                and all(finite(value) for value in metrics.values()),
                "无效Perf诊断数值",
            )
        else:
            require(metrics is None, "无效测量携带数值")
        require(isinstance(row.get("reason_code"), str), "缺少缺测原因")
    require(isinstance(row.get("provenance"), dict), "缺少provenance")
    if schema == PERF:
        require(row.get("status") in {"measured", "failed"}, "未知perf状态")
        return
    extra = (
        {"metric_name", "samples", "expected_samples"}
        if schema == ACCURACY
        else {"config_components", "counts"}
    )
    require(COMMON <= set(row) and set(row) <= COMMON | extra, "非白名单趋势字段")
    require(row["status"] in {"measured", "unmeasured"}, "未知测量状态")
    quality = row["quality_verdict"]
    require(quality in {"passed", "failed", "not_evaluated"}, "未知质量判决")
    require(measured or quality == "not_evaluated", "缺测不可判质量")
    require(measured or row["reason_code"] in MISSING_REASONS, "未知缺测原因")
    reasons = row["quality_reason_codes"]
    require(
        isinstance(reasons, list)
        and all(isinstance(x, str) and x in QUALITY_REASONS for x in reasons),
        "未知质量原因",
    )
    require(bool(reasons) == (quality == "failed"), "质量原因与判决不符")
    validate_provenance(row["provenance"])
    if schema == ANSWER:
        validate_components(row.get("config_components"))
        require(
            row["config_digest"] == digest(row["config_components"]), "组合摘要不符"
        )
        require(
            row["measurement_id"] == "suite"
            or row["measurement_id"].startswith("case:"),
            "无效Answer测量标识",
        )
        if measured:
            require(set(metrics) == allowed, "Answer指标缺失")
            value = next(iter(metrics.values()))
            require(0 <= value <= 1, "通过率越界")
            require(
                row["measurement_id"] == "suite" or value in (0, 1),
                "case指标必须为0或1",
            )
        require(
            row["measurement_id"] == "suite" or "counts" not in row,
            "case不可携带suite计数",
        )
        if row["measurement_id"] == "suite":
            counts = row.get("counts")
            require(
                isinstance(counts, dict)
                and set(counts) == {"total", "measured", "unmeasured"},
                "缺少suite计数",
            )
            require(
                all(type(x) is int and x >= 0 for x in counts.values())
                and counts["total"] > 0
                and counts["total"] == counts["measured"] + counts["unmeasured"],
                "无效suite计数",
            )
            require(measured == (counts["unmeasured"] == 0), "suite缺测状态错误")
    else:
        name = row.get("metric_name")
        require(
            name is None or (isinstance(name, str) and ID.fullmatch(name)),
            "无效metric名称",
        )
        for key in ("samples", "expected_samples"):
            require(
                row.get(key) is None or (type(row[key]) is int and row[key] >= 0),
                "无效样本数",
            )
        if measured:
            require(
                "score" in metrics
                and name is not None
                and row.get("samples") == row.get("expected_samples")
                and type(row.get("samples")) is int
                and row["samples"] > 0,
                "不完整Accuracy测量",
            )
