import json
import logging
import sys

import pytest

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.arg_groups.validation_hook import check_watermark_server_args
from sglang.srt.sampling.watermark import redact_watermark_secrets
from sglang.srt.sampling.watermark_config import (
    WatermarkConfigError,
    load_watermark_config,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _write_config(path, *, key="0123456789abcdef", context_window=4):
    path.write_text(
        json.dumps({"key": key, "context_window": context_window}), encoding="utf-8"
    )


def test_file_config_resolution_contract(tmp_path):
    config_path = tmp_path / "watermark.json"
    _write_config(config_path, context_window=2)
    server_args = ServerArgs(
        model_path="dummy",
        device="cuda",
        enable_watermark=True,
        watermark_config=str(config_path),
        watermark_context_window=9,
    )

    server_args.resolve_once()
    check_watermark_server_args(server_args)

    assert resolution_result(server_args, "watermark_key") == "0123456789abcdef"
    assert resolution_result(server_args, "watermark_context_window") == 2
    assert server_args.watermark_key is None
    assert server_args.watermark_context_window == 9


def test_default_key_source_validation(tmp_path):
    config_path = tmp_path / "watermark.json"
    _write_config(config_path)

    with pytest.raises(ValueError, match="mutually exclusive"):
        ServerArgs(
            model_path="dummy",
            enable_watermark=True,
            watermark_key="0123456789abcdef",
            watermark_config=str(config_path),
        ).resolve_once()

    server_args = ServerArgs(
        model_path="dummy",
        watermark_config=str(config_path),
    )
    server_args.resolve_once()
    with pytest.raises(ValueError, match="require --enable-watermark"):
        check_watermark_server_args(server_args)


def test_config_errors_and_logs_do_not_expose_secrets(tmp_path, caplog):
    secret = "fedcba9876543210"
    config_path = tmp_path / f"watermark-{secret}.json"
    _write_config(config_path, key=secret)
    config = load_watermark_config(str(config_path))
    assert secret not in repr(config)

    server_args = ServerArgs(
        model_path="dummy",
        enable_watermark=True,
        watermark_config=str(config_path),
    )
    server_args.resolve_once()
    logged_args = redact_watermark_secrets(server_args.resolved_dict())
    with caplog.at_level(logging.INFO):
        logging.getLogger(__name__).info("server_args=%s", logged_args)
    assert secret not in repr(server_args)
    assert secret not in caplog.text
    assert str(config_path) not in caplog.text
    assert logged_args["watermark_key"] == "<redacted>"
    assert logged_args["watermark_config"] == "<redacted>"

    config_path.write_text(
        json.dumps({"key": secret, "context_window": 4, secret: "value"}),
        encoding="utf-8",
    )
    with pytest.raises(WatermarkConfigError) as error:
        load_watermark_config(str(config_path))
    assert secret not in str(error.value)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
