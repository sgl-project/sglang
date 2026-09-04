import json
import logging
import os
import pickle
import sys
import traceback
from types import SimpleNamespace

import pytest

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.arg_groups.validation_hook import check_watermark_server_args
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import (
    TokenizerManager,
    _validate_watermark_request,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.sampling.watermark import (
    redact_watermark_command_line,
    redact_watermark_secrets,
)
from sglang.srt.sampling.watermark_config import (
    WatermarkConfigError,
    load_watermark_config,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.request_logger import (
    _dataclass_to_string_truncated,
    _transform_data_for_logging,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _write_config(path, *, key="0123456789abcdef", context_window=4):
    path.write_text(
        json.dumps({"key": key, "context_window": context_window}), encoding="utf-8"
    )
    os.chmod(path, 0o600)


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

    manager = object.__new__(TokenizerManager)
    manager.server_args = server_args
    assert b"0123456789abcdef" not in pickle.dumps(manager._server_args_for_dump())


def test_default_key_source_validation(tmp_path, monkeypatch):
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

    server_args = ServerArgs(
        model_path="dummy",
        device="cuda",
        enable_watermark=True,
        watermark_key="0123456789abcdef",
        disaggregation_mode="decode",
    )
    server_args.resolve_once()
    with pytest.raises(ValueError, match="not supported with PD disaggregation"):
        check_watermark_server_args(server_args)

    server_args = ServerArgs(
        model_path="dummy",
        device="cuda",
        enable_watermark=True,
        watermark_key="0123456789abcdef",
        dllm_algorithm="LowConfidence",
    )
    server_args.resolve_once()
    with pytest.raises(ValueError, match="not supported with diffusion LLM"):
        check_watermark_server_args(server_args)

    server_args = ServerArgs(
        model_path="dummy",
        device="cuda",
        enable_watermark=True,
        watermark_key="0123456789abcdef",
        speculative_algorithm="DFLASH",
    )
    server_args.resolve_once()
    with pytest.raises(ValueError, match="supports speculative algorithms"):
        check_watermark_server_args(server_args)

    server_args = ServerArgs(
        model_path="dummy",
        device="cuda",
        enable_watermark=True,
        watermark_key="0123456789abcdef",
        speculative_algorithm="EAGLE",
        speculative_use_rejection_sampling=True,
    )
    server_args.resolve_once()
    with pytest.raises(ValueError, match="speculative-use-rejection-sampling"):
        check_watermark_server_args(server_args)

    server_args = ServerArgs(
        model_path="dummy",
        device="cuda",
        enable_watermark=True,
        watermark_key="0123456789abcdef",
    )
    server_args.resolve_once()
    monkeypatch.setenv("SGLANG_RUST_SERVER", "1")
    with pytest.raises(ValueError, match="not supported with SGLANG_RUST_SERVER"):
        check_watermark_server_args(server_args)


def test_watermarked_beam_search_is_rejected(monkeypatch):
    features = SimpleNamespace(
        enable_watermark=True,
        watermark_key="0123456789abcdef",
        watermark_context_window=4,
    )
    monkeypatch.setattr(
        "sglang.srt.managers.tokenizer_manager.get_exec",
        lambda: SimpleNamespace(features=features),
    )
    with pytest.raises(ValueError, match="beam search is not supported"):
        _validate_watermark_request(SamplingParams(beam_width=2))


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

    missing_path = tmp_path / "secret-config-name.json"
    with pytest.raises(WatermarkConfigError) as error:
        load_watermark_config(str(missing_path))
    formatted_error = "".join(traceback.format_exception(error.value))
    assert str(missing_path) not in formatted_error

    request = GenerateReqInput(
        text="hello", watermark={"key": secret, "context_window": 4}
    )
    assert secret not in str(_transform_data_for_logging(request))
    assert secret not in _dataclass_to_string_truncated(request)
    assert secret not in _dataclass_to_string_truncated([request])

    dump_request = redact_watermark_secrets(request)
    assert secret not in repr(dump_request.watermark)
    assert request.watermark["key"] == secret

    request = GenerateReqInput(
        text="hello",
        sampling_params={"watermark": {"key": secret, "context_window": 4}},
    )
    dump_request = redact_watermark_secrets(request)
    assert secret not in repr(dump_request.sampling_params)
    assert request.sampling_params["watermark"]["key"] == secret

    command = redact_watermark_command_line(
        [
            "python",
            "-m",
            "sglang.launch_server",
            "--watermark-key",
            secret,
            "--watermark-config=/run/secrets/watermark.json",
        ]
    )
    assert secret not in command
    assert "/run/secrets/watermark.json" not in command


def test_config_file_security_guards(tmp_path, caplog):
    config_path = tmp_path / "watermark.json"
    _write_config(config_path)
    os.chmod(config_path, 0o644)
    with caplog.at_level(logging.WARNING):
        load_watermark_config(str(config_path))
    assert "readable by group or other users" in caplog.text
    assert str(config_path) not in caplog.text

    config_path.write_text("x" * 4097, encoding="utf-8")
    with pytest.raises(WatermarkConfigError, match="exceeds 4096 bytes"):
        load_watermark_config(str(config_path))

    _write_config(config_path, context_window=65)
    with pytest.raises(WatermarkConfigError, match="from 1 to 64"):
        load_watermark_config(str(config_path))

    fifo_path = tmp_path / "watermark.fifo"
    os.mkfifo(fifo_path)
    with pytest.raises(WatermarkConfigError, match="regular file"):
        load_watermark_config(str(fifo_path))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
