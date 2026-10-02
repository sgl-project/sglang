import json
import logging
import os
import pickle
import sys
import traceback
from array import array
from types import SimpleNamespace

import pytest

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.arg_groups.serving_hook import handle_other_validations
from sglang.srt.arg_groups.validation_hook import check_watermark_server_args
from sglang.srt.managers.io_struct import GenerateReqInput, TokenizedGenerateReqInput
from sglang.srt.managers.tokenizer_manager import (
    TokenizerManager,
    _validate_watermark_request,
)
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.sampling.watermarking.config import (
    WatermarkConfigError,
    load_watermark_config,
)
from sglang.srt.sampling.watermarking.core import (
    redact_watermark_command_line,
    redact_watermark_secrets,
)
from sglang.srt.server_args import ServerArgs, prepare_server_args
from sglang.srt.utils.request_logger import (
    _dataclass_to_string_truncated,
    _transform_data_for_logging,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _write_config(path, **fields):
    config = {"key": "0123456789abcdef", "context_window": 4, **fields}
    path.write_text(
        json.dumps(
            {name: value for name, value in config.items() if value is not None}
        ),
        encoding="utf-8",
    )
    os.chmod(path, 0o600)


_FULL_CONFIG = {
    "key": "0123456789abcdef",
    "key_b": "fedcba9876543210",
    "context_window": 2,
    "mixing_probability": 0.25,
    "max_probability": 0.9,
    "default_enabled": True,
    "enforce_all": True,
}


def test_inline_config_resolves_all_fields():
    server_args = ServerArgs(
        model_path="dummy",
        device="cuda",
        enable_watermark=True,
        watermark_config=json.dumps(_FULL_CONFIG),
    )
    server_args.resolve_once()
    check_watermark_server_args(server_args)
    for field, value in _FULL_CONFIG.items():
        assert resolution_result(server_args, f"watermark_{field}") == value

    manager = object.__new__(TokenizerManager)
    manager.server_args = server_args
    publish(server_args, role="test")
    try:
        dumped = pickle.dumps(manager._server_args_for_dump())
        assert b"0123456789abcdef" not in dumped
        assert b"fedcba9876543210" not in dumped
    finally:
        reset_context()


_KEY = "0123456789abcdef"
_KEY_B = "fedcba9876543210"


@pytest.mark.parametrize(
    ("overrides", "config", "match"),
    [
        pytest.param(
            {"enable_watermark": False},
            {},
            "requires --enable-watermark",
            id="config-without-capability",
        ),
        pytest.param(
            {"disaggregation_mode": "decode"},
            {},
            "not supported with PD disaggregation",
            id="pd-disaggregation",
        ),
        pytest.param(
            {"dllm_algorithm": "LowConfidence"},
            {},
            "not supported with diffusion LLM",
            id="diffusion-llm",
        ),
        pytest.param(
            {"speculative_algorithm": "DFLASH"},
            {},
            "supports speculative algorithms",
            id="unsupported-spec",
        ),
        pytest.param(
            {
                "speculative_algorithm": "EAGLE",
                "speculative_use_rejection_sampling": True,
            },
            {},
            "speculative-use-rejection-sampling",
            id="rejection-sampling",
        ),
        pytest.param(
            {"pp_size": 2, "speculative_algorithm": "EAGLE"},
            {},
            "pipeline-parallel speculative decoding",
            id="pp-spec",
        ),
        pytest.param(
            {"sampling_backend": "token_oracle"},
            {},
            "sampling-backend token_oracle",
            id="token-oracle-sampler",
        ),
        pytest.param(
            {},
            {"key_b": _KEY_B, "mixing_probability": 0.0},
            "strictly between 0 and 1",
            id="mixing-lower",
        ),
        pytest.param(
            {},
            {"key_b": _KEY_B, "mixing_probability": 1.0},
            "strictly between 0 and 1",
            id="mixing-upper",
        ),
        pytest.param(
            {},
            {"mixing_probability": 0.25},
            "requires key_b",
            id="mixing-without-key-b",
        ),
        pytest.param(
            {},
            {"max_probability": 0.0},
            "greater than 0 and at most 1",
            id="max-probability-lower",
        ),
        pytest.param(
            {},
            {"max_probability": 1.01},
            "greater than 0 and at most 1",
            id="max-probability-upper",
        ),
        pytest.param(
            {},
            {"key": None, "key_b": _KEY_B},
            "key_b requires a server key",
            id="key-b-without-key",
        ),
        pytest.param(
            {},
            {"key": None, "default_enabled": True},
            "require a server key",
            id="default-enabled-without-key",
        ),
        pytest.param(
            {},
            {"key": None, "enforce_all": True},
            "require a server key",
            id="enforce-all-without-key",
        ),
    ],
)
def test_startup_validation_fails_closed(overrides, config, match):
    config = {"key": _KEY, **config}
    kwargs = {
        "model_path": "dummy",
        "device": "cuda",
        "enable_watermark": True,
        "watermark_config": json.dumps(
            {name: value for name, value in config.items() if value is not None}
        ),
        **overrides,
    }
    server_args = ServerArgs(**kwargs)
    server_args.resolve_once()
    with pytest.raises(ValueError, match=match):
        check_watermark_server_args(server_args)


def test_rust_server_is_rejected(monkeypatch):
    server_args = ServerArgs(
        model_path="dummy",
        device="cuda",
        enable_watermark=True,
        watermark_config=json.dumps({"key": _KEY}),
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
        watermark_default_enabled=False,
        watermark_enforce_all=False,
    )
    monkeypatch.setattr(
        "sglang.srt.managers.tokenizer_manager.get_exec",
        lambda: SimpleNamespace(features=features),
    )
    with pytest.raises(ValueError, match="beam search is not supported"):
        _validate_watermark_request(
            SamplingParams(beam_width=2, watermark={"enabled": True})
        )

    with pytest.raises(ValueError, match="requires a server default key"):
        features.watermark_key = None
        _validate_watermark_request(SamplingParams(watermark={"enabled": True}))


def test_config_errors_and_logs_do_not_expose_secrets(tmp_path, caplog):
    secret = "fedcba9876543210"
    secret_b = "0123456789abcdee"
    config_path = tmp_path / f"watermark-{secret}.json"
    _write_config(config_path, key=secret, key_b=secret_b)
    config = load_watermark_config(str(config_path))
    assert secret not in repr(config)
    assert secret_b not in repr(config)
    redacted_config = redact_watermark_secrets(config)
    assert redacted_config.key == "<redacted>"
    assert redacted_config.key_b == "<redacted>"

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
    assert secret_b not in repr(server_args)
    assert secret not in caplog.text
    assert secret_b not in caplog.text
    assert str(config_path) not in caplog.text
    assert logged_args["watermark_key"] == "<redacted>"
    assert logged_args["watermark_key_b"] == "<redacted>"
    assert logged_args["watermark_config"] == "<redacted>"

    config_path.write_text(
        json.dumps({"key": secret, "context_window": 4, secret: "value"}),
        encoding="utf-8",
    )
    with pytest.raises(WatermarkConfigError) as error:
        load_watermark_config(str(config_path))
    assert secret not in str(error.value)

    with pytest.raises(WatermarkConfigError) as error:
        load_watermark_config(f'{{"key":"{secret}"')
    assert error.value.__cause__ is None

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

    tokenized_request = TokenizedGenerateReqInput(
        input_text="hello",
        input_ids=array("q", [1]),
        input_embeds=None,
        mm_inputs=None,
        token_type_ids=None,
        sampling_params=SamplingParams(watermark={"key": secret}),
        return_logprob=False,
        logprob_start_len=-1,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=False,
    )
    dump_request = redact_watermark_secrets(tokenized_request)
    assert dump_request.sampling_params.watermark.key == "<redacted>"
    assert tokenized_request.sampling_params.watermark.key == secret

    inline_config = json.dumps({"key": secret, "key_b": secret_b})
    command = redact_watermark_command_line(
        [
            "python",
            "-m",
            "sglang.launch_server",
            "--watermark-config",
            inline_config,
            "--watermark-config=/run/secrets/watermark.json",
        ]
    )
    assert secret not in command
    assert secret_b not in command
    assert "/run/secrets/watermark.json" not in command

    for option in ("--watermark-conf", "--watermark-c", "--watermark"):
        server_args = prepare_server_args(
            [
                "--model-path",
                "dummy",
                "--enable-watermark",
                option,
                inline_config,
            ]
        )
        assert secret not in server_args.launch_command
        assert secret_b not in server_args.launch_command

    server_args = prepare_server_args(
        [
            "--model-path",
            "dummy",
            "--enable-watermark",
            f"--watermark-conf={inline_config}",
        ]
    )
    assert secret not in server_args.launch_command
    assert secret_b not in server_args.launch_command


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("watermark_key", "not-hex", "only hex digits"),
        ("watermark_key_b", "not-hex", "only hex digits"),
        ("watermark_context_window", 0, "integer from 1 to 64"),
        ("watermark_context_window", 65, "integer from 1 to 64"),
    ],
)
def test_direct_server_args_validate_watermark_fields(field, value, match):
    fields = {"watermark_key": _KEY, field: value}
    server_args = ServerArgs(
        model_path="dummy", device="cuda", enable_watermark=True, **fields
    )
    with pytest.raises(ValueError, match=match):
        check_watermark_server_args(server_args)


def test_preferred_sampling_params_reject_watermark():
    server_args = ServerArgs(
        model_path="dummy",
        preferred_sampling_params=json.dumps(
            {"temperature": 0.8, "watermark": {"enabled": False}}
        ),
    )
    with pytest.raises(ValueError, match="not supported"):
        handle_other_validations(server_args)


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

    _write_config(config_path, key_b="not-hex")
    with pytest.raises(WatermarkConfigError, match="only hex digits"):
        load_watermark_config(str(config_path))

    _write_config(config_path, enforce_all="true")
    with pytest.raises(WatermarkConfigError, match="must be a boolean"):
        load_watermark_config(str(config_path))

    _write_config(config_path, mixing_probability=True)
    with pytest.raises(WatermarkConfigError, match="must be a number"):
        load_watermark_config(str(config_path))

    fifo_path = tmp_path / "watermark.fifo"
    os.mkfifo(fifo_path)
    with pytest.raises(WatermarkConfigError, match="regular file"):
        load_watermark_config(str(fifo_path))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
