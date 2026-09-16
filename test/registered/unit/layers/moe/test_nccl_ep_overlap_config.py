"""NCCL EP overlap configuration contracts without native EP or model weights."""

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from registered.unit.layers.moe.test_nccl_ep_graph_config import (  # noqa: F401
    ep_bindings,
    server_args,
)

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture
def model_path(tmp_path, monkeypatch):
    from transformers import DeepseekV2Config, GenerationConfig

    torch.cuda.init()
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    config = DeepseekV2Config(
        hidden_size=2048,
        num_hidden_layers=2,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=256,
    )
    config.architectures = ["DeepseekV2ForCausalLM"]
    config.quantization_config = dict(
        quant_method="fp8",
        activation_scheme="dynamic",
        weight_block_size=[128, 128],
    )
    config.save_pretrained(tmp_path)
    GenerationConfig().save_pretrained(tmp_path)
    return str(tmp_path)


@pytest.mark.parametrize("graph", [False, True])
def test_public_overlap_modes(model_path, monkeypatch, graph):
    from nccl_ep_test.fake_ep import dispatcher_environment

    from sglang.srt.layers.moe.utils import initialize_moe_config
    from sglang.srt.runtime_context import get_flags

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (9, 0))
    args = server_args(
        model_path,
        tp_size=2,
        enable_nccl_ep_cuda_graph=graph,
        enable_single_batch_overlap=True,
    )
    # Restore every materialized flag, including flags changed by initialization.
    with get_flags().moe.override(**vars(get_flags().moe)):
        initialize_moe_config(args)
        with dispatcher_environment(capacity=8) as environment:
            dispatcher = environment.dispatcher(layer_id=0)
            assert dispatcher._comm is not None
            assert dispatcher._comm.stream != torch.cuda.current_stream()


@pytest.mark.parametrize(
    "change",
    [
        {"moe_runner_backend": "deep_gemm"},
        {"enable_eplb": True},
        {"enable_two_batch_overlap": True},
    ],
)
def test_unsupported_overlap_is_rejected(model_path, monkeypatch, change):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (9, 0))
    options = dict(enable_single_batch_overlap=True)
    options.update(change)
    with pytest.raises(ValueError, match="NCCL EP overlap requires"):
        server_args(model_path, tp_size=2, **options)


def test_overlap_does_not_silently_fallback(model_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 9))
    with pytest.raises(ValueError, match="NCCL EP overlap is unavailable"):
        server_args(
            model_path,
            enable_single_batch_overlap=True,
        )


def test_serial_nccl_ep_keeps_the_current_stream(model_path, monkeypatch):
    from nccl_ep_test.fake_ep import dispatcher_environment

    from sglang.srt.layers.moe.utils import initialize_moe_config
    from sglang.srt.runtime_context import get_flags

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (9, 0))
    args = server_args(model_path)
    with get_flags().moe.override(**vars(get_flags().moe)):
        initialize_moe_config(args)
        with dispatcher_environment(capacity=8) as environment:
            assert environment.dispatcher(layer_id=0)._comm is None


def test_overlap_rejects_unsupported_fp8_block_shape(model_path, monkeypatch):
    path = Path(model_path) / "config.json"
    config = json.loads(path.read_text())
    config["quantization_config"]["weight_block_size"] = [64, 128]
    path.write_text(json.dumps(config))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (9, 0))
    with pytest.raises(ValueError, match="overlap requires DeepSeek V2/V3 FP8"):
        server_args(
            model_path,
            tp_size=2,
            enable_single_batch_overlap=True,
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
