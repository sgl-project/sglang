import pytest
import torch
from sglang.srt.entrypoints.engine import Engine
from dataclasses import dataclass


def simple_task_func(**kwargs):
    # This function will be run on all model workers
    return 12345


def get_model_param_info(**kwargs):
    # kwargs will contain 'model' and 'model_runner'
    model = kwargs["model"]
    param_info = []
    for name, param in model.named_parameters():
        param_info.append(
            {
                "name": name,
                "numel": param.numel(),
                "shape": tuple(param.shape),
                "dtype": str(param.dtype),
            }
        )
    return param_info


@dataclass
class ParamInfo:
    name: str
    shape: tuple
    numel: int
    dtype: torch.dtype


def get_model_param_info_struct(**kwargs):
    # kwargs will contain 'model' and 'model_runner'
    model = kwargs["model"]
    param_info = []
    for name, param in model.named_parameters():
        param_info.append(
            ParamInfo(
                name=name,
                shape=tuple(param.shape),
                numel=param.numel(),
                dtype=param.dtype,
            )
        )
    return param_info


@pytest.mark.parametrize(
    "tp_size,dp_size,enable_dp_attention,expected_len",
    [
        (1, 1, False, 1),
        (4, 1, False, 4),
        (4, 2, True, 8),  # 4 TP * 2 DP
    ],
)
def test_execute_task_in_model_worker(
    tp_size, dp_size, enable_dp_attention, expected_len
):
    # You may need to adjust model_path to the actual path or huggingface repo
    model_path = "DeepSeek-R1-Distill-Qwen-1.5B"
    engine = Engine(
        model_path=model_path,
        served_model_name=model_path,
        attention_backend="torch_native",  # make test faster
        tp_size=tp_size,
        trust_remote_code=True,
        dp_size=dp_size,
        enable_dp_attention=enable_dp_attention,
        # Add any other required ServerArgs here, e.g. device, port, etc.
        # For local single GPU: device="cuda:0"
        # For multi-GPU: device="cuda"
        # For test: log_level="error"
        log_level="error",
        stream_output=False,
    )

    # Test simple_task_func
    results = engine.execute_task_in_model_worker(simple_task_func)
    assert isinstance(results, list)
    assert len(results) == expected_len
    for r in results:
        assert r == 12345

    # Test get_model_param_info
    param_info_results = engine.execute_task_in_model_worker(get_model_param_info)
    # print(param_info_results)
    assert isinstance(param_info_results, list)
    assert len(param_info_results) == expected_len
    for param_info in param_info_results:
        assert isinstance(param_info, list)
        for param in param_info:
            assert "name" in param
            assert "numel" in param
            assert "shape" in param
            assert "dtype" in param
    all_weight_names = [
        param["name"] for param_info in param_info_results for param in param_info
    ]
    assert "model.embed_tokens.weight" in all_weight_names
    assert "model.layers.0.self_attn.qkv_proj.weight" in all_weight_names
    assert "model.layers.0.mlp.gate_up_proj.weight" in all_weight_names
    assert "model.layers.1.input_layernorm.weight" in all_weight_names

    # Test struct serialization
    param_info_results = engine.execute_task_in_model_worker(
        get_model_param_info_struct
    )
    for param_info in param_info_results:
        assert isinstance(param_info, list)
        for param in param_info:
            assert param.name
            assert param.numel > 0
            assert len(param.shape) >= 1
            assert isinstance(param.dtype, torch.dtype)
    engine.shutdown()  # Clean up subprocesses
