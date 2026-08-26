import sys
from types import SimpleNamespace

import pytest

from sglang.benchmark import one_batch
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_load_model_initializes_runtime_configs_before_model(monkeypatch):
    events = []

    monkeypatch.setattr(one_batch, "suppress_other_loggers", lambda: None)
    monkeypatch.setattr(
        one_batch,
        "initialize_moe_config",
        lambda server_args: events.append("moe"),
    )
    monkeypatch.setattr(
        one_batch,
        "initialize_fp8_gemm_config",
        lambda server_args: events.append("fp8"),
    )
    monkeypatch.setattr(
        one_batch,
        "initialize_fp4_gemm_config",
        lambda server_args: events.append("fp4"),
    )
    monkeypatch.setattr(
        one_batch.ModelConfig,
        "from_server_args",
        lambda server_args: object(),
    )
    monkeypatch.setattr(
        one_batch,
        "compute_dp_attention_world_info",
        lambda *args: (0, 1, 0, 1),
    )
    monkeypatch.setattr(one_batch, "ParallelState", lambda **kwargs: object())
    monkeypatch.setattr(one_batch, "use_mlx", lambda: False)
    monkeypatch.setattr(one_batch, "get_tokenizer", lambda *args, **kwargs: "tokenizer")

    class FakeModelRunner:
        max_total_num_tokens = 1

        def __init__(self, **kwargs):
            assert events == ["moe", "fp8", "fp4"]

        def alloc_memory_pool(self):
            pass

        def init_attention_backends(self):
            pass

        def init_cuda_graphs(self):
            pass

    monkeypatch.setattr(one_batch, "ModelRunner", FakeModelRunner)

    server_args = SimpleNamespace(
        tp_size=1,
        ep_size=1,
        dp_size=1,
        enable_dp_attention=False,
        attn_cp_size=1,
        dcp_size=1,
        moe_dp_size=1,
        mem_fraction_static=0.8,
        is_startup_weight_load_overlap=False,
        tokenizer_path="tokenizer",
        tokenizer_mode="auto",
        trust_remote_code=False,
    )
    port_args = SimpleNamespace(nccl_port=12345)

    _, tokenizer = one_batch.load_model(server_args, port_args, gpu_id=0, tp_rank=0)

    assert tokenizer == "tokenizer"
    assert events == ["moe", "fp8", "fp4"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
