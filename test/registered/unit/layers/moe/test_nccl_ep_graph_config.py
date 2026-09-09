"""Public configuration contracts; no model weights or EP communication.

External EP bindings and GPU capability queries are replaced. The public
ServerArgs constructor, dispatcher, and CUDA Graph backend remain real.
"""

import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="The CUDA configuration path is required"
)


@pytest.fixture(autouse=True)
def ep_bindings(monkeypatch):
    from nccl_ep_test.fake_ep import FakeLibrary, replace_ep_modules

    from sglang.srt.layers.moe.token_dispatcher import nccl_ep

    library = FakeLibrary()
    root = ModuleType("nccl")
    root.__path__ = []
    root.core, root.ep = library.core, library.ep
    library.core.get_version = lambda: SimpleNamespace(
        libnccl=SimpleNamespace(version="2.30.7")
    )
    modules = {"nccl": root, "nccl.core": library.core, "nccl.ep": library.ep}
    for name, module in modules.items():
        module.__spec__ = ModuleSpec(name, loader=None, is_package=name == "nccl")
    monkeypatch.setattr(nccl_ep, "_nccl_ep_runtime", None)
    with replace_ep_modules(modules):
        yield library


@pytest.fixture
def model_path(tmp_path, monkeypatch):
    from transformers import GenerationConfig, Qwen2MoeConfig

    torch.cuda.init()  # Initialize the real device before overriding capability.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    config = Qwen2MoeConfig(
        hidden_size=2048,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=16,
        num_key_value_heads=16,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=256,
        shared_expert_intermediate_size=256,
    )
    config.architectures = ["Qwen2MoeForCausalLM"]
    config.save_pretrained(tmp_path)
    GenerationConfig().save_pretrained(tmp_path)
    return str(tmp_path)


def server_args(model_path, **overrides):
    from sglang.srt.server_args import ServerArgs

    options = dict(
        model_path=model_path,
        device="cuda",
        attention_backend="triton",
        moe_a2a_backend="nccl_ep",
        moe_runner_backend="triton",
        tp_size=1,
        dtype="bfloat16",
    )
    options.update(overrides)
    return ServerArgs(**options)


def test_nccl_ep_defaults_to_eager_without_graph_opt_in(model_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kw: (9, 0))
    args = server_args(model_path)
    assert args.cuda_graph_config.decode.backend == "disabled"
    assert args.cuda_graph_config.prefill.backend == "disabled"


def test_public_opt_in_reaches_the_persistent_full_graph_backend(
    model_path, monkeypatch
):
    from nccl_ep_test.dispatcher import forward_layer
    from nccl_ep_test.fake_ep import dispatcher_environment

    from sglang.srt.model_executor.runner.shape_key import ShapeKey
    from sglang.srt.model_executor.runner_backend.utils import resolve_decode_backend
    from sglang.srt.runtime_context import get_context, get_exec

    with monkeypatch.context() as capability:
        capability.setattr(
            torch.cuda, "get_device_capability", lambda *args, **kw: (9, 0)
        )
        args = server_args(model_path, enable_nccl_ep_cuda_graph=True)
    assert args.cuda_graph_config.decode.backend == "full"
    assert args.cuda_graph_config.prefill.backend == "disabled"
    with dispatcher_environment(capacity=16) as environment:
        get_context().set_server_args(args)
        assert get_exec().moe.enable_nccl_ep_cuda_graph
        runner = SimpleNamespace(
            device_module=torch.cuda,
            max_num_token=16,
            model_runner=SimpleNamespace(
                server_args=args,
                device="cuda",
                tp_group=environment.coordinator,
            ),
        )
        backend = resolve_decode_backend(runner)
        dispatcher = environment.dispatcher(layer_id=0)
        x = torch.ones(8, 2048, dtype=torch.bfloat16, device="cuda")
        ids = torch.tensor([[0, 1]] * 8, device="cuda")
        weights = torch.tensor([[0.25, 0.75]] * 8, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        try:
            with torch.cuda.stream(stream), backend.capture_session(stream):
                backend.capture_one(
                    ShapeKey(8), lambda: forward_layer(dispatcher, x, ids, weights, 0)
                )
            with backend.replay_session():
                result = backend.replay(ShapeKey(8), None)[2]
            torch.testing.assert_close(result, torch.full_like(x, 1.75), rtol=0, atol=0)
            assert environment.events.count("handle_create") == 1
        finally:
            backend.cleanup()


def test_opt_in_does_not_fall_back_on_an_unsupported_gpu(model_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kw: (8, 9))
    with pytest.raises(ValueError, match="NCCL EP.*sm_8"):
        server_args(model_path, enable_nccl_ep_cuda_graph=True)


@pytest.mark.parametrize(
    "overrides",
    [
        {"moe_a2a_backend": "none"},
        {"cuda_graph_backend_decode": "breakable"},
        {"cuda_graph_backend_decode": "tc_piecewise"},
        {"disable_cuda_graph": True},
        {"cuda_graph_backend_prefill": "full"},
        {"enable_two_batch_overlap": True},
        {"enable_single_batch_overlap": True},
        {"enable_pdmux": True},
        {"enable_eplb": True},
        {"elastic_ep_backend": "nixl"},
        {"enable_elastic_expert_backup": True},
        {"speculative_algorithm": "EAGLE"},
        {"enable_torch_compile": True},
        {"enable_memory_saver": True},
        {"nnodes": 2},
        {"nccl_ep_mode": "normal"},
        {"nccl_ep_num_max_dispatch_tokens_per_rank": 2048},
    ],
)
def test_opt_in_rejects_unsupported_execution_configs(
    model_path, monkeypatch, overrides
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kw: (9, 0))
    with pytest.raises(ValueError, match="NCCL EP CUDA Graph"):
        server_args(model_path, enable_nccl_ep_cuda_graph=True, **overrides)


def test_opt_in_requires_the_handle_update_capability(model_path, monkeypatch):
    import nccl.ep as ep

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kw: (9, 0))
    monkeypatch.setattr(ep.Handle, "update", None)
    with pytest.raises(ValueError, match="Handle.update"):
        server_args(model_path, enable_nccl_ep_cuda_graph=True)


def test_nccl_ep_gate_prefers_the_binding_library_version(model_path, monkeypatch):
    # The binding reports 2.30.7. Torch's compile-time version query must not
    # override that primary library identity.
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kw: (9, 0))
    monkeypatch.setattr(torch.cuda.nccl, "version", lambda: (2, 28, 9))
    args = server_args(model_path, enable_nccl_ep_cuda_graph=True)
    assert args.moe_a2a_backend == "nccl_ep"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
