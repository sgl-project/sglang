"""Run with DWDP_TEST_DEVICES=4,5 python -m pytest .../test_npu_dwdp.py -s.

Exercises real FusedMoE kernels, IPC, ND/NZ source weights, slot reuse and
unequal forward counts (no matching collective may be required by a forward).
"""

import datetime
import json
import os
import socket
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank, devices, port, dtype):
    import torch_npu  # noqa: F401

    from sglang.srt.layers.moe import MoeA2ABackend, MoeRunnerBackend
    from sglang.srt.layers.moe.dwdp import DwdpManager
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.runtime_context import (
        SpawnRanks,
        get_flags,
        get_parallel,
        publish,
        set_global_dwdp_manager,
    )
    from sglang.srt.server_args import ServerArgs

    assert "sglang.srt.layers.moe.dwdp.dwdp_manager" not in sys.modules
    assert "sglang.srt.layers.moe.dwdp.transport" not in sys.modules
    assert DwdpManager.__name__ == "NPUDwdpManager"

    torch.npu.set_device(devices[rank])
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=90),
    )
    args = ServerArgs(
        model_path="dummy",
        device="npu",
        tp_size=2,
        dp_size=2,
        ep_size=2,
        dwdp_size=2,
        moe_dense_tp_size=1,
        enable_dp_attention=True,
        disable_cuda_graph=True,
    )
    publish(
        args,
        role="model_runner",
        ranks=SpawnRanks(world_rank=rank, gpu_id=devices[rank]),
    )
    group = SimpleNamespace(cpu_group=dist.group.WORLD, world_size=2)
    with (
        get_parallel().override(tp_group=group),
        get_flags().moe.override(
            a2a_backend=MoeA2ABackend.NONE,
            runner_backend=MoeRunnerBackend.ASCEND,
        ),
    ):
        # Non-contiguous layer IDs and different intermediate widths catch
        # incorrect layer-index-based slot assignment and view sizing.
        model = torch.nn.ModuleList()
        refs = []
        for pos, layer_id in enumerate((2, 5, 8, 11)):
            width = 64 * (1 + pos % 2)
            with torch.device(f"npu:{devices[rank]}"):
                layer = FusedMoE(
                    num_experts=4,
                    hidden_size=64,
                    intermediate_size=width,
                    layer_id=layer_id,
                    top_k=2,
                    params_dtype=dtype,
                    with_bias=True,
                )
            generator = torch.Generator().manual_seed(100 + layer_id)
            w13 = torch.randn(4, width * 2, 64, generator=generator) * 0.03
            w2 = torch.randn(4, 64, width, generator=generator) * 0.03
            b13 = torch.randn(4, width * 2, generator=generator) * 0.01
            b2 = torch.randn(4, 64, generator=generator) * 0.01
            for name, full in (
                ("w13_weight", w13),
                ("w2_weight", w2),
                ("w13_weight_bias", b13),
                ("w2_weight_bias", b2),
            ):
                target = getattr(layer, name)
                target.data.copy_(full[rank * 2 : (rank + 1) * 2].to(target.dtype))
            layer.quant_method.process_weights_after_loading(layer)
            refs.append((w13.to(dtype).float(), w2.to(dtype).float(), b13, b2))
            model.append(layer)
        manager = DwdpManager(args)
        manager.setup(model)
        set_global_dwdp_manager(manager)
        assert len(manager._slots) == 4
        # One rank stops executing MoE while the other continues reading it.
        for step in range(2 + 3 * rank):
            manager.prefetch_first_layers()
            for layer, (w13, w2, b13, b2) in zip(model, refs):
                tokens = 3 + step + rank
                gen = torch.Generator().manual_seed(900 + step + rank)
                x = (torch.randn(tokens, 64, generator=gen) * 0.1).to(dtype)
                ids = torch.stack(
                    (torch.arange(tokens) % 4, (torch.arange(tokens) + 2) % 4), -1
                )
                weights = (
                    torch.tensor([0.25, 0.75]).expand(tokens, 2).contiguous().to(dtype)
                )
                topk = StandardTopKOutput(
                    weights.npu(), ids.npu(), torch.zeros(tokens, 4, device="npu")
                )
                with torch.no_grad():
                    out = layer(x.npu(), topk).float().cpu()
                expected = torch.zeros(tokens, 64)
                for token in range(tokens):
                    for k in range(2):
                        expert = ids[token, k]
                        gate, up = (w13[expert] @ x[token].float() + b13[expert]).chunk(
                            2
                        )
                        val = (
                            w2[expert] @ (torch.nn.functional.silu(gate) * up)
                            + b2[expert]
                        )
                        expected[token] += weights[token, k].float() * val
                torch.testing.assert_close(out, expected, rtol=0.04, atol=5e-4)
        manager.cleanup()
        manager.cleanup()  # idempotent
        set_global_dwdp_manager(None)
    dist.destroy_process_group()
    print(f"rank {rank}: NPU DWDP MoE parity + independent forwards passed", flush=True)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_npu_dwdp(dtype):
    if not os.environ.get("DWDP_TEST_DEVICES"):
        pytest.skip("Set DWDP_TEST_DEVICES to two idle NPU device IDs")
    devices = [int(value) for value in os.environ["DWDP_TEST_DEVICES"].split(",")]
    assert len(devices) == 2
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(devices, port, dtype), nprocs=2, join=True)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"nnodes": 2}, "single host"),
        ({"quantization": "w8a8_int8"}, "unquantized BF16/FP16"),
        ({"enable_lora": True}, "immutable device-resident"),
        ({"enable_memory_saver": True}, "immutable device-resident"),
        ({"cpu_offload_gb": 1}, "immutable device-resident"),
        ({"enable_pdmux": True}, "layer-split"),
        ({"cuda_graph_backend_decode": "full"}, "explicit graph capture"),
        (
            {"cuda_graph_config": {"prefill": {"backend": "full"}}},
            "explicit graph capture",
        ),
    ],
)
def test_invalid_npu_dwdp_args(kwargs, message):
    from sglang.srt.arg_groups.parallel_hook import handle_dwdp
    from sglang.srt.server_args import ServerArgs

    args = ServerArgs(
        model_path="dummy", device="npu", tp_size=2, dwdp_size=2, **kwargs
    )
    with pytest.raises(ValueError, match=message):
        handle_dwdp(args)


def test_npu_dwdp_resolution():
    from sglang.srt.arg_groups.overrides import resolving_view
    from sglang.srt.arg_groups.parallel_hook import handle_dwdp
    from sglang.srt.server_args import ServerArgs

    args = ServerArgs(model_path="dummy", device="npu", tp_size=2, dwdp_size=2)
    handle_dwdp(args)
    cfg = resolving_view(args)
    assert cfg.disable_shared_experts_fusion
    assert cfg.dp_size == cfg.ep_size == 2
    assert cfg.moe_dense_tp_size == 1
    assert cfg.moe_a2a_backend == "none"
    assert cfg.enable_dp_attention
    assert cfg.enable_dp_lm_head


def test_npu_dwdp_full_resolution(tmp_path):
    from sglang.srt.arg_groups.overrides import resolved_view
    from sglang.srt.server_args import ServerArgs

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3MoeForCausalLM"],
                "model_type": "qwen3_moe",
                "hidden_size": 64,
                "intermediate_size": 128,
                "moe_intermediate_size": 64,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "num_hidden_layers": 4,
                "vocab_size": 128,
                "max_position_embeddings": 2048,
                "num_experts": 4,
                "num_experts_per_tok": 2,
            }
        )
    )
    args = ServerArgs(model_path=str(tmp_path), device="npu", tp_size=2, dwdp_size=2)
    args.resolve_once()
    cfg = resolved_view(args)
    assert cfg.dp_size == cfg.ep_size == 2
    assert cfg.disable_shared_experts_fusion
    assert cfg.cuda_graph_config.prefill.backend == "disabled"
    assert cfg.cuda_graph_config.decode.backend == "disabled"


def test_qwen_dwdp_shared_expert_replication():
    from sglang.srt.layers.moe import MoeA2ABackend, MoeRunnerBackend
    from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock
    from sglang.srt.runtime_context import (
        SpawnRanks,
        get_flags,
        publish,
        reset_context,
    )
    from sglang.srt.server_args import ServerArgs

    args = ServerArgs(
        model_path="dummy",
        device="npu",
        tp_size=2,
        dp_size=2,
        ep_size=2,
        dwdp_size=2,
        moe_dense_tp_size=1,
        enable_dp_attention=True,
    )
    publish(args, role="model_runner", ranks=SpawnRanks(world_rank=0, gpu_id=0))
    config = SimpleNamespace(
        num_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        hidden_size=64,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=128,
        hidden_act="silu",
        model_type="qwen3_5_moe_text",
    )
    try:
        with (
            get_flags().moe.override(
                a2a_backend=MoeA2ABackend.NONE,
                runner_backend=MoeRunnerBackend.ASCEND,
            ),
            torch.device("meta"),
        ):
            block = Qwen2MoeSparseMoeBlock(layer_id=0, config=config)
        shared = block.shared_expert
        assert shared.gate_up_proj.tp_size == shared.down_proj.tp_size == 1
        assert tuple(shared.gate_up_proj.weight.shape) == (256, 64)
        assert tuple(shared.down_proj.weight.shape) == (64, 128)
    finally:
        reset_context()
