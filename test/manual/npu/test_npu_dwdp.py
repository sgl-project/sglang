"""Run with DWDP_TEST_DEVICES=0,1 pytest test/manual/npu/test_npu_dwdp.py -s.

Exercises real FusedMoE kernels, IPC, ND/NZ source weights, slot reuse and
unequal forward counts (no matching collective may be required by a forward).
"""

import datetime
import json
import os
import socket
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _int8_layer(layer_id, hidden, width, rank, large):
    import torch_npu

    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.quantization.modelslim.modelslim import ModelSlimConfig

    config = ModelSlimConfig(
        {
            f"experts.0.{name}.weight": "W8A8_DYNAMIC"
            for name in ("gate_proj", "up_proj", "down_proj")
        }
    )
    layer = FusedMoE(
        num_experts=4,
        hidden_size=hidden,
        intermediate_size=width,
        layer_id=layer_id,
        top_k=2,
        params_dtype=torch.bfloat16,
        quant_config=config,
        prefix="experts",
    )
    gen = torch.Generator(device="cpu").manual_seed(100 + layer_id)
    for prefix, rows, cols in (("w13", width * 2, hidden), ("w2", hidden, width)):
        weight = torch.randint(
            -100, 101, (4, rows, cols), generator=gen, dtype=torch.int8, device="cpu"
        )
        scale = 0.0002 + torch.rand(4, rows, 1, generator=gen, device="cpu") * 0.0004
        for name, full in (
            ("weight", weight),
            ("weight_scale", scale),
            ("weight_offset", torch.zeros_like(scale)),
        ):
            data = full if rank is None else full[rank * 2 : (rank + 1) * 2]
            getattr(layer, f"{prefix}_{name}").data = data.npu().contiguous()
    layer.quant_method.process_weights_after_loading(layer)
    for name in ("w13_weight", "w2_weight"):
        param = getattr(layer, name)
        param.data = torch_npu.npu_format_cast(param.data, 29 if large else 2)
    if rank is None:
        # Full-weight single-device oracle; never enters DWDP event hooks.
        layer.bind_full_expert_weights({})
        layer._dwdp_bound = False
    return layer


def _worker(rank, devices, port, dtype, large):
    import torch_npu  # noqa: F401

    from sglang.srt.hardware_backend.npu.dwdp.vmm import NPUWeightVMM
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

    assert "sglang.srt.layers.moe.dwdp.cuda_backend" not in sys.modules
    assert "sglang.srt.layers.moe.dwdp.transport" not in sys.modules
    assert DwdpManager.__name__ == "NPUDwdpManager"
    assert "sglang.srt.layers.moe.dwdp.dwdp_manager" not in sys.modules

    torch.npu.set_device(devices[rank])
    torch.npu.config.allow_internal_format = True
    torch.set_num_threads(1)
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
        dp_size=1,
        attn_dp_size=2,
        ep_size=2,
        dwdp_size=2,
        moe_dense_tp_size=1,
        disable_cuda_graph=True,
        quantization="modelslim" if dtype == torch.int8 else None,
    )
    publish(
        args,
        role="model_runner",
        ranks=SpawnRanks(world_rank=rank, gpu_id=devices[rank]),
    )
    device_group = dist.new_group(
        [0, 1], backend="hccl", timeout=datetime.timedelta(seconds=90)
    )
    group = SimpleNamespace(
        cpu_group=dist.group.WORLD, device_group=device_group, world_size=2
    )
    with (
        get_parallel().override(tp_group=group),
        get_flags().moe.override(
            a2a_backend=MoeA2ABackend.NONE,
            runner_backend=MoeRunnerBackend.ASCEND,
        ),
    ):
        # Non-contiguous layer IDs and different intermediate widths catch
        # incorrect layer-index-based slot assignment and view sizing.
        moe_layers = []
        refs = []
        int8 = dtype == torch.int8
        input_dtype = torch.bfloat16 if int8 else dtype
        hidden = (512 if int8 else 256) if large else 64
        for pos, layer_id in enumerate((2, 4, 7, 9)):
            width = (2048 + 256 * (pos // 2)) if large else 64 * (1 + pos % 2)
            if int8:
                with torch.device(f"npu:{devices[rank]}"):
                    moe_layers.append(_int8_layer(layer_id, hidden, width, rank, large))
                    refs.append(_int8_layer(layer_id, hidden, width, None, large))
                continue
            with torch.device(f"npu:{devices[rank]}"):
                layer = FusedMoE(
                    num_experts=4,
                    hidden_size=hidden,
                    intermediate_size=width,
                    layer_id=layer_id,
                    top_k=2,
                    params_dtype=dtype,
                    with_bias=True,
                )
            generator = torch.Generator().manual_seed(100 + layer_id)
            w13 = torch.randn(4, width * 2, hidden, generator=generator) * 0.03
            w2 = torch.randn(4, hidden, width, generator=generator) * 0.03
            b13 = torch.randn(4, width * 2, generator=generator) * 0.01
            b2 = torch.randn(4, hidden, generator=generator) * 0.01
            for name, full in (
                ("w13_weight", w13),
                ("w2_weight", w2),
                ("w13_weight_bias", b13),
                ("w2_weight_bias", b2),
            ):
                target = getattr(layer, name)
                target.data.copy_(full[rank * 2 : (rank + 1) * 2].to(target.dtype))
            layer.quant_method.process_weights_after_loading(layer)
            for name in ("w13_weight", "w2_weight"):
                weight = getattr(layer, name)
                weight.data = torch_npu.npu_format_cast(weight.data, 29 if large else 2)
                assert torch_npu.get_npu_format(weight) == (29 if large else 2)
            refs.append((w13.to(dtype).float(), w2.to(dtype).float(), b13, b2))
            moe_layers.append(layer)
        model = torch.nn.Module()
        model.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(10)])
        for layer in moe_layers:
            model.layers[layer.layer_id] = layer
        manager = DwdpManager(args)
        original_migrate = manager._migrate_weight
        migrated = []

        def checked_migrate(key, param, peers):
            assert all(source.untyped_storage().nbytes() == 0 for source in migrated)
            handle = original_migrate(key, param, peers)
            assert param.untyped_storage().nbytes() == 0
            migrated.append(param)
            return handle

        manager._migrate_weight = checked_migrate
        mappings, original_map = {}, NPUWeightVMM._map

        def record_map(vmm, ptr, size, handle):
            original_map(vmm, ptr, size, handle)
            mappings[ptr] = (size, handle)

        with patch.object(NPUWeightVMM, "_map", record_map):
            manager.setup(model)
        assert manager._moe_layer_indices == [2, 4, 7, 9]
        assert len(migrated) == len(moe_layers) * 2
        if large:
            pool = manager._vmm._page_pool
            assert all(pool.num_pages(slot) >= 2 for slot in (0, 1))
            for li in (2, 7):
                assert manager._buffer_index_for_layer(li) == 0
                layout = manager._vmm.layout(manager._specs[li, "w13_weight"], rank)
                base = (
                    manager._weights[li, "w13_weight"].data_ptr() - layout.pre_padding
                )
                offset = 0 if layout.pre_size else layout.mnnvl_size
                assert mappings[base + offset] == (
                    pool.page_size,
                    pool._page_handles[0][0],
                )
        set_global_dwdp_manager(manager)
        # Every size uses the same remote-expert copy path, including remote
        # bytes inside local boundary pages. Large weights also reuse remote
        # page slots across layers and exercise partial-page local shards.
        assert len(manager._peer_views) == len(moe_layers) * 2
        for (peer, li, name), view in manager._peer_views.items():
            assert peer != rank
            spec = manager._specs[li, name]
            assert tuple(view.shape) == spec.chunk_shape
            assert view.dtype == spec.dtype
            assert view.is_contiguous()
        local_snapshots = {
            (layer.layer_id, name): getattr(layer, name)[rank * 2 : (rank + 1) * 2]
            .cpu()
            .clone()
            for layer in moe_layers
            for name in ("w13_weight", "w2_weight")
        }
        for li, name in local_snapshots:
            remote = manager._get_remote_slices(li, name)
            assert sum(dst.shape[0] for dst, *_ in remote) == 2
        # One rank stops executing MoE while the other continues reading it.
        for step in range(2 + 3 * rank):
            # This also destroys resident remote edge bytes: every forward
            # must restore them from peers, without changing any local source.
            for li, name in local_snapshots:
                for dst, *_ in manager._get_remote_slices(li, name):
                    dst.fill_(-77)
            torch.npu.synchronize()
            manager.prefetch_first_layers()
            for layer, reference in zip(moe_layers, refs):
                tokens = 3 + step + rank
                gen = torch.Generator().manual_seed(900 + step + rank)
                x = (torch.randn(tokens, hidden, generator=gen) * 0.1).to(input_dtype)
                ids = torch.stack(
                    (torch.arange(tokens) % 4, (torch.arange(tokens) + 2) % 4), -1
                )
                weights = (
                    torch.tensor([0.25, 0.75])
                    .expand(tokens, 2)
                    .contiguous()
                    .to(input_dtype)
                )
                topk = StandardTopKOutput(
                    weights.npu(), ids.npu(), torch.zeros(tokens, 4, device="npu")
                )
                with torch.no_grad():
                    out = layer(x.npu(), topk).float().cpu()
                    if int8:
                        expected = reference(x.npu(), topk).float().cpu()
                        torch.testing.assert_close(out, expected, rtol=0.02, atol=5e-4)
                        continue
                w13, w2, b13, b2 = reference
                expected = torch.zeros(tokens, hidden)
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
            for layer in moe_layers:
                for name in ("w13_weight", "w2_weight"):
                    saved = local_snapshots[layer.layer_id, name]
                    torch.testing.assert_close(
                        getattr(layer, name)[rank * 2 : (rank + 1) * 2].cpu(),
                        saved,
                        rtol=0,
                        atol=0,
                    )
        manager.cleanup()
        manager.cleanup()  # idempotent
        set_global_dwdp_manager(None)
    dist.destroy_process_group(device_group)
    dist.destroy_process_group()
    print(f"rank {rank}: NPU DWDP MoE parity + independent forwards passed", flush=True)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.int8])
@pytest.mark.parametrize("large", [False, True])
def test_npu_dwdp(dtype, large):
    if not os.environ.get("DWDP_TEST_DEVICES"):
        pytest.skip("Set DWDP_TEST_DEVICES to two idle NPU device IDs")
    devices = [int(value) for value in os.environ["DWDP_TEST_DEVICES"].split(",")]
    assert len(devices) == 2
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(devices, port, dtype, large), nprocs=2, join=True)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"nnodes": 2}, "single host"),
        ({"enable_lora": True}, "LoRA"),
        ({"enable_pdmux": True}, "layer-split"),
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


@pytest.mark.parametrize("device", ["cuda", "npu"])
@pytest.mark.parametrize(
    "kwargs",
    [
        {"cpu_offload_gb": 1},
        {"offload_group_size": 2},
        {"enable_memory_saver": True},
    ],
)
def test_dwdp_preserves_offload_and_memory_saver_settings(device, kwargs):
    from sglang.srt.arg_groups.overrides import resolving_view
    from sglang.srt.arg_groups.parallel_hook import handle_dwdp
    from sglang.srt.server_args import ServerArgs

    args = ServerArgs(
        model_path="dummy", device=device, tp_size=2, dwdp_size=2, **kwargs
    )
    handle_dwdp(args)
    cfg = resolving_view(args)
    for name, value in kwargs.items():
        assert getattr(cfg, name) == value


@pytest.mark.parametrize("device", ["cuda", "npu"])
@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({}, ("disabled", "disabled")),
        ({"disable_cuda_graph": False}, ("disabled", "disabled")),
        ({"cuda_graph_backend_decode": "full"}, ("full", "disabled")),
        ({"cuda_graph_backend_prefill": "full"}, ("disabled", "full")),
        (
            {
                "cuda_graph_config": {
                    "decode": {"backend": "full"},
                    "prefill": {"backend": "full"},
                }
            },
            ("full", "full"),
        ),
        (
            {
                "cuda_graph_backend_decode": "full",
                "cuda_graph_config": {
                    "decode": {"backend": "disabled"},
                    "prefill": {"backend": "full"},
                },
            },
            ("disabled", "full"),
        ),
        (
            {
                "cuda_graph_config": {
                    "decode": {"backend": "breakable"},
                    "prefill": {"backend": "breakable"},
                }
            },
            ("breakable", "breakable"),
        ),
    ],
)
def test_dwdp_graph_config_matches_cuda(device, kwargs, expected):
    from sglang.srt.arg_groups.cuda_graph_hook import parse_cuda_graph_config
    from sglang.srt.arg_groups.overrides import resolving_view
    from sglang.srt.arg_groups.parallel_hook import handle_dwdp
    from sglang.srt.server_args import ServerArgs

    args = ServerArgs(
        model_path="dummy", device=device, tp_size=2, dwdp_size=2, **kwargs
    )
    handle_dwdp(args)
    parse_cuda_graph_config(args)
    cfg = resolving_view(args)
    assert cfg.disable_cuda_graph
    assert (
        cfg.cuda_graph_config.decode.backend,
        cfg.cuda_graph_config.prefill.backend,
    ) == expected


@pytest.mark.parametrize("device", ["cuda", "npu"])
@pytest.mark.parametrize("shared_tp1", [False, True])
@pytest.mark.parametrize("disable_fusion", [False, True])
def test_dwdp_preserves_shared_expert_settings(device, shared_tp1, disable_fusion):
    from sglang.srt.arg_groups.overrides import resolving_view
    from sglang.srt.arg_groups.parallel_hook import handle_dwdp
    from sglang.srt.environ import envs
    from sglang.srt.server_args import ServerArgs

    with envs.SGLANG_SHARED_EXPERT_TP1.override(shared_tp1):
        args = ServerArgs(
            model_path="dummy",
            device=device,
            tp_size=2,
            dwdp_size=2,
            disable_shared_experts_fusion=disable_fusion,
        )
        handle_dwdp(args)
        cfg = resolving_view(args)
        assert envs.SGLANG_SHARED_EXPERT_TP1.get() == shared_tp1
        assert cfg.disable_shared_experts_fusion == disable_fusion
        assert cfg.attn_dp_size == cfg.ep_size == 2
        assert cfg.dp_size == 1
        assert cfg.moe_dense_tp_size == 1
        assert cfg.moe_a2a_backend == "none"
        assert cfg.enable_dp_lm_head


def test_npu_dwdp_full_resolution(tmp_path):
    from sglang.srt.arg_groups.overrides import resolved_view
    from sglang.srt.environ import envs
    from sglang.srt.server_args import ServerArgs

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DeepseekV2ForCausalLM"],
                "model_type": "deepseek_v2",
                "hidden_size": 64,
                "intermediate_size": 128,
                "moe_intermediate_size": 64,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "num_hidden_layers": 4,
                "vocab_size": 128,
                "max_position_embeddings": 2048,
                "n_routed_experts": 4,
                "n_shared_experts": 1,
                "num_experts_per_tok": 2,
            }
        )
    )
    with envs.SGLANG_SHARED_EXPERT_TP1.override(False):
        args = ServerArgs(
            model_path=str(tmp_path), device="npu", tp_size=2, dwdp_size=2
        )
        args.resolve_once()
        cfg = resolved_view(args)
        assert not envs.SGLANG_SHARED_EXPERT_TP1.get()
        assert cfg.attn_dp_size == cfg.ep_size == 2
        assert cfg.dp_size == 1
        assert not cfg.disable_shared_experts_fusion
        assert cfg.cuda_graph_config.prefill.backend == "disabled"
        assert cfg.cuda_graph_config.decode.backend == "disabled"
