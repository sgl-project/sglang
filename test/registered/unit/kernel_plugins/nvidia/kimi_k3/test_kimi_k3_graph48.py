# SPDX-License-Identifier: Apache-2.0
"""Selected-width host preparation/admission; no CUDA kernels or collectives execute."""

import importlib
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize("profile", ["k3_native_mla", "k3_full_decode"])
@pytest.mark.parametrize(
    "capture_widths,expected", [(None, (48,)), ((37, 48, 96), (37, 48))]
)
def test_selected_graph_widths_are_prepared_before_capture_and_eager_stays_native(
    monkeypatch, profile, capture_widths, expected
):
    from sglang.kernels.plugin_kernels.nvidia.spec import BuildContext
    from sglang.srt import runtime_context
    from sglang.srt.kernel_plugins.nvidia import binding
    from sglang.srt.kernel_plugins.nvidia.kimi_k3 import compat, providers, spec

    try:
        with monkeypatch.context() as env:
            env.setenv("SGLANG_NVIDIA_MEGAKERNEL_SLOTS", profile)
            importlib.reload(spec)
            assert 48 in spec.WIDTHS
            assert 48 not in spec.BASE_WIDTHS
            context = BuildContext(
                widths=(37, 48, 96, 128),
                capture_widths=capture_widths,
                tokens_per_request=1,
                deadline_s=30,
                binding=SimpleNamespace(native_mla_api_version=1),
            )
            resources = SimpleNamespace()
            with (
                patch.object(compat, "validate_runtime_compatibility"),
                patch.object(compat, "validate_tokenspeed_compatibility"),
                patch.object(
                    spec,
                    "get_model",
                    return_value=SimpleNamespace(context_length=1048576),
                ),
                patch.object(
                    runtime_context,
                    "get_parallel",
                    return_value=SimpleNamespace(attn_dcp_size=8),
                ),
                patch.object(binding, "megakernel_agree"),
                patch.object(torch.cuda, "current_device", return_value=0),
                patch.object(
                    torch.cuda,
                    "get_device_properties",
                    return_value=SimpleNamespace(multi_processor_count=148),
                ),
                patch.object(providers, "_cuda_memory", return_value={}),
                patch.object(
                    providers, "build_resources", return_value=resources
                ) as build,
            ):
                built = spec._build(
                    SimpleNamespace(language_model=object()), None, context
                )
            assert built.widths == expected
            assert build.call_args.kwargs["batches"] == expected
            assert build.call_args.kwargs[
                "native_mla" if profile == "k3_native_mla" else "full_decode"
            ]
            mode = SimpleNamespace(is_decode=lambda: True)
            with patch.object(
                torch.cuda, "is_current_stream_capturing", return_value=False
            ):
                assert (
                    built.plan(
                        SimpleNamespace(shape=(48,)),
                        SimpleNamespace(forward_mode=mode, batch_size=48),
                    )
                    is None
                )
            with patch.object(
                torch.cuda, "is_current_stream_capturing", return_value=True
            ):
                for rows in expected:
                    plan = built.plan(
                        SimpleNamespace(shape=(rows,)),
                        SimpleNamespace(forward_mode=mode, batch_size=rows),
                    )
                    assert plan.rows == rows and plan.mode == "decode"
                # Unprepared post-KV clamp bins never trigger a late build.
                for rows in (24, 32, 64, 96, 128):
                    assert (
                        built.plan(
                            SimpleNamespace(shape=(rows,)),
                            SimpleNamespace(forward_mode=mode, batch_size=rows),
                        )
                        is None
                    )
    finally:
        importlib.reload(spec)
    assert spec.WIDTHS == spec.BASE_WIDTHS


@pytest.mark.parametrize("native,width", [(False, 37), (True, 48)])
def test_actual_selected_width_build_reaches_all_93_layers_and_workspaces(
    monkeypatch, native, width
):
    from sglang.srt.kernel_plugins.nvidia.kimi_k3 import prefetch, providers

    trace = []

    def meta(*shape, dtype=torch.bfloat16):
        return torch.empty(shape, dtype=dtype, device="meta")

    class Stream:
        def __init__(self, *args, **kwargs):
            pass

        def synchronize(self):
            pass

    class Event:
        def record(self, stream):
            pass

    class Node:
        def __init__(self, *args, **kwargs):
            if "cooperative" in kwargs:
                trace.append(("expert_cooperative", kwargs["cooperative"]))

        def bind_ring(self, tensor):
            self.ring = tensor

        @classmethod
        def for_target(cls, *args, **kwargs):
            return cls()

    class Projection(Node):
        def __init__(self, weight, policies):
            self.policies = policies

        def prepare(self):
            return self

    class Core(Node):
        def __init__(self, proxy, policies, *args, **kwargs):
            self.policies = policies
            self.heads, self.kv_shards, self.kv_rank = (
                12,
                kwargs.get("kv_shards", 1),
                kwargs.get("kv_rank", 0),
            )
            trace.append(("core", policies))

    class Box(Node):
        def __init__(self, *args, **kwargs):
            self.front_views, self.hint_views = (
                {0: meta(1), 1: meta(1)},
                {0: meta(1), 1: meta(1)},
            )
            self.out = SimpleNamespace(buffer=meta(1, 64, 7168))

    class DcpBox(Node):
        def __init__(self, group, device, size, rank, heads, **kwargs):
            self.size, self.rank = size, rank

    class State(Node):
        def __init__(self, tp, device, batches):
            self.batches = batches
            self.stream, self.begin, self.shared_done, self.gate_done = (
                Stream(),
                Event(),
                Event(),
                Event(),
            )
            self.front, self.plan, self.shared = (), meta(1), meta(1)

    class Oproj(Node):
        def __init__(self, decoder, box, batches, *args, **kwargs):
            self.batches = batches

    class Tail(Node):
        def __init__(self, op, norm, up, batches, *args, **kwargs):
            self.batches = batches

    class Front(Node):
        def __init__(self, moe, packed, box, storage, batches, *args, **kwargs):
            self.batches = batches

    class MoE(Node):
        def __init__(self, front, experts, tail, state, **kwargs):
            self.front, self.tail, self.state = front, tail, state

    def workspace(device, heads, max_seq, *, batches):
        trace.append(("workspace", heads, max_seq, batches))
        return meta(1, dtype=torch.int8)

    modules = {
        "gemm.projection": SimpleNamespace(PreparedProjection=Projection),
        "mla.prepared": SimpleNamespace(PreparedMla=Core),
        "kda.prepared": SimpleNamespace(PreparedKda=Core),
        "attn_res.prepared": SimpleNamespace(PreparedAttnRes=Node),
        "oproj.prepared": SimpleNamespace(OprojMailbox=Box, PreparedOproj=Oproj),
        "common.twoshot": SimpleNamespace(TwoShot=Node),
        "moe.dense": SimpleNamespace(PreparedDenseMLP=Node),
        "moe.experts_runtime": SimpleNamespace(PersistentExperts=Node),
        "moe.front": SimpleNamespace(SharedFront=Front),
        "moe.prepared": SimpleNamespace(MoEScratch=State, PreparedMoE=MoE),
        "moe.small_batch.chain": SimpleNamespace(
            SmallBatchBox=Box, SmallBatchChain=Node
        ),
        "moe.tail_runtime": SimpleNamespace(MoeTailBox=Box, PreparedTail=Tail),
        "mla.dcp": SimpleNamespace(
            DcpBox=DcpBox,
            prepare_workspace=workspace,
            local_max=lambda n, d: ((n + 64 * d - 1) // (64 * d)) * 64,
        ),
        "common.embedding": SimpleNamespace(PreparedEmbedding=Node),
    }
    full_ids = [i for i in range(93) if i % 4 == 0]
    linear_ids = [i for i in range(93) if i % 4]
    mlp = SimpleNamespace(
        experts=SimpleNamespace(
            w13_weight=meta(1),
            w13_weight_scale=meta(1),
            w2_weight=meta(1),
            w2_weight_scale=meta(1),
        ),
        shared_experts=object(),
        gate=SimpleNamespace(weight=meta(896, 7168)),
        routed_expert_down_proj=SimpleNamespace(weight=meta(3584, 7168)),
        routed_expert_norm=SimpleNamespace(weight=meta(3584)),
        routed_expert_up_proj=SimpleNamespace(weight=meta(7168, 3584)),
        routed_scaling_factor=1.0,
    )
    layers = tuple(
        SimpleNamespace(
            layer_idx=i,
            _is_moe_layer=True,
            mlp=mlp,
            self_attn=SimpleNamespace(
                **({"q_b_proj": object()} if i in full_ids else {})
            ),
        )
        for i in range(93)
    )
    backbone = SimpleNamespace(
        layers=layers,
        start_layer=0,
        end_layer=93,
        config=SimpleNamespace(
            activation_situ_beta=4.0, activation_situ_linear_beta=25.0
        ),
        embed_tokens=SimpleNamespace(
            weight=meta(64, 7168),
            num_added_embeddings=0,
            tp_size=8,
            shard_indices=SimpleNamespace(org_vocab_start_index=0),
        ),
    )
    model = SimpleNamespace(
        model=backbone,
        parameters=lambda: iter((meta(1),)),
        config=SimpleNamespace(
            full_attention_layer_ids=full_ids,
            linear_layer_ids=linear_ids,
            rms_norm_eps=1e-6,
            moe_renormalize=True,
        ),
    )
    parallel = SimpleNamespace(
        tp_size=8,
        tp_rank=0,
        attn_dcp_size=8,
        attn_dcp_rank=0,
        tp_group=SimpleNamespace(device_group=object(), cpu_group=object()),
        dcp_group=SimpleNamespace(device_group=object(), cpu_group=object()),
    )
    # Group rejection/order is exercised by test_kimi_k3_init_barrier; this
    # host-width fixture isolates the CUDA-dependent host module itself.
    host = ModuleType("sglang.srt.kernel_plugins.nvidia.kimi_k3.cutedsl.common.host")
    host.validate_init_barrier_group = lambda device_group, cpu_group: cpu_group
    monkeypatch.setitem(sys.modules, host.__name__, host)
    monkeypatch.setattr(providers, "_module", lambda name, origins=None: modules[name])
    monkeypatch.setattr(providers, "get_parallel", lambda: parallel)
    monkeypatch.setattr(providers, "_mla_proxy", lambda module: (object(), ()))
    monkeypatch.setattr(providers, "_kda_proxy", lambda module: object())
    monkeypatch.setattr(
        providers, "coalesce_mla_projection_storage", lambda module: meta(3648, 7168)
    )
    monkeypatch.setattr(
        providers, "coalesce_kda_projection_storage", lambda module: meta(6288, 7168)
    )
    monkeypatch.setattr(providers, "_layer_proxy", lambda layer: layer)
    monkeypatch.setattr(providers, "_final_proxy", lambda model: model)
    monkeypatch.setattr(providers, "_cuda_memory", lambda: {})
    monkeypatch.setattr(providers.ResourceSet, "refresh_inventory", lambda self: None)
    monkeypatch.setattr(prefetch, "PrefetchController", Node)
    monkeypatch.setattr(torch.cuda, "Stream", Stream)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=148),
    )

    resources = providers.build_resources(
        model,
        context_length=1048576,
        memory_before={},
        deadline_ns=30_000_000_000,
        native_mla=native,
        full_decode=not native,
        batches=(width,),
    )
    assert resources.batches == (width,)
    assert len(resources.attention) == len(resources.feed_forward) == 93
    assert trace.count(("expert_cooperative", False)) == 93
    assert [
        key for key, value in resources.attention.items() if value.kind == "mla"
    ] == full_ids
    for value in resources.attention.values():
        projection = value.projection if native else value.projection.fallback
        assert projection.policies == {
            width: (64, 64, 2, 5) if value.kind == "mla" else (64, 64, 1, 6)
        }
        assert value.core.policies == {
            width: 3648 if value.kind == "mla" else (6288, 0)
        }
        assert value.output.batches == (width,)
    assert all(
        value.state.batches == value.front.batches == value.tail.batches == (width,)
        for value in resources.feed_forward.values()
    )
    if not native:
        assert ("workspace", 96, 131072, (width,)) in trace
        assert resources.embedding is not None
    else:
        assert not any(row[0] == "workspace" for row in trace)
    assert all(
        row["batches"] == [width]
        for row in resources.compilation_inventory
        if "batches" in row
    )


@pytest.mark.parametrize(
    "batches,extended",
    [
        ((48,), False),
        ((96,), True),
        ((128,), True),
        ((9,), False),
        ((0,), True),
        ((True,), True),
        ((), True),
    ],
)
def test_unadmitted_resource_widths_are_rejected_before_cuda_import(batches, extended):
    from sglang.srt.kernel_plugins.nvidia.kimi_k3 import providers

    with patch.object(providers, "_module") as load:
        with pytest.raises(ValueError, match="Graph48|prepared rows"):
            providers.build_resources(
                None,
                context_length=1048576,
                memory_before={},
                deadline_ns=1,
                full_decode=extended,
                batches=batches,
            )
        load.assert_not_called()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
