# SPDX-License-Identifier: Apache-2.0
"""CPU checks for host ordering; these do not compile or execute CUDA kernels."""

import importlib
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.kernel_plugins.nvidia.kimi_k3 import prefetch
from sglang.srt.kernel_plugins.nvidia.kimi_k3.profile import selected_profile
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_profile_selection_is_explicit_and_exclusive(monkeypatch):
    monkeypatch.delenv("SGLANG_NVIDIA_MEGAKERNEL_SLOTS", raising=False)
    assert selected_profile() == "community"
    monkeypatch.setenv("SGLANG_NVIDIA_MEGAKERNEL_SLOTS", "k3_full_decode")
    assert selected_profile() == "k3_full_decode"
    monkeypatch.setenv("SGLANG_NVIDIA_MEGAKERNEL_SLOTS", "k3_full_decode,k3_native_mla")
    with pytest.raises(ValueError, match="select one"):
        selected_profile()


def test_full_profile_requires_extended_runtime_and_checks_dependency_before_allocation(
    monkeypatch,
):
    from sglang.kernels.plugin_kernels.nvidia.spec import BuildContext, Feature
    from sglang.srt.kernel_plugins.nvidia.kimi_k3 import compat, spec

    try:
        with monkeypatch.context() as env:
            env.setenv("SGLANG_NVIDIA_MEGAKERNEL_SLOTS", "k3_full_decode")
            importlib.reload(spec)
            assert spec.FULL_DECODE and not spec.NATIVE_MLA
            assert spec.SPEC.requires.resident_blocks == 148
            assert Feature.CONTEXT_PARALLEL in spec.SPEC.features
            assert Feature.SPECULATIVE_DECODING not in spec.SPEC.features
            context = BuildContext(
                widths=(1,), tokens_per_request=1, deadline_s=30.0, binding=object()
            )
            with pytest.raises(ValueError, match="API v1"):
                spec._build(SimpleNamespace(), None, context)
            context = BuildContext(
                widths=(1,),
                tokens_per_request=1,
                deadline_s=30.0,
                binding=SimpleNamespace(native_mla_api_version=1),
            )
            with (
                patch.object(compat, "validate_runtime_compatibility"),
                patch.object(
                    compat,
                    "validate_tokenspeed_compatibility",
                    side_effect=RuntimeError("ABI rejected"),
                ),
                patch.object(torch.cuda, "get_device_properties") as gpu,
            ):
                with pytest.raises(RuntimeError, match="ABI rejected"):
                    spec._build(SimpleNamespace(), None, context)
                gpu.assert_not_called()
    finally:
        importlib.reload(spec)


def test_weight_ranges_follow_strides_alignment_and_byte_cap():
    backing = torch.empty(16, 16, dtype=torch.bfloat16)
    weights = (backing[:0], backing[:, ::2], torch.empty(32, dtype=torch.bfloat16))
    assert prefetch.weight_ranges(weights, 32) == [(backing.data_ptr(), 32)]
    spans = prefetch.weight_ranges(weights)
    # Last element's physical offset is 254 BF16 elements, rounded to 16 bytes.
    assert spans == [(backing.data_ptr(), 496), (weights[-1].data_ptr(), 64)]
    with pytest.raises(ValueError, match="aligned"):
        prefetch.weight_ranges((backing.flatten()[1:],))
    with pytest.raises(ValueError, match="at least 16"):
        prefetch.weight_ranges(weights, 1)


def test_resource_inventory_does_not_traverse_borrowed_collective_owners():
    from sglang.srt.kernel_plugins.nvidia.kimi_k3.providers import (
        _owned_resource_inventory,
    )

    class FrameworkGroup:
        @property
        def __dict__(self):
            raise AssertionError("framework communicator was charged to plugin")

    group = FrameworkGroup()
    root = SimpleNamespace(
        boxes={"dcp": SimpleNamespace(group=group)},
        scratch={},
        workspaces={},
        runtime_handles={},
        attention={},
        moe_front_weights={},
        embedding=SimpleNamespace(group=group),
    )
    model = SimpleNamespace(
        modules=lambda: (), parameters=lambda: (), buffers=lambda: ()
    )
    assert _owned_resource_inventory(root, model) == []


def test_prefetch_policy_and_stream_fork_join():
    trace = []
    main = SimpleNamespace(wait_stream=lambda stream: trace.append("join"))
    side = SimpleNamespace(cuda_stream=1, wait_event=lambda event: trace.append("wait"))

    class Event:
        def record(self, stream):
            trace.append("record")

    class Ranges:
        def __init__(self, weights, **kwargs):
            self.weights, self.options = tuple(weights), kwargs

        def launch(self, stream):
            trace.append("launch")

    weight = torch.empty(16)
    layers = tuple(
        SimpleNamespace(
            layer_idx=i,
            _is_moe_layer=True,
            mlp=SimpleNamespace(
                shared_experts=SimpleNamespace(
                    gate_up_proj=SimpleNamespace(weight=weight),
                    down_proj=SimpleNamespace(weight=weight),
                )
            ),
            self_attn=SimpleNamespace(o_proj=SimpleNamespace(weight=weight)),
        )
        for i in range(3)
    )
    providers = {
        i: SimpleNamespace(
            kind="kda" if i == 1 else "mla",
            projection_weight=weight,
            core=SimpleNamespace(weights=(weight,)),
        )
        for i in range(3)
    }
    with (
        patch.object(torch.cuda, "Stream", return_value=side),
        patch.object(torch.cuda, "Event", Event),
        patch.object(torch.cuda, "current_stream", return_value=main),
        patch.object(prefetch, "PreparedWeightRanges", Ranges),
        patch.object(prefetch, "_stream_handle", return_value=1),
    ):
        controller = prefetch.PrefetchController(
            SimpleNamespace(device=torch.device("cpu"), tp=8),
            layers,
            {i: weight for i in range(3)},
            providers,
        )
        assert ("attention_moe", 0, 7) not in controller.tables
        assert ("attention_moe", 0, 8) in controller.tables
        assert ("attention_moe", 1, 5) in controller.tables
        assert all(key[2] <= 8 for key in controller.tables)
        assert (
            controller.tables["next_attention", 0, 1].options["byte_cap"]
            == 96 * 1024**2
        )
        assert (
            controller.tables["next_attention", 0, 8].options["byte_cap"]
            == 32 * 1024**2
        )
        trace.clear()
        assert controller.fork("attention_moe", 1, 5)
        controller.join()
        controller.join()
        assert trace == ["record", "wait", "launch", "join"]
        assert not controller.fork("attention_moe", 1, 16)


@pytest.mark.parametrize("fail", [False, True])
def test_full_executor_uses_projection_return_embedding_and_joins_on_error(fail):
    # The registry contract deliberately reloads plugin modules. Import here so
    # this class and the validation patch always refer to the same live module.
    from sglang.srt.kernel_plugins.nvidia.kimi_k3.executor import (
        KimiK3MegakernelExecutor,
    )

    trace = []
    scratch = {
        name: torch.zeros(1, 7168)
        for name in (
            "prefix",
            "blocks",
            "hidden.0",
            "hidden.1",
            "hidden.out",
            "mla.projected",
            "kda.projected",
            "attention.local",
            "embedding.out",
        )
    }
    projected = torch.ones(1, 7)
    attention = {}
    for i in range(93):
        kind = "mla" if i % 4 == 0 else "kda"

        def core(value, out, *, index=i, **kwargs):
            assert value is projected
            trace.append(("core", index))
            if fail and index == 1:
                raise RuntimeError("core failed")

        attention[i] = SimpleNamespace(
            kind=kind,
            cache=object(),
            caches=(object(),),
            projection=SimpleNamespace(run=lambda *args: projected),
            core=SimpleNamespace(run=core),
            output=SimpleNamespace(
                run=lambda *args, index=i: trace.append(("output", index))
            ),
        )
    nop = SimpleNamespace(run=lambda x, *args: x)
    embedded = torch.zeros(1, 7168)
    embedding = Mock()
    embedding.run.return_value = embedded
    resources = SimpleNamespace(
        binding_fingerprint="bound",
        native_mla=False,
        full_decode=True,
        boxes={"dcp": object()},
        scratch=scratch,
        layers=tuple(SimpleNamespace(layer_idx=i) for i in range(93)),
        attention=attention,
        before={i: nop for i in range(93)},
        fused_before={},
        feed_forward={i: nop for i in range(93)},
        final=nop,
        embedding=embedding,
        model=SimpleNamespace(model=SimpleNamespace(embed_tokens=Mock())),
        prefetch=SimpleNamespace(
            fork=lambda point, layer, batch: trace.append((point, layer)),
            join=lambda: trace.append("join"),
        ),
    )
    view = SimpleNamespace(
        mla=SimpleNamespace(
            write_slots=torch.zeros(1),
            sequence_lengths=torch.zeros(1),
            global_sequence_lengths=torch.ones(1),
            block_table=torch.zeros(1, 1),
        ),
        kda=SimpleNamespace(
            state_indices=torch.zeros(1), track_decode_state=lambda i: None
        ),
        prepare_layer=lambda i: trace.append(("prepare", i)),
    )
    with patch(
        "sglang.srt.kernel_plugins.nvidia.kimi_k3.executor._validate_step_inputs"
    ):
        if fail:
            with pytest.raises(RuntimeError, match="core failed"):
                KimiK3MegakernelExecutor(resources).run(
                    torch.zeros(1, dtype=torch.int64), None, view
                )
        else:
            result = KimiK3MegakernelExecutor(resources).run(
                torch.zeros(1, dtype=torch.int64), None, view
            )
            assert result.data_ptr() == scratch["hidden.out"].data_ptr()
    assert trace[:5] == [
        ("prepare", 0),
        ("core", 0),
        ("attention_moe", 0),
        ("output", 0),
        ("oproj_down", 0),
    ]
    assert trace[5:8] == [("prepare", 1), ("attention_moe", 1), ("core", 1)]
    assert trace[-1] == "join"
    embedding.run.assert_called_once()
    resources.model.model.embed_tokens.assert_not_called()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
