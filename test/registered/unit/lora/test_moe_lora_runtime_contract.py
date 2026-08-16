"""Runtime contracts at the MoE LoRA config/runner boundary.

This file intentionally covers only two durable properties:

* every config runner is admitted against the resident BF16 MoE layout before
  it can execute; and
* decode and prefill CUDA graphs consume distinct, graph-stable routing
  buffers whose contents may be refreshed in place between replays.

The latter is exercised with a real ``torch.cuda.CUDAGraph``.  Metadata is
updated outside capture, exactly as it is in serving, while a captured
consumer reads the stable allocations on each replay.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest import mock

import pytest
import torch

pytest.importorskip("triton")

from sglang.test.ci.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")

if TYPE_CHECKING:
    from sglang.srt.lora.backend.base_backend import BaseLoRABackend


class _FakeMoeLayer:
    """Minimal layer surface needed by metadata allocation."""

    def __init__(self, device: torch.device) -> None:
        self.base_layer = SimpleNamespace(
            w13_weight=torch.empty(1, device=device, dtype=torch.bfloat16),
        )


def _graph_backend(device: torch.device) -> BaseLoRABackend:
    from sglang.srt.lora.backend.base_backend import BaseLoRABackend

    backend = object.__new__(BaseLoRABackend)
    backend._is_moe_lora = True
    backend.device = device
    backend.max_loras_per_batch = 4
    backend.prefill_cuda_graph_batch_info = None
    backend.prefill_moe_cg_buffers = None
    backend.init_cuda_graph_moe_buffers(
        max_bs=8,
        max_loras=4,
        compute_dtype=torch.bfloat16,
        moe_layer=_FakeMoeLayer(device),
        include_legacy_kernel_buffers=False,
    )
    backend.init_prefill_cuda_graph_moe_buffers(max_num_tokens=12)
    return backend


def _batch_info(
    *,
    device: torch.device,
    weight_indices: list[int],
    lora_ranks: list[int],
    seg_lens: list[int],
) -> SimpleNamespace:
    seg_indptr = torch.tensor(
        [0, *torch.tensor(seg_lens).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    return SimpleNamespace(
        use_cuda_graph=True,
        bs=len(seg_lens),
        num_segments=len(seg_lens),
        seg_indptr=seg_indptr,
        weight_indices=torch.tensor(
            weight_indices,
            dtype=torch.int32,
            device=device,
        ),
        lora_ranks=torch.tensor(lora_ranks, dtype=torch.int32, device=device),
        req_seg_indptr=None,
        req_weight_indices=None,
        moe_lora_info=None,
    )


def _refresh_metadata(
    backend: BaseLoRABackend,
    batch_info: SimpleNamespace,
    *,
    seg_lens: list[int],
    is_prefill: bool,
) -> None:
    if is_prefill:
        backend.prefill_cuda_graph_batch_info = batch_info
    forward_batch = SimpleNamespace(
        batch_size=len(seg_lens),
        extend_seq_lens_cpu=seg_lens,
        forward_mode=SimpleNamespace(is_extend=lambda: is_prefill),
    )
    backend._add_moe_lora_info(forward_batch, batch_info)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph required")
def test_decode_and_prefill_metadata_refresh_through_real_graph_replay() -> None:
    """A replay must observe new routing without changing captured pointers."""
    device = torch.device("cuda")
    backend = _graph_backend(device)
    decode_buffers = backend.moe_cg_buffers
    prefill_buffers = backend.prefill_moe_cg_buffers
    assert prefill_buffers is not None

    decode_mapping_ptr = decode_buffers["token_lora_mapping"].data_ptr()
    prefill_mapping_ptr = prefill_buffers["token_lora_mapping"].data_ptr()
    assert decode_mapping_ptr != prefill_mapping_ptr

    decode_batch = _batch_info(
        device=device,
        weight_indices=[1, 2, 1, 0, 2, 1, 0, 2],
        lora_ranks=[0, 16, 32, 0],
        seg_lens=[1] * 8,
    )
    prefill_batch = _batch_info(
        device=device,
        weight_indices=[1, 2, 0],
        lora_ranks=[0, 16, 32, 0],
        seg_lens=[4, 5, 3],
    )
    _refresh_metadata(
        backend,
        decode_batch,
        seg_lens=[1] * 8,
        is_prefill=False,
    )
    _refresh_metadata(
        backend,
        prefill_batch,
        seg_lens=[4, 5, 3],
        is_prefill=True,
    )

    observed_decode = torch.empty_like(decode_buffers["token_lora_mapping"])
    observed_prefill = torch.empty_like(prefill_buffers["token_lora_mapping"])
    observed_decode_enabled = torch.empty_like(decode_buffers["adapter_enabled"])
    observed_prefill_enabled = torch.empty_like(prefill_buffers["adapter_enabled"])

    # Compile the copy path before capture; metadata routing was already
    # compiled by the first refresh above.
    observed_decode.copy_(decode_buffers["token_lora_mapping"])
    observed_prefill.copy_(prefill_buffers["token_lora_mapping"])
    observed_decode_enabled.copy_(decode_buffers["adapter_enabled"])
    observed_prefill_enabled.copy_(prefill_buffers["adapter_enabled"])
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        observed_decode.copy_(decode_buffers["token_lora_mapping"])
        observed_prefill.copy_(prefill_buffers["token_lora_mapping"])
        observed_decode_enabled.copy_(decode_buffers["adapter_enabled"])
        observed_prefill_enabled.copy_(prefill_buffers["adapter_enabled"])

    first_decode = observed_decode.clone()
    first_prefill = observed_prefill.clone()
    assert first_decode.tolist() == [1, 2, 1, -1, 2, 1, -1, 2]
    assert first_prefill.tolist() == [
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        -1,
        -1,
        -1,
    ]

    decode_batch = _batch_info(
        device=device,
        weight_indices=[3, 3, 0, 3],
        lora_ranks=[0, 0, 0, 8],
        seg_lens=[1] * 4,
    )
    prefill_batch.seg_indptr = torch.tensor([0, 2, 5], dtype=torch.int32, device=device)
    prefill_batch.weight_indices = torch.tensor(
        [3, 0], dtype=torch.int32, device=device
    )
    prefill_batch.lora_ranks = torch.tensor(
        [0, 0, 0, 8], dtype=torch.int32, device=device
    )
    prefill_batch.bs = 2
    prefill_batch.num_segments = 2
    _refresh_metadata(
        backend,
        decode_batch,
        seg_lens=[1] * 4,
        is_prefill=False,
    )
    _refresh_metadata(
        backend,
        prefill_batch,
        seg_lens=[2, 3],
        is_prefill=True,
    )

    # The serving scheduler completes the metadata refresh before launching
    # the model graph.  Make that ordering explicit even if PyTorch chooses a
    # distinct internal replay stream.
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()

    assert decode_buffers["token_lora_mapping"].data_ptr() == decode_mapping_ptr
    assert prefill_buffers["token_lora_mapping"].data_ptr() == prefill_mapping_ptr
    assert torch.equal(observed_decode, decode_buffers["token_lora_mapping"])
    assert torch.equal(observed_prefill, prefill_buffers["token_lora_mapping"])
    assert torch.equal(observed_decode_enabled, decode_buffers["adapter_enabled"])
    assert torch.equal(observed_prefill_enabled, prefill_buffers["adapter_enabled"])
    assert not torch.equal(observed_decode, first_decode)
    assert not torch.equal(observed_prefill, first_prefill)

    # Smaller replays must clear the graph bucket tail rather than inherit
    # routes from the earlier, larger assignment.
    assert observed_decode[4:].tolist() == [-1] * 4
    assert observed_prefill[5:].tolist() == [-1] * 7
    assert observed_decode_enabled.tolist() == [0, 0, 0, 1]
    assert observed_prefill_enabled.tolist() == [0, 0, 0, 1]
    assert observed_decode.tolist() == [3, 3, -1, 3, -1, -1, -1, -1]
    assert observed_prefill.tolist() == [
        3,
        3,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
    ]

    # A later base-only replay must not retain any active assignment from
    # either of the preceding batches.
    decode_batch = _batch_info(
        device=device,
        weight_indices=[0, 0],
        lora_ranks=[0, 0, 0, 0],
        seg_lens=[1, 1],
    )
    prefill_batch = _batch_info(
        device=device,
        weight_indices=[0],
        lora_ranks=[0, 0, 0, 0],
        seg_lens=[3],
    )
    _refresh_metadata(
        backend,
        decode_batch,
        seg_lens=[1, 1],
        is_prefill=False,
    )
    _refresh_metadata(
        backend,
        prefill_batch,
        seg_lens=[3],
        is_prefill=True,
    )
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()

    assert decode_buffers["token_lora_mapping"].data_ptr() == decode_mapping_ptr
    assert prefill_buffers["token_lora_mapping"].data_ptr() == prefill_mapping_ptr
    assert observed_decode.tolist() == [-1] * 8
    assert observed_prefill.tolist() == [-1] * 12
    assert observed_decode_enabled.tolist() == [0, 0, 0, 0]
    assert observed_prefill_enabled.tolist() == [0, 0, 0, 0]


class TestRunnerAdmission:
    """Reject resident states no precreated config runner can consume."""

    @staticmethod
    def _base_layer(**overrides):
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatcher
        from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

        runner_backend = SimpleNamespace(
            is_deep_gemm=lambda: overrides.get("deep_gemm", True),
        )
        quant_method = object.__new__(UnquantizedFusedMoEMethod)
        quant_method.runner = SimpleNamespace(runner_backend=runner_backend)
        dispatcher = object.__new__(StandardDispatcher)
        dispatcher.skip_local_expert_mapping = overrides.get("skip_local", False)
        is_gated = overrides.get("is_gated", True)
        # Resident gate/up width follows the gating declaration unless a test
        # forces a disagreement.
        gateup_width = overrides.get("gateup_width", (2 if is_gated else 1) * 4)
        return SimpleNamespace(
            quant_method=quant_method,
            dispatcher=dispatcher,
            w13_weight=torch.zeros(2, gateup_width, 4, dtype=torch.bfloat16),
            w2_weight=torch.zeros(2, 4, 4, dtype=torch.bfloat16),
            should_fuse_routed_scaling_factor_in_topk=overrides.get(
                "fused_scaling", False
            ),
            moe_runner_config=SimpleNamespace(
                activation=overrides.get("activation", "silu"),
                is_gated=is_gated,
                gemm1_alpha=None,
                gemm1_clamp_limit=None,
                swiglu_limit=None,
                apply_router_weight_on_input=False,
                no_combine=False,
                num_fused_shared_experts=0,
                top_k=2,
                routed_scaling_factor=1.0,
            ),
        )

    def _admit(self, **overrides) -> None:
        from sglang.srt.lora.moe.moe_lora_runner import MoeLoraRunner

        with mock.patch(
            "sglang.srt.layers.deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM",
            overrides.pop("jit_deep_gemm", True),
        ):
            MoeLoraRunner._admit(self._base_layer(**overrides))

    @pytest.mark.parametrize(
        ("activation", "is_gated"),
        [("silu", True), ("silu", False), ("relu2", False), ("relu2", True)],
    )
    def test_accepts_supported_activation_contracts(
        self,
        activation: str,
        is_gated: bool,
    ) -> None:
        self._admit(activation=activation, is_gated=is_gated)

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"deep_gemm": False}, "deep_gemm"),
            ({"jit_deep_gemm": False}, "JIT DeepGEMM"),
            ({"skip_local": True}, "EP-local"),
            ({"fused_scaling": True}, "routed scaling"),
            (
                {"activation": "gelu", "is_gated": False},
                "SiLU or ReLU2",
            ),
            (
                {"is_gated": False, "gateup_width": 8},
                "disagrees with",
            ),
        ],
    )
    def test_rejects_unsupported_resident_contracts(
        self,
        overrides: dict[str, object],
        message: str,
    ) -> None:
        with pytest.raises(NotImplementedError, match=message):
            self._admit(**overrides)
