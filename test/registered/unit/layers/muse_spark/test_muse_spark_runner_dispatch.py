"""Muse Spark's custom experts: the TRT-LLM dispatch contract and the runner
backends they admit below Blackwell."""

import unittest
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.moe import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.layers.moe.token_dispatcher import flashinfer
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import FlashinferA2ADispatchType
from sglang.srt.layers.muse_spark import muse_spark_fp4_moe
from sglang.srt.layers.muse_spark.muse_spark_bf16_moe import MuseSpark1xBf16Experts
from sglang.srt.layers.muse_spark.muse_spark_fp4_moe import MuseSpark1xFp4Experts
from sglang.srt.layers.muse_spark.muse_spark_nvfp4_moe import (
    MuseSparkNvFp4FusedMoEMethod,
)
from sglang.srt.layers.quantization import modelopt_quant
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptNvFp4FusedMoEMethod,
)
from sglang.srt.runtime_context import get_context, get_flags, get_forward, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _CpuCommunication:
    def __init__(self, _bootstrap, params, **_kwargs):
        self.params = params

    def dispatch(self, hidden_states, topk_ids, topk_weights, **_kwargs):
        return SimpleNamespace(
            hidden_states=hidden_states,
            hidden_states_scale=None,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )


class TestMuseSparkRunnerDispatch(CustomTestCase):
    def test_custom_expert_config_selects_trtllm_receive_sentinel(self):
        """Unoccupied SP receive slots must use the direct TRT-LLM consumer's
        sentinel even though these experts do not construct a generic runner.
        """
        with (
            get_context().override_server_args(model_path="dummy"),
            get_flags().moe.override(
                runner_backend=MoeRunnerBackend.FLASHINFER_TRTLLM,
                a2a_backend=MoeA2ABackend.NONE,
            ),
            get_parallel().override(
                tp_size=2,
                tp_rank=1,
                attn_tp_size=2,
                attn_tp_rank=1,
                moe_ep_size=2,
                moe_ep_rank=1,
            ),
            get_forward().scoped(sp_active=True),
            patch.object(
                modelopt_quant,
                "get_platform",
                return_value=SimpleNamespace(is_blackwell=True),
            ),
        ):
            experts = (
                MuseSpark1xBf16Experts(
                    num_experts=4,
                    num_local_experts=2,
                    local_expert_offset=2,
                    expert_hidden_size=128,
                    intermediate_size=128,
                    top_k=2,
                ),
                MuseSpark1xFp4Experts(
                    num_experts=4,
                    num_local_experts=2,
                    local_expert_offset=2,
                    hidden_size=128,
                    intermediate_size=128,
                    top_k=2,
                    quant_config=ModelOptFp4Config(
                        is_checkpoint_nvfp4_serialized=True, group_size=16
                    ),
                ),
            )
            # Replace FlashInfer's distributed workspace allocation, not the
            # SGLang constructor or its runner/sentinel selection.
            with ExitStack() as stack:
                for name, value in {
                    "use_flashinfer": True,
                    "BootstrapConfig": lambda **_: None,
                    "MoEEpCommParams": SimpleNamespace,
                    "NVLinkOneSidedConfig": lambda **_: None,
                    "TorchDistributedCommBackend": lambda _: None,
                    "create_communication": _CpuCommunication,
                    "get_flashinfer_a2a_dispatch_type": lambda: (
                        FlashinferA2ADispatchType.BF16
                    ),
                    "get_is_extend_in_batch": lambda: True,
                    "get_dp_global_num_tokens": lambda: None,
                    "is_dp_attention_enabled": lambda: False,
                }.items():
                    stack.enter_context(
                        patch.object(flashinfer, name, value, create=True)
                    )
                for expert in experts:
                    with self.subTest(expert=type(expert).__name__):
                        dispatcher = flashinfer.FlashinferDispatcher(
                            group=SimpleNamespace(size=lambda: 2, rank=lambda: 1),
                            router_topk=2,
                            num_experts=4,
                            num_local_experts=2,
                            hidden_size=128,
                            moe_runner_config=expert.moe_runner_config,
                            enable_sp_prefill_a2a=True,
                        )
                        topk = StandardTopKOutput(
                            torch.full((2, 2), 0.5),
                            torch.tensor([[0, 3], [2, 1]], dtype=torch.int32),
                            None,
                        )
                        output = dispatcher.dispatch(
                            torch.zeros((2, 128), dtype=torch.bfloat16), topk
                        )
                        self.assertEqual(
                            dispatcher.sp_prefill_moe_a2a.params.invalid_expert_id, -1
                        )
                        torch.testing.assert_close(
                            output.topk_output.topk_ids, topk.topk_ids
                        )


def _nvfp4_config():
    return ModelOptFp4Config(is_checkpoint_nvfp4_serialized=True, group_size=16)


@contextmanager
def _cuda_device(backend: MoeRunnerBackend, capability=(9, 0)):
    with (
        get_context().override_server_args(model_path="dummy"),
        get_flags().moe.override(
            runner_backend=backend, a2a_backend=MoeA2ABackend.NONE
        ),
        patch.object(
            modelopt_quant,
            "get_platform",
            return_value=SimpleNamespace(is_blackwell=capability >= (10, 0)),
        ),
        patch.object(muse_spark_fp4_moe, "is_cuda", return_value=True),
        patch.object(
            muse_spark_fp4_moe, "get_device_capability", return_value=capability
        ),
    ):
        yield


def _fp4_experts():
    return MuseSpark1xFp4Experts(
        num_experts=4,
        num_local_experts=2,
        local_expert_offset=2,
        hidden_size=128,
        intermediate_size=128,
        top_k=2,
        quant_config=_nvfp4_config(),
    )


class TestMuseSparkFp4BackendAdmission(CustomTestCase):
    def test_hopper_runs_packed_nvfp4_experts_as_w4a16(self):
        for backend in (MoeRunnerBackend.MARLIN, MoeRunnerBackend.HUMMING):
            with self.subTest(backend=backend.value), _cuda_device(backend):
                experts = _fp4_experts()
                self.assertTrue(experts.use_w4a16)
                self.assertEqual(experts.use_humming, backend.is_humming())

    def test_hopper_rejects_the_trtllm_runner(self):
        with _cuda_device(MoeRunnerBackend.FLASHINFER_TRTLLM):
            with self.assertRaisesRegex(ValueError, "Blackwell"):
                _fp4_experts()

    def test_blackwell_rejects_humming(self):
        """Muse admits humming only where auto picks a W4A16 backend (SM80-SM90);
        on Blackwell it must fail at startup, not serve an unvalidated path."""
        with _cuda_device(MoeRunnerBackend.HUMMING, capability=(10, 0)):
            with self.assertRaisesRegex(ValueError, "SM80-SM90 only"):
                _fp4_experts()

    def test_humming_checks_the_scale_layout_before_repacking(self):
        """Humming skips its own tensor validation for unpadded experts, so a
        checkpoint without one E4M3 scale per 16 weights must fail here."""
        with _cuda_device(MoeRunnerBackend.HUMMING):
            method = MuseSparkNvFp4FusedMoEMethod(_nvfp4_config())
        method._moe_runner_backend = MoeRunnerBackend.HUMMING
        packed = torch.zeros((2, 256, 64), dtype=torch.uint8)
        layer = SimpleNamespace(
            gate_up_proj=packed,
            gate_up_proj_weight_scale=torch.zeros(
                (2, 256, 4), dtype=torch.float8_e4m3fn
            ),
            down_proj=packed,
            down_proj_weight_scale=torch.zeros((2, 256, 8), dtype=torch.float8_e4m3fn),
        )
        with self.assertRaisesRegex(ValueError, "one FP8 scale per 16"):
            method.process_weights_after_loading(layer)

    def test_humming_admission_below_blackwell_is_muse_only(self):
        """Muse prepares and runs Humming's experts itself; the generic ModelOpt
        NVFP4 MoE method has no Humming path and must keep rejecting it."""
        with _cuda_device(MoeRunnerBackend.HUMMING):
            MuseSparkNvFp4FusedMoEMethod(_nvfp4_config())
            with self.assertRaisesRegex(ValueError, "does not support NVFP4"):
                ModelOptNvFp4FusedMoEMethod(_nvfp4_config())


if __name__ == "__main__":
    unittest.main()
