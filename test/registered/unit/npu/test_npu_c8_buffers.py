"""CPU contracts for A2 packed C8, typed L2 mirrors and sparse attention."""

import sys
import unittest
from enum import Enum
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

import sglang.srt.utils.hf_transformers  # noqa: F401
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

for name in (
    "torch_npu",
    "torch_npu.contrib",
    "sgl_kernel_npu",
    "sgl_kernel_npu.attention",
    "sgl_kernel_npu.attention.sinks_attention",
):
    sys.modules.setdefault(name, MagicMock())

from sglang.srt.arg_groups import kv_cache_hook
from sglang.srt.hardware_backend.npu.attention import ascend_backend
from sglang.srt.hardware_backend.npu.kv_capability import resolve_kv_capability
from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool
from sglang.srt.layers.quantization.modelslim.modelslim import ModelSlimConfig
from sglang.srt.mem_cache.pool_host import mla
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import get_context, get_parallel


class TestC8Configuration(CustomTestCase):
    def test_only_supported_model_and_parallel_layouts_are_accepted(self):
        for npu, dsa, cp, dcp, accepted in (
            (True, True, 1, 1, True),
            (False, True, 1, 1, False),
            (True, False, 1, 1, False),
            (True, True, 2, 1, False),
            (True, True, 1, 2, False),
        ):
            args = SimpleNamespace(kv_cache_dtype="int8", attn_cp_size=cp, dcp_size=dcp)
            with (
                self.subTest(npu=npu, dsa=dsa, cp=cp, dcp=dcp),
                patch.object(
                    kv_cache_hook, "resolving_view", side_effect=lambda value: value
                ),
                patch.object(
                    kv_cache_hook,
                    "get_platform",
                    return_value=SimpleNamespace(is_npu=npu),
                ),
                patch.object(
                    kv_cache_hook,
                    "model_config_of",
                    return_value=SimpleNamespace(
                        hf_config=SimpleNamespace(architectures=["model"])
                    ),
                ),
                patch(
                    "sglang.srt.configs.model_config.is_deepseek_dsa", return_value=dsa
                ),
            ):
                if accepted:
                    kv_cache_hook.handle_int8_kv_cache_compatibility(args)
                else:
                    with self.assertRaises(ValueError):
                        kv_cache_hook.handle_int8_kv_cache_compatibility(args)

    def test_modelslim_defers_runtime_int8_cache_to_the_npu_pool(self):
        config = ModelSlimConfig.__new__(ModelSlimConfig)
        config.requested_kv_cache_dtype = "int8"
        self.assertIsNone(config.get_modelslim_kv_scheme("model.layers.0.self_attn"))


class TestC8BufferContracts(CustomTestCase):
    def setUp(self):
        super().setUp()
        scope = get_context().override_server_args(page_size=4)
        scope.install()
        self.addCleanup(scope.restore)
        self.enterContext(get_parallel().override(attn_dcp_rank=0))

    def pool(self, *, packed=True, mixed=False, indexer="auto"):
        dtype = torch.int8 if packed else torch.bfloat16
        capability = resolve_kv_capability(
            kv_cache_dtype=dtype,
            is_arch35=False,
            soc_name="Ascend910B4",
            requested_indexer_kv_dtype=indexer,
            int8_indexer_op_available=True,
        )
        return NPUMLATokenToKVPool(
            size=8,
            page_size=4,
            dtype=dtype,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            layer_num=2,
            device="cpu",
            enable_memory_saver=False,
            index_head_dim=128,
            kv_cache_dim=656 if packed else None,
            kv_capability=capability,
            modelslim_cache_layout=SimpleNamespace(
                key_dtype=torch.int8, value_dtype=torch.bfloat16
            )
            if mixed
            else None,
        )

    def test_packed_pd_descriptors_match_k_and_indexer_storage(self):
        pool = self.pool()
        pointers, lengths, strides = pool.get_contiguous_buf_infos()
        self.assertEqual(len(pointers), 4)
        self.assertEqual(pool.get_kv_layer_ids(), [0, 1, 0, 1])
        self.assertEqual(strides, [4 * 656] * 2 + [4 * 128 * 2] * 2)
        self.assertEqual(
            sum(lengths), pool.k_buffer.nbytes + pool.index_k_buffer.nbytes
        )
        self.assertTrue(all(pointers))

    def test_l2_bytes_and_roundtrip_follow_component_dtypes(self):
        class Direction(Enum):
            H2D = 1
            D2H = 2

        def exchange(**kwargs):
            for name in ("k", "v", "index_k", "index_k_scale"):
                device = kwargs["device_" + name]
                host = kwargs["host_" + name]
                if device is None or host is None or device.numel() == 0:
                    continue
                self.assertEqual(device.dtype, host.dtype)
                for dev, cpu in zip(
                    kwargs["device_indices"][::4], kwargs["host_indices"][::4]
                ):
                    if kwargs["direction"] == Direction.D2H:
                        host[int(cpu) // 4].view_as(device[:, int(dev) // 4]).copy_(
                            device[:, int(dev) // 4]
                        )
                    else:
                        device[:, int(dev) // 4].copy_(
                            host[int(cpu) // 4].view_as(device[:, int(dev) // 4])
                        )

        for packed, mixed, indexer in (
            (True, False, "auto"),
            (False, False, "auto"),
            (False, True, "auto"),
            (True, False, "int8"),
            (False, False, "int8"),
        ):
            with self.subTest(packed=packed, mixed=mixed, indexer=indexer):
                pool = self.pool(packed=packed, mixed=mixed, indexer=indexer)
                device_components = [pool.k_buffer, pool.v_buffer, pool.index_k_buffer]
                if pool.index_k_scale_buffer is not None:
                    device_components.append(pool.index_k_scale_buffer)
                for tensor in device_components:
                    tensor.fill_(7)
                with (
                    patch.object(mla, "_is_npu", True),
                    patch.object(mla, "ascendc_io_enabled", return_value=False),
                    patch.object(
                        mla, "memfabric_host_memory_enabled", return_value=False
                    ),
                    patch.object(mla, "make_kernel_ptr_table", return_value=None),
                    patch.object(mla, "TransferDirection", Direction, create=True),
                    patch.object(
                        mla,
                        "transfer_kv_dim_exchange",
                        side_effect=exchange,
                        create=True,
                    ),
                    patch.dict(
                        mla.ALLOC_MEMORY_FUNCS,
                        {
                            "cpu": lambda dims, dtype, **kwargs: torch.empty(
                                dims, dtype=dtype
                            )
                        },
                    ),
                    patch(
                        "sglang.srt.mem_cache.pool_host.base.host_memory_budget_bytes",
                        return_value=1 << 40,
                    ),
                ):
                    host = mla.MLATokenToKVPoolHost(
                        pool._mla_buffer_info(),
                        host_to_device_ratio=2,
                        host_size=0,
                        page_size=4,
                        layout="page_first_kv_split",
                        pin_memory=False,
                        device="cpu",
                        device_capacity_tokens=8,
                        start_layer=0,
                        end_layer=2,
                        target_layer_num=2,
                    )
                    host_components = [
                        host.k_buffer,
                        host.v_buffer,
                        host.index_k_buffer,
                    ]
                    if host.index_k_scale_buffer is not None:
                        host_components.append(host.index_k_scale_buffer)
                    allocated = sum(tensor.nbytes for tensor in host_components)
                    self.assertEqual(
                        allocated,
                        host.get_size_per_token() * host.page_num * host.page_size,
                    )
                    device_ids, host_ids = torch.arange(4), torch.arange(8, 12)
                    pointers, lengths = host.get_page_buffer_meta(host_ids)
                    components = host_components
                    components = [tensor for tensor in components if tensor.numel()]
                    self.assertEqual(
                        lengths, [tensor[2].nbytes for tensor in components]
                    )
                    self.assertEqual(
                        pointers, [tensor[2].data_ptr() for tensor in components]
                    )
                    host.backup_from_device_all_layer(
                        pool, host_ids, device_ids, "kernel_ascend"
                    )
                    for tensor in device_components:
                        tensor[:, 0].zero_()
                    host.load_to_device_per_layer(
                        pool, host_ids, device_ids, 0, "kernel_ascend"
                    )
                    for tensor in device_components:
                        self.assertTrue(torch.all(tensor == 7))

    def test_sparse_attention_receives_packed_int8_and_bf16_queries(self):
        pool = self.pool()
        backend = ascend_backend.AscendAttnBackend.__new__(
            ascend_backend.AscendAttnBackend
        )
        backend.token_to_kv_pool = pool
        backend.kv_capability = pool.kv_capability
        backend.dsa_dcp_target = False
        backend.qk_rope_head_dim = 64
        backend._pad_topk_indices = lambda indices, count: indices
        backend.forward_metadata = SimpleNamespace(
            actual_seq_lengths_q=torch.tensor([1], dtype=torch.int32),
            actual_seq_lengths_kv=torch.tensor([4], dtype=torch.int32),
            block_tables=torch.tensor([[0]], dtype=torch.int32),
        )
        output = torch.zeros(1, 1, 512, dtype=torch.bfloat16)
        operator = MagicMock(return_value=output)
        with patch.object(
            ascend_backend.torch_npu, "npu_kv_quant_sparse_flash_attention", operator
        ):
            result = backend.forward_sparse(
                torch.ones(1, 1, 512, dtype=torch.float16),
                None,
                None,
                SimpleNamespace(layer_id=0, scaling=0.1),
                SimpleNamespace(forward_mode=ForwardMode.DECODE),
                save_kv_cache=False,
                q_rope=torch.ones(1, 1, 64, dtype=torch.float16),
                topk_indices=torch.tensor([[0]]),
            )
        self.assertIs(result, output)
        args = operator.call_args.kwargs
        self.assertEqual(args["query"].dtype, torch.bfloat16)
        self.assertEqual(args["key"].dtype, torch.int8)
        self.assertIs(args["key"], args["value"])
        self.assertEqual(args["quant_scale_repo_mode"], 1)
        self.assertEqual(args["tile_size"], 128)


if __name__ == "__main__":
    unittest.main()
