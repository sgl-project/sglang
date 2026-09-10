import math
from dataclasses import dataclass
from unittest import mock

import torch

from sglang.srt.model_executor import cuda_graph_buffer_registry, input_buffers
from sglang.srt.model_executor.cuda_graph_buffer_registry import GraphSlot
from sglang.srt.model_executor.input_buffers import ForwardInputBuffers
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@dataclass
class _NestedBuffers:
    token_table: torch.Tensor
    offsets: torch.Tensor


@dataclass
class _TestBuffers(ForwardInputBuffers):
    pooled: torch.Tensor
    next_token_logits_buffer: torch.Tensor
    ngram_embedding_info: _NestedBuffers


class TestForwardInputBufferPool(CustomTestCase):
    def setUp(self) -> None:
        input_buffers._forward_input_buffer_pool.clear()

    def tearDown(self) -> None:
        input_buffers._forward_input_buffer_pool.clear()

    @staticmethod
    def _cuda_tensor_mock(*, clone=None, shape=(2, 2)):
        tensor = mock.Mock()
        tensor.numel.return_value = math.prod(shape)
        tensor.dtype = torch.int64
        tensor.device = torch.device("cuda")
        tensor.size.return_value = torch.Size(shape)
        tensor.stride.return_value = torch.empty(shape).stride()
        if clone is not None:
            tensor.clone.return_value = clone
        return tensor

    def test_first_cuda_registration_uses_selected_pool_and_is_reused(self) -> None:
        pool = object()
        canonical = mock.Mock()
        canonical.as_strided.return_value = object()
        first = self._cuda_tensor_mock(clone=canonical)
        second = self._cuda_tensor_mock()

        with mock.patch.object(
            torch.cuda, "use_mem_pool", return_value=mock.MagicMock()
        ) as use_mem_pool:
            input_buffers.share_input_buffer("seq_lens", first, pool)
            input_buffers.share_input_buffer("seq_lens", second, pool)

        use_mem_pool.assert_called_once_with(pool)
        first.clone.assert_called_once_with()
        second.clone.assert_not_called()
        self.assertIs(
            input_buffers._forward_input_buffer_pool[
                ("seq_lens", 4, torch.int64, torch.device("cuda"))
            ],
            canonical,
        )

    def test_eager_first_registry_canonical_uses_selected_pool(self) -> None:
        pool = object()
        canonical = mock.Mock()
        canonical.as_strided.return_value = mock.Mock()
        eager_buffer = self._cuda_tensor_mock(clone=canonical, shape=(4,))
        later_decode_buffer = self._cuda_tensor_mock(shape=(4,))
        registry = cuda_graph_buffer_registry.CudaGraphBufferRegistry(
            device=torch.device("cuda"),
            max_bs=4,
            max_num_tokens=4,
            share_pool=True,
            memory_pool=pool,
        )
        slot = GraphSlot(
            name="req_pool_indices",
            shape_fn=lambda bs, _num_tokens: (bs,),
            dtype=torch.int64,
            axis="bs",
        )

        with (
            mock.patch.object(
                cuda_graph_buffer_registry.torch, "zeros", return_value=eager_buffer
            ),
            mock.patch.object(
                torch.cuda, "use_mem_pool", return_value=mock.MagicMock()
            ) as use_mem_pool,
        ):
            registry.register_slot(slot)
            input_buffers.share_input_buffer(
                "req_pool_indices", later_decode_buffer, pool
            )

        use_mem_pool.assert_called_once_with(pool)
        eager_buffer.clone.assert_called_once_with()
        later_decode_buffer.clone.assert_not_called()
        self.assertIs(
            input_buffers._forward_input_buffer_pool[
                ("req_pool_indices", 4, torch.int64, torch.device("cuda"))
            ],
            canonical,
        )

    def test_build_eager_registry_forwards_selected_pool(self) -> None:
        pool = object()
        sentinel = object()
        with mock.patch.object(
            cuda_graph_buffer_registry,
            "build_decode_registry",
            return_value=sentinel,
        ) as build_decode:
            result = cuda_graph_buffer_registry.build_eager_registry(
                device=torch.device("cuda"),
                max_bs=4,
                max_num_token=16,
                cache_loc_dtype=torch.int64,
                memory_pool=pool,
            )

        self.assertIs(result, sentinel)
        self.assertIs(build_decode.call_args.kwargs["memory_pool"], pool)

    def test_share_buffers_routes_only_owned_storage_to_pool(self) -> None:
        pool = object()
        buffers = _TestBuffers(
            pooled=torch.zeros(2),
            next_token_logits_buffer=torch.zeros(2),
            ngram_embedding_info=_NestedBuffers(
                token_table=torch.zeros(2),
                offsets=torch.zeros(2),
            ),
        )

        with mock.patch.object(
            input_buffers,
            "share_input_buffer",
            side_effect=lambda _name, tensor, _pool: tensor,
        ) as share:
            buffers.share_buffers(
                memory_pool=pool,
                memory_pool_exclusions=frozenset(
                    {
                        "next_token_logits_buffer",
                        "ngram_embedding_info.token_table",
                    }
                ),
            )

        routed_pools = {call.args[0]: call.args[2] for call in share.call_args_list}
        self.assertIs(routed_pools["pooled"], pool)
        self.assertIsNone(routed_pools["next_token_logits_buffer"])
        self.assertIsNone(routed_pools["ngram_embedding_info.token_table"])
        self.assertIs(routed_pools["ngram_embedding_info.offsets"], pool)


if __name__ == "__main__":
    import unittest

    unittest.main()
