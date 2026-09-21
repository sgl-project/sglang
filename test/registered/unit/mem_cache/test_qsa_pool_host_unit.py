"""Real QSA compressed-key host transfers through the declared mirror."""

import unittest

import torch

from sglang.srt.mem_cache.pool_host.common import (
    ALLOC_MEMORY_FUNCS,
    alloc_with_pin_memory,
)
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.pool_host.qsa import (
    QSAIndexerHostPoolBuilder,
    qsa_indexer_pool_decl,
)
from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import is_cuda
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=9, stage="base-b", runner_config="1-gpu-small")

PAGE = 64
RATIO = 4


def _slots(pages, device):
    return (
        torch.tensor(pages, device=device)[:, None] * PAGE
        + torch.arange(PAGE, device=device)
    ).reshape(-1)


def _groups(slots):
    return slots[::RATIO] // RATIO


class TestQSAIndexerHostTransfer(unittest.TestCase):
    def setUp(self):
        if not (torch.cuda.is_available() and is_cuda()):
            self.skipTest("CUDA is required for QSA host transfer tests.")
        publish(ServerArgs(model_path="dummy"), role="test")
        self.addCleanup(reset_context)
        self._alloc = ALLOC_MEMORY_FUNCS["cuda"]
        ALLOC_MEMORY_FUNCS["cuda"] = alloc_with_pin_memory
        self.addCleanup(ALLOC_MEMORY_FUNCS.__setitem__, "cuda", self._alloc)

    @staticmethod
    def _device_pool(layers, start_layer=0):
        return QSATokenToKVPool(
            size=PAGE * 8,
            dtype=torch.bfloat16,
            page_size=PAGE,
            head_num=1,
            head_dim=128,
            full_attention_layer_ids=layers,
            start_layer=start_layer,
            device="cuda",
            mamba_pool=None,
            qsa_index_kv_heads=1,
            qsa_index_head_dim=128,
            qsa_compress_ratio=RATIO,
            qsa_token_topk=128,
            num_request_slots=4,
        )

    def _mirror(self, pool, layout, drafts=()):
        anchor = MHATokenToKVPoolHost(pool.full_kv_pool, 2, 0, PAGE, layout)
        host = QSAIndexerHostPoolBuilder().build(
            decl=qsa_indexer_pool_decl(pool),
            anchor_host=anchor,
            allocator_type="default",
            packed_draft_device_pools=drafts,
        )
        return anchor, host

    def _indices(self, host, backend):
        # host-side row indices live where each kernel path reads them
        src, dst = _slots([3, 1], "cuda"), _slots([4, 2], "cuda")
        host_idx = _slots([2, 0], "cpu")
        if backend == "kernel" and not host.can_use_write_back_jit:
            host_idx = host_idx.cuda()
        return src, dst, host_idx

    def _roundtrip(self, layout, backend):
        pool = self._device_pool([7, 11], start_layer=4)
        anchor, host = self._mirror(pool, layout)
        decl = qsa_indexer_pool_decl(pool)
        src, dst, host_idx = self._indices(host, backend)
        torch.manual_seed(42)
        pool.qsa_compressed_flat.normal_()
        expected = [b[_groups(src)].clone() for b in pool.qsa_compressed_k_buffer_pool]

        host.backup_from_device_all_layer(pool, host_idx, src, backend)
        torch.cuda.synchronize()
        pool.qsa_compressed_flat.fill_(-99)
        load_host_idx = host_idx.cuda() if backend == "kernel" else host_idx
        for local_layer in range(2):
            host.load_to_device_per_layer(
                pool, load_host_idx, dst, local_layer, backend
            )
        torch.cuda.synchronize()

        for layer_id, saved in zip((7, 11), expected):
            restored = pool.get_qsa_compressed_k_buffer(layer_id)[_groups(dst)]
            torch.testing.assert_close(restored, saved, rtol=0, atol=0)
        # declared bytes are what the mirror allocated
        self.assertEqual(
            decl.storage_info.host_bytes(
                page_num=anchor.page_num, layer_num=2, page_size=PAGE
            ),
            sum(b.nbytes for b in host.get_hybrid_pool_buffer()),
        )
        self.assertEqual(host.size, anchor.size)

    def test_roundtrip_page_first_kernel(self):
        self._roundtrip("page_first", "kernel")

    def test_roundtrip_layer_first_direct(self):
        self._roundtrip("layer_first", "direct")

    def test_packed_draft_layers_follow_target_layers(self):
        pool = self._device_pool([7, 11], start_layer=4)
        draft = self._device_pool([0])
        _, host = self._mirror(pool, "layer_first", drafts=(draft,))
        self.assertEqual(host.layer_num, 3)
        src, dst, host_idx = self._indices(host, "direct")
        src, dst = src.cpu(), dst.cpu()
        buffers = [
            *pool.qsa_compressed_k_buffer_pool,
            *draft.qsa_compressed_k_buffer_pool,
        ]
        for b in buffers:
            b.normal_()
        expected = [b[_groups(src).cuda()].clone() for b in buffers]

        host.backup_from_device_all_layer(pool, host_idx, src, "direct")
        torch.cuda.synchronize()
        for b in buffers:
            b.fill_(-99)
        for local_layer in range(2):
            host.load_to_device_per_layer(pool, host_idx, dst, local_layer, "direct")
        # the controller hands packed drafts target_layer_num + depth
        host.load_to_device_per_layer(
            draft, host_idx, dst, pool.full_kv_pool.layer_num, "direct", is_draft=True
        )
        torch.cuda.synchronize()

        for b, saved in zip(buffers, expected):
            torch.testing.assert_close(b[_groups(dst).cuda()], saved, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
