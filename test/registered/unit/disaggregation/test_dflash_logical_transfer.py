"""PD transfer of full DFlash draft KV in its logical token-id domain."""

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.dflash_kv import (
    DFlashDraftTransfer,
    draft_transfer_window,
    resolve_dflash_draft_transfer,
)
from sglang.srt.disaggregation.nixl.conn import NixlKVManager
from sglang.srt.disaggregation.utils import TransferBackend, setup_state_kv_args
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, enter_scope, published_topology

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

WIRE_PAGE = 64
TOKEN_BYTES = 512


def make_draft_pool(dcp_size):
    # A replicated draft pool pages the widened (logical) id space.
    page_size = WIRE_PAGE * dcp_size
    return SimpleNamespace(
        page_size=page_size,
        get_contiguous_buf_infos=lambda: (
            [1000, 2000],
            [1 << 20, 1 << 20],
            [page_size * TOKEN_BYTES] * 2,
        ),
    )


def make_draft_worker(layer_types, sliding_window=4096):
    hf_config = SimpleNamespace(layer_types=layer_types, sliding_window=sliding_window)
    return SimpleNamespace(
        draft_model_runner=SimpleNamespace(
            model_config=SimpleNamespace(hf_config=hf_config)
        )
    )


class RecordingAgent:
    def __init__(self):
        self.descs = []

    def get_xfer_descs(self, reqs, mem_kind):
        self.descs.append(np.asarray(reqs))
        return len(self.descs) - 1

    def initialize_xfer(self, op, src, dst, peer, notif):
        return (src, dst, notif)

    def transfer(self, handle):
        return "PROC"


class TestDFlashDraftTransfer(CustomTestCase):
    def setUp(self):
        enter_scope(self, published_topology(speculative_algorithm="DFLASH"))

    def resolve(self, *, dcp_size, backend=TransferBackend.NIXL, registered=True):
        enter_scope(self, get_parallel().override(attn_dcp_size=dcp_size))
        allocator = SimpleNamespace(
            full_draft_kv_pool=make_draft_pool(dcp_size) if registered else None
        )
        return resolve_dflash_draft_transfer(
            allocator=allocator,
            draft_worker=make_draft_worker(["full_attention"]),
            transfer_backend=backend,
        )

    def test_logical_component_only_under_dcp_on_nixl(self):
        """Without DCP the draft must keep riding the target KV entries."""
        for dcp_size, backend, registered, enabled in (
            (8, TransferBackend.NIXL, True, True),
            (1, TransferBackend.NIXL, True, False),
            (8, TransferBackend.MOONCAKE, True, False),
            (8, TransferBackend.NIXL, False, False),
        ):
            with self.subTest(
                dcp_size=dcp_size, backend=backend, registered=registered
            ):
                transfer = self.resolve(
                    dcp_size=dcp_size, backend=backend, registered=registered
                )
                kv_args = SimpleNamespace(page_size=WIRE_PAGE)
                setup_state_kv_args(
                    kv_args, SimpleNamespace(), dflash_draft_transfer=transfer
                )
                self.assertEqual(
                    kv_args.state_types, [StateType.DFLASH_KV] if enabled else []
                )
                if enabled:
                    # Item lengths are per target wire page, not per draft page.
                    self.assertEqual(
                        kv_args.state_item_lens, [[WIRE_PAGE * TOKEN_BYTES] * 2]
                    )

    def test_window_tail_only_when_every_draft_layer_slides(self):
        """A full-attention draft layer reads the whole prompt; HF windows
        include the current token."""
        sliding = ["sliding_attention"] * 2
        for layer_types, expected in (
            (sliding, 4096),
            (["sliding_attention", "full_attention"], None),
            (["full_attention"] * 2, None),
            (None, None),
        ):
            with self.subTest(layer_types=layer_types):
                hf_config = SimpleNamespace(
                    layer_types=layer_types, sliding_window=4096
                )
                self.assertEqual(draft_transfer_window(hf_config), expected)

    def test_payload_skips_decode_prefix_and_nixl_writes_logical_rows(self):
        """Draft KV is addressed by logical token ids even when the target's
        DCP rows are rank-local, and the decode prefix already holds it."""
        dcp_size = 8
        transfer = DFlashDraftTransfer(pool=make_draft_pool(dcp_size), window=None)
        # Logical ids of a 512-token request on prefill and on decode.
        prefill_row = torch.arange(4096, 4096 + 512)
        decode_row = torch.arange(8192, 8192 + 512)
        src_pages = transfer.page_indices(
            req_to_token=prefill_row,
            prefix_len=128,
            seq_len=512,
            wire_page_size=WIRE_PAGE,
        )
        dst_pages = transfer.page_indices(
            req_to_token=decode_row,
            prefix_len=128,
            seq_len=512,
            wire_page_size=WIRE_PAGE,
        )
        np.testing.assert_array_equal(src_pages, np.arange(66, 72))
        np.testing.assert_array_equal(dst_pages, np.arange(130, 136))

        ptrs, _, item_lens = transfer.buffer_infos(WIRE_PAGE)
        manager = NixlKVManager.__new__(NixlKVManager)
        manager.agent = RecordingAgent()
        manager.is_mla_backend = True
        manager.attn_tp_size = 8
        manager.pp_size = 1
        manager.kv_args = SimpleNamespace(
            gpu_id=0,
            kv_data_ptrs=[],
            state_types=[StateType.DFLASH_KV],
            state_data_ptrs=[ptrs],
            state_item_lens=[item_lens],
            state_dim_per_tensor=[[]],
            state_conv_shard_groups=[[]],
            state_slice_outer_counts=[[]],
            state_layer_ids=[[]],
            mla_compression_ratios=None,
        )
        handles = manager.maybe_send_extra(
            "decode",
            prefill_state_indices=[src_pages.tolist()],
            dst_state_data_ptrs=[[5000, 6000]],
            dst_state_indices=[dst_pages.tolist()],
            dst_gpu_id=1,
            notif="room_state_0",
            decode_tp_size=8,
            dst_state_item_lens=[item_lens],
        )
        self.assertEqual(len(handles), 1)
        src_descs, dst_descs = manager.agent.descs
        page_bytes = WIRE_PAGE * TOKEN_BYTES
        # One contiguous run of six pages per draft buffer.
        np.testing.assert_array_equal(
            src_descs[:, :2],
            [
                [1000 + 66 * page_bytes, 6 * page_bytes],
                [2000 + 66 * page_bytes, 6 * page_bytes],
            ],
        )
        np.testing.assert_array_equal(
            dst_descs[:, :2],
            [
                [5000 + 130 * page_bytes, 6 * page_bytes],
                [6000 + 130 * page_bytes, 6 * page_bytes],
            ],
        )

    def test_window_tail_starts_at_page_boundary(self):
        transfer = DFlashDraftTransfer(pool=make_draft_pool(1), window=100)
        pages = transfer.page_indices(
            req_to_token=torch.arange(1024),
            prefix_len=0,
            seq_len=1000,
            wire_page_size=WIRE_PAGE,
        )
        # 1000 - 100 = 900 floors to 896, the first page decode still reads.
        np.testing.assert_array_equal(pages, np.arange(14, 16))


if __name__ == "__main__":
    unittest.main()
