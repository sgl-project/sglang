import concurrent.futures
import enum
import logging
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from sglang.srt.disaggregation.ascend.transfer_engine import AscendTransferEngine
from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.utils import group_concurrent_contiguous
from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVBootstrapServer,
    MooncakeKVManager,
    MooncakeKVReceiver,
    MooncakeKVSender,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils.network import get_local_ip_auto

logger = logging.getLogger(__name__)


class AscendStateType(str, enum.Enum):
    """DSV4-on-NPU PD components without a cross-hardware equivalent."""

    DSV4_C128 = "dsv4_c128"
    # C4 compress-state rows (attention + indexer) addressed within each
    # req_pool_idx bank on A5 (CYCLE cache_mode).  Separate from StateType.SWA
    # because each peer maps logical positions into its own local ring.
    DSV4_C4_STATE = "dsv4_c4_state"


_DSV4_KVCACHE_STATE_TYPES = tuple(AscendStateType)


def _build_page_interleaved_dcp_plan(
    src_page_indices: npt.NDArray[np.int32],
    dst_page_indices: npt.NDArray[np.int32],
    *,
    page_size: int,
    dcp_size: int,
    dcp_rank: int,
    src_page_offset: int,
    decode_prefix_len: int,
    num_kv_tokens: int,
) -> Tuple[npt.NDArray[np.int64], ...]:
    """Map physical prefill pages to local/global decode page slots."""
    if not 0 <= dcp_rank < dcp_size:
        raise ValueError(f"Invalid DCP rank {dcp_rank} for size {dcp_size}")
    virtual_page_size = page_size * dcp_size
    if decode_prefix_len % virtual_page_size:
        raise ValueError(
            "Ascend PD DCP requires decode_prefix_len to align to the virtual "
            f"page size ({virtual_page_size}), got {decode_prefix_len}"
        )
    if src_page_offset < 0 or num_kv_tokens < 0:
        raise ValueError(
            "Ascend PD DCP page offset and token count must be nonnegative"
        )

    src_pages = np.asarray(src_page_indices, dtype=np.int64)
    dst_pages = np.asarray(dst_page_indices, dtype=np.int64)
    max_src_pages = (num_kv_tokens + page_size - 1) // page_size
    if src_pages.size > max_src_pages:
        raise ValueError(
            "Ascend PD DCP source page count exceeds the token count: "
            f"pages={src_pages.size}, tokens={num_kv_tokens}, page_size={page_size}"
        )
    if src_pages.size == 0:
        empty = np.empty((0,), dtype=np.int64)
        return empty, empty.copy(), empty.copy(), empty.copy()

    # CP may assign this sender only a contiguous subset of the chunk pages;
    # index_slice.start still carries that subset's suffix-relative offset.
    relative_pages = src_page_offset + np.arange(src_pages.size, dtype=np.int64)
    dst_positions = relative_pages // dcp_size
    if dst_positions[-1] >= dst_pages.size:
        raise ValueError(
            "Ascend PD DCP destination does not contain enough virtual pages: "
            f"required={dst_positions[-1] + 1}, available={dst_pages.size}"
        )
    dst_super_pages = dst_pages[dst_positions]
    owners = (decode_prefix_len // page_size + relative_pages) % dcp_size
    local = owners == dcp_rank

    return (
        src_pages[local],
        dst_super_pages[local],
        src_pages,
        dst_super_pages * dcp_size + owners,
    )


class AscendKVManager(MooncakeKVManager):
    def __init__(
        self,
        args,
        disaggregation_mode,
        server_args,
        is_mla_backend: Optional[bool] = False,
        dcp_remote_decode_layout=None,
    ):
        self._dcp_remote_decode_layout = (
            None if dcp_remote_decode_layout is None else list(dcp_remote_decode_layout)
        )
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)

    def _is_npu_dsa_layout(self) -> bool:
        return getattr(self.kv_args, "kv_buf_groups", 1) == 3

    def _requires_exact_state_index_match(self, st: StateType) -> bool:
        return (
            super()._requires_exact_state_index_match(st)
            or st in _DSV4_KVCACHE_STATE_TYPES
        )

    def init_engine(self):
        # TransferEngine initialized on ascend.
        local_ip = get_local_ip_auto()
        self.engine = AscendTransferEngine(
            hostname=local_ip,
            npu_id=self.kv_args.gpu_id,
            disaggregation_mode=self.disaggregation_mode,
        )

    def register_buffer_to_engine(self):
        # MemFabric aligns registered buffers to 2 MiB. Register everything in
        # one batch so overlapping aligned ranges from small tensors are merged
        # before they are published to the peer.
        ptrs = list(self.kv_args.kv_data_ptrs)
        lens = list(self.kv_args.kv_data_lens)
        ptrs.extend(self.kv_args.aux_data_ptrs)
        lens.extend(self.kv_args.aux_data_lens)
        for component_ptrs, component_lens in zip(
            self.kv_args.state_data_ptrs or [],
            self.kv_args.state_data_lens or [],
        ):
            ptrs.extend(component_ptrs)
            lens.extend(component_lens)
        if ptrs:
            self.engine.batch_register(ptrs, lens)

    def requires_dcp_relayout(self, dst_dcp_size: int, dst_dcp_rank: int) -> bool:
        if self._is_npu_dsa_layout() and self.dcp_size != dst_dcp_size:
            if self.dcp_size == 1 and dst_dcp_size > 1:
                return True
            raise RuntimeError(
                "NPU DSA PD supports prefill DCP 1 -> decode DCP N only, got "
                f"prefill={self.dcp_size}, decode={dst_dcp_size}"
            )
        return super().requires_dcp_relayout(dst_dcp_size, dst_dcp_rank)

    def _get_dcp_remote_decode_layout(self) -> List[bool]:
        layout = self._dcp_remote_decode_layout
        if layout is None or len(layout) != len(self.kv_args.kv_data_ptrs):
            raise RuntimeError(
                "Ascend PD DCP layout does not match its source KV entries"
            )
        return layout

    def prepare_dcp_token_item_lens(
        self, dst_page_item_lens: List[Optional[int]], dst_dcp_size: int
    ) -> List[int]:
        if not self._is_npu_dsa_layout():
            return super().prepare_dcp_token_item_lens(dst_page_item_lens, dst_dcp_size)
        self._get_dcp_remote_decode_layout()
        page_size = self.kv_args.page_size
        token_item_lens = []
        for entry, item_len in enumerate(self.kv_args.kv_item_lens):
            token_item_len, remainder = divmod(item_len, page_size)
            if remainder:
                raise RuntimeError(
                    f"Ascend PD DCP source entry {entry} is not page aligned"
                )
            token_item_lens.append(token_item_len)
        return token_item_lens

    def _init_dcp_pack_buffers_once(
        self, dcp_size: int, *, include_draft: bool = False
    ) -> None:
        # Page-interleaved NPU DCP transfers whole pages directly.
        self._dcp_pack_buffers = []

    def get_mla_kv_ptrs_with_pp(
        self, src_kv_ptrs: List[int], dst_kv_ptrs: List[int], state_type=None
    ) -> Tuple[List[int], List[int], int]:
        mla_ratios = getattr(self.kv_args, "mla_compression_ratios", None)
        if mla_ratios:
            if len(src_kv_ptrs) == len(dst_kv_ptrs):
                return src_kv_ptrs, dst_kv_ptrs, len(src_kv_ptrs)

            start_layer = self.kv_args.prefill_start_layer
            end_layer = self.kv_args.prefill_end_layer
            c4_full = sum(ratio == 4 for ratio in mla_ratios)
            c4_start = sum(ratio == 4 for ratio in mla_ratios[:start_layer])
            c4_end = sum(ratio == 4 for ratio in mla_ratios[:end_layer])
            c128_start = sum(ratio == 128 for ratio in mla_ratios[:start_layer])
            c128_end = sum(ratio == 128 for ratio in mla_ratios[:end_layer])

            if state_type == AscendStateType.DSV4_C128:
                dst = dst_kv_ptrs[c128_start:c128_end]
                return src_kv_ptrs, dst, len(src_kv_ptrs)

            if state_type == AscendStateType.DSV4_C4_STATE:
                # Layout: [attn_state_0..attn_{c4_full-1},
                #          idx_state_0..idx_{c4_full-1}]
                # Two groups, each c4_full entries; slice both by PP stage.
                dst = []
                for offset in (0, c4_full):
                    dst.extend(dst_kv_ptrs[offset + c4_start : offset + c4_end])
                return src_kv_ptrs, dst, len(src_kv_ptrs)

            # NPU main KV layout: [C4 KV, index K, index scale].
            if state_type is None and len(dst_kv_ptrs) == 3 * c4_full:
                dst = []
                for offset in (0, c4_full, 2 * c4_full):
                    dst.extend(dst_kv_ptrs[offset + c4_start : offset + c4_end])
                return src_kv_ptrs, dst, len(src_kv_ptrs)

            # On A5 (CYCLE cache_mode), StateType.SWA only contains SWA KV
            # buffers (C4 compress state is registered separately as
            # DSV4_C4_STATE).  The common _mla_slice_ptrs_for_pp assumes
            # SWA + C4 state are bundled (swa_L + 2*c4_full), so intercept
            # here and slice SWA KV by layer index directly.
            if state_type == StateType.SWA and AscendStateType.DSV4_C4_STATE in (
                self.kv_args.state_types or []
            ):
                dst = list(dst_kv_ptrs[start_layer:end_layer])
                return src_kv_ptrs, dst, len(src_kv_ptrs)

            return super().get_mla_kv_ptrs_with_pp(src_kv_ptrs, dst_kv_ptrs, state_type)

        # src_kv_ptrs: k_data, v_data, index_k_data(optional)
        # dst_kv_ptrs: k_data, v_data, index_k_data(optional)
        # state_type is accepted for parity with the common disaggregation path;
        # the NPU kv_buf_groups slicing below is state-type agnostic.
        kv_buf_groups = getattr(self.kv_args, "kv_buf_groups", 1)
        hidden_kv_layers = getattr(self.kv_args, "hidden_kv_layers", 0)
        draft_kv_layers = getattr(self.kv_args, "draft_kv_layers", 0)
        src_layers = len(src_kv_ptrs) // kv_buf_groups
        dst_layers = len(dst_kv_ptrs) // kv_buf_groups
        if src_layers == dst_layers:
            sliced_dst_kv_ptrs = dst_kv_ptrs
        else:
            sliced_dst_kv_ptrs = []
            start_layer = self.kv_args.prefill_start_layer
            transfer_draft_kv = get_parallel().pp_group.is_last_rank and draft_kv_layers
            if transfer_draft_kv:
                end_layer = start_layer + src_layers - draft_kv_layers
            else:
                end_layer = start_layer + src_layers

            # target kv
            for i in range(kv_buf_groups):
                layer_offset = i * hidden_kv_layers
                sliced_dst_kv_ptrs.extend(
                    dst_kv_ptrs[layer_offset + start_layer : layer_offset + end_layer]
                )
            # draft kv
            if transfer_draft_kv:
                for i in range(kv_buf_groups):
                    layer_offset = (
                        i * draft_kv_layers + kv_buf_groups * hidden_kv_layers
                    )
                    sliced_dst_kv_ptrs.extend(
                        dst_kv_ptrs[layer_offset : layer_offset + draft_kv_layers]
                    )
        layers_current_pp_stage = len(src_kv_ptrs)
        return src_kv_ptrs, sliced_dst_kv_ptrs, layers_current_pp_stage

    def send_kvcache(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
        executor: concurrent.futures.ThreadPoolExecutor,
        dst_layer_ids: Optional[List[int]] = None,
        dst_device_kv_indices: Optional[npt.NDArray[np.int32]] = None,
        dst_kv_item_len: Optional[int] = None,
        dst_attn_tp_size: Optional[int] = None,
    ):
        if dst_device_kv_indices is not None:
            raise NotImplementedError(
                "Ascend PD transfer does not support HiSparse "
                "destination device KV indices"
            )
        self._validate_envelope_kv_layout(
            dst_kv_ptrs, dst_kv_item_len, dst_attn_tp_size
        )
        # Hybrid MLA prefill stages expose PP-local entries, while a PP=1
        # decode peer registers all model layers. Pair only this layout by
        # global layer id; every other Ascend layout keeps the legacy path.
        if self.is_hybrid_mla_backend and self.pp_size > 1:
            return self._send_kvcache_generic(
                mooncake_session_id=mooncake_session_id,
                src_data_ptrs=self.kv_args.kv_data_ptrs,
                dst_data_ptrs=dst_kv_ptrs,
                item_lens=self.kv_args.kv_item_lens,
                prefill_data_indices=prefill_kv_indices,
                dst_data_indices=dst_kv_indices,
                executor=executor,
                src_layer_ids=self.kv_args.kv_layer_ids,
                dst_layer_ids=dst_layer_ids,
            )

        # Group by indices
        prefill_kv_blocks, dst_kv_blocks = group_concurrent_contiguous(
            prefill_kv_indices, dst_kv_indices
        )

        if self.pp_size > 1:
            if self.is_mla_backend:
                src_kv_ptrs, sliced_dst_kv_ptrs, layers_current_pp_stage = (
                    self.get_mla_kv_ptrs_with_pp(self.kv_args.kv_data_ptrs, dst_kv_ptrs)
                )
                layers_params = [
                    (
                        src_kv_ptrs[layer_id],
                        sliced_dst_kv_ptrs[layer_id],
                        self.kv_args.kv_item_lens[layer_id],
                    )
                    for layer_id in range(layers_current_pp_stage)
                ]
            else:
                (
                    src_k_ptrs,
                    src_v_ptrs,
                    dst_k_ptrs,
                    dst_v_ptrs,
                    layers_current_pp_stage,
                ) = self.get_mha_kv_ptrs_with_pp(self.kv_args.kv_data_ptrs, dst_kv_ptrs)

                layers_params = [
                    (
                        src_k_ptrs[layer_id],
                        dst_k_ptrs[layer_id],
                        self.kv_args.kv_item_lens[layer_id],
                    )
                    for layer_id in range(layers_current_pp_stage)
                ] + [
                    (
                        src_v_ptrs[layer_id],
                        dst_v_ptrs[layer_id],
                        self.kv_args.kv_item_lens[layers_current_pp_stage + layer_id],
                    )
                    for layer_id in range(layers_current_pp_stage)
                ]
        else:
            num_layers = len(self.kv_args.kv_data_ptrs)
            layers_params = [
                (
                    self.kv_args.kv_data_ptrs[layer_id],
                    dst_kv_ptrs[layer_id],
                    self.kv_args.kv_item_lens[layer_id],
                )
                for layer_id in range(num_layers)
            ]

        def set_transfer_blocks(
            src_ptr: int, dst_ptr: int, item_len: int
        ) -> List[Tuple[int, int, int]]:
            transfer_blocks = []
            for prefill_index, decode_index in zip(prefill_kv_blocks, dst_kv_blocks):
                src_addr = src_ptr + int(prefill_index[0]) * item_len
                dst_addr = dst_ptr + int(decode_index[0]) * item_len
                length = item_len * len(prefill_index)
                transfer_blocks.append((src_addr, dst_addr, length))
            return transfer_blocks

        # Worker function for processing a single layer
        def process_layer(src_ptr: int, dst_ptr: int, item_len: int) -> int:
            transfer_blocks = set_transfer_blocks(src_ptr, dst_ptr, item_len)
            return self._transfer_data(mooncake_session_id, transfer_blocks)

        # Worker function for processing all layers in a batch
        def process_layers(layers_params: List[Tuple[int, int, int]]) -> int:
            transfer_blocks = []
            for src_ptr, dst_ptr, item_len in layers_params:
                transfer_blocks.extend(set_transfer_blocks(src_ptr, dst_ptr, item_len))
            return self._transfer_data(mooncake_session_id, transfer_blocks)

        if self.enable_custom_mem_pool:
            futures = [
                executor.submit(
                    process_layer,
                    src_ptr,
                    dst_ptr,
                    item_len,
                )
                for (src_ptr, dst_ptr, item_len) in layers_params
            ]
            for future in concurrent.futures.as_completed(futures):
                status = future.result()
                if status != 0:
                    for f in futures:
                        f.cancel()
                    return status
        else:
            # Combining all layers' params in one batch transfer is more efficient
            # compared to using multiple threads
            return process_layers(layers_params)

        return 0

    def send_kvcache_dcp(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
        *,
        dcp_token_item_lens: List[int],
        dst_dcp_size: int,
        dst_dcp_rank: int,
        src_page_offset: int,
        decode_prefix_len: int,
        num_kv_tokens: int,
        executor: concurrent.futures.ThreadPoolExecutor,
        dst_layer_ids: List[int],
        pack_buffer=None,
        dst_kv_item_lens: Optional[List[int]] = None,
        dst_tp_rank: int = 0,
        dst_attn_tp_size: Optional[int] = None,
    ) -> int:
        if not self._is_npu_dsa_layout():
            return super().send_kvcache_dcp(
                mooncake_session_id,
                prefill_kv_indices,
                dst_kv_ptrs,
                dst_kv_indices,
                dcp_token_item_lens=dcp_token_item_lens,
                dst_dcp_size=dst_dcp_size,
                dst_dcp_rank=dst_dcp_rank,
                src_page_offset=src_page_offset,
                decode_prefix_len=decode_prefix_len,
                num_kv_tokens=num_kv_tokens,
                executor=executor,
                dst_layer_ids=dst_layer_ids,
                pack_buffer=pack_buffer,
                dst_kv_item_lens=dst_kv_item_lens,
                dst_tp_rank=dst_tp_rank,
                dst_attn_tp_size=dst_attn_tp_size,
            )

        src_kv_ptrs = self.kv_args.kv_data_ptrs
        _, dst_kv_ptrs, _ = self.get_mla_kv_ptrs_with_pp(src_kv_ptrs, dst_kv_ptrs)
        if dst_kv_item_lens:
            _, dst_kv_item_lens, _ = self.get_mla_kv_ptrs_with_pp(
                self.kv_args.kv_item_lens, dst_kv_item_lens
            )

        layout = self._get_dcp_remote_decode_layout()
        num_entries = len(src_kv_ptrs)
        if not (
            len(dst_kv_ptrs)
            == len(self.kv_args.kv_item_lens)
            == len(dcp_token_item_lens)
            == len(layout)
            == num_entries
        ):
            raise RuntimeError("Ascend PD DCP KV entry metadata is inconsistent")
        if dst_kv_item_lens and len(dst_kv_item_lens) != num_entries:
            raise RuntimeError("Ascend PD DCP destination KV metadata is inconsistent")

        local_src, local_dst, global_src, global_dst = _build_page_interleaved_dcp_plan(
            prefill_kv_indices,
            dst_kv_indices,
            page_size=self.kv_args.page_size,
            dcp_size=dst_dcp_size,
            dcp_rank=dst_dcp_rank,
            src_page_offset=src_page_offset,
            decode_prefix_len=decode_prefix_len,
            num_kv_tokens=num_kv_tokens,
        )

        transfer_blocks: List[Tuple[int, int, int]] = []
        for entry, uses_global_slots in enumerate(layout):
            page_bytes = self.kv_args.kv_item_lens[entry]
            if page_bytes != dcp_token_item_lens[entry] * self.kv_args.page_size:
                raise RuntimeError(
                    f"Ascend PD DCP source geometry differs at entry {entry}"
                )
            if dst_kv_item_lens:
                expected_dst_bytes = page_bytes * (
                    dst_dcp_size if uses_global_slots else 1
                )
                if dst_kv_item_lens[entry] != expected_dst_bytes:
                    raise RuntimeError(
                        "Ascend PD DCP destination geometry differs at entry "
                        f"{entry}: expected={expected_dst_bytes}, "
                        f"actual={dst_kv_item_lens[entry]}"
                    )

            src_pages = global_src if uses_global_slots else local_src
            dst_pages = global_dst if uses_global_slots else local_dst
            src_groups, dst_groups = group_concurrent_contiguous(src_pages, dst_pages)
            for src_group, dst_group in zip(src_groups, dst_groups):
                transfer_blocks.append(
                    (
                        src_kv_ptrs[entry] + int(src_group[0]) * page_bytes,
                        dst_kv_ptrs[entry] + int(dst_group[0]) * page_bytes,
                        len(src_group) * page_bytes,
                    )
                )

        return self._transfer_data(mooncake_session_id, transfer_blocks)

    def _is_generic_kvcache_state_type(self, st) -> bool:
        # DSV4 per-pool components also use the page-indexed send path.
        return (
            super()._is_generic_kvcache_state_type(st)
            or st in _DSV4_KVCACHE_STATE_TYPES
        )


class AscendKVSender(MooncakeKVSender):
    pass


class AscendKVReceiver(MooncakeKVReceiver):
    pass


class AscendKVBootstrapServer(MooncakeKVBootstrapServer):
    pass
