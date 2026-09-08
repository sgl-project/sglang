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
from sglang.srt.disaggregation.utils import build_transfer_entry_pairs
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


class AscendKVManager(MooncakeKVManager):
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
        start_layer = self.kv_args.prefill_start_layer
        kv_buf_groups = getattr(self.kv_args, "kv_buf_groups", 1)
        total_kv_layers = getattr(self.kv_args, "total_kv_layers", 0)
        src_layers = len(src_kv_ptrs) // kv_buf_groups
        # When only speculative-algorithm is enabled for decode
        # the KV has one more layer than prefill.
        # The draft layer needs to be skipped.
        dst_total_layers = (
            min(len(dst_kv_ptrs) // kv_buf_groups, total_kv_layers)
            if total_kv_layers
            else len(dst_kv_ptrs) // kv_buf_groups
        )
        end_layer = start_layer + src_layers
        if src_layers == dst_total_layers:
            sliced_dst_kv_ptrs = dst_kv_ptrs
        else:
            sliced_dst_kv_ptrs = []
            for i in range(kv_buf_groups):
                layer_offset = i * dst_total_layers
                sliced_dst_kv_ptrs.extend(
                    dst_kv_ptrs[layer_offset + start_layer : layer_offset + end_layer]
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
        if self.is_hybrid_mla_backend and self.kv_args.draft_total_kv_head_num > 0:
            return self._send_hybrid_draft_kvcache(
                mooncake_session_id,
                prefill_kv_indices,
                dst_kv_ptrs,
                dst_kv_indices,
                dst_layer_ids,
                dst_attn_tp_size,
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

    def _send_hybrid_draft_kvcache(
        self,
        session_id,
        src_indices,
        dst_ptrs,
        dst_indices,
        dst_layer_ids,
        dst_tp_size,
    ):
        # Existing bootstrap routing maps each decode rank to rank // ratio
        # on prefill. This covers both split heads and replicated GQA heads
        # when decode TP is an integer multiple of prefill TP (e.g. 16 -> 32).
        if (
            dst_tp_size is None
            or dst_tp_size < self.attn_tp_size
            or dst_tp_size % self.attn_tp_size
        ):
            raise ValueError(
                "Ascend hybrid draft KV requires decode attention TP to be an "
                "integer multiple of prefill attention TP."
            )
        heads = self.kv_args.draft_total_kv_head_num
        if heads <= 0 or any(
            max(heads, tp) % min(heads, tp) for tp in (self.attn_tp_size, dst_tp_size)
        ):
            raise ValueError("Unsupported Ascend draft KV head/TP partition.")
        src_heads = max(1, heads // self.attn_tp_size)
        dst_heads = max(1, heads // dst_tp_size)
        src_rank = self.kv_args.engine_rank % self.attn_tp_size
        dst_rank = self.decode_kv_args_table[session_id].dst_tp_rank % dst_tp_size
        src_head_start = src_rank // max(1, self.attn_tp_size // heads) * src_heads
        dst_head_start = dst_rank // max(1, dst_tp_size // heads) * dst_heads
        head_offset = dst_head_start - src_head_start
        if head_offset < 0 or head_offset + dst_heads > src_heads:
            raise ValueError(
                "Decode rank requested draft KV from the wrong prefill rank."
            )

        args = self.kv_args
        if len(src_indices) != len(dst_indices):
            raise ValueError("Prefill/decode draft KV page counts must match.")
        num_target_entries = len(args.kv_data_ptrs) - args.num_draft_kv_entries
        pairs = build_transfer_entry_pairs(
            args.kv_layer_ids,
            dst_layer_ids or [],
            len(args.kv_data_ptrs),
            len(dst_ptrs),
            allow_positional_fallback=False,
        )
        src_blocks, dst_blocks = group_concurrent_contiguous(src_indices, dst_indices)
        page_size = args.page_size
        tokens = np.arange(page_size, dtype=np.int64)
        transfer_blocks = []
        for i, j in pairs:
            src_ptr, dst_ptr = args.kv_data_ptrs[i], dst_ptrs[j]
            src_item_len = args.kv_item_lens[i]
            if i < num_target_entries or src_heads == dst_heads:
                # Target MLA and replicated draft heads can copy whole pages.
                for src, dst in zip(src_blocks, dst_blocks):
                    transfer_blocks.append(
                        (
                            src_ptr + int(src[0]) * src_item_len,
                            dst_ptr + int(dst[0]) * src_item_len,
                            len(src) * src_item_len,
                        )
                    )
                continue

            if src_item_len % (page_size * src_heads):
                raise ValueError("Draft KV page is not a token-major head partition.")
            head_bytes = src_item_len // page_size // src_heads
            dst_item_len = page_size * dst_heads * head_bytes
            # NPU MHA pools are [page, token, head, dim]. A head slice is
            # contiguous within each token, not across adjacent tokens.
            src_addrs = (
                (
                    src_ptr
                    + np.asarray(src_indices, dtype=np.int64)[:, None] * src_item_len
                    + tokens * (src_heads * head_bytes)
                    + head_offset * head_bytes
                )
                .reshape(-1)
                .tolist()
            )
            dst_addrs = (
                (
                    dst_ptr
                    + np.asarray(dst_indices, dtype=np.int64)[:, None] * dst_item_len
                    + tokens * (dst_heads * head_bytes)
                )
                .reshape(-1)
                .tolist()
            )
            transfer_blocks.extend(
                (src, dst, dst_heads * head_bytes)
                for src, dst in zip(src_addrs, dst_addrs)
            )
        return self._transfer_data(session_id, transfer_blocks)

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
