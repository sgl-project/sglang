import logging
from typing import Dict, List, Optional

import torch

from sglang.srt.utils import get_device_module
from sglang.srt.utils.common import get_num_new_pages

device_module = get_device_module()

logger = logging.getLogger(__name__)

HISPARSE_RESIDENCY_NONE = "none"
HISPARSE_RESIDENCY_STAGING = "staging"
HISPARSE_RESIDENCY_RESIDENT_BACKED = "resident_backed"
HISPARSE_RESIDENCY_SWAP = "swap"


class HiSparseDynamicMixin:
    def _init_dynamic_hisparse_state(self, max_num_reqs: int, device: str) -> None:
        self.active_reqs = {}
        self.req_residency = [HISPARSE_RESIDENCY_NONE] * max_num_reqs
        self.host_backed_len = [0] * max_num_reqs
        self.req_is_swap = torch.zeros(max_num_reqs, dtype=torch.int8, device=device)
        self.num_dynamic_demotions = 0
        self.num_dynamic_demoted_tokens = 0
        self.token_to_kv_pool_allocator.register_hisparse_coordinator(self)

    def _mark_hisparse_staging(self, req) -> None:
        self._set_residency(req, HISPARSE_RESIDENCY_STAGING)

    def _mark_hisparse_resident(self, req, host_backed_len: int) -> None:
        self._set_host_backed_len(req.req_pool_idx, host_backed_len)
        self._set_residency(req, HISPARSE_RESIDENCY_RESIDENT_BACKED)

    def _mark_hisparse_swap(
        self, req, host_backed_len: Optional[int] = None
    ) -> None:
        if host_backed_len is not None:
            self._set_host_backed_len(req.req_pool_idx, host_backed_len)
        self._set_residency(req, HISPARSE_RESIDENCY_SWAP)

    def _finish_hisparse_staging(self, req) -> None:
        req.hisparse_staging = False
        if self.dynamic:
            self._mark_hisparse_resident(req, len(req.fill_ids))
            return
        self.alloc_device_buffer(req)
        self._skip_first_backup[req.req_pool_idx] = True

    def _dynamic_swap_kernel_kwargs(self) -> dict:
        if not self.dynamic:
            return {}
        return {
            "req_to_token": self.req_to_token_pool.req_to_token,
            "full_to_hisparse_device_index_mapping": (
                self.token_to_kv_pool_allocator.full_to_hisparse_device_index_mapping
            ),
            "req_is_swap": self.req_is_swap,
        }

    def cuda_graph_capture_variants(self, forward_mode) -> tuple:
        if self.dynamic and forward_mode.is_decode():
            return ("resident", "swap")
        return (None,)

    def cuda_graph_variant(self, forward_batch) -> Optional[str]:
        if not self.dynamic or not forward_batch.forward_mode.is_decode():
            return None
        return "swap" if self.forward_batch_uses_swap(forward_batch) else "resident"

    def cuda_graph_variant_label(
        self,
        variant_label: Optional[str],
        *,
        forward_batch=None,
        hisparse_variant: Optional[str] = None,
    ) -> Optional[str]:
        if hisparse_variant is None and forward_batch is not None:
            hisparse_variant = self.cuda_graph_variant(forward_batch)
        if hisparse_variant is None:
            return variant_label
        hisparse_label = f"hisparse_{hisparse_variant}"
        return (
            hisparse_label
            if variant_label is None
            else f"{variant_label}_{hisparse_label}"
        )

    def prepare_cuda_graph_forward_batch(
        self,
        forward_batch,
        bs: int,
        hisparse_variant: Optional[str] = None,
    ) -> None:
        forward_batch.hisparse_coordinator = self
        if self.dynamic and hisparse_variant is not None:
            forward_batch.hisparse_use_swap = hisparse_variant == "swap"
        self.num_real_reqs.fill_(bs)

    def set_cuda_graph_replay(self, attn_backend, forward_batch) -> None:
        set_replay = getattr(attn_backend, "set_hisparse_cuda_graph_replay", None)
        if set_replay is not None:
            set_replay(self, self.forward_batch_uses_swap(forward_batch))

    def _set_residency(self, req, residency: str) -> None:
        req_idx = req.req_pool_idx
        self.req_residency[req_idx] = residency
        if self.dynamic:
            if residency == HISPARSE_RESIDENCY_NONE:
                self.active_reqs.pop(req_idx, None)
            else:
                self.active_reqs[req_idx] = req
        self.req_is_swap[req_idx] = 1 if residency == HISPARSE_RESIDENCY_SWAP else 0

    def _set_host_backed_len(self, req_idx: int, length: int) -> None:
        self.host_backed_len[req_idx] = max(self.host_backed_len[req_idx], length)

    def _maybe_map_last_loc_dynamic(
        self,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> bool:
        if not self.dynamic:
            return False

        self._dynamic_backup_previous_token(seq_lens_cpu, req_pool_indices_cpu)
        self._map_last_loc_to_buffer_dynamic(
            seq_lens,
            out_cache_loc,
            req_pool_indices,
            seq_lens_cpu,
            req_pool_indices_cpu,
        )
        return True

    def _maybe_abort_staging_request_dynamic(self, req) -> bool:
        if not self.dynamic:
            return False
        self._dynamic_abort_staging_request(req)
        return True

    def _maybe_request_finished_dynamic(self, req) -> bool:
        if not self.dynamic:
            return False
        self._dynamic_request_finished(req)
        return True

    def batch_uses_swap(self, reqs: List) -> bool:
        if not reqs:
            return False
        if not self.dynamic:
            return True
        return any(
            self.req_residency[req.req_pool_idx] == HISPARSE_RESIDENCY_SWAP
            for req in reqs
        )

    def forward_batch_uses_swap(self, forward_batch) -> bool:
        if not self.dynamic or not forward_batch.forward_mode.is_decode():
            return True
        return forward_batch.hisparse_use_swap

    def should_wait_for_pending_backup_before_forward(self, forward_batch) -> bool:
        if not self._has_pending_backup:
            return False
        if not forward_batch.forward_mode.is_decode():
            return False
        if not self.dynamic:
            return True
        return self.forward_batch_uses_swap(forward_batch)

    def _map_last_loc_to_buffer_dynamic(
        self,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> None:
        resident_positions = self._get_batch_positions_by_residency(
            req_pool_indices_cpu, HISPARSE_RESIDENCY_RESIDENT_BACKED
        )
        if resident_positions:
            self._alloc_resident_decode_locs(
                resident_positions,
                seq_lens,
                seq_lens_cpu,
                out_cache_loc,
                req_pool_indices,
                req_pool_indices_cpu,
            )

        swap_positions = self._get_batch_positions_by_residency(
            req_pool_indices_cpu, HISPARSE_RESIDENCY_SWAP
        )
        if not swap_positions:
            return

        pos_tensor = torch.tensor(swap_positions, dtype=torch.int64, device=self.device)
        reserved_buffer_loc = self._grow_device_buffers(
            seq_lens[pos_tensor],
            req_pool_indices[pos_tensor],
            seq_lens_cpu[swap_positions],
            req_pool_indices_cpu[swap_positions],
        )
        swap_req_pool_indices = req_pool_indices[pos_tensor]
        self.req_device_buffer_token_locs[
            :, swap_req_pool_indices, self.device_buffer_size
        ] = reserved_buffer_loc.to(torch.int32)
        self.mem_pool_device.full_to_hisparse_device_index_mapping[
            out_cache_loc[pos_tensor]
        ] = reserved_buffer_loc

    def _get_batch_positions_by_residency(
        self, req_pool_indices_cpu: torch.Tensor, residency: str
    ) -> List[int]:
        return [
            i
            for i, req_idx in enumerate(req_pool_indices_cpu.tolist())
            if self.req_residency[int(req_idx)] == residency
        ]

    def _alloc_resident_decode_locs(
        self,
        resident_positions: List[int],
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        out_cache_loc: torch.Tensor,
        req_pool_indices: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> None:
        decode_backup_limits = {
            int(req_pool_indices_cpu[i]): max(0, int(seq_lens_cpu[i]) - 1)
            for i in range(len(req_pool_indices_cpu))
        }
        while resident_positions:
            resident_seq_lens_cpu = seq_lens_cpu[resident_positions]
            need_pages = get_num_new_pages(
                seq_lens=resident_seq_lens_cpu,
                page_size=self.mem_pool_device.page_size,
                decode=True,
            )
            need_tokens = need_pages * self.mem_pool_device.page_size
            if (
                need_tokens
                > self.token_to_kv_pool_allocator.hisparse_attn_allocator.available_size()
            ):
                if not self.demote_until_hisparse_available(
                    need_tokens,
                    host_backed_len_overrides=decode_backup_limits,
                ):
                    raise RuntimeError(
                        f"HiSparse dynamic decode allocation failed (need_tokens={need_tokens})"
                    )
                resident_positions = self._get_batch_positions_by_residency(
                    req_pool_indices_cpu, HISPARSE_RESIDENCY_RESIDENT_BACKED
                )
                continue

            pos_tensor = torch.tensor(
                resident_positions, dtype=torch.int64, device=self.device
            )
            resident_req_pool_indices = req_pool_indices[pos_tensor]
            prev_token_pos = seq_lens[pos_tensor] - 2
            prev_logical_locs = self.req_to_token_pool.req_to_token[
                resident_req_pool_indices, prev_token_pos
            ]
            prev_device_locs = self.mem_pool_device._translate_loc_to_hisparse_device(
                prev_logical_locs
            )
            hisparse_indices = (
                self.token_to_kv_pool_allocator.hisparse_attn_allocator.alloc_decode(
                    seq_lens[pos_tensor],
                    resident_seq_lens_cpu,
                    prev_device_locs,
                )
            )
            if hisparse_indices is None:
                if not self.demote_until_hisparse_available(
                    max(need_tokens, 1),
                    host_backed_len_overrides=decode_backup_limits,
                ):
                    raise RuntimeError("HiSparse dynamic decode allocation failed")
                resident_positions = self._get_batch_positions_by_residency(
                    req_pool_indices_cpu, HISPARSE_RESIDENCY_RESIDENT_BACKED
                )
                continue

            self.mem_pool_device.full_to_hisparse_device_index_mapping[
                out_cache_loc[pos_tensor]
            ] = hisparse_indices
            return

    def _device_locs_for_token_positions(
        self, req_indices: List[int], token_positions: List[int]
    ) -> torch.Tensor:
        device_locs = []
        mapping = self.mem_pool_device.full_to_hisparse_device_index_mapping
        for req_idx, token_pos in zip(req_indices, token_positions):
            if self.req_residency[req_idx] == HISPARSE_RESIDENCY_RESIDENT_BACKED:
                logical_loc = self.req_to_token_pool.req_to_token[req_idx, token_pos]
                device_locs.append(mapping[logical_loc])
            else:
                buffer_slot = min(token_pos, self.device_buffer_size)
                device_locs.append(self.req_to_device_buffer[req_idx, buffer_slot])
        return torch.stack(device_locs).to(device=self.device)

    def _backup_token_positions(
        self, req_indices: List[int], token_positions: List[int]
    ) -> None:
        filtered_req_indices = []
        filtered_token_positions = []
        for req_idx, token_pos in zip(req_indices, token_positions):
            if token_pos < 0 or token_pos < self.host_backed_len[req_idx]:
                continue
            filtered_req_indices.append(req_idx)
            filtered_token_positions.append(token_pos)

        if not filtered_req_indices:
            return

        device_locs = self._device_locs_for_token_positions(
            filtered_req_indices, filtered_token_positions
        )
        host_locs = self.mem_pool_host.alloc(len(device_locs))
        if host_locs is None:
            logger.error(
                "HiSparse: host mem pool alloc failed for %d backup tokens",
                len(device_locs),
            )
            raise RuntimeError(
                f"HiSparse host mem pool alloc failed for {len(device_locs)} backup tokens"
            )
        host_locs = host_locs.to(device=self.device)

        backup_req_indices = torch.tensor(
            filtered_req_indices, dtype=torch.int64, device=self.device
        )
        actual_token_pos = torch.tensor(
            filtered_token_positions, dtype=torch.int64, device=self.device
        )
        self.req_to_host_pool[backup_req_indices, actual_token_pos] = host_locs

        schedule_stream = device_module.current_stream()
        with device_module.stream(self.decode_backup_stream):
            self.decode_backup_stream.wait_stream(schedule_stream)
            if self.decode_producer_stream is not None:
                self.decode_backup_stream.wait_stream(self.decode_producer_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device,
                host_locs,
                device_locs,
                io_backend="kernel",
            )
            self._backup_done_event.record()
            if host_locs.is_cuda:
                host_locs.record_stream(self.decode_backup_stream)
            if backup_req_indices.is_cuda:
                backup_req_indices.record_stream(self.decode_backup_stream)
            if actual_token_pos.is_cuda:
                actual_token_pos.record_stream(self.decode_backup_stream)
            if device_locs.is_cuda:
                device_locs.record_stream(self.decode_backup_stream)
        self._has_pending_backup = True

        for req_idx, token_pos in zip(filtered_req_indices, filtered_token_positions):
            self._set_host_backed_len(req_idx, token_pos + 1)

    def _dynamic_backup_previous_token(
        self,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> None:
        req_indices = []
        token_positions = []
        for i in range(len(seq_lens_cpu)):
            req_idx = int(req_pool_indices_cpu[i])
            token_pos = int(seq_lens_cpu[i]) - 2
            if token_pos >= self.host_backed_len[req_idx]:
                req_indices.append(req_idx)
                token_positions.append(token_pos)
        self._backup_token_positions(req_indices, token_positions)

    def ensure_host_backed(self, req, length: int) -> None:
        req_idx = req.req_pool_idx
        start = self.host_backed_len[req_idx]
        if start >= length:
            if self._has_pending_backup:
                self.wait_for_pending_backup()
            return
        self._backup_token_positions(
            [req_idx] * (length - start), list(range(start, length))
        )
        self.wait_for_pending_backup()

    def _device_buffer_alloc_size(self, kv_allocated_len: int) -> int:
        page_size = self.mem_pool_device.page_size
        alloc_size = min(
            ((kv_allocated_len + page_size - 1) // page_size) * page_size,
            self.device_buffer_size,
        )
        if alloc_size == self.device_buffer_size:
            alloc_size = self.padded_buffer_size
        return alloc_size

    def reclaimable_resident_tokens(self) -> int:
        if not self.dynamic:
            return 0

        total = 0
        for req_idx, req in self.active_reqs.items():
            if self.req_residency[req_idx] != HISPARSE_RESIDENCY_RESIDENT_BACKED:
                continue
            total += max(
                0,
                req.kv_allocated_len
                - self._device_buffer_alloc_size(req.kv_allocated_len),
            )
        return total

    def demote_until_hisparse_available(
        self,
        need_tokens: int,
        host_backed_len_overrides: Optional[Dict[int, int]] = None,
    ) -> bool:
        allocator = self.token_to_kv_pool_allocator.hisparse_attn_allocator
        if allocator.available_size() >= need_tokens:
            return True

        host_backed_len_overrides = host_backed_len_overrides or {}
        candidates = [
            req
            for req_idx, req in self.active_reqs.items()
            if self.req_residency[req_idx] == HISPARSE_RESIDENCY_RESIDENT_BACKED
            and req.kv_allocated_len
            > self._device_buffer_alloc_size(req.kv_allocated_len)
        ]
        candidates.sort(key=lambda req: req.kv_allocated_len, reverse=True)
        for req in candidates:
            self.demote_request_to_swap(
                req,
                host_backed_len=host_backed_len_overrides.get(req.req_pool_idx),
            )
            if allocator.available_size() >= need_tokens:
                return True
        return allocator.available_size() >= need_tokens

    def demote_request_to_swap(
        self, req, host_backed_len: Optional[int] = None
    ) -> int:
        req_idx = req.req_pool_idx
        if self.req_residency[req_idx] != HISPARSE_RESIDENCY_RESIDENT_BACKED:
            return 0

        if host_backed_len is None:
            host_backed_len = req.kv_allocated_len
        self.ensure_host_backed(req, host_backed_len)
        old_buffer_size = int(self.req_device_buffer_size[req_idx])
        before = (
            self.token_to_kv_pool_allocator.hisparse_attn_allocator.available_size()
        )
        self.alloc_device_buffer(req)
        self._set_residency(req, HISPARSE_RESIDENCY_SWAP)
        after = self.token_to_kv_pool_allocator.hisparse_attn_allocator.available_size()
        released_tokens = max(0, after - before)
        self.num_dynamic_demotions += 1
        self.num_dynamic_demoted_tokens += released_tokens
        logger.info(
            "HiSparse dynamic demoted request %s to swap "
            "(kv_len=%d, old_buffer=%d, new_buffer=%d, released_tokens=%d, "
            "total_demotions=%d, total_released_tokens=%d)",
            req.rid,
            req.kv_allocated_len,
            old_buffer_size,
            int(self.req_device_buffer_size[req_idx]),
            released_tokens,
            self.num_dynamic_demotions,
            self.num_dynamic_demoted_tokens,
        )
        return released_tokens

    def _dynamic_abort_staging_request(self, req) -> None:
        self.ack_staging_queue = [
            act for act in self.ack_staging_queue if act.req is not req
        ]
        self.write_staging_stream.synchronize()

        host_indices = self.req_to_host_pool[req.req_pool_idx, : req.kv_allocated_len]
        host_indices = host_indices[host_indices >= 0]
        if host_indices.numel() > 0:
            self.mem_pool_host.free(host_indices)
        self.req_to_host_pool[req.req_pool_idx, :] = -1
        self.host_backed_len[req.req_pool_idx] = 0
        self._set_residency(req, HISPARSE_RESIDENCY_NONE)
        req.hisparse_staging = False

    def _dynamic_request_finished(self, req) -> None:
        if self.decode_producer_stream is not None:
            device_module.current_stream().wait_stream(self.decode_producer_stream)
        if self._has_pending_backup:
            self._backup_done_event.wait(device_module.current_stream())
            self._has_pending_backup = False

        if self.req_residency[req.req_pool_idx] == HISPARSE_RESIDENCY_SWAP:
            current_cap = int(self.req_device_buffer_size[req.req_pool_idx])
            buffer_indices = self.req_to_device_buffer[req.req_pool_idx, :current_cap]
            self.token_to_kv_pool_allocator.free_hisparse_indices(buffer_indices)

            allocated_locs = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : req.kv_allocated_len
            ]
            self.token_to_kv_pool_allocator.full_to_hisparse_device_index_mapping[
                allocated_locs
            ] = 0

        host_indices = self.req_to_host_pool[req.req_pool_idx, : req.kv_allocated_len]
        host_indices = host_indices[host_indices >= 0]
        if host_indices.numel() > 0:
            self.mem_pool_host.free(host_indices)

        self.req_device_buffer_tokens[:, req.req_pool_idx, :] = -1
        self.req_device_buffer_token_locs[:, req.req_pool_idx, :] = -1
        self.req_to_device_buffer[req.req_pool_idx, :] = 0
        self.req_device_buffer_size[req.req_pool_idx] = 0
        self.req_to_host_pool[req.req_pool_idx, :] = -1
        self.lru_slots[:, req.req_pool_idx, :].copy_(self._lru_init)
        self.host_backed_len[req.req_pool_idx] = 0
        self._set_residency(req, HISPARSE_RESIDENCY_NONE)
