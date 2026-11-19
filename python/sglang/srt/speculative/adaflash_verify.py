"""Bridge AdaFlash's direct lengths to upstream DSpark verification."""

import logging
import math
import os
from typing import Optional

import numpy as np
import torch
from sglang.kernels.ops.speculative.dflash import (
    compute_dflash_correct_drafts_and_bonus,
)
from sglang.kernels.ops.speculative.dspark.dspark_accept import AcceptGreedy
from sglang.kernels.ops.speculative.dspark.dspark_verify_window import (
    RaggedVerifyWindow,
)
from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.models.adaflash import DFlashAdaFlashDraftModel
from sglang.srt.runtime_context import get_exec, get_parallel
from sglang.srt.speculative.dspark_components.dspark_planner import VerifyWindow
from sglang.srt.speculative.dspark_components.dspark_verify import TargetVerifyExecutor
from sglang.srt.speculative.ragged_verify import (
    RaggedVerifyLayout,
    RaggedVerifyMode,
    read_ragged_verify_mode,
    round_up_grid,
)
from sglang.srt.utils.device import is_pin_memory_available

logger = logging.getLogger(__name__)


def _read_length_scale() -> float:
    length_scale = float(os.environ.get("ADAFLASH_LENGTH_SCALE", "1.2"))
    if not math.isfinite(length_scale) or length_scale <= 0:
        raise ValueError(
            f"ADAFLASH_LENGTH_SCALE must be positive and finite, got {length_scale}"
        )
    return length_scale


def _read_min_dynamic_bs() -> int:
    raw = os.environ.get("ADAFLASH_MIN_DYNAMIC_BS", "12")
    min_bs = int(raw)
    if min_bs < 0:
        raise ValueError(
            f"ADAFLASH_MIN_DYNAMIC_BS must be non-negative, got {min_bs}"
        )
    return min_bs


class AdaFlashVerify:
    def __init__(self, worker):
        self.model = worker.draft_model
        self.block_size = int(worker.block_size)
        self.model_runner = worker.model_runner
        self.device = worker.device
        self.length_scale = _read_length_scale()
        self.min_dynamic_bs = _read_min_dynamic_bs()
        self.executor = TargetVerifyExecutor(
            target_worker=worker.target_worker,
            gamma=self.block_size - 1,
            verify_num_draft_tokens=self.block_size,
            model_runner=worker.model_runner,
            kv_injector=None,
            tp_sync=worker._tp_sync,
        )
        self.num_steps = 0
        self.num_verify_tokens = 0
        self.num_full_width_tokens = 0
        self.last_verify_lens = None
        self.num_materialized_tokens = 0
        self.last_packed_positions: Optional[torch.Tensor] = None

        self._full_width_layouts: dict[tuple[int, str], RaggedVerifyLayout] = {}
        self._buf_cap_bs = 0
        self._buf_slot = 0
        self._meta_cpu_bufs: Optional[list[torch.Tensor]] = None
        self._meta_dev_bufs: Optional[list[torch.Tensor]] = None
        self._rows_cpu_bufs: Optional[list[torch.Tensor]] = None
        self._rows_dev_bufs: Optional[list[torch.Tensor]] = None
        self._grid_np: Optional[np.ndarray] = None
        self._col_np: Optional[np.ndarray] = None
        self._row_ids_buf: Optional[torch.Tensor] = None
        self._dense_predict_buf: Optional[torch.Tensor] = None
        self._target_argmax_buf: Optional[torch.Tensor] = None
        self._target_argmax_captured = False
        self._fused_draft_kv_checked = False
        self._fused_draft_kv_ok = False

        if (
            torch.device(worker.device).type == "npu"
            and get_exec().graph.cuda_graph_config.decode.backend != Backend.DISABLED
        ):
            decode_bs = get_exec().graph.cuda_graph_config.decode.bs
            max_bs = max(decode_bs) if decode_bs else 1
            max_running = getattr(self.model_runner, "max_running_requests", None)
            if max_running is not None:
                max_bs = max(max_bs, int(max_running))
            max_tokens = max_bs * self.block_size
            self._target_argmax_buf = torch.empty(
                (max_tokens,), dtype=torch.int64, device=worker.device
            )
            self._dense_predict_buf = torch.empty(
                (max_tokens,), dtype=torch.int64, device=worker.device
            )
            self.model_runner.capture_tail_hooks.append(self._capture_target_argmax)
            self._ensure_scratch_buffers(max_bs, torch.device(worker.device))

    def _capture_target_argmax(self, runner, out, forward_batch, num_tokens) -> None:
        if (
            runner.model_runner.is_draft_worker
            or not runner.ragged_verify_mode
            or not forward_batch.forward_mode.is_target_verify()
        ):
            return
        if (
            not isinstance(out, LogitsProcessorOutput)
            or out.next_token_logits is None
            or self._target_argmax_buf is None
        ):
            return
        if num_tokens <= self._target_argmax_buf.numel():
            ids = torch.argmax(out.next_token_logits[:num_tokens], dim=-1)
            self._target_argmax_buf[:num_tokens].copy_(ids)
            self._target_argmax_captured = True

    def _ensure_scratch_buffers(self, bs: int, device: torch.device) -> None:
        device = torch.device(device)
        if (
            self._buf_cap_bs >= bs
            and self._meta_dev_bufs is not None
            and self._meta_dev_bufs[0].device.type == device.type
        ):
            return
        cap_bs = max(bs, self._buf_cap_bs * 2 if self._buf_cap_bs > 0 else bs, 32)
        pin_memory = is_pin_memory_available(device)
        meta_len = 2 * cap_bs + 1
        max_tokens = cap_bs * self.block_size
        self._meta_cpu_bufs = [
            torch.empty((meta_len,), dtype=torch.int32, pin_memory=pin_memory)
            for _ in range(2)
        ]
        self._meta_dev_bufs = [
            torch.empty((meta_len,), dtype=torch.int32, device=device)
            for _ in range(2)
        ]
        self._rows_cpu_bufs = [
            torch.empty((max_tokens,), dtype=torch.int64, pin_memory=pin_memory)
            for _ in range(2)
        ]
        self._rows_dev_bufs = [
            torch.empty((max_tokens,), dtype=torch.int64, device=device)
            for _ in range(2)
        ]
        self._grid_np = np.arange(max_tokens, dtype=np.int64).reshape(
            cap_bs, self.block_size
        )
        self._col_np = np.arange(self.block_size, dtype=np.int32)[None, :]
        self._row_ids_buf = torch.arange(cap_bs, dtype=torch.long, device=device)
        if (
            self._dense_predict_buf is None
            or self._dense_predict_buf.numel() < max_tokens
            or self._dense_predict_buf.device.type != device.type
        ):
            self._dense_predict_buf = torch.empty(
                (max_tokens,), dtype=torch.int64, device=device
            )
        self._buf_cap_bs = cap_bs

    def _resolve_grid(self, total: int) -> list[int]:
        model_runner = getattr(self, "model_runner", None)
        graph_runner = (
            model_runner.decode_cuda_graph_runner if model_runner is not None else None
        )
        if graph_runner is not None and graph_runner.ragged_verify_mode:
            return graph_runner.capture_num_tokens
        return [total]

    def is_full_width(self, layout: RaggedVerifyLayout) -> bool:
        return (
            layout.total_verify_tokens is not None
            and layout.total_verify_tokens == layout.bs * self.block_size
            and layout.graph_num_tokens == layout.total_verify_tokens
        )

    def full_width_layout(self, bs: int, device: torch.device) -> RaggedVerifyLayout:
        """Return a cached full-width layout without D2H or host-to-device staging."""
        total = bs * self.block_size
        grid = self._resolve_grid(total)
        tier = round_up_grid(total, grid)
        cache = getattr(self, "_full_width_layouts", None)
        key = (bs, torch.device(device).type, tier)
        layout = cache.get(key) if cache is not None else None
        if layout is None:
            verify_lens_cpu = [self.block_size] * bs
            verify_lens = torch.full(
                (bs,), self.block_size, dtype=torch.int32, device=device
            )
            qo_indptr = torch.arange(
                0,
                total + 1,
                self.block_size,
                dtype=torch.int32,
                device=device,
            )
            layout = RaggedVerifyLayout(
                verify_lens=verify_lens,
                graph_num_tokens=tier,
                extend_start_loc=qo_indptr[:-1],
                qo_indptr_device=qo_indptr,
                verify_lens_cpu=verify_lens_cpu,
                total_verify_tokens=total,
            )
            if cache is not None:
                cache[key] = layout
        self.num_steps += 1
        self.num_verify_tokens += total
        self.num_full_width_tokens += total
        self.last_verify_lens = layout.verify_lens_cpu
        if self.num_steps <= 3:
            logger.info(
                "AdaFlash compact verify: full-width fallback bs=%d, target tokens=%d/%d",
                bs,
                total,
                total,
            )
        return layout

    def _build_layout_fast(
        self,
        lengths_cpu: list[int],
        tier: int,
        device: torch.device,
    ) -> RaggedVerifyLayout:
        bs = len(lengths_cpu)
        if not hasattr(self, "_buf_cap_bs"):
            return RaggedVerifyLayout.from_verify_lens(
                verify_lens_cpu=lengths_cpu,
                device=device,
                grid=[tier],
            )
        self._ensure_scratch_buffers(bs, device)
        slot = self._buf_slot
        self._buf_slot = slot ^ 1
        meta_cpu = self._meta_cpu_bufs[slot]
        meta_dev = self._meta_dev_bufs[slot]
        meta_np = meta_cpu.numpy()
        lengths_arr = np.asarray(lengths_cpu, dtype=np.int32)
        meta_np[:bs] = lengths_arr
        meta_np[bs] = 0
        np.cumsum(lengths_arr, out=meta_np[bs + 1 : 2 * bs + 1])
        count = 2 * bs + 1
        meta_dev[:count].copy_(meta_cpu[:count], non_blocking=True)
        verify_lens = meta_dev[:bs]
        qo_indptr = meta_dev[bs:count]
        return RaggedVerifyLayout(
            verify_lens=verify_lens,
            graph_num_tokens=tier,
            extend_start_loc=qo_indptr[:-1],
            qo_indptr_device=qo_indptr,
            verify_lens_cpu=lengths_cpu,
            total_verify_tokens=tier,
        )

    @torch.no_grad()
    def layout(
        self,
        hidden: Optional[torch.Tensor] = None,
        *,
        precomputed_lens: Optional[torch.Tensor] = None,
        bs: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> RaggedVerifyLayout:
        if bs is None:
            if precomputed_lens is not None:
                bs = int(precomputed_lens.shape[0])
            elif hidden is not None:
                bs = int(hidden.shape[0])
            else:
                raise ValueError("AdaFlashVerify.layout requires hidden or precomputed_lens.")
        if device is None:
            device = (
                precomputed_lens.device
                if precomputed_lens is not None
                else hidden.device
            )

        min_dynamic_bs = getattr(self, "min_dynamic_bs", 0)
        if min_dynamic_bs > 0 and bs <= min_dynamic_bs:
            return self.full_width_layout(bs, device)

        if precomputed_lens is not None:
            lengths_cpu = precomputed_lens.cpu().tolist()
        else:
            ratios = self.model.thresh_head_two_model(hidden).squeeze(-1)
            length_scale = _read_length_scale()
            lengths = torch.floor(
                ratios.float() * (self.block_size * length_scale)
            ).clamp(1, self.block_size)
            lengths_cpu = lengths.to(torch.int32).cpu().tolist()

        # Reuse the DSpark runner's token buckets when graph capture is enabled.
        total = sum(lengths_cpu)
        grid = self._resolve_grid(total)
        tier = round_up_grid(total, grid)
        # FIA needs every query assigned to a request. Fill the small gap with
        # real draft candidates that follow the predicted prefixes. This cannot
        # change the greedy result because the target verifies every extra token.
        remaining = tier - total
        for row in range(bs - 1, -1, -1):
            if not remaining:
                break
            extra = min(remaining, self.block_size - lengths_cpu[row])
            lengths_cpu[row] += extra
            remaining -= extra
        if remaining:
            raise RuntimeError("AdaFlash Graph tier exceeds the live draft windows.")
        total = tier
        if total == bs * self.block_size and hasattr(self, "_full_width_layouts"):
            return self.full_width_layout(bs, device)

        layout = self._build_layout_fast(lengths_cpu, tier, device)
        self.num_steps += 1
        self.num_verify_tokens += total
        self.num_full_width_tokens += bs * self.block_size
        self.last_verify_lens = lengths_cpu
        if self.num_steps <= 3:
            logger.info(
                "AdaFlash compact verify: lengths=%s, target tokens=%d/%d",
                lengths_cpu,
                total,
                bs * self.block_size,
            )
        return layout

    def packed_target_indices(
        self, layout: RaggedVerifyLayout, device: torch.device
    ) -> Optional[torch.Tensor]:
        """Map packed verify rows into the [bs, block_size] window without draft rows."""
        if (
            torch.device(device).type != "npu"
            or layout.verify_lens_cpu is None
            or layout.total_verify_tokens != layout.graph_num_tokens
        ):
            return None
        bs = len(layout.verify_lens_cpu)
        total = layout.total_verify_tokens
        if hasattr(self, "_buf_cap_bs"):
            self._ensure_scratch_buffers(bs, torch.device(device))
            # Use the buffer slot matching the most recent _build_layout_fast call.
            slot = self._buf_slot ^ 1
            rows_cpu = self._rows_cpu_bufs[slot]
            rows_dev = self._rows_dev_bufs[slot]
            lengths_arr = np.asarray(layout.verify_lens_cpu, dtype=np.int32)
            mask = self._col_np < lengths_arr[:, None]
            rows_cpu[:total].numpy()[:] = self._grid_np[:bs][mask]
            rows_dev[:total].copy_(rows_cpu[:total], non_blocking=True)
            return rows_dev[:total]

        target_rows = []
        for row, length in enumerate(layout.verify_lens_cpu):
            start = row * self.block_size
            target_rows.extend(range(start, start + length))
        return torch.tensor(target_rows, dtype=torch.long, device=device)

    def packed_indices(self, layout, device):
        """Map live verification rows and non-anchor draft rows to the block."""
        if (
            torch.device(device).type != "npu"
            or layout.verify_lens_cpu is None
            or layout.total_verify_tokens != layout.graph_num_tokens
        ):
            return None, None

        target_rows = []
        draft_rows = []
        for row, length in enumerate(layout.verify_lens_cpu):
            start = row * self.block_size
            candidates = range(start + 1, start + length)
            target_rows.append(start)
            target_rows.extend(candidates)
            draft_rows.extend(candidates)
        return (
            torch.tensor(target_rows, dtype=torch.long, device=device),
            torch.tensor(draft_rows, dtype=torch.long, device=device),
        )

    def can_use_fused_draft_kv(
        self, draft_model, token_to_kv_pool, swa_cache_loc
    ) -> bool:
        if (
            swa_cache_loc is not None
            or torch.device(self.device).type != "npu"
            or not envs.SGLANG_NPU_ADAFLASH_FUSED_DRAFT_KV.get()
        ):
            return False
        if not self._fused_draft_kv_checked:
            from sglang.srt.hardware_backend.npu.adaflash_kv import (
                can_store_packed_adaflash_kv,
            )

            self._fused_draft_kv_ok = can_store_packed_adaflash_kv(
                draft_model, token_to_kv_pool
            )
            self._fused_draft_kv_checked = True
        return self._fused_draft_kv_ok

    def run(
        self,
        *,
        batch,
        layout,
        block_ids,
        draft_tokens,
        positions_2d,
        cache_loc_2d,
        iteration_ids,
        device,
        packed_indices=None,
        greedy_argmax_only=False,
    ):
        verify_window = VerifyWindow(
            positions_2d=positions_2d,
            verify_cache_loc=cache_loc_2d.reshape(-1),
            verify_cache_loc_2d=cache_loc_2d,
            iteration_ids=iteration_ids,
        )
        ragged_window = None
        if packed_indices is not None:
            # Select from the target window already planned by the worker.
            packed_positions = positions_2d.flatten().index_select(0, packed_indices)
            ragged_window = RaggedVerifyWindow(
                positions=packed_positions,
                verify_cache_loc=verify_window.verify_cache_loc.index_select(
                    0, packed_indices
                ).to(torch.int32),
                verify_ids=draft_tokens.flatten().index_select(0, packed_indices),
                window_index=packed_indices,
            )
            self.last_packed_positions = packed_positions
        elif greedy_argmax_only and self.is_full_width(layout):
            packed_positions = positions_2d.reshape(-1)
            ragged_window = RaggedVerifyWindow(
                positions=packed_positions,
                verify_cache_loc=verify_window.verify_cache_loc.to(torch.int32),
                verify_ids=draft_tokens.reshape(-1),
                window_index=None,
            )
            self.last_packed_positions = packed_positions
        else:
            self.last_packed_positions = None

        precomputed_argmax_buf = (
            self._target_argmax_buf
            if getattr(self, "_target_argmax_captured", False)
            else None
        )
        result, _ = self.executor.run_compact(
            batch=batch,
            layout=layout,
            draft_block_ids=block_ids,
            draft_tokens=draft_tokens[:, 1:],
            bs=batch.batch_size(),
            device=device,
            # The DFlash worker applies its ordinary logits adjustments once.
            sampling_info=None,
            verify_window=verify_window,
            ragged_window=ragged_window,
            greedy_argmax_only=greedy_argmax_only and ragged_window is not None,
            precomputed_argmax_buf=precomputed_argmax_buf,
            dense_predict_buf=getattr(self, "_dense_predict_buf", None),
        )
        actual_tokens = batch.input_ids.numel()
        if actual_tokens != layout.graph_num_tokens:
            raise RuntimeError(
                f"AdaFlash materialized {actual_tokens} rows for "
                f"{layout.graph_num_tokens} graph input tokens."
            )
        self.num_materialized_tokens += actual_tokens
        if self.num_steps <= 3:
            logger.info(
                "AdaFlash target input: materialized=%d, full_width=%d",
                actual_tokens,
                batch.batch_size() * self.block_size,
            )
        return result

    def accept(self, *, candidates, logits, layout, target_predict=None):
        if target_predict is not None:
            bs, width = candidates.shape
            target_predict_2d = target_predict.view(bs, width)
            correct, bonus = compute_dflash_correct_drafts_and_bonus(
                candidates=candidates,
                target_predict=target_predict_2d,
            )
            if self.is_full_width(layout):
                return correct, bonus
            ell = (layout.verify_lens.to(device=correct.device) - 1).to(correct.dtype)
            correct = torch.minimum(correct, ell).clamp_(min=0)
            row_ids_buf = getattr(self, "_row_ids_buf", None)
            if (
                row_ids_buf is not None
                and row_ids_buf.numel() >= bs
                and row_ids_buf.device.type == target_predict_2d.device.type
            ):
                row_ids = row_ids_buf[:bs]
            else:
                row_ids = torch.arange(bs, device=target_predict_2d.device)
            bonus = target_predict_2d[row_ids, correct.to(torch.long)].to(torch.int64)
            return correct, bonus

        correct, bonus, _ = AcceptGreedy.execute(
            candidates=candidates,
            target_logits=logits,
            verify_num_draft_tokens=candidates.shape[1],
            cutoff_verify_lens=layout.verify_lens,
            target_predict=target_predict,
        )
        return correct, bonus


def make_adaflash_verify(worker):
    if not isinstance(worker.draft_model, DFlashAdaFlashDraftModel):
        return None
    mode = read_ragged_verify_mode()
    if mode is RaggedVerifyMode.STATIC:
        return None
    if mode is not RaggedVerifyMode.COMPACT:
        raise ValueError(
            "AdaFlash supports SGLANG_RAGGED_VERIFY_MODE=static or compact."
        )
    if get_parallel().tp_size != 1 or get_parallel().attn_dp_enabled:
        raise ValueError(
            "AdaFlash compact verification currently requires TP1 without DP."
        )
    if get_exec().graph.cuda_graph_config.decode.backend != Backend.DISABLED:
        if torch.device(worker.model_runner.device).type != "npu":
            raise ValueError(
                "AdaFlash graph verification requires the Ascend dense MHA backend."
            )
    architectures = worker.model_runner.model_config.hf_config.architectures
    if architectures != ["Qwen3ForCausalLM"]:
        raise ValueError(
            "AdaFlash compact verification currently supports Qwen3ForCausalLM."
        )
    if envs.SGLANG_SIMULATE_ACC_LEN.get() > 0:
        raise ValueError(
            "AdaFlash compact verification requires real acceptance lengths."
        )
    if worker.selector is not None or worker.lilicorr is not None:
        raise ValueError(
            "AdaFlash direct lengths require the ordinary DFlash draft head."
        )
    return AdaFlashVerify(worker)
