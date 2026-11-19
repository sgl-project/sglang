"""Bridge AdaFlash's direct lengths to upstream DSpark verification."""

import logging
import math
import os

import torch
from sglang.kernels.ops.speculative.dspark.dspark_accept import AcceptGreedy
from sglang.kernels.ops.speculative.dspark.dspark_verify_window import (
    RaggedVerifyWindow,
)
from sglang.srt.environ import envs
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

logger = logging.getLogger(__name__)


class AdaFlashVerify:
    def __init__(self, worker):
        self.model = worker.draft_model
        self.block_size = int(worker.block_size)
        self.model_runner = worker.model_runner
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

    @torch.no_grad()
    def layout(self, hidden: torch.Tensor) -> RaggedVerifyLayout:
        ratios = self.model.thresh_head_two_model(hidden).squeeze(-1)
        length_scale = float(os.environ.get("ADAFLASH_LENGTH_SCALE", "1.2"))
        if not math.isfinite(length_scale) or length_scale <= 0:
            raise ValueError(
                f"ADAFLASH_LENGTH_SCALE must be positive and finite, got {length_scale}"
            )
        lengths = torch.floor(ratios * self.block_size * length_scale).clamp(
            1, self.block_size
        )
        lengths_cpu = lengths.to(torch.int32).cpu().tolist()
        # Reuse the DSpark runner's token buckets when graph capture is enabled.
        total = sum(lengths_cpu)
        model_runner = getattr(self, "model_runner", None)
        graph_runner = (
            model_runner.decode_cuda_graph_runner if model_runner is not None else None
        )
        grid = (
            graph_runner.capture_num_tokens
            if graph_runner is not None and graph_runner.ragged_verify_mode
            else [total]
        )
        tier = round_up_grid(total, grid)
        # FIA needs every query assigned to a request. Fill the small gap with
        # real draft candidates that follow the predicted prefixes. This cannot
        # change the greedy result because the target verifies every extra token.
        remaining = tier - total
        for row in range(len(lengths_cpu) - 1, -1, -1):
            if not remaining:
                break
            extra = min(remaining, self.block_size - lengths_cpu[row])
            lengths_cpu[row] += extra
            remaining -= extra
        if remaining:
            raise RuntimeError("AdaFlash Graph tier exceeds the live draft windows.")
        total = tier
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=lengths_cpu,
            device=hidden.device,
            grid=grid,
        )
        self.num_steps += 1
        self.num_verify_tokens += total
        self.num_full_width_tokens += len(lengths_cpu) * self.block_size
        self.last_verify_lens = lengths_cpu
        if self.num_steps <= 3:
            logger.info(
                "AdaFlash compact verify: lengths=%s, target tokens=%d/%d",
                lengths_cpu,
                total,
                len(lengths_cpu) * self.block_size,
            )
        return layout

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
            ragged_window = RaggedVerifyWindow(
                positions=positions_2d.flatten().index_select(0, packed_indices),
                verify_cache_loc=verify_window.verify_cache_loc.index_select(
                    0, packed_indices
                ).to(torch.int32),
                verify_ids=draft_tokens.flatten().index_select(0, packed_indices),
                window_index=packed_indices,
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

    @staticmethod
    def accept(*, candidates, logits, layout, target_predict=None):
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
