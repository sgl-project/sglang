"""Small DFlash2 selector tree (SGLang v0.5.21).

The primary seven-token path stays intact. Seven one-node rescue branches carry
the selector's rank-2 token at each draft position. If the primary path first
fails at position p and the target's token was rank 2 there, the tree accepts
that rescue token and the target bonus. This targets the measured +0.28 tau
rank-2 recovery without materializing a wide second path.

Greedy only. The experiment intentionally uses the existing FA3 custom-mask
TARGET_VERIFY path and captures a width-15 target graph.
"""
import os
import types
from dataclasses import dataclass

import torch


def _log(message):
    print(f"[coral] {message}", file=os.sys.stderr, flush=True)


@dataclass
class _TargetState:
    length: int
    attn_backend: object
    graph_runner: object


def _tree_width():
    return int(os.environ.get("CORAL_DFLASH_TREE_WIDTH", "15"))


def _secondary_count():
    return int(os.environ.get("CORAL_DFLASH_TREE_SECONDARY", "7"))


def _rescue_enabled():
    return os.environ.get("CORAL_DFLASH_TREE_RESCUE", "1") == "1"


def _mask_mode():
    return os.environ.get("CORAL_DFLASH_TREE_MASK", "tree")


def _force_target_state():
    return os.environ.get("CORAL_DFLASH_TREE_FORCE_STATE", "0") == "1"


def _build_target_state(worker, length, capture_batch_sizes):
    from sglang.srt.model_executor.runner import DecodeCudaGraphRunner
    from sglang.srt.model_executor.graph_shared_output import GraphSharedOutput
    from sglang.srt.runtime_context import get_context

    model_runner = worker.model_runner
    required_logits_rows = length * max(1, int(getattr(model_runner, "max_running_requests", 1)))
    current_shared_output = model_runner.graph_shared_output
    if (
        current_shared_output is None
        or current_shared_output.max_rows < required_logits_rows
    ):
        tree_shared_output = GraphSharedOutput(
            device=torch.device(model_runner.device),
            max_rows=required_logits_rows,
        )
        GraphSharedOutput._process_shared = tree_shared_output
        model_runner.graph_shared_output = tree_shared_output
    backup_init = model_runner.init_new_workspace
    overrides = {
        "speculative_num_steps": length - 1,
        "speculative_num_draft_tokens": length,
    }
    if capture_batch_sizes is not None:
        overrides["cuda_graph_bs_decode"] = capture_batch_sizes
    context = get_context()
    saved = {name: context.config_leaf(name) for name in overrides}
    try:
        context.override("dflash_tree_capture", **overrides)
        attn_backend = model_runner._get_attention_backend(init_new_workspace=True)
        attn_backend.topk = 2
        full_attn_backend = getattr(attn_backend, "full_attn_backend", None)
        if full_attn_backend is not None:
            full_attn_backend.topk = 2
        graph_runner = DecodeCudaGraphRunner(
            model_runner,
            attn_backend=attn_backend,
            speculative_num_steps=length - 1,
            speculative_num_draft_tokens=length,
        )
    finally:
        context.override("dflash_tree_restore", **saved)
        model_runner.init_new_workspace = backup_init
    return _TargetState(length=length, attn_backend=attn_backend, graph_runner=graph_runner)


def _tree_retrieve_arrays(batch_size, device):
    width = _tree_width()
    full = 8
    next_token = torch.full((width,), -1, dtype=torch.int32, device=device)
    next_sibling = torch.full((width,), -1, dtype=torch.int32, device=device)
    for node in range(full - 1):
        next_token[node] = node + 1
    for node in range(1, full):
        next_sibling[node] = full + node - 1
    next_token = next_token.unsqueeze(0).expand(batch_size, -1).contiguous()
    next_sibling = next_sibling.unsqueeze(0).expand(batch_size, -1).contiguous()
    return next_token, next_sibling


def _wrap_sampler(mod, current_worker):
    sampler_cls = getattr(mod, "_SelectorDraftSampler", None)
    if sampler_cls is None or not hasattr(mod, "_selector_lattice"):
        raise RuntimeError("DFlash tree requires _SelectorDraftSampler")

    original_lattice = mod._selector_lattice
    original_init = sampler_cls.__init__
    original_call = sampler_cls.__call__
    last_lattice = {}

    def selector_lattice(draft_model, pred_hidden, anchor_token_ids):
        result = original_lattice(draft_model, pred_hidden, anchor_token_ids)
        last_lattice["value"] = result
        return result

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        max_bs = int(self.out.shape[0]) // (self.block_size - 1)
        self.out = torch.empty(
            (max_bs * (_tree_width() - 1),), dtype=torch.int64, device=self.out.device
        )

    def call(self, hidden_states, input_ids):
        result = original_call(self, hidden_states, input_ids)
        candidate_ids, scores = last_lattice.pop("value")
        batch_size, gamma, top_k = candidate_ids.shape
        primary = self.out[: batch_size * gamma].view(batch_size, gamma)
        primary_index = (candidate_ids == primary[..., None]).long().argmax(dim=-1)

        if _secondary_count() > 0:
            secondary = []
            for position in range(gamma):
                if position == 0:
                    rows = scores[:, 0, 0].float()
                else:
                    rows = scores[:, position].float()[
                        torch.arange(batch_size, device=scores.device),
                        primary_index[:, position - 1],
                    ]
                second_index = rows.topk(k=min(2, top_k), dim=-1).indices[:, -1]
                secondary.append(
                    candidate_ids[:, position]
                    .gather(1, second_index[:, None])[:, 0]
                    .to(primary.dtype)
                )
            secondary = torch.stack(secondary, dim=1)
            self.out[batch_size * gamma : batch_size * (_tree_width() - 1)].copy_(
                secondary.reshape(-1)
            )

        worker = current_worker.get("worker")
        if worker is not None and current_worker.get("tree_active", False):
            worker.block_size = _tree_width()
        return result

    mod._selector_lattice = selector_lattice
    sampler_cls.__init__ = init
    sampler_cls.__call__ = call


def _tree_allow_mask(batch_size, device):
    width = _tree_width()
    if _mask_mode() == "causal":
        return (
            torch.tril(torch.ones((width, width), dtype=torch.bool, device=device))
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
            .contiguous()
        )
    secondary_count = _secondary_count()
    mask = torch.zeros((batch_size, width, width), dtype=torch.bool, device=device)
    primary_count = width - secondary_count
    for node in range(primary_count):
        mask[:, node, : node + 1] = True
    for branch in range(secondary_count):
        node = primary_count + branch
        mask[:, node, : branch + 1] = True
        mask[:, node, node] = True
    return mask


def _build_custom_mask(batch, positions, worker):
    width = _tree_width()
    batch_size = positions.numel() // width
    device = positions.device
    if worker is None:
        raise RuntimeError("DFlash tree mask requires the active worker")
    block_mask = getattr(worker, "_tree_block_mask", None)
    if block_mask is None or block_mask.shape[0] < batch_size:
        block_mask = _tree_allow_mask(max(batch_size, 1), device)
        worker._tree_block_mask = block_mask

    seq_lens_cpu = batch.seq_lens_cpu
    if seq_lens_cpu is not None:
        seq_lens = [int(value) for value in seq_lens_cpu[:batch_size].tolist()]
    else:
        seq_lens = [int(value) for value in batch.seq_lens[:batch_size].tolist()]

    max_context = int(
        getattr(
            worker.model_runner.server_args,
            "context_length",
            262144,
        )
    )
    max_total = batch_size * width * (max_context + width)
    mask_buffer = getattr(worker, "_tree_mask_buffer", None)
    if mask_buffer is None or mask_buffer.numel() < max_total:
        mask_buffer = torch.full((max_total,), True, dtype=torch.bool, device=device)
        worker._tree_mask_buffer = mask_buffer

    row_indices = torch.arange(width, device=device)
    column_indices = torch.arange(width, device=device)
    base = 0
    for row, seq_len in enumerate(seq_lens):
        segment_length = width * (seq_len + width)
        row_offsets = base + seq_len + row_indices * (seq_len + width)
        extraction_indices = (
            row_offsets[:, None] + column_indices[None, :]
        ).reshape(-1)
        mask_buffer[extraction_indices] = block_mask[row].reshape(-1)
        base += segment_length
    return mask_buffer[:base]


def install(mod, log=_log):
    width = _tree_width()
    full = 8
    if width != 8 + _secondary_count():
        raise ValueError("CORAL_DFLASH_TREE_WIDTH must equal 8 + CORAL_DFLASH_TREE_SECONDARY")

    worker_cls = mod.DFlashWorkerV2
    original_verify_input = mod.DFlashVerifyInput
    original_verify_prepare = original_verify_input.prepare_for_verify
    original_prepare_block = mod._prepare_dflash_draft_block_unchecked
    original_ensure_buffers = worker_cls._ensure_draft_block_buffers
    original_append = worker_cls._append_target_hidden_to_draft_kv_by_loc
    original_init_graphs = worker_cls.init_cuda_graphs
    original_worker_init = worker_cls.__init__
    original_forward = worker_cls.forward_batch_generation
    current_worker = {}

    _wrap_sampler(mod, current_worker)

    def worker_init(self, *args, **kwargs):
        result = original_worker_init(self, *args, **kwargs)
        current_worker["worker"] = self
        current_worker["tree_active"] = False

        runner = self.draft_model_runner
        if not hasattr(runner, "_coral_tree_original_forward"):
            runner._coral_tree_original_forward = runner.forward

            def draft_forward(runner_self, forward_batch, *args, **kwargs):
                if not current_worker.get("tree_active", False):
                    return runner_self._coral_tree_original_forward(
                        forward_batch, *args, **kwargs
                    )
                token_count = forward_batch.input_ids.numel()
                backup_positions = forward_batch.positions
                backup_cache = forward_batch.out_cache_loc
                forward_batch.positions = backup_positions[:token_count]
                forward_batch.out_cache_loc = backup_cache[:token_count]
                try:
                    result = runner_self._coral_tree_original_forward(
                        forward_batch, *args, **kwargs
                    )
                    if current_worker.get("tree_active", False):
                        current_worker["worker"].block_size = _tree_width()
                    return result
                finally:
                    forward_batch.positions = backup_positions
                    forward_batch.out_cache_loc = backup_cache

            runner.forward = types.MethodType(draft_forward, runner)
        return result

    def ensure_buffers(self, batch_size):
        original_ensure_buffers(self, batch_size)
        cap = max(int(batch_size), int(self._draft_block_ids_buf.shape[0]))
        device = self.device
        if self._draft_block_positions_buf.shape[1] != width:
            self._draft_block_positions_buf = torch.empty(
                (cap, width), dtype=torch.int64, device=device
            )
        if self._draft_verify_out_cache_loc_buf.shape[1] != width:
            self._draft_verify_out_cache_loc_buf = torch.empty(
                (cap, width), dtype=torch.int64, device=device
            )
        if self._draft_block_tokens_buf.shape[1] != width:
            self._draft_block_tokens_buf = torch.empty(
                (cap, width), dtype=torch.long, device=device
            )

    def prepare_block(*args, **kwargs):
        block_ids = kwargs["block_ids_out"]
        positions = kwargs["positions_out"]
        cache_locs = kwargs["cache_loc_out"]
        positions_full_width = positions[:, :full].contiguous()
        cache_locs_full_width = cache_locs[:, :full].contiguous()
        kwargs["positions_out"] = positions_full_width
        kwargs["cache_loc_out"] = cache_locs_full_width
        original_prepare_block(*args, **kwargs)
        positions[:, :full].copy_(positions_full_width)
        cache_locs[:, :full].copy_(cache_locs_full_width)

        prefix_lens = kwargs["prefix_lens"]
        batch_size = block_ids.shape[0]
        positions[:, full:].copy_(
            prefix_lens.view(-1, 1)
            + torch.arange(1, width - full + 1, device=positions.device).view(1, -1)
        )
        all_cache_locs = mod.assign_extend_cache_locs_func(
            req_pool_indices=kwargs["req_pool_indices"],
            req_to_token=kwargs["req_to_token"],
            start_offset=prefix_lens,
            end_offset=prefix_lens + width,
            batch_size=batch_size,
            draft_token_num=width,
            device=positions.device,
        )
        cache_locs.copy_(all_cache_locs.view(batch_size, width))

    def make_verify_input(*args, **kwargs):
        worker = current_worker.get("worker")
        if worker is None or not hasattr(worker, "_tree_target_state"):
            return original_verify_input(*args, **kwargs)
        draft_token = kwargs.get("draft_token", args[0] if args else None)
        positions = kwargs.get("positions")
        batch = current_worker.get("batch")
        if draft_token is None or positions is None or batch is None:
            raise RuntimeError("DFlash tree verify missing draft tokens, positions, or batch")
        if draft_token.numel() % width or positions.numel() % width:
            raise RuntimeError("DFlash tree verify tensor width mismatch")
        kwargs["custom_mask"] = _build_custom_mask(batch, positions, worker)
        kwargs["topk"] = 2
        verify_input = original_verify_input(*args, **kwargs)
        verify_input._tree_verify = True
        verify_input._tree_full_block_size = full
        retrieve_next_token, retrieve_next_sibling = _tree_retrieve_arrays(
            draft_token.numel() // width, draft_token.device
        )
        verify_input.retrieve_next_token = retrieve_next_token
        verify_input.retrieve_next_sibling = retrieve_next_sibling
        state = worker._tree_target_state
        if state is None:
            return verify_input
        worker._tree_original_attn_backend = worker.model_runner.attn_backend
        worker._tree_original_graph_runner = worker.model_runner.decode_cuda_graph_runner
        worker.model_runner.attn_backend = state.attn_backend
        worker.model_runner.decode_cuda_graph_runner = state.graph_runner
        return verify_input

    def verify_prepare(self, batch, target_worker):
        if not getattr(self, "_tree_verify", False):
            return original_verify_prepare(self, batch, target_worker)
        batch_size = self.draft_token.numel() // width
        if batch.seq_lens_cpu is not None:
            batch.seq_lens_cpu = batch.seq_lens_cpu + (width - full)
            batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
        elif self.live_seq_lens_cpu is not None:
            self.live_seq_lens_cpu = self.live_seq_lens_cpu + (width - full)
        result = original_verify_prepare(self, batch, target_worker)
        worker = current_worker.get("worker")
        if worker is not None and not result[1] and not getattr(worker, "_tree_warned_no_graph", False):
            worker._tree_warned_no_graph = True
            log("DFlash tree width-15 verify cannot use CUDA graph; expect eager-target slowdown")
        return result

    def accept_block(self, *, candidates, next_token_logits, sampling_info, draft_input, prefix_lens, bs):
        if sampling_info is None or not sampling_info.is_all_greedy:
            raise RuntimeError("DFlash tree requires greedy requests")
        target_predict = torch.argmax(next_token_logits, dim=-1).view(bs, width)
        primary_matches = candidates[:, 1 : full] == target_predict[:, : full - 1]
        primary_correct = primary_matches.to(torch.int32).cumprod(dim=1).sum(dim=1)
        primary_bonus = target_predict.gather(1, primary_correct[:, None])[:, 0]

        if _secondary_count() == 0:
            bonus = primary_bonus.to(torch.int64)
            out_tokens = candidates.new_zeros((bs, width), dtype=torch.int64)
            out_tokens[:, : full - 1] = candidates[:, 1:full]
            out_tokens.scatter_(1, primary_correct.to(torch.int64)[:, None], bonus[:, None])
            commit_lens = primary_correct + 1
            new_seq_lens = prefix_lens + commit_lens.to(prefix_lens.dtype)
            return primary_correct.to(torch.int32), commit_lens.to(torch.int32), bonus, out_tokens, new_seq_lens, target_predict

        secondary = candidates[:, full:]
        secondary_matches = secondary == target_predict[:, : width - full]
        positions = torch.arange(width - full, device=candidates.device).view(1, -1)
        rescue = (
            secondary_matches & (positions == primary_correct[:, None])
            if _rescue_enabled()
            else torch.zeros_like(secondary_matches)
        )
        rescue_any = rescue.any(dim=1)
        rescue_position = rescue.to(torch.int32).argmax(dim=1) + 1
        accept_len = torch.where(rescue_any, rescue_position, primary_correct).to(torch.int32)

        rescue_bonus_index = full + rescue_position - 1
        rescue_bonus = target_predict.gather(1, rescue_bonus_index[:, None])[:, 0]
        bonus = torch.where(rescue_any, rescue_bonus, primary_bonus).to(torch.int64)

        out_tokens = candidates.new_zeros((bs, width), dtype=torch.int64)
        out_tokens[:, : full - 1] = candidates[:, 1:full]
        rescue_rows = rescue_any.nonzero(as_tuple=False).squeeze(1)
        if rescue_rows.numel():
            out_tokens[rescue_rows, rescue_position[rescue_rows] - 1] = secondary[
                rescue_rows, rescue_position[rescue_rows] - 1
            ]
        out_tokens.scatter_(1, accept_len.to(torch.int64)[:, None], bonus[:, None])
        commit_lens = accept_len + 1
        new_seq_lens = prefix_lens + commit_lens.to(prefix_lens.dtype)
        return accept_len, commit_lens.to(torch.int32), bonus, out_tokens, new_seq_lens, target_predict

    def append_target_hidden(self, *args, **kwargs):
        result = original_append(self, *args, **kwargs)
        current_worker["worker"].block_size = full
        return result

    def init_graphs(self):
        original_init_graphs(self)
        if int(self.block_size) != full:
            raise ValueError(f"DFlash tree expects a full block of {full}")
        capture_bs = [
            int(x)
            for x in os.environ.get("CORAL_DFLASH_TREE_CAPTURE_BS", "1").split(",")
            if x
        ]
        if width > full or _force_target_state():
            self._tree_target_state = _build_target_state(self, width, capture_bs)
        log(
            f"DFlash tree: width {width}, {width - full} rank-2 rescue branches, "
            "greedy only"
        )

    def forward(self, batch, *args, **kwargs):
        worker = self
        current_worker["batch"] = batch
        current_worker["tree_active"] = True
        worker.block_size = full
        backup_attn = worker.model_runner.attn_backend
        backup_graph = worker.model_runner.decode_cuda_graph_runner
        try:
            return original_forward(worker, batch, *args, **kwargs)
        finally:
            worker.block_size = full
            worker.model_runner.attn_backend = backup_attn
            worker.model_runner.decode_cuda_graph_runner = backup_graph
            current_worker["tree_active"] = False
            current_worker["batch"] = None

    mod._prepare_dflash_draft_block_unchecked = prepare_block
    original_verify_input.prepare_for_verify = verify_prepare
    mod.DFlashVerifyInput = make_verify_input
    worker_cls._ensure_draft_block_buffers = ensure_buffers
    worker_cls._append_target_hidden_to_draft_kv_by_loc = append_target_hidden
    worker_cls._accept_block = accept_block
    worker_cls.init_cuda_graphs = init_graphs
    worker_cls.__init__ = worker_init
    worker_cls.forward_batch_generation = forward
