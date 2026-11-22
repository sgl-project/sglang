from collections.abc import Callable
from typing import TYPE_CHECKING, Optional

import msgspec
import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.disaggregation.utils import pre_alloc_fill_len
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.speculative.dflash_draft_ring import DFlashDraftKVRing
from sglang.srt.speculative.domino_utils import corrector_unary_scores
from sglang.srt.speculative.dspark_components.dspark_draft import resolve_greedy_mask

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch


def _rope_tables(positions: torch.Tensor, head_dim: int, dtype: torch.dtype):
    compute_dtype = torch.promote_types(dtype, torch.float32)
    dims = torch.arange(0, head_dim, 2, device=positions.device, dtype=compute_dtype)
    angles = positions[:, None] * (10000.0 ** (-dims / head_dim))
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def _rotate(x, cos, sin):
    left, right = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-right, left), dim=-1) * sin


def _on_kernels(x: torch.Tensor) -> bool:
    """Whether the rollout's CUDA kernels take these activations: a CPU or a
    wider tensor runs the reference path, on any machine."""
    return x.is_cuda and x.dtype in (torch.bfloat16, torch.float32)


def _rms_norm(x: torch.Tensor, norm: nn.RMSNorm) -> torch.Tensor:
    """``weight * (x * rsqrt(mean(x^2) + eps))`` with FP32 statistics, rounded once."""
    if _on_kernels(x):
        from sglang.kernels.ops.layernorm.rmsnorm_fp32 import rmsnorm_fp32

        return rmsnorm_fp32(x, norm.weight, norm.eps)
    wide = x.to(torch.promote_types(x.dtype, torch.float32))
    wide = wide * torch.rsqrt(wide.square().mean(-1, keepdim=True) + norm.eps)
    return (wide * norm.weight).to(x.dtype)


def _fused_attention(x: torch.Tensor, head_dim: int) -> bool:
    """Whether the one-launch attention kernels serve these activations."""
    return _on_kernels(x) and head_dim >= 2 and head_dim & (head_dim - 1) == 0


def _history_attention(
    qkv: torch.Tensor, history: torch.Tensor, slot: int, heads: int
) -> torch.Tensor:
    """Attend a slot's two streams to the slots up to it, and record the slot.

    The contract is ``corrector_history_attention``'s; this is its reference.
    """
    bs, width = qkv.shape[0], qkv.shape[-1] // 3
    head_dim = width // heads
    if _fused_attention(qkv, head_dim):
        from sglang.kernels.ops.speculative.domino_corrector import (
            corrector_history_attention,
        )

        return corrector_history_attention(qkv, history, slot, heads=heads)
    # Both streams query the same history. Only the token stream contributes keys
    # and values, so a slot's draft hidden state never reaches later slots.
    history[:, :, slot] = qkv[:, 0, width:].view(bs, heads, 2 * head_dim)
    seen = history[:, :, : slot + 1]
    query = qkv[..., :width].unflatten(-1, (heads, head_dim))
    probs = torch.einsum("bshd,bhjd->bshj", query, seen[..., :head_dim]).softmax(-1)
    return torch.einsum("bshj,bhjd->bshd", probs, seen[..., head_dim:]).flatten(2)


def _context_attention(
    query: torch.Tensor,
    context_kv: torch.Tensor,
    valid: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    layer: int,
    heads: int,
) -> torch.Tensor:
    """Attend a slot's two streams to the window of projected target context.

    The contract is ``corrector_context_attention``'s; this is its reference.
    """
    head_dim = query.shape[-1] // heads
    if _fused_attention(query, head_dim):
        from sglang.kernels.ops.speculative.domino_corrector import (
            corrector_context_attention,
        )

        return corrector_context_attention(
            query, context_kv, valid, cos, sin, layer=layer, heads=heads
        )
    key, value = (
        context_kv[:, :, layer]
        .to(cos.dtype)
        .unflatten(-1, (2 * heads, head_dim))
        .chunk(2, dim=2)
    )
    key = _rotate(key, cos.repeat(1, 2)[:, None, :], sin.repeat(1, 2)[:, None, :])
    mask = valid.bool()[:, None, None, :]
    scores = torch.einsum(
        "bshd,bwhd->bshw", query.to(cos.dtype).unflatten(-1, (heads, head_dim)), key
    )
    scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    # A request without a window row attends to nothing.
    probs = scores.softmax(dim=-1) * mask.any(dim=-1, keepdim=True)
    return torch.einsum("bshw,bwhd->bshd", probs, value).flatten(2).to(query.dtype)


class CorrectorSampling(msgspec.Struct, frozen=True, kw_only=True):
    """How each request of a block draws its proposals."""

    # [batch], clamped above 0: a greedy row's temperature may be 0 and still divides.
    temperatures: torch.Tensor
    # [batch]; these rows take the argmax.
    greedy_mask: torch.Tensor
    # [batch, K] FP32 scratch, redrawn for every slot.
    noise: torch.Tensor


def _sampling_params(
    *, bs: int, sampling_info, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    if sampling_info is None:
        return (
            torch.ones(bs, dtype=torch.float32, device=device),
            torch.ones(bs, dtype=torch.bool, device=device),
        )
    return (
        sampling_info.temperatures.view(-1)[:bs].float().clamp_min(1e-5),
        resolve_greedy_mask(bs=bs, sampling_info=sampling_info, device=device),
    )


# ``corrector_draw`` holds a request's whole pool in one program.
_FUSED_DRAW_MAX_POOL = 8192


def _draw(
    scores: torch.Tensor, candidate_ids: torch.Tensor, sampling: CorrectorSampling
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw [batch] tokens from q = softmax(scores / T) over the candidates, and
    return them with q, [batch, K].

    A greedy row takes its argmax and carries the point mass there, so one
    captured graph serves greedy and sampled rows. The contract is
    ``corrector_draw``'s; this is its reference.
    """
    noise = sampling.noise.exponential_()
    if (
        scores.is_cuda
        and scores.dtype == torch.float32
        and scores.shape[-1] <= _FUSED_DRAW_MAX_POOL
    ):
        from sglang.kernels.ops.speculative.domino_corrector import corrector_draw

        return corrector_draw(
            scores,
            noise,
            sampling.temperatures,
            sampling.greedy_mask.view(torch.uint8),
            candidate_ids,
        )
    q = (scores.float() / sampling.temperatures[:, None]).softmax(dim=-1)
    # Exponential race: argmax(q / E), E ~ Exp(1), is an exact draw from q.
    noise = noise.clamp_min(torch.finfo(torch.float32).tiny)
    greedy = sampling.greedy_mask[:, None]
    greedy_choice = scores.argmax(dim=-1, keepdim=True)
    choice = torch.where(
        greedy, greedy_choice, (q / noise).argmax(dim=-1, keepdim=True)
    )
    point_mass = torch.zeros_like(q).scatter_(1, greedy_choice, 1.0)
    return candidate_ids.gather(1, choice).squeeze(1), torch.where(
        greedy, point_mass, q
    )


class _CorrectorBlock(nn.Module):
    """One block's parameters; ``_RolloutPlan`` lays them out for the rollout."""

    def __init__(self, hidden_size: int, width: int, heads: int, eps: float):
        super().__init__()
        self.heads = heads
        self.self_norm = nn.RMSNorm(width, eps)
        self.self_qkv = nn.Linear(width, 3 * width, bias=False)
        self.self_out = nn.Linear(width, width, bias=False)
        self.cross_norm = nn.RMSNorm(width, eps)
        self.cross_q = nn.Linear(width, width, bias=False)
        self.cross_kv = nn.Linear(hidden_size, 2 * width, bias=False)
        self.cross_out = nn.Linear(width, width, bias=False)
        self.mlp_norm = nn.RMSNorm(width, eps)
        self.mlp_gate_up = nn.Linear(width, 8 * width, bias=False)
        self.mlp_down = nn.Linear(4 * width, width, bias=False)


class TransformerCorrector(nn.Module):
    """Two-stream Domino head conditioned on prior picks and committed context."""

    def __init__(self, hidden_size: int, config: dict, eps: float):
        super().__init__()
        width, heads = config["d_model"], config["heads"]
        self.head_dim = width // heads
        self.context_window = config["context_window"]
        self.input_hidden = nn.Linear(hidden_size, width, bias=False)
        self.input_token = nn.Linear(hidden_size, width, bias=False)
        self.blocks = nn.ModuleList(
            _CorrectorBlock(hidden_size, width, heads, eps)
            for _ in range(config["layers"])
        )
        self.output_norm = nn.RMSNorm(width, eps)
        self.mlp_act = SiluAndMul()
        self._rollout_plans: dict = {}

    def project_context(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project committed target context to [..., layers, 2 * d_model]."""
        return torch.stack([block.cross_kv(hidden) for block in self.blocks], dim=-2)

    @torch.no_grad()
    def token_table(self, embedding: Callable, vocab_size: int) -> torch.Tensor:
        """[vocab, width]: what each token id adds to a slot's streams."""
        ids = torch.arange(vocab_size, device=self.input_token.weight.device)
        # Chunked: the embeddings of the whole vocabulary are tens of times the table.
        return torch.cat(
            [self.input_token(embedding(chunk)) for chunk in ids.split(8192)]
        )

    def _rollout_plan(self, num_slots: int) -> "_RolloutPlan":
        weight = self.input_token.weight
        key = (num_slots, weight.device, weight.dtype)
        if key not in self._rollout_plans:
            self._rollout_plans[key] = _RolloutPlan.build(self, num_slots)
        return self._rollout_plans[key]

    @torch.no_grad()
    def propose(
        self,
        *,
        hidden: torch.Tensor,
        anchors: torch.Tensor,
        context_kv: torch.Tensor,
        context_mask: torch.Tensor,
        token_inputs: torch.Tensor,
        embed_proj: nn.Sequential,
        unary_picks: torch.Tensor,
        unary_scores: torch.Tensor,
        candidate_ids: Optional[torch.Tensor] = None,
        sampling: Optional[CorrectorSampling] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Propose [batch, slots] tokens. Slot 1 follows the anchor.

        The first ``unary_picks.shape[1]`` slots keep those picks. The others add
        the correction to ``unary_scores``, which span ``candidate_ids`` or the
        vocabulary (see ``corrector_unary_scores``), and take the argmax. With
        ``sampling``, which needs ``candidate_ids``, they draw from the corrected
        scores instead and their q, [batch, corrected slots, K], is returned too.

        ``token_inputs`` is ``token_table``'s [vocab, width] table.

        A block is a few hundred small kernels, so the rollout is laid out for few
        launches (see ``_RolloutPlan``). In the blocks, as in training, products
        round in the activation dtype and norms accumulate in FP32. The output head
        runs in FP32: its sums are the scores the picks are compared on.
        """
        bs, num_slots, hidden_size = hidden.shape
        num_unary_slots = unary_picks.shape[1]
        plan = self._rollout_plan(num_slots)
        wide, width, head_dim = plan.wide, plan.width, self.head_dim
        heads = width // head_dim

        # The correction's first layer, split into its draft hidden and state inputs.
        head_weight = embed_proj[0].weight.to(wide)
        hidden_part = F.linear(hidden.to(wide), head_weight[:, :hidden_size])
        state_t = head_weight[:, hidden_size:].T
        if candidate_ids is not None:
            # [batch, K, emb_dim]: the output rows of the pool, shared by the block.
            candidate_weight = F.embedding(candidate_ids, embed_proj[2].weight).to(wide)
            unary_scores = unary_scores.to(wide)

        context_valid = context_mask.to(torch.uint8)
        # [batch, slots, 2 streams, width]: what each stream adds to the token.
        hidden_query = self.input_hidden(hidden)
        slot_inputs = torch.stack((torch.zeros_like(hidden_query), hidden_query), dim=2)
        # [batch, heads, slots, 2 * head_dim]: a slot's key, then its value.
        history = [
            hidden_query.new_empty((bs, heads, num_slots, 2 * head_dim))
            for _ in plan.blocks
        ]
        previous = anchors
        picks, q_rows = [], []
        for i in range(num_slots):
            streams = token_inputs[previous][:, None, :] + slot_inputs[:, i]
            flat = streams.view(bs * 2, width)
            for j, (block, norms) in enumerate(zip(plan.blocks, self.blocks)):
                inputs = _rms_norm(streams, norms.self_norm)
                attended = _history_attention(
                    torch.matmul(inputs, block.query_key_value_t[i]),
                    history[j],
                    i,
                    heads,
                )
                flat.addmm_(attended.reshape(bs * 2, width), block.self_out_t)

                inputs = _rms_norm(streams, norms.cross_norm)
                attended = _context_attention(
                    torch.matmul(inputs, block.cross_query_t[i]),
                    context_kv,
                    context_valid,
                    plan.context_cos,
                    plan.context_sin,
                    layer=j,
                    heads=heads,
                )
                flat.addmm_(attended.reshape(bs * 2, width), block.cross_out_t)

                inputs = _rms_norm(streams, norms.mlp_norm)
                gate_up = torch.matmul(inputs, block.gate_up_t)
                # The fused op picks its backend by machine, not by tensor.
                hidden_mlp = (
                    self.mlp_act(gate_up)
                    if _on_kernels(gate_up)
                    else self.mlp_act.forward_native(gate_up)
                )
                flat.addmm_(hidden_mlp.view(bs * 2, -1), block.down_t)
            if i < num_unary_slots:
                previous = unary_picks[:, i]
            else:
                state = _rms_norm(streams[:, 1].to(wide), self.output_norm)
                correction = embed_proj[1](
                    torch.addmm(hidden_part[:, i], state, state_t)
                )
                scores = unary_scores[:, i - num_unary_slots]
                if candidate_ids is None:
                    previous = (
                        scores + embed_proj[2](correction.to(scores.dtype))
                    ).argmax(dim=-1)
                else:
                    scores = torch.baddbmm(
                        scores[:, :, None], candidate_weight, correction[:, :, None]
                    )[:, :, 0]
                    if sampling is None:
                        choice = scores.argmax(dim=-1, keepdim=True)
                        previous = candidate_ids.gather(1, choice).squeeze(1)
                    else:
                        previous, q = _draw(scores, candidate_ids, sampling)
                        q_rows.append(q)
            picks.append(previous)
        return torch.stack(picks, dim=1), torch.stack(q_rows, dim=1) if q_rows else None


class _BlockPlan(msgspec.Struct, frozen=True, kw_only=True):
    """One block's weights as the rollout multiplies them, input-major.

    A leading [slots] axis holds a slot's rotation.
    """

    # [slots, width, 3 * width]: the query columns, then each head's key columns
    # and its value columns.
    query_key_value_t: torch.Tensor
    self_out_t: torch.Tensor  # [width, width]
    cross_query_t: torch.Tensor  # [slots, width, width]
    cross_out_t: torch.Tensor  # [width, width]
    gate_up_t: torch.Tensor  # [width, 8 * width]
    down_t: torch.Tensor  # [4 * width, width]


class _RolloutPlan(msgspec.Struct, frozen=True, kw_only=True):
    """A corrector's weights rearranged so a rollout step is few kernel launches.

    A slot's rotation and the attention scale move into the query and key rows,
    and every weight is stored input-major.
    """

    wide: torch.dtype  # what the output head runs in, and the plan is derived in
    width: int
    # [window, head_dim / 2], wide: the rotation of each window row, oldest first,
    # pairing dimension d with d + head_dim / 2.
    context_cos: torch.Tensor
    context_sin: torch.Tensor
    blocks: tuple[_BlockPlan, ...]

    @classmethod
    def build(cls, corrector: "TransformerCorrector", num_slots: int) -> "_RolloutPlan":
        token_weight = corrector.input_token.weight
        dtype = token_weight.dtype
        wide = torch.promote_types(dtype, torch.float32)
        width, head_dim = token_weight.shape[0], corrector.head_dim
        heads = width // head_dim
        positions = torch.arange(1, num_slots + 1, device=token_weight.device)
        cos, sin = _rope_tables(positions, head_dim, wide)
        offsets = torch.arange(-corrector.context_window, 0, device=token_weight.device)
        context_cos, context_sin = _rope_tables(offsets, head_dim, wide)

        def rotated(weight):
            # [slots, width, in]: each head's block of output rows, rotated per slot.
            rows = weight.unflatten(0, (heads, head_dim)).transpose(1, 2)
            rows = _rotate(rows[None], cos[:, None, None, :], sin[:, None, None, :])
            return rows.transpose(2, 3).flatten(1, 2)

        def input_major(weight):
            return weight.mT.to(dtype).contiguous()

        blocks = []
        for block in corrector.blocks:
            query, key, value = block.self_qkv.weight.to(wide).chunk(3, dim=0)
            scale = head_dim**-0.5
            blocks.append(
                _BlockPlan(
                    query_key_value_t=input_major(
                        torch.cat(
                            (
                                rotated(query) * scale,
                                torch.stack(
                                    (
                                        rotated(key).unflatten(1, (heads, head_dim)),
                                        value.unflatten(0, (heads, head_dim)).expand(
                                            num_slots, -1, -1, -1
                                        ),
                                    ),
                                    dim=2,
                                ).flatten(1, 3),
                            ),
                            dim=1,
                        )
                    ),
                    self_out_t=input_major(block.self_out.weight),
                    cross_query_t=input_major(
                        rotated(block.cross_q.weight.to(wide)) * scale
                    ),
                    cross_out_t=input_major(block.cross_out.weight),
                    gate_up_t=input_major(block.mlp_gate_up.weight),
                    down_t=input_major(block.mlp_down.weight),
                )
            )
        return cls(
            wide=wide,
            width=width,
            context_cos=context_cos[:, : head_dim // 2].contiguous(),
            context_sin=context_sin[:, : head_dim // 2].contiguous(),
            blocks=tuple(blocks),
        )


class CorrectorContext:
    """The last window of projected target keys and values for each request."""

    def __init__(self, corrector: TransformerCorrector, *, pool, req_pool_rows: int):
        window = corrector.context_window
        kv_weight = corrector.blocks[0].cross_kv.weight
        self.ring = DFlashDraftKVRing(
            rows_per_req=window,
            req_to_token_rows=req_pool_rows,
            window_left=window - 1,
            dcp_size=1,
            dcp_rank=0,
            pool=pool,
            device=kv_weight.device,
        )
        self.offsets = torch.arange(-window, 0, device=kv_weight.device)
        self.kv = kv_weight.new_zeros(
            (1 + req_pool_rows * window, len(corrector.blocks), kv_weight.shape[0])
        )

    def extend_locs(
        self,
        batch: "ScheduleBatch",
        positions: torch.Tensor,
        extend_lens: torch.Tensor,
        req_generation: torch.Tensor,
    ) -> torch.Tensor:
        self.ring.record_floors(batch.reqs, batch.prefix_lens, req_generation)
        return self.ring.extend_locs(
            req_pool_indices=batch.req_pool_indices,
            positions=positions,
            extend_lens=extend_lens,
            seq_ends=batch.seq_lens,
        )

    def start_windows(self, reqs, req_generation: torch.Tensor) -> None:
        """PD decode: a transferred request arrives without its window, so the
        window starts at the request's first decode position."""
        self.ring.record_floors(
            reqs, [pre_alloc_fill_len(req) for req in reqs], req_generation
        )

    def verify_locs(
        self,
        req_pool_indices: torch.Tensor,
        positions: torch.Tensor,
        commit_lens: torch.Tensor,
    ) -> torch.Tensor:
        steps = torch.arange(positions.shape[1], device=positions.device)
        # A block may be longer than the window. Only its last committed window
        # has unique ring addresses and will be visible to the next proposal.
        end = commit_lens[:, None]
        keep = (steps < end) & (steps >= end - self.ring.span)
        locs = self.ring.locs(req_pool_indices[:, None], positions)
        return torch.where(keep, locs, 0).flatten()

    def write(self, locs: torch.Tensor, context_kv: torch.Tensor) -> None:
        self.kv[locs] = context_kv

    def gather(
        self,
        req_pool_indices: torch.Tensor,
        anchor_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = anchor_positions[:, None] + self.offsets
        valid = positions >= self.ring.min_pos[req_pool_indices, None]
        locs = self.ring.locs(req_pool_indices[:, None], positions)
        context = self.kv[locs].masked_fill_(~valid[:, :, None, None], 0)
        return context, valid


class CorrectorDraftSampler:
    """Corrector rollout over a draft block: eager, or captured in the draft graph."""

    def __init__(
        self,
        *,
        corrector: TransformerCorrector,
        context: CorrectorContext,
        embedding: Callable,
        lm_head: nn.Module,
        embed_proj: nn.Sequential,
        tp_group,
        vocab_size: int,
        block_size: int,
        num_unary_slots: int,
        candidate_pool_size: int,
        sampling_enabled: bool,
        max_bs: int,
    ):
        self.corrector = corrector
        self.context = context
        self.token_inputs = corrector.token_table(embedding, vocab_size)
        self.lm_head = lm_head
        self.embed_proj = embed_proj
        self.tp_group = tp_group
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.num_unary_slots = num_unary_slots
        self.candidate_pool_size = candidate_pool_size
        self.sampling_enabled = sampling_enabled
        self.max_bs = max_bs
        device, slots = lm_head.weight.device, block_size - 1
        # Read by the worker after a replay; the address is baked into the graph.
        self.out = torch.empty((max_bs * slots,), dtype=torch.int64, device=device)
        # The sampled draw's inputs, staged before a replay, and its outputs.
        pool = candidate_pool_size if sampling_enabled else 0
        self.temperatures = torch.ones((max_bs,), dtype=torch.float32, device=device)
        self.greedy_mask = torch.ones((max_bs,), dtype=torch.bool, device=device)
        self.noise = torch.empty((max_bs, pool), dtype=torch.float32, device=device)
        self.candidate_out = torch.empty(
            (max_bs, slots, pool), dtype=torch.int64, device=device
        )
        self.q_out = torch.empty(
            (max_bs, slots, pool), dtype=torch.float32, device=device
        )

    def stage_sampling_params(self, *, bs: int, sampling_info) -> None:
        if not self.sampling_enabled:
            return
        temperatures, greedy_mask = _sampling_params(
            bs=bs, sampling_info=sampling_info, device=self.out.device
        )
        self.temperatures[:bs].copy_(temperatures)
        self.greedy_mask[:bs].copy_(greedy_mask)

    def eager_sampling(self, *, bs: int, sampling_info) -> Optional[CorrectorSampling]:
        """The sampling state of a block that runs outside the draft graph."""
        if not self.sampling_enabled:
            return None
        device = self.out.device
        temperatures, greedy_mask = _sampling_params(
            bs=bs, sampling_info=sampling_info, device=device
        )
        return CorrectorSampling(
            temperatures=temperatures,
            greedy_mask=greedy_mask,
            noise=torch.empty(
                (bs, self.candidate_pool_size), dtype=torch.float32, device=device
            ),
        )

    def propose(
        self,
        *,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        req_pool_indices: torch.Tensor,
        positions: torch.Tensor,
        in_cuda_graph: bool,
        sampling: Optional[CorrectorSampling] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """[batch, block_size - 1] proposals for a draft block given token-major.

        With ``sampling``, also each slot's candidates and their q, both
        [batch, block_size - 1, K], for the accept.
        """
        bs = hidden_states.shape[0] // self.block_size
        hidden = hidden_states.view(bs, self.block_size, -1)[:, 1:]
        # A block's first position is its anchor's.
        context_kv, context_mask = self.context.gather(
            req_pool_indices, positions.view(bs, self.block_size)[:, 0]
        )
        unary_picks, unary_scores, candidate_ids = corrector_unary_scores(
            hidden,
            self.lm_head,
            tp_group=self.tp_group,
            vocab_size=self.vocab_size,
            num_unary_slots=self.num_unary_slots,
            candidate_pool_size=self.candidate_pool_size,
            in_cuda_graph=in_cuda_graph,
        )
        if candidate_ids is None:
            sampling = None
        tokens, q_rows = self.corrector.propose(
            hidden=hidden,
            anchors=input_ids.view(bs, self.block_size)[:, 0],
            context_kv=context_kv,
            context_mask=context_mask,
            token_inputs=self.token_inputs,
            embed_proj=self.embed_proj,
            unary_picks=unary_picks,
            unary_scores=unary_scores,
            candidate_ids=candidate_ids,
            sampling=sampling,
        )
        if q_rows is None:
            return tokens, None, None
        num_unary_slots, pool = unary_picks.shape[1], q_rows.shape[-1]
        # An uncorrected slot is a point mass on a pick that may sit outside the
        # pool: its row repeats the pick, with q = 1 on every copy.
        candidates = torch.cat(
            (
                unary_picks[:, :, None].expand(-1, -1, pool),
                candidate_ids[:, None, :].expand(-1, q_rows.shape[1], -1),
            ),
            dim=1,
        )
        q_rows = torch.cat(
            (q_rows.new_ones((bs, num_unary_slots, pool)), q_rows), dim=1
        )
        return tokens, candidates, q_rows

    def capture_hook(self, runner, out, forward_batch, num_tokens) -> None:
        """Draft graph tail hook. The request slots and positions it reads are the
        graph's own input buffers, so a replay stages only the sampling parameters."""
        del runner, num_tokens
        bs = out.hidden_states.shape[0] // self.block_size
        sampling = None
        if self.sampling_enabled:
            sampling = CorrectorSampling(
                temperatures=self.temperatures[:bs],
                greedy_mask=self.greedy_mask[:bs],
                noise=self.noise[:bs],
            )
        tokens, candidates, q_rows = self.propose(
            hidden_states=out.hidden_states,
            input_ids=forward_batch.input_ids,
            req_pool_indices=forward_batch.req_pool_indices,
            positions=forward_batch.positions,
            in_cuda_graph=True,
            sampling=sampling,
        )
        self.out[: tokens.numel()].copy_(tokens.reshape(-1))
        if q_rows is not None:
            self.candidate_out[:bs].copy_(candidates)
            self.q_out[:bs].copy_(q_rows)
