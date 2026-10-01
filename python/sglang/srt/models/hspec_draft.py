"""H-Spec (mamba_attn_hybrid) draft model for SGLang speculative decoding.

Ports the speculators ``mamba_attn_hybrid`` drafter (hybrid Mamba-2 /
attention / MLP recipe over Qwen3) onto SGLang primitives, following the
same serving pattern as ``models/dflash.py``:

- no embeddings / no target lm_head: block token embeddings come from the
  target model, and (for full-vocab drafters) sampling reuses the target head
- draft attention sub-layers are ``DFlashAttention`` (RadixAttention over the
  draft KV pool); the context prefix K/V are materialized from the *target*
  paged KV pool at ``attn_kv_layer_ids`` (post-RoPE) by the H-Spec worker,
  which is exactly the semantics the drafter was trained with
- the Mamba mixer uses a reference SSD scan in fp32 (no CUDA mamba_ssm
  dependency), which is also the correct kernel on Ascend NPU
"""

from __future__ import annotations

import logging
import os
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.models.dflash import (
    DFlashAttention,
    DFlashDraftModel,
    DFlashMLP,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader

logger = logging.getLogger(__name__)


class _RMSNormGated(nn.Module):
    """Gated RMSNorm used by the Mamba mixer (fp32 statistics)."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(
        self, x: torch.Tensor, gate: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        if gate is not None:
            x = x * F.silu(gate.float())
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(input_dtype)


def _ssd_reference(x, dt, A, B, C, D, initial_states):
    """Reference Mamba-2 SSD scan (fp32), portable across backends."""
    b, L, h, p = x.shape
    g, n = B.shape[2], B.shape[3]
    rep = h // g
    if g == 1:
        B = B.expand(b, L, h, n)
        C = C.expand(b, L, h, n)
    else:
        B = B.repeat_interleave(rep, dim=2)
        C = C.repeat_interleave(rep, dim=2)
    state = initial_states
    ys = []
    for t in range(L):
        dA = torch.exp(dt[:, t] * A)
        dBx = dt[:, t][..., None, None] * (x[:, t][..., None] * B[:, t][:, :, None, :])
        state = dA[..., None, None] * state + dBx
        y = (state * C[:, t][:, :, None, :]).sum(-1) + D[None, :, None] * x[:, t]
        ys.append(y)
    return torch.stack(ys, dim=1)


class HSpecMambaMixer(nn.Module):
    """Mamba-2 mixer for the hybrid drafter (reference SSD scan).

    The convolution is evaluated as an explicit fp32 sum over kernel taps,
    which avoids Conv1d ACL-layout pitfalls on Ascend NPU while matching the
    CUDA reference numerics.
    """

    def __init__(
        self,
        hidden_size: int,
        d_state: int,
        num_heads: int,
        head_dim: int,
        n_groups: int,
        conv_kernel: int,
        expand: int,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.d_inner = num_heads * head_dim
        self.num_heads, self.head_dim = num_heads, head_dim
        self.n_groups, self.d_state = n_groups, d_state
        self.conv_kernel = conv_kernel
        self.conv_dim = self.d_inner + 2 * n_groups * d_state

        self.in_proj = nn.Linear(
            hidden_size,
            2 * self.d_inner + 2 * n_groups * d_state + num_heads,
            bias=False,
        )
        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            conv_kernel,
            groups=self.conv_dim,
            padding=conv_kernel - 1,
            bias=True,
        )
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, num_heads + 1, dtype=torch.float32))
        )
        self.D = nn.Parameter(torch.ones(num_heads))
        self.dt_bias = nn.Parameter(torch.zeros(num_heads))
        self.norm = _RMSNormGated(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        initial_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, L, _ = hidden_states.shape
        gate, xBC, dt = torch.split(
            self.in_proj(hidden_states),
            [self.d_inner, self.conv_dim, self.num_heads],
            dim=-1,
        )
        _kc = self.conv_kernel
        _dtype = xBC.dtype
        _w = self.conv1d.weight.squeeze(1).float()
        _xpad = F.pad(xBC.float(), (0, 0, _kc - 1, 0))
        _conv = self.conv1d.bias.float() + sum(
            _w[:, k] * _xpad[:, k : k + L, :] for k in range(_kc)
        )
        xBC = F.silu(_conv.to(_dtype))
        x, B, C = torch.split(
            xBC,
            [self.d_inner, self.n_groups * self.d_state, self.n_groups * self.d_state],
            dim=-1,
        )
        A = -torch.exp(self.A_log.float())

        if initial_states is None:
            initial_states = torch.zeros(
                b,
                self.num_heads,
                self.head_dim,
                self.d_state,
                dtype=torch.float32,
                device=hidden_states.device,
            )
        dt_r = F.softplus(dt.float() + self.dt_bias.float())
        y = _ssd_reference(
            x.reshape(b, L, self.num_heads, self.head_dim).float(),
            dt_r,
            A,
            B.reshape(b, L, self.n_groups, self.d_state).float(),
            C.reshape(b, L, self.n_groups, self.d_state).float(),
            self.D.float(),
            initial_states.float(),
        )
        y = y.to(hidden_states.dtype).reshape(b, L, self.d_inner)

        y = self.norm(y, gate)
        return self.out_proj(y)


class HSpecMambaSubLayer(nn.Module):
    """Residual Mamba sub-layer: h + Mixer(norm(h), seed_state)."""

    def __init__(self, config, *, mixer_kwargs: dict) -> None:
        super().__init__()
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.ssm = HSpecMambaMixer(hidden_size=int(config.hidden_size), **mixer_kwargs)
        self.ssm_norm = RMSNorm(int(config.hidden_size), eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        num_reqs: int,
        ssm_state0: Optional[torch.Tensor],
    ) -> torch.Tensor:
        AK, d = hidden_states.shape
        K = AK // num_reqs
        h = hidden_states.reshape(num_reqs, K, d)
        h = h + self.ssm(self.ssm_norm(h), ssm_state0)
        return h.reshape(AK, d)


class HSpecAttentionSubLayer(nn.Module):
    """Residual attention sub-layer over the draft KV pool.

    The attention weights/qk-norm/rope come from ``DFlashAttention``; the
    context prefix K/V in the draft pool are materialized by the H-Spec
    worker from the target paged KV pool.
    """

    def __init__(self, config, *, layer_id: int, quant_config=None) -> None:
        super().__init__()
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.input_layernorm = RMSNorm(int(config.hidden_size), eps=rms_norm_eps)
        self.self_attn = DFlashAttention(
            config=config, layer_id=layer_id, quant_config=quant_config
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        return residual + hidden_states


class HSpecMLPSubLayer(nn.Module):
    """Residual MLP sub-layer."""

    def __init__(self, config, *, quant_config=None) -> None:
        super().__init__()
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.mlp_norm = RMSNorm(int(config.hidden_size), eps=rms_norm_eps)
        self.mlp = DFlashMLP(config=config, quant_config=quant_config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.mlp(self.mlp_norm(hidden_states))


class HSpecVanillaMarkov(nn.Module):
    """Sequential Markov head over a (possibly reduced) draft vocab.

    ``markov_w1`` embeds target-vocab token ids; ``markov_w2`` maps the
    Markov latent to draft-vocab logits bias, matching the speculators
    checkpoint layout (``markov_head.markov_w1/w2``).
    """

    def __init__(
        self, *, vocab_size: int, draft_vocab_size: int, markov_rank: int
    ) -> None:
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab_size, markov_rank)
        self.markov_w2 = nn.Linear(markov_rank, draft_vocab_size, bias=False)

    def bias(self, prev_target_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w2(self.markov_w1(prev_target_ids))


class MambaAttnHybridDraftModel(DFlashDraftModel):
    """H-Spec hybrid drafter (block_pattern-driven Mamba/attention/MLP)."""

    supports_fused_context_kv = False

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        # Build the hybrid stack directly; the base class' uniform decoder
        # layers do not match a block_pattern recipe.
        nn.Module.__init__(self)
        from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config

        self.config = config
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))
        hidden_size = int(config.hidden_size)
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)

        draft_config = parse_dflash_draft_config(draft_hf_config=config)
        # The fusion latent width is sized by the draft config's own target
        # layer ids (attn_kv/latent ids index the *target* model, whose depth
        # the draft config does not know) — use them directly instead of
        # resolve_target_layer_ids, which validates against a target depth.
        target_layer_ids = list(draft_config.target_layer_ids or [])
        self.num_context_features = int(len(target_layer_ids))
        self.fc = nn.Linear(
            self.num_context_features * hidden_size, hidden_size, bias=False
        )
        self.hidden_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        # speculators mamba_attn_hybrid checkpoints ship no fc_norm
        # (fc_norm=false); _build_seed_states skips per-chunk normalization.
        self.fc_norm = None
        self.block_size = draft_config.resolve_block_size(default=16)
        # Attributes the shared dflash-family worker reads on any drafter.
        # H-Spec has no candidate selector, LiLiCorr head, or Domino projector,
        # and reuses the target lm_head instead of owning one.
        self.candidate_selector: Optional[nn.Module] = None
        self.lilicorr: Optional[nn.Module] = None
        self.is_nemotron_35_draft = False
        self.embed_tokens = None
        self.prefix_gru = None
        self.embed_proj = None
        self.shift_label = False
        self.lm_head = None

        block_pattern = list(getattr(config, "block_pattern", None) or [])
        if not block_pattern:
            raise ValueError(
                "mamba_attn_hybrid draft requires a non-empty block_pattern."
            )
        attn_kv_layer_ids = list(getattr(config, "attn_kv_layer_ids", None) or [])
        num_attn = block_pattern.count("attention")
        if len(attn_kv_layer_ids) != num_attn:
            raise ValueError(
                f"attn_kv_layer_ids has {len(attn_kv_layer_ids)} entries but "
                f"block_pattern has {num_attn} attention sub-layer(s)."
            )
        if int(getattr(config, "num_hidden_layers", num_attn)) != num_attn:
            raise ValueError(
                f"num_hidden_layers ({getattr(config, 'num_hidden_layers')}) must "
                f"equal the number of attention sub-layers ({num_attn})."
            )
        self.block_pattern = block_pattern
        self.attn_kv_layer_ids = attn_kv_layer_ids

        self.mamba_d_state = int(getattr(config, "mamba_d_state", 16))
        self.mamba_num_heads = int(getattr(config, "mamba_num_heads", 64))
        self.mamba_head_dim = int(getattr(config, "mamba_head_dim", 64))
        self.mamba_n_groups = int(getattr(config, "mamba_n_groups", 1))
        self.mamba_conv_kernel = int(getattr(config, "mamba_conv_kernel", 4))
        self.mamba_expand = int(getattr(config, "mamba_expand", 1))
        self.mamba_seed_mode = str(getattr(config, "mamba_seed_mode", "per_layer"))
        mixer_kwargs = dict(
            d_state=self.mamba_d_state,
            num_heads=self.mamba_num_heads,
            head_dim=self.mamba_head_dim,
            n_groups=self.mamba_n_groups,
            conv_kernel=self.mamba_conv_kernel,
            expand=self.mamba_expand,
        )

        layers: list[nn.Module] = []
        self._sub_type: list[str] = []
        self._sub_seed_idx: list[int] = []
        attn_i = 0
        mamba_i = 0
        for flat_idx, token in enumerate(block_pattern):
            if token == "mamba":
                layers.append(HSpecMambaSubLayer(config, mixer_kwargs=mixer_kwargs))
                self._sub_type.append("mamba")
                if self.mamba_seed_mode == "per_layer":
                    self._sub_seed_idx.append(mamba_i)
                elif self.mamba_seed_mode == "shared":
                    self._sub_seed_idx.append(0)
                else:
                    self._sub_seed_idx.append(0 if mamba_i == 0 else -1)
                mamba_i += 1
            elif token == "attention":
                layers.append(
                    HSpecAttentionSubLayer(
                        config, layer_id=attn_i, quant_config=quant_config
                    )
                )
                self._sub_type.append("attention")
                self._sub_seed_idx.append(-1)
                attn_i += 1
            elif token == "mlp":
                layers.append(HSpecMLPSubLayer(config, quant_config=quant_config))
                self._sub_type.append("mlp")
                self._sub_seed_idx.append(-1)
            else:
                raise ValueError(
                    f"Unsupported mamba_attn_hybrid block token {token!r}."
                )
        # Replace the base class' uniform decoder layers with the hybrid stack.
        self.layers = nn.ModuleList(layers)
        self.attn_sublayers = [
            (i, layer.self_attn)
            for i, layer in enumerate(self.layers)
            if self._sub_type[i] == "attention"
        ]

        num_mamba = mamba_i
        seed_out = self.mamba_num_heads * self.mamba_head_dim * self.mamba_d_state
        num_seed_projs = num_mamba if self.mamba_seed_mode == "per_layer" else 1
        self.seed_projs = nn.ModuleList(
            [
                nn.Linear(int(config.hidden_size), seed_out, bias=False)
                for _ in range(num_seed_projs)
            ]
        )

        # Latent seed (fused target hidden of the anchor token) is provided by
        # the worker right before each draft forward.
        self._latent_seed: Optional[torch.Tensor] = None

        # Draft-vocab head + identity mapping. The speculators checkpoint
        # stores a fused gate_up projection and an optional d2t offset array.
        draft_vocab_size = int(
            getattr(config, "draft_vocab_size", 0) or getattr(config, "vocab_size")
        )
        self.draft_vocab_size = draft_vocab_size
        self.draft_lm_head = nn.Linear(
            int(config.hidden_size), draft_vocab_size, bias=False
        )
        if draft_vocab_size != int(getattr(config, "vocab_size")):
            self.draft_id_to_target_offset = nn.Parameter(
                torch.zeros(draft_vocab_size, dtype=torch.long), requires_grad=False
            )
        else:
            self.draft_id_to_target_offset = None

        markov_rank = int(getattr(config, "markov_rank", 0) or 0)
        if markov_rank > 0:
            markov_head_type = str(getattr(config, "markov_head_type", "vanilla"))
            if markov_head_type != "vanilla":
                raise ValueError(
                    "mamba_attn_hybrid serving supports only "
                    f"markov_head_type='vanilla', got {markov_head_type!r}."
                )
            self.markov_head: Optional[HSpecVanillaMarkov] = HSpecVanillaMarkov(
                vocab_size=int(getattr(config, "vocab_size")),
                draft_vocab_size=draft_vocab_size,
                markov_rank=markov_rank,
            )
        else:
            self.markov_head = None

    # ------------------------------------------------------------------
    # Latent seed plumbing (worker -> draft forward).
    # ------------------------------------------------------------------
    def set_block_size(self, block_size: int) -> None:
        # The hybrid stack has no DFLASH convolutions indexed by block_size;
        # only adopt the worker-resolved value.
        self.block_size = int(block_size)

    def prepare_context_hidden_for_kv(
        self, layer, ctx_hidden: torch.Tensor
    ) -> torch.Tensor:
        # Attention sub-layers reuse the target KV pool in place; the context
        # hidden needs no per-layer transformation before KV materialization.
        return ctx_hidden

    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        """Project concatenated target-layer hidden states into draft hidden_size."""
        expected = int(self.fc.in_features)
        if target_hidden.ndim != 2 or int(target_hidden.shape[-1]) != expected:
            raise ValueError(
                "H-Spec target_hidden feature dim mismatch. "
                f"Expected shape [N, {expected}] "
                f"(num_context_features={self.num_context_features}, hidden_size={int(self.config.hidden_size)}), "
                f"but got shape={tuple(target_hidden.shape)}."
            )
        return self.hidden_norm(self.fc(target_hidden))

    def set_latent_seed(self, latent_seed: Optional[torch.Tensor]) -> None:
        self._latent_seed = latent_seed

    def _build_seed_states(self, num_reqs: int) -> list[Optional[torch.Tensor]]:
        num_slots = len(self._sub_seed_idx) and (
            max(idx for idx in self._sub_seed_idx if idx >= 0) + 1
            if any(idx >= 0 for idx in self._sub_seed_idx)
            else 0
        )
        if self._latent_seed is None:
            return [None] * num_slots
        latent = self._latent_seed[:num_reqs]
        if self.fc_norm is not None:
            chunks = latent.chunk(len(self.fc_norm), dim=-1)
            latent = torch.cat(
                [norm(c) for norm, c in zip(self.fc_norm, chunks, strict=True)],
                dim=-1,
            )
        z = self.hidden_norm(self.fc(latent))
        seed_states: list[Optional[torch.Tensor]] = []
        for proj in self.seed_projs:
            seed_states.append(
                proj(z).view(
                    num_reqs,
                    self.mamba_num_heads,
                    self.mamba_head_dim,
                    self.mamba_d_state,
                )
            )
        if self.mamba_seed_mode != "per_layer":
            seed_states = [s.float() for s in seed_states]
        return seed_states

    # ------------------------------------------------------------------
    # Forward.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
        pp_proxy_tensors=None,
    ) -> LogitsProcessorOutput:
        if input_embeds is None:
            if hasattr(self, "forward_embed"):
                input_embeds = self.forward_embed(input_ids)
            else:
                raise ValueError(
                    "HSpecDraftForCausalLM requires input_embeds (target embedding)."
                )
        hidden_states = input_embeds
        num_reqs = hidden_states.shape[0] // int(self.block_size)

        seed_states = self._build_seed_states(num_reqs)
        for i, layer in enumerate(self.layers):
            token = self._sub_type[i]
            if token == "mamba":
                seed_idx = self._sub_seed_idx[i]
                hidden_states = layer(
                    hidden_states,
                    num_reqs,
                    seed_states[seed_idx] if seed_idx >= 0 else None,
                )
            elif token == "attention":
                hidden_states = layer(positions, hidden_states, forward_batch)
            else:
                hidden_states = layer(hidden_states)

        if hidden_states.numel() != 0:
            hidden_states = self.norm(hidden_states)

        return LogitsProcessorOutput(
            next_token_logits=None,
            hidden_states=hidden_states,
        )

    # ------------------------------------------------------------------
    # Draft sampling (draft vocab -> target vocab), used by the worker.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_draft_ids(
        self,
        draft_hidden: torch.Tensor,
        first_prev_target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Greedy-sample the draft block and map ids into the target vocab.

        ``draft_hidden`` is [bs, K - 1, hidden] (the draft positions after the
        anchor); ``first_prev_target_ids`` is [bs] target-vocab ids of the
        token preceding the first drafted position (the previous bonus).
        Returns [bs, K - 1] target-vocab token ids.
        """
        bs, steps, _ = draft_hidden.shape
        prev = first_prev_target_ids.long()
        disable_markov = bool(int(os.environ.get("HSPEC_DISABLE_MARKOV_BIAS", "0")))
        outs = []
        for step in range(steps):
            h = draft_hidden[:, step, :]
            logits = F.linear(h, self.draft_lm_head.weight)
            if self.markov_head is not None and not disable_markov:
                logits = logits + self.markov_head.bias(prev).to(logits.dtype)
            ids = logits.argmax(dim=-1)
            if self.draft_id_to_target_offset is not None:
                ids = ids + self.draft_id_to_target_offset[ids]
            outs.append(ids)
            prev = ids
        return torch.stack(outs, dim=1)

    # ------------------------------------------------------------------
    # Context K/V materialization: gather from the target paged KV pool.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def write_target_pool_kv(
        self,
        *,
        target_token_to_kv_pool,
        draft_token_to_kv_pool,
        cache_loc: torch.Tensor,
        cache_loc_2d: Optional[torch.Tensor] = None,
        commit_lens: Optional[torch.Tensor] = None,
    ) -> None:
        """Copy the target layers' post-RoPE K/V into the draft KV pool.

        ``cache_loc`` indexes both pools identically (both are laid out over
        the shared req_to_token slot mapping).
        """
        loc = cache_loc
        if loc.dtype != torch.int64:
            loc = loc.to(torch.int64)
        for sub_i, (layer_idx, attn) in enumerate(self.attn_sublayers):
            key_cache, value_cache = target_token_to_kv_pool.get_kv_buffer(
                self.attn_kv_layer_ids[sub_i]
            )
            if key_cache.dim() == 4:
                flat_k = key_cache.reshape(-1, *key_cache.shape[2:])
                flat_v = value_cache.reshape(-1, *value_cache.shape[2:])
            else:
                flat_k, flat_v = key_cache, value_cache
            actual_kv_heads = (
                flat_k.shape[1]
                if flat_k.dim() == 3
                else flat_k.shape[-1] // attn.head_dim
            )
            if actual_kv_heads != attn.num_kv_heads:
                raise ValueError(
                    "MAMBA_ATTN_HYBRID: target KV pool head layout "
                    f"({actual_kv_heads} kv heads) does not match the draft "
                    f"attention ({attn.num_kv_heads} kv heads x "
                    f"{attn.head_dim} head_dim); check the GQA config and TP "
                    "support."
                )
            k = flat_k.index_select(0, loc).view(-1, attn.num_kv_heads, attn.head_dim)
            v = flat_v.index_select(0, loc).view(-1, attn.num_kv_heads, attn.head_dim)
            if k.dtype != v.dtype:
                v = v.to(k.dtype)
            if cache_loc_2d is not None and commit_lens is not None:
                draft_token_to_kv_pool.set_kv_buffer_prefix_valid(
                    attn.attn,
                    cache_loc_2d,
                    commit_lens,
                    k,
                    v,
                    attn.attn.k_scale,
                    attn.attn.v_scale,
                )
            else:
                draft_token_to_kv_pool.set_kv_buffer(
                    attn.attn,
                    loc,
                    k,
                    v,
                    attn.attn.k_scale,
                    attn.attn.v_scale,
                )

    # ------------------------------------------------------------------
    # Weights.
    # ------------------------------------------------------------------
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())

        def resolve_param_name(name: str) -> Optional[str]:
            if name in params_dict:
                return name
            if name.startswith("model."):
                stripped = name[len("model.") :]
                if stripped in params_dict:
                    return stripped
            else:
                prefixed = f"model.{name}"
                if prefixed in params_dict:
                    return prefixed
            return None

        for name, loaded_weight in weights:
            if name.startswith("embed_tokens"):
                # The drafter reuses the target embedding.
                continue
            if "verifier_lm_head" in name or "verifier_norm" in name:
                continue
            if "d2t" in name:
                name = name.replace("d2t", "draft_id_to_target_offset")
            if name == "lm_head.weight" and "draft_lm_head.weight" in params_dict:
                name = "draft_lm_head.weight"
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if f".{weight_name}." not in f".{name}.":
                    continue
                mapped = name.replace(weight_name, param_name)
                resolved = resolve_param_name(mapped)
                if resolved is None:
                    break
                param = params_dict[resolved]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                resolved = resolve_param_name(name)
                if resolved is None:
                    # Unknown / auxiliary weights (rotary caches etc.) are ignored.
                    continue
                param = params_dict[resolved]
                if resolved.endswith("fc.weight") and tuple(
                    loaded_weight.shape
                ) != tuple(param.shape):
                    raise ValueError(
                        "H-Spec fc.weight shape mismatch: expected "
                        f"{tuple(param.shape)} (latent_fusion_layer_ids sized), "
                        f"got {tuple(loaded_weight.shape)} for {name!r}."
                    )
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = [MambaAttnHybridDraftModel]
