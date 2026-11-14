from collections.abc import Iterable

import torch
from torch import nn

from sglang.kernels.ops.inkling2.mtp import (
    mix_rmsnorm,
    rnn_gates,
    rnn_inputs,
)
from sglang.srt.distributed.device_communicators.vocab_gather import (
    make_vocab_gather,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import QKVParallelLinear
from sglang.srt.layers.logits_processor import should_apply_lm_head_quant_method
from sglang.srt.mem_cache.memory_pool import KVWriteLoc, layer_window_locs
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.dflash import DFlashDraftModel
from sglang.srt.models.dspark import (
    BaseDSparkMarkovHead,
    DSparkDraftModel,
    VocabShard,
    gather_and_crop_vocab,
    project_through_lm_head,
)
from sglang.srt.runtime_context import get_parallel, get_platform


class Inkling2MTPHead(BaseDSparkMarkovHead):
    def __init__(self, vocab_size, markov_rank, hidden_size, rms_norm_eps):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.markov_rank = markov_rank
        self.markov_w1 = nn.Embedding(vocab_size, markov_rank)
        self.rnn_proj = nn.Linear(2 * markov_rank + 2 * hidden_size, 3 * markov_rank)
        self.hidden_out = nn.Linear(markov_rank, hidden_size)
        self.dflare_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self._fold_weight = None
        self._fold_bias = None
        self._gather_logits = None
        self._get_staged_f = None

    @torch.no_grad()
    def attach_fold(self, weight, multiplier):
        assert weight.ndim == 2 and weight.shape[1] == self.hidden_size
        self._fold_weight = weight.new_empty((weight.shape[0], self.markov_rank))
        self._fold_bias = weight.new_empty((weight.shape[0],))
        self.refresh_fold(weight, multiplier)

    @torch.no_grad()
    def refresh_fold(self, weight, multiplier):
        assert self._fold_weight is not None and self._fold_bias is not None
        assert weight.shape == (self._fold_weight.shape[0], self.hidden_size)
        projection = self.hidden_out.weight.double()
        bias = self.hidden_out.bias.double()
        for start in range(0, weight.shape[0], 4096):
            rows = slice(start, min(start + 4096, weight.shape[0]))
            head = weight[rows].double()
            folded_weight, folded_bias = head @ projection, head @ bias
            if multiplier:
                folded_weight /= multiplier
                folded_bias /= multiplier
            self._fold_weight[rows].copy_(folded_weight)
            self._fold_bias[rows].copy_(folded_bias)

    def get_prev_embeddings(self, token_ids):
        return self.markov_w1(token_ids.long())

    def _rnn_step(self, state, prev_tokens, hidden, f_rnn):
        inputs = rnn_inputs(state, prev_tokens, self.markov_w1.weight, hidden, f_rnn)
        return rnn_gates(self.rnn_proj(inputs), state)

    def init_state(self, hidden_states):
        batch_size = hidden_states.shape[0]
        assert self._get_staged_f is not None
        f_rnn = self._get_staged_f()
        assert f_rnn is not None and f_rnn.shape[0] >= batch_size
        assert f_rnn.shape[1] == self.hidden_size
        return (
            hidden_states.new_zeros(batch_size, self.markov_rank),
            f_rnn[:batch_size].to(hidden_states.dtype),
        )

    def step_bias(self, prev_tokens, *, hidden_states, state):
        state, f_rnn = state
        state, latent = self._rnn_step(state, prev_tokens, hidden_states, f_rnn)
        assert self._fold_weight is not None and self._fold_bias is not None
        assert self._gather_logits is not None
        bias = torch.addmm(
            self._fold_bias, latent.to(self._fold_weight.dtype), self._fold_weight.T
        )
        if self.vocab_shard is None:
            bias = self._gather_logits(bias)
        return bias, (state, f_rnn)

    def step_logits_dtype(self, base_dtype):
        return torch.promote_types(base_dtype, self._fold_weight.dtype)


class Inkling2ForConditionalGenerationMTP(DSparkDraftModel):
    fall_back_to_pt_during_load = False
    allow_patterns_overrides = ["mtp.safetensors"]
    safetensors_index_file = None
    supports_pre_gather_target_hidden_projection = False
    supports_fused_context_kv = False
    folds_static_verify_commit = True

    def __init__(self, config, quant_config=None, prefix=""):
        if quant_config is not None:
            raise ValueError("Inkling2 MTP requires BF16 checkpoint weights")
        if config.q_bias or config.o_bias:
            raise ValueError(
                "Inkling2 MTP does not support attention projection biases"
            )
        DFlashDraftModel.__init__(
            self, config, quant_config=quant_config, prefix=prefix
        )
        hidden, eps = config.hidden_size, config.rms_norm_eps
        self.block_size = config.block_size
        self.gamma = self.block_size - 1
        self.sample_from_anchor = False
        self.num_stages = config.num_hidden_layers
        self.num_context_features = len(config.target_layer_ids)
        self.logits_mup_width_multiplier = None
        self.fc = self.hidden_norm = None
        self.lm_head = None
        self.confidence_head = None
        self.markov_head = Inkling2MTPHead(
            config.padded_vocab_size, config.markov_rank, hidden, eps
        )
        self.markov_head.step_softcap = config.final_logit_softcapping or None
        self.markov_head.step_vocab_limit = config.unpadded_vocab_size
        self.fuse_weight = nn.Parameter(
            torch.zeros(self.num_stages + 1, self.num_context_features),
            requires_grad=False,
        )
        self.mask_embed = nn.Parameter(torch.zeros(hidden), requires_grad=False)
        self._fuse_alpha = None
        self._f_rnn_rows = None
        self._f_rnn_buf = None
        for layer_id, layer in enumerate(self.layers):
            attn = layer.self_attn
            attn.attn.window_anchor_at_start = layer_id in config.local_layer_ids
            attn.attn.num_splits = (
                0
                if get_platform().is_sm90 or layer_id not in config.local_layer_ids
                else 1
            )
            attn.attn.pack_gqa = True
            attn.kv_proj_ctx = QKVParallelLinear(
                hidden_size=hidden,
                head_size=attn.head_dim,
                total_num_heads=0,
                total_num_kv_heads=attn.total_num_kv_heads,
                bias=False,
            )
            attn.k_norm_ctx = RMSNorm(attn.head_dim, eps=eps)
            layer.dflare_norm = RMSNorm(hidden, eps=eps)

    def set_block_size(self, block_size):
        if block_size != self.block_size:
            raise ValueError("Inkling2 MTP requires the checkpoint's block size")

    def attach_shared_modules(self, *, embed_tokens, lm_head):
        if should_apply_lm_head_quant_method(lm_head, lm_head.quant_method):
            raise ValueError("Inkling2 MTP requires an unquantized target LM head")
        self.embed_tokens, self.lm_head = embed_tokens, lm_head
        self.markov_head.attach_fold(lm_head.weight, self.logits_mup_width_multiplier)
        self.markov_head._gather_logits = self._gather_logits
        self.markov_head._get_staged_f = lambda: self._f_rnn_rows
        if lm_head.tp_size > 1:
            group = (
                get_parallel().attn_tp_group
                if lm_head.use_attn_tp_group
                else get_parallel().tp_group
            )
            assert group.world_size == lm_head.tp_size
            assert (
                lm_head.num_embeddings_per_partition * lm_head.tp_size
                == lm_head.num_embeddings_padded
            )
            self.markov_head.vocab_size = int(lm_head.org_vocab_size)
            shard = lm_head.shard_indices
            self.markov_head.vocab_shard = VocabShard(
                start=int(shard.org_vocab_start_index),
                width=int(shard.org_vocab_end_index - shard.org_vocab_start_index),
                gather=make_vocab_gather(
                    group,
                    local_width=int(lm_head.num_embeddings_per_partition),
                    dtype=lm_head.weight.dtype,
                ),
            )

    def _gather_logits(self, logits):
        return gather_and_crop_vocab(logits, self.lm_head)

    def compute_base_logits(self, hidden):
        assert self.lm_head is not None
        if self.logits_mup_width_multiplier:
            hidden = hidden / self.logits_mup_width_multiplier
        logits = project_through_lm_head(hidden, self.lm_head)
        if self.markov_head.vocab_shard is None:
            logits = self._gather_logits(logits)
        return logits, None

    def forward_embed(self, input_ids):
        assert self.embed_tokens is not None
        ids = input_ids.view(-1, self.block_size)
        hidden = (
            self.mask_embed.to(self.embed_tokens.weight.dtype)
            .expand(ids.shape[0], self.block_size, -1)
            .clone()
        )
        hidden[:, 0, :] = self.embed_tokens(ids[:, 0]).to(hidden.dtype)
        return hidden.view(-1, hidden.shape[-1])

    def ensure_propose_staging(self, req_pool_size):
        if self._f_rnn_buf is None:
            self._f_rnn_buf = self.mask_embed.new_zeros(
                req_pool_size + 1, self.config.hidden_size
            )
        elif self._f_rnn_buf.shape[0] != req_pool_size + 1:
            raise ValueError(
                "MTP request pool changed after recurrent state allocation"
            )

    def _mix_norm(self, targets, row, norm):
        assert self._fuse_alpha is not None
        alpha = self._fuse_alpha[row].to(targets.dtype)
        return mix_rmsnorm(targets, alpha, norm.weight, norm.variance_epsilon)

    def compute_f_rnn(self, target_hidden_last):
        targets = target_hidden_last.view(
            target_hidden_last.shape[0],
            self.num_context_features,
            self.config.hidden_size,
        )
        return self._mix_norm(targets, -1, self.markov_head.dflare_norm)

    def store_f_rnn(self, req_pool_indices, target_hidden_rows):
        assert self._f_rnn_buf is not None
        features = self.compute_f_rnn(target_hidden_rows)
        self._f_rnn_buf.index_copy_(
            0, req_pool_indices.view(-1).long(), features.to(self._f_rnn_buf.dtype)
        )

    def store_verify_commit(
        self, *, req_pool_indices, verify_hidden, commit_lens, stride, valid=None
    ):
        bs = commit_lens.shape[0]
        rows = torch.arange(bs, device=verify_hidden.device)
        last_hidden = verify_hidden.view(bs, stride, -1)[rows, commit_lens.long() - 1]
        indices = req_pool_indices[:bs].view(-1).long()
        if valid is not None:
            indices = torch.where(valid, indices, self._f_rnn_buf.shape[0] - 1)
        self.store_f_rnn(indices, last_hidden)

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch, input_embeds=None, **kwargs):
        batch_size, remainder = divmod(input_ids.numel(), self.block_size)
        assert remainder == 0
        assert self._f_rnn_buf is not None
        self._f_rnn_rows = self._f_rnn_buf.index_select(
            0, forward_batch.req_pool_indices.view(-1)[:batch_size].long()
        )
        input_embeds = self.forward_embed(input_ids)
        return DFlashDraftModel.forward(
            self,
            input_ids,
            positions,
            forward_batch,
            input_embeds=input_embeds,
            **kwargs,
        )

    def project_target_hidden(self, target_hidden):
        raise RuntimeError(
            "Inkling2 MTP requires a separate target mixture for every layer"
        )

    def write_target_hidden_kv(
        self,
        *,
        target_hidden,
        pool,
        positions,
        cache_loc,
        cache_loc_2d=None,
        commit_lens=None,
        target_hidden_is_projected=False,
        swa_loc=None,
    ):
        if target_hidden_is_projected:
            raise ValueError("Inkling2 MTP requires unprojected target hidden states")
        targets = target_hidden.view(
            target_hidden.shape[0], self.num_context_features, self.config.hidden_size
        )
        if swa_loc is None and isinstance(pool, SWAKVPool):
            # Static verify capture can supply only full-space locations.
            swa_loc = pool.translate_loc_from_full_to_swa(cache_loc)
        for i, layer in enumerate(self.layers):
            attn = layer.self_attn
            context = self._mix_norm(targets, i, layer.dflare_norm)
            kv, _ = attn.kv_proj_ctx(context.to(attn.kv_proj_ctx.weight.dtype))
            k, v = kv.split([attn.kv_size, attn.kv_size], dim=-1)
            k = attn.k_norm_ctx(k.reshape(-1, attn.head_dim)).view_as(k)
            k = attn.apply_k_rope(positions, k)
            k = k.view(-1, attn.num_kv_heads, attn.head_dim)
            v = v.view(-1, attn.num_kv_heads, attn.head_dim)
            if cache_loc_2d is not None and commit_lens is not None:
                pool.layer_router.layer(attn.attn.layer_id).store_prefix(
                    attn.attn,
                    KVWriteLoc(
                        cache_loc_2d,
                        window_locs=layer_window_locs(
                            pool,
                            attn.attn.layer_id,
                            None if swa_loc is None else swa_loc.view_as(cache_loc_2d),
                        ),
                        physical=True,
                    ),
                    commit_lens,
                    k,
                    v,
                    attn.attn.k_scale,
                    attn.attn.v_scale,
                )
            else:
                pool.layer_router.layer(attn.attn.layer_id).store(
                    attn.attn,
                    KVWriteLoc(
                        cache_loc,
                        window_locs=layer_window_locs(
                            pool, attn.attn.layer_id, swa_loc
                        ),
                        physical=True,
                    ),
                    k,
                    v,
                    attn.attn.k_scale,
                    attn.attn.v_scale,
                )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = {
            name: param
            for name, param in self.named_parameters()
            if not name.startswith(("embed_tokens.", "lm_head."))
        }
        mapping = (
            (".q_proj.", ".qkv_proj.", "q"),
            (".k_proj.", ".qkv_proj.", "k"),
            (".v_proj.", ".qkv_proj.", "v"),
            (".k_proj_ctx.", ".kv_proj_ctx.", "k"),
            (".v_proj_ctx.", ".kv_proj_ctx.", "v"),
            (".gate_proj.", ".gate_up_proj.", 0),
            (".up_proj.", ".gate_up_proj.", 1),
        )
        loaded, shards = set(), {}
        for checkpoint_name, weight in weights:
            name, shard = checkpoint_name, None
            for source, destination, shard_id in mapping:
                if source in name:
                    name, shard = name.replace(source, destination), shard_id
                    break
            if name not in params:
                raise ValueError(
                    f"Unrecognized MTP checkpoint tensor: {checkpoint_name}"
                )
            param = params[name]
            if weight.dtype != torch.bfloat16 or param.dtype != weight.dtype:
                raise ValueError(
                    f"{checkpoint_name}: MTP requires BF16 parameters and checkpoint weights"
                )
            loader = getattr(param, "weight_loader", default_weight_loader)
            if shard is None:
                if name in loaded:
                    raise ValueError(
                        f"Duplicate MTP checkpoint tensor: {checkpoint_name}"
                    )
                loader(param, weight)
            else:
                if name in loaded and name not in shards:
                    raise ValueError(
                        f"MTP checkpoint mixes packed and unpacked tensors: {checkpoint_name}"
                    )
                pieces = shards.setdefault(name, set())
                if shard in pieces:
                    raise ValueError(
                        f"Duplicate MTP checkpoint tensor: {checkpoint_name}"
                    )
                loader(param, weight, shard)
                pieces.add(shard)
            loaded.add(name)
        missing = set(params) - loaded
        for name, pieces in shards.items():
            expected = (
                {"q", "k", "v"}
                if ".qkv_proj." in name
                else {"k", "v"}
                if ".kv_proj_ctx." in name
                else {0, 1}
            )
            if pieces != expected:
                missing.add(f"{name} shards {sorted(expected - pieces)}")
        if missing:
            raise ValueError(f"Missing MTP checkpoint tensors: {sorted(missing)}")
        fuse_alpha = torch.softmax(self.fuse_weight.float(), dim=-1).to(
            self.fuse_weight.dtype
        )
        if self._fuse_alpha is None:
            self._fuse_alpha = fuse_alpha
        else:
            self._fuse_alpha.copy_(fuse_alpha)
        return loaded


EntryClass = Inkling2ForConditionalGenerationMTP
