import torch
from dsv41_args import DeepseekV41Args
from dsv41_compressor import Compressor
from dsv41_indexer import Indexer
from dsv41_linear import Linear
from dsv41_norm import RMSNorm
from dsv41_rope import apply_rotary_emb_tail, precompute_freqs_cis
from dsv41_shared import SharedAttentionRuntime
from torch import nn

from sglang.srt.layers.attention.dsv4.torch_quant import fake_quant_fp4, fake_quant_fp8


def sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """q [b, m, h, d], kv [b, n, d] (one latent is both key and value), attn_sink [h] fp32,
    topk_idxs [b, m, topk] int32 with -1 for empty slots -> [b, m, h, d] in q's dtype.
    Probabilities are rounded to bf16 before the value sum, as the kernel does."""
    b, m, h, d = q.shape
    topk = topk_idxs.size(-1)
    valid = topk_idxs >= 0
    gather_idx = topk_idxs.clamp_min(0).reshape(b, m * topk, 1).expand(-1, -1, d)
    keys = kv.gather(1, gather_idx).view(b, m, topk, d).float()
    scores = torch.einsum("bmhd,bmtd->bmht", q.float(), keys) * softmax_scale
    scores = scores.masked_fill(~valid[:, :, None, :], -torch.inf)
    # A finite floor keeps a row with no valid slot at an all-zero output instead of NaN.
    row_max = scores.amax(dim=-1, keepdim=True).clamp_min(-1e30)
    probs = torch.exp(scores - row_max)
    denom = probs.sum(dim=-1, keepdim=True) + torch.exp(
        attn_sink.view(1, 1, h, 1) - row_max
    )
    out = torch.einsum("bmht,bmtd->bmhd", probs.to(q.dtype).float(), keys)
    return (out / denom).to(q.dtype)


def get_window_topk_idxs(
    window_size: int, bsz: int, seqlen: int, start_pos: int, device
) -> torch.Tensor:
    """Sliding-window cache slots each query attends to, -1 for empty. Prefill rows see
    their own causal window; a decode step sees the whole ring, oldest first."""
    if start_pos == 0:
        end = torch.arange(seqlen, device=device).unsqueeze(1)
        idxs = (end - window_size + 1).clamp(0) + torch.arange(
            min(seqlen, window_size), device=device
        )
        idxs = torch.where(idxs > end, -1, idxs)
    else:
        oldest = start_pos % window_size + 1
        idxs = torch.cat(
            [
                torch.arange(oldest, window_size, device=device),
                torch.arange(oldest, device=device),
            ]
        )
        idxs = torch.where(idxs > start_pos, -1, idxs)
    return idxs.int().unsqueeze(0).expand(bsz, -1, -1).contiguous()


class Attention(nn.Module):
    """Latent attention over a sliding window of raw KV plus, when compress_ratio > 0,
    index_topk compressed positions, in one sparse_attn call. Only kv_source layers
    compress their own KV; the others read the shared cache."""

    def __init__(self, args: DeepseekV41Args, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.n_heads = args.n_heads
        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.n_groups = args.o_groups
        self.window_size = args.window_size
        self.compress_ratio = args.compress_ratios[layer_id]
        self.softmax_scale = self.head_dim**-0.5

        fp8 = torch.float8_e4m3fn
        self.attn_sink = nn.Parameter(torch.empty(self.n_heads, dtype=torch.float32))
        self.wq_a = Linear(args.dim, self.q_lora_rank, dtype=fp8)
        self.q_norm = RMSNorm(self.q_lora_rank, args.norm_eps)
        self.wq_b = Linear(self.q_lora_rank, self.n_heads * self.head_dim, dtype=fp8)
        self.wkv = Linear(args.dim, self.head_dim, dtype=fp8)
        self.kv_norm = RMSNorm(self.head_dim, args.norm_eps)
        self.wo_a = Linear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            dtype=torch.bfloat16,
        )
        self.wo_b = Linear(self.n_groups * self.o_lora_rank, args.dim, dtype=fp8)

        is_backbone = layer_id < args.n_layers
        self.is_kv_source = is_backbone and layer_id in args.kv_source_layers
        self.is_index_source = is_backbone and layer_id in args.index_source_layers
        self.compressor = None
        self.indexer = None
        if self.is_kv_source:
            self.compressor = Compressor(
                dim=args.dim,
                head_dim=self.head_dim,
                compress_ratio=self.compress_ratio,
                norm_eps=args.norm_eps,
                max_batch_size=args.max_batch_size,
            )
        if self.is_index_source:
            self.indexer = Indexer(args, layer_id)

        cache_dtype = torch.bfloat16
        self.register_buffer(
            "window_kv_cache",
            torch.zeros(
                args.max_batch_size, args.window_size, self.head_dim, dtype=cache_dtype
            ),
            persistent=False,
        )
        if self.is_kv_source:
            self.register_buffer(
                "compress_kv_cache",
                torch.zeros(
                    args.max_batch_size,
                    args.max_seq_len // self.compress_ratio,
                    self.head_dim,
                    dtype=cache_dtype,
                ),
                persistent=False,
            )
        # Pure sliding-window layers use the base theta without YaRN.
        if self.compress_ratio:
            original_seq_len, rope_theta = (
                args.original_seq_len,
                args.compress_rope_theta,
            )
        else:
            original_seq_len, rope_theta = 0, args.rope_theta
        freqs_cis = precompute_freqs_cis(
            self.rope_head_dim,
            args.max_seq_len,
            original_seq_len,
            rope_theta,
            args.rope_factor,
            args.beta_fast,
            args.beta_slow,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def _window_kv(self, x, freqs_cis, start_pos):
        """This layer's window K (fp8-rounded over the whole post-RoPE vector) and the
        window slots each query attends to."""
        bsz, seqlen, _ = x.size()
        win = self.window_size
        kv = self.kv_norm(self.wkv(x))
        kv = fake_quant_fp8(apply_rotary_emb_tail(kv, self.rope_head_dim, freqs_cis))
        if start_pos == 0:
            if seqlen <= win:
                self.window_kv_cache[:bsz, :seqlen] = kv
            else:
                cutoff = seqlen % win
                (
                    self.window_kv_cache[:bsz, cutoff:win],
                    self.window_kv_cache[:bsz, :cutoff],
                ) = kv[:, -win:].split([win - cutoff, cutoff], dim=1)
            window_kv = kv
        else:
            self.window_kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
            window_kv = self.window_kv_cache[:bsz]
        return window_kv, get_window_topk_idxs(win, bsz, seqlen, start_pos, x.device)

    def _compress_topk_idxs(
        self, x, qr, latent, start_pos, offset, compress_len, shared
    ):
        if not self.is_index_source:
            return shared.topk_idxs
        bsz, seqlen, _ = x.size()
        if compress_len == 0:
            idxs = torch.empty(bsz, seqlen, 0, dtype=torch.int32, device=x.device)
        else:
            idxs = self.indexer(
                x,
                qr,
                latent,
                start_pos=start_pos,
                offset=offset,
                freqs_cis=self.freqs_cis,
                shared=shared,
            )
        shared.topk_idxs = idxs
        return idxs

    def _compress_kv(self, x, qr, start_pos, offset, shared):
        """The shared compressed KV and the compressed positions each query attends to."""
        bsz, seqlen, _ = x.size()
        ratio = self.compress_ratio
        compress_len = (start_pos + seqlen) // ratio
        latent = None
        if self.is_kv_source:
            latent = self.compressor(x, start_pos)
            shared.compress_kv = self.compress_kv_cache
        # The indexer needs the pre-RoPE latent, so it runs before the cache is written.
        idxs = self._compress_topk_idxs(
            x, qr, latent, start_pos, offset, compress_len, shared
        )
        if latent is not None:
            if start_pos == 0:
                freqs = self.freqs_cis[: seqlen - seqlen % ratio : ratio]
            else:
                freqs = self.freqs_cis[start_pos + 1 - ratio].unsqueeze(0)
            latent = fake_quant_fp4(
                apply_rotary_emb_tail(latent, self.rope_head_dim, freqs)
            )
            start = start_pos // ratio
            self.compress_kv_cache[:bsz, start : start + latent.size(1)] = latent
        return shared.compress_kv[:bsz, :compress_len], idxs

    def forward(
        self, x: torch.Tensor, start_pos: int, shared: SharedAttentionRuntime
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        rd = self.rope_head_dim

        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))
        q = apply_rotary_emb_tail(q, rd, freqs_cis)

        kv, topk_idxs = self._window_kv(x, freqs_cis, start_pos)
        if self.compress_ratio:
            compress_kv, compress_idxs = self._compress_kv(
                x, qr, start_pos, kv.size(1), shared
            )
            kv = torch.cat([kv, compress_kv], dim=1)
            topk_idxs = torch.cat([topk_idxs, compress_idxs], dim=-1)

        o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
        o = apply_rotary_emb_tail(o, rd, freqs_cis, inverse=True)

        # wo_a is block-diagonal over groups: each group projects only its own heads.
        o = o.view(bsz, seqlen, self.n_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        return self.wo_b(o.flatten(2))
