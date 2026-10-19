# Copyright 2026 Blockway
# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Agens Volundr.

Volundr is a hybrid language model with an optional vision tower. It shares its tokenizer and base
projections with Qwen3.5-family models; the mixers, hyper-connections and n-gram memory are its own:

  * KDA layers (Kimi Delta Attention: linear attention with a per-channel forget gate), served by
    the KDA linear-attention backend (KDAAttnBackend) through RadixLinearAttention. Volundr keeps
    q/k/v, beta and the low-rank gate in separate projections; the calling convention follows
    models/kimi_linear.py.
  * BCSA layers (Blockway Compressed-Sparse Attention, VolundrAttention below): a sliding window over
    the paged KV cache plus a mean-pooled far field selected by a learned indexer, merged into one
    softmax through the window's log-sum-exp. `attn` layers in mixer_types are plain dense attention.
  * mHC (manifold hyper-connections, VolundrMHC): n residual streams, mixed around every layer.
  * Engram (VolundrEngram): hashed n-gram conditional memory. The n-gram hash needs the 1-2 token ids
    before each token, which a ForwardBatch does not carry at decode time; a per-request
    previous-token table keyed by the mamba state slot supplies them.
  * The BCSA indexer key is a function of the layer input, not of K or V, so each BCSA layer keeps a
    per-token side cache of it, indexed by the same out_cache_loc as the KV write.
  * Vision: VolundrForConditionalGeneration serves the vision tower (the qwen3_vl vision model)
    with `--enable-multimodal`. Image tokens carry 3-D MRoPE positions through the BCSA layers' rotary
    embedding; everything that indexes the KV cache or the n-gram table keeps the 1-D token index.
    VolundrForCausalLM is the text-only class.

Serving constraints (defaults set by arg_groups/model_overrides/volundr.py, verified here):
  * page size 1: BCSA reads pooled far-field K/V straight out of the token-indexed KV cache.
  * radix cache off: a prefix hit would resume a sequence whose Engram previous-token ids were never
    seeded.
  * no prefill CUDA graph: the Engram and BCSA extend paths read per-batch host metadata.
  * the FlashAttention-3 backend for the BCSA sliding window.

Layer typing comes from `config.mixer_types` ("kda" | "bcsa" | "attn").

Every RMSNorm of the attention/MLP stack (input/post_attention/final/q_norm/k_norm) is GemmaRMSNorm,
the zero-centred `xhat * (1 + weight)` form — see the note on q_norm below. mHC's own norm and the KDA
output norm are the plain `weight *` form. Getting this wrong is silent and total.
"""

from typing import Iterable, Optional

import torch
from torch import nn

from sglang.kernels.ops.attention.fla.fused_norm_gate import FusedRMSNormGated
from sglang.kernels.ops.attention.flash_attention import flash_attn_with_kvcache
from sglang.kernels.ops.attention.volundr_bcsa import (
    bcsa_block_keys_decode,
    far_sparse_fused,
    merge_local_far,
)
from sglang.kernels.ops.layernorm.volundr_mhc import mhc_combine, mhc_mix
from sglang.srt.distributed.parallel_state import get_pp_group
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import GemmaRMSNorm, RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import Modality, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from sglang.srt.runtime_context import get_parallel, get_server_args
from sglang.srt.utils import add_prefix, set_weight_attrs
from sglang.srt.utils.hf_transformers_utils import get_rope_config


def _text_cfg(config):
    """The checkpoint nests the language model under text_config."""
    return getattr(config, "text_config", config)


def _full_attn_backend():
    """The softmax-attention half of the hybrid backend (the other half drives the KDA layers)."""
    b = get_attn_backend()
    return getattr(b, "full_attn_backend", None) or b


def _is_flash_attention(backend) -> bool:
    """Is this the FlashAttention backend (the one the BCSA local branch is written for)?"""
    from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend

    return isinstance(backend, FlashAttentionBackend)


def _attn_tp():
    """(rank, world) of the attention tensor-parallel group.

    attn_tp, not the global tp: it is the group `sharded_weight_loader` slices against and the one
    VolundrTextConfig.mamba2_cache_params already sizes the KDA recurrent state with. With DP
    attention off — which is the only configuration this class is served in — the two coincide.
    """
    p = get_parallel()
    return p.attn_tp_rank, p.attn_tp_size


def _set_loader(param, fn) -> None:
    """Replace a parameter's weight_loader, whatever kind of parameter it is.

    sglang's parallel Linear classes give their weight a ModelWeightParameter whose `weight_loader`
    is a read-only property over `_weight_loader`; a plain nn.Parameter just carries the attribute.
    set_weight_attrs refuses to overwrite either (it asserts), so neither can be used directly.
    """
    if hasattr(param, "_weight_loader"):
        param._weight_loader = fn
    else:
        if hasattr(param, "weight_loader"):
            delattr(param, "weight_loader")
        set_weight_attrs(param, {"weight_loader": fn})


def _blockwise_column_loader(block_sizes, tp_rank: int, tp_size: int, dim: int = 0):
    """Loader for a fused projection whose output dim is a concatenation of independent blocks.

    Volundr ships `in_proj_qkv` (and the conv1d beside it) as ONE tensor laid out [q | k | v], with
    q and k 16 heads wide and v 48. A plain column-parallel load takes a single contiguous slice of
    that, which on two ranks hands rank 0 all of q plus half of k — silently, and the KDA kernel
    accepts the shape. This takes each block's own per-rank slice and concatenates them, which is
    what sglang's mamba_v2_sharded_weight_loader does for the same reason.
    """

    def loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        parts, off = [], 0
        for full in block_sizes:
            shard = full // tp_size
            parts.append(loaded_weight.narrow(dim, off + tp_rank * shard, shard))
            off += full
        assert off == loaded_weight.shape[dim], (
            f"block sizes {block_sizes} do not tile the loaded dim {loaded_weight.shape[dim]}"
        )
        param.data.copy_(torch.cat(parts, dim=dim))

    return loader


def _mixer_kind(c, layer_id: int) -> str:
    """'kda' | 'bcsa' | 'attn' for a layer.

    Prefers `mixer_types`, which is the checkpoint's own record, and falls back to `layer_types`.
    """
    mt = getattr(c, "mixer_types", None)
    if mt:
        return mt[layer_id]
    lt = getattr(c, "layer_types", None)
    if lt:
        return "kda" if lt[layer_id] == "linear_attention" else "bcsa"
    raise ValueError(
        "config carries neither mixer_types nor layer_types; cannot type the layers"
    )


class VolundrMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, quant_config=None, prefix=""):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        return self.down_proj(self.act_fn(gate_up))[0]


class VolundrKDA(nn.Module):
    """Kimi Delta Attention. Volundr keeps the projections separate; sglang's KDA backend takes
    (mixed_qkv, a, b) and owns the conv state and the recurrent state.

    Tensor parallelism. Every head-indexed quantity is split across attn_tp ranks and everything the
    KDA backend sees is already per-rank — the backend and its kernels read no tp_size anywhere, so
    the division has to happen here and must not be repeated downstream (models/qwen3_next.py and
    models/kimi_linear.py are the references, and this follows qwen3_next's discipline of sharding
    everything on attn_tp because that is the group `sharded_weight_loader` hard-codes, and the
    group VolundrTextConfig.mamba2_cache_params already sizes the recurrent state with):

      in_proj_qkv  column-parallel, but BLOCKWISE — [q | k | v] are three independent head groups
      in_proj_b    column-parallel over the 48 value heads (one scalar each)
      in_proj_z    column-parallel over value channels
      f_down       replicated (rank-128 bottleneck; every rank computes the same 128 numbers)
      f_up         column-parallel over value heads, i.e. THE SAME head split as in_proj_qkv's v
                   block and as dt_bias — the per-channel gate must line up with the channels it
                   gates, which is the one thing that fails silently if it is got wrong
      conv1d       blockwise column-parallel like in_proj_qkv (it is the same [q|k|v] layout)
      A_log        per value head, sliced on load
      dt_bias      per (value head, key channel), head-major, so a contiguous dim-0 slice is a
                   contiguous head range — sliced on load
      norm         replicated (it is per head_v_dim, not per head)
      out_proj     row-parallel, and it owns the all-reduce (reduce_results defaults to True; this
                   class threads the residual itself and has no LayerCommunicator)

    At tp_size == 1 every projection stays a ReplicatedLinear, so packed INT4 weights load without
    being re-split by the merged and column-parallel loaders; that is also why tp_size > 1 is a
    bf16-only configuration here.
    """

    def __init__(self, config, layer_id: int, quant_config=None, prefix: str = ""):
        super().__init__()
        c = _text_cfg(config)
        tp_rank, tp_size = _attn_tp()
        self.tp_size = tp_size
        total_k_heads = c.linear_num_key_heads  # 16
        total_v_heads = (
            c.linear_num_value_heads
        )  # 48 — grouped value attention, 3 v per k
        self.head_k_dim = c.linear_key_head_dim
        self.head_v_dim = c.linear_value_head_dim
        if tp_size > 1:
            for name, n in (
                ("linear_num_key_heads", total_k_heads),
                ("linear_num_value_heads", total_v_heads),
            ):
                if n % tp_size:
                    raise ValueError(
                        f"KDA {name}={n} is not divisible by attn_tp_size={tp_size}"
                    )
            if quant_config is not None:
                raise NotImplementedError(
                    "Volundr tp_size > 1 is bf16 only: the KDA projections become blockwise "
                    "column-parallel at tp > 1 and this class does not re-split packed quantized "
                    "weights. Serve INT4 at tp 1 (it fits one 48 GB GPU) or bf16 at tp 2."
                )
        # per-rank counts and widths from here down; `total_*` only ever reaches a constructor arg
        self.num_k_heads = total_k_heads // tp_size
        self.num_v_heads = total_v_heads // tp_size
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        total_key_dim = self.head_k_dim * total_k_heads
        total_value_dim = self.head_v_dim * total_v_heads
        conv_dim = self.key_dim * 2 + self.value_dim
        qkv_blocks = (total_key_dim, total_key_dim, total_value_dim)
        gate_rank = c.kda_gate_rank

        if tp_size == 1:
            self.in_proj_qkv = ReplicatedLinear(
                c.hidden_size,
                conv_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("in_proj_qkv", prefix),
            )
            self.in_proj_b = ReplicatedLinear(
                c.hidden_size,
                self.num_v_heads,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("in_proj_b", prefix),
            )
            self.in_proj_z = ReplicatedLinear(
                c.hidden_size,
                self.value_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("in_proj_z", prefix),
            )
            self.f_up = ReplicatedLinear(
                gate_rank,
                self.num_v_heads * self.head_k_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("f_up", prefix),
            )
            self.out_proj = ReplicatedLinear(
                self.value_dim,
                c.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("out_proj", prefix),
            )
        else:
            self.in_proj_qkv = ColumnParallelLinear(
                c.hidden_size,
                total_key_dim * 2 + total_value_dim,
                bias=False,
                quant_config=quant_config,
                parallel_group="attn_tp",
                prefix=add_prefix("in_proj_qkv", prefix),
            )
            # the default column-parallel loader takes one contiguous slice; [q|k|v] needs three
            _set_loader(
                self.in_proj_qkv.weight,
                _blockwise_column_loader(qkv_blocks, tp_rank, tp_size),
            )
            self.in_proj_b = ColumnParallelLinear(
                c.hidden_size,
                total_v_heads,
                bias=False,
                quant_config=quant_config,
                parallel_group="attn_tp",
                prefix=add_prefix("in_proj_b", prefix),
            )
            self.in_proj_z = ColumnParallelLinear(
                c.hidden_size,
                total_value_dim,
                bias=False,
                quant_config=quant_config,
                parallel_group="attn_tp",
                prefix=add_prefix("in_proj_z", prefix),
            )
            self.f_up = ColumnParallelLinear(
                gate_rank,
                total_v_heads * self.head_k_dim,
                bias=False,
                quant_config=quant_config,
                parallel_group="attn_tp",
                prefix=add_prefix("f_up", prefix),
            )
            self.out_proj = RowParallelLinear(
                total_value_dim,
                c.hidden_size,
                bias=False,
                quant_config=quant_config,
                input_is_parallel=True,
                reduce_results=True,
                parallel_group="attn_tp",
                prefix=add_prefix("out_proj", prefix),
            )
        # low-rank channel-wise gate: this is what makes it KDA rather than GDN. f_down stays
        # replicated on every rank (128 outputs, no head structure); f_up above splits its output
        # over the same value heads in_proj_qkv's v block was split over.
        self.f_down = ReplicatedLinear(
            c.hidden_size,
            gate_rank,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("f_down", prefix),
        )

        # fp32 conv weights and fp32 A_log / dt_bias, matching models/kimi_linear.py. The checkpoint
        # stores all three in bf16; widening on load is exact and is what both the HF reference and
        # these kernels do internally anyway (every read is `.to(tl.float32)`).
        self.conv1d = nn.Conv1d(
            conv_dim,
            conv_dim,
            kernel_size=c.linear_conv_kernel_dim,
            groups=conv_dim,
            padding=c.linear_conv_kernel_dim - 1,
            bias=False,
            dtype=torch.float32,
        )
        # A_log is per-head (48,); dt_bias is per-channel and flat (48*128,): the channel resolution
        # that, alongside the low-rank f_down/f_up gate, makes this KDA rather than GDN.
        self.A_log = nn.Parameter(torch.empty(self.num_v_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(
            torch.empty(self.num_v_heads * self.head_k_dim, dtype=torch.float32)
        )
        if tp_size > 1:
            _set_loader(
                self.conv1d.weight,
                _blockwise_column_loader(qkv_blocks, tp_rank, tp_size),
            )
            _set_loader(self.A_log, sharded_weight_loader(0, parallel_group="attn_tp"))
            _set_loader(
                self.dt_bias, sharded_weight_loader(0, parallel_group="attn_tp")
            )
        # gated RMSNorm, not the fused add-residual form: KDA's output norm takes a gate (z),
        # whereas sglang's RMSNorm(x, y) means "add y as residual" and returns a tuple. The
        # activation is the checkpoint's output_gate_type ('swish' == SiLU, the config default).
        self.norm = FusedRMSNormGated(
            self.head_v_dim, eps=c.rms_norm_eps, activation=c.output_gate_type
        )

        self.attn = RadixLinearAttention(
            layer_id=layer_id,
            num_q_heads=self.num_k_heads,
            num_k_heads=self.num_k_heads,
            num_v_heads=self.num_v_heads,
            head_q_dim=self.head_k_dim,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            conv_weights=self.conv1d.weight.squeeze(1),
            bias=None,
            activation="silu",
            A_log=self.A_log,
            dt_bias=self.dt_bias,
        )

    @staticmethod
    def _is_flat_gate_mode(forward_batch: ForwardBatch) -> bool:
        mode = forward_batch.forward_mode
        return mode.is_decode() or mode.is_target_verify()

    def forward(self, hidden_states: torch.Tensor, forward_batch: ForwardBatch):
        mixed_qkv, _ = self.in_proj_qkv(hidden_states)
        beta, _ = self.in_proj_b(hidden_states)
        gate, _ = self.f_up(self.f_down(hidden_states)[0])
        # Calling convention of models/kimi_linear.py, which drives the same backend: the extend
        # kernel wants the gate shaped [1, T, HV, K], the decode and target-verify kernels take the
        # flat gate [T, HV*K]. Beta stays raw in every mode (the kernels apply the sigmoid).
        if not self._is_flat_gate_mode(forward_batch):
            gate = gate.unflatten(-1, (-1, self.head_k_dim)).unsqueeze(0)
        beta = beta.unsqueeze(0)
        core = self.attn(forward_batch, mixed_qkv=mixed_qkv, a=gate, b=beta)
        z, _ = self.in_proj_z(hidden_states)
        o = self.norm(core.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
        return self.out_proj(o.reshape(-1, self.value_dim))[0]


class VolundrAttention(nn.Module):
    """BCSA — Blockway compressed-sparse attention — or, on the single `attn` layer, plain dense
    attention.

    BCSA splits the keys a query may
    see into two disjoint sets and puts them under ONE softmax:

      * local  : every key in (t - W, t],  W = config.bcsa_window. Dense, causal, sliding.
      * far    : keys older than that window, mean-pooled into blocks of C = config.bcsa_compress
                 consecutive positions. A DSA-style indexer (bcsa_index_heads query heads, 1 key
                 head, bcsa_index_head_dim dims, its own RMSNorm) scores every complete far block
                 and the top config.bcsa_topk are attended to; the rest are dropped.

    Block n covers positions [nC, nC+C-1] and is "far" for query t iff its LAST position is at or
    before t - W:  (n+1)C - 1 <= t - W. That is exactly complementary to the window (which starts at
    t - W + 1), so the two sets tile the causal prefix with no gap and no overlap, and the valid far
    blocks of a query are always the contiguous prefix n <= (t - W - C + 1) // C. For t <= W the far
    set is empty and BCSA degenerates to dense attention.

    How it is served here. sglang has no kernel for "sliding window plus a separately pooled,
    indexer-selected far field under a shared softmax", and its NSA/DSA sparse backend selects raw
    cache tokens rather than pooled blocks, so it cannot be pointed at the far branch. Instead:

      * the local branch is one flash_attn_with_kvcache call with window_size=(W-1, 0) over the
        ordinary paged KV cache, asking for the log-sum-exp as well as the output. This is the same
        kernel, page table and cache the dense path used, so the local half stays at full speed.
      * the far branch is computed here: pool K/V of the selected blocks straight out of the KV
        cache (they are already there — pooling 4 rows costs 4 gathers), score, top-k, softmax.
      * the two are merged exactly, using the local LSE:
            out = (e^(lse-m) * out_local + sum_n e^(s_n-m) v_n) / (e^(lse-m) + sum_n e^(s_n-m))
        which is algebraically the single softmax the reference takes over the concatenation.

    The one thing that cannot be recomputed from the KV cache is the indexer's key, idx_norm(idx_k
    h): it is a function of the layer INPUT, which no cache holds. So this class keeps its own
    per-token side cache of that 128-wide vector, indexed by the very same out_cache_loc sglang
    hands the KV write — which means sglang's own allocator manages its lifetime for free, and a
    slot's indexer key is always as live as the K/V beside it. It costs 128 * 2 bytes per token per
    BCSA layer. It is allocated after the KV pool is sized, so --mem-fraction-static has to
    leave room for it.

    Decode stays inside CUDA graphs: every shape here is static given the batch size (the pooled
    indexer keys are read for max_blocks = context_length / 4 blocks and masked, the top-k is a
    fixed 512), no host sync, no data-dependent branch. The extend path runs per request and is
    chunked over queries, because its far scores are [tokens, 24 heads, blocks] and at 12k tokens
    that is gigabytes if taken in one bite.
    """

    def __init__(
        self,
        config,
        layer_id: int,
        quant_config=None,
        prefix: str = "",
        bcsa: bool = True,
    ):
        super().__init__()
        c = _text_cfg(config)
        # An `attn` layer carries no idx_q/idx_k/idx_norm tensors and is plain dense attention at
        # any length. Only `bcsa` layers get the sparse path.
        self.bcsa = bool(bcsa)
        self.window = c.bcsa_window
        self.compress = c.bcsa_compress
        self.bcsa_topk = c.bcsa_topk
        self.idx_heads, self.idx_dim = c.bcsa_index_heads, c.bcsa_index_head_dim
        self.idx_scale = self.idx_dim**-0.5
        self.idx_cache = None  # [pool_size + 1, 128], allocated on the first forward
        # The mean-pooled indexer key of every COMPLETE block, stored at the KV slot of the
        # block's last position (so sglang's allocator owns its lifetime exactly like idx_cache).
        # Written once when a block completes (prefill, or the decode step that fills it); read by
        # both far paths, so decode never re-pools the per-position keys of the whole context.
        self.blk_cache = None  # [pool_size + 1, 128]
        self.max_blocks = 0  # context_length // C, fixed once the pools exist
        self.r2t = None  # the backend's req_to_token table, [max_reqs, max_ctx]
        tp_rank, tp = _attn_tp()
        self.tp_size = tp
        self.total_heads = c.num_attention_heads
        self.total_kv_heads = c.num_key_value_heads
        self.head_dim = getattr(c, "head_dim", None) or (
            c.hidden_size // c.num_attention_heads
        )
        if tp > 1:
            if self.total_heads % tp or self.total_kv_heads % tp:
                raise ValueError(
                    f"attention heads ({self.total_heads} q / {self.total_kv_heads} kv) are not "
                    f"divisible by attn_tp_size={tp}; BCSA's far field groups q heads under their "
                    "own kv head and needs both splits to be whole"
                )
            if quant_config is not None:
                raise NotImplementedError(
                    "Volundr tp_size > 1 is bf16 only (see VolundrKDA)"
                )
        self.num_heads = self.total_heads // tp
        self.num_kv_heads = max(1, self.total_kv_heads // tp)
        self.scaling = self.head_dim**-0.5

        # Gated attention (config.attn_output_gate): q_proj emits query AND an output
        # gate, interleaved PER HEAD — [h0_q | h0_gate | h1_q | h1_gate | ...], not two contiguous
        # halves. Splitting it as two halves loads eleven heads' gates as queries and still runs.
        # That interleaving is also exactly what makes a plain contiguous column-parallel slice the
        # RIGHT split under TP: rank r's slice of 2*H/tp head-widths is its own heads WITH their own
        # gates. (If the checkpoint ever concatenates [all_q | all_gate] instead, this silently
        # gives rank 0 only queries and rank 1 only gates — the same trap sglang's Qwen3-Next
        # qkv_proj sits in, and the reason this comment exists.)
        self.output_gate = bool(getattr(c, "attn_output_gate", False))
        q_out = self.total_heads * self.head_dim * (2 if self.output_gate else 1)
        if tp == 1:
            self.q_proj = ReplicatedLinear(
                c.hidden_size,
                q_out,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("q_proj", prefix),
            )
            self.k_proj = ReplicatedLinear(
                c.hidden_size,
                self.total_kv_heads * self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("k_proj", prefix),
            )
            self.v_proj = ReplicatedLinear(
                c.hidden_size,
                self.total_kv_heads * self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("v_proj", prefix),
            )
            self.o_proj = ReplicatedLinear(
                self.total_heads * self.head_dim,
                c.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("o_proj", prefix),
            )
        else:
            self.q_proj = ColumnParallelLinear(
                c.hidden_size,
                q_out,
                bias=False,
                quant_config=quant_config,
                parallel_group="attn_tp",
                prefix=add_prefix("q_proj", prefix),
            )
            self.k_proj = ColumnParallelLinear(
                c.hidden_size,
                self.total_kv_heads * self.head_dim,
                bias=False,
                quant_config=quant_config,
                parallel_group="attn_tp",
                prefix=add_prefix("k_proj", prefix),
            )
            self.v_proj = ColumnParallelLinear(
                c.hidden_size,
                self.total_kv_heads * self.head_dim,
                bias=False,
                quant_config=quant_config,
                parallel_group="attn_tp",
                prefix=add_prefix("v_proj", prefix),
            )
            self.o_proj = RowParallelLinear(
                self.total_heads * self.head_dim,
                c.hidden_size,
                bias=False,
                quant_config=quant_config,
                input_is_parallel=True,
                reduce_results=True,
                parallel_group="attn_tp",
                prefix=add_prefix("o_proj", prefix),
            )
        # The attention/MLP norms are the zero-centred (Gemma) form, `xhat * (1 + weight)`, with a
        # zero-initialised weight. sglang's plain RMSNorm multiplies by `weight` instead; with these
        # checkpoints (weight rms 0.05 on input_layernorm) that silently scales every norm output
        # by ~1/14 and the model decodes noise.
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=c.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=c.rms_norm_eps)
        if self.bcsa:
            # The indexer: 4 query heads and 1 key head of 128 dims, both through the SAME norm
            # (the reference applies idx_norm to the query view and to the key), in the
            # zero-centred Gemma form like every other norm here.
            # Replicated under TP, and not merely because it is small: the block scores are the
            # MEAN over its 4 query heads, so a head split would change the score rather than
            # partition it, and the top-512 selection has to be the same set on every rank.
            self.idx_q = ReplicatedLinear(
                c.hidden_size,
                self.idx_heads * self.idx_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("idx_q", prefix),
            )
            self.idx_k = ReplicatedLinear(
                c.hidden_size,
                self.idx_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("idx_k", prefix),
            )
            self.idx_norm = GemmaRMSNorm(self.idx_dim, eps=c.rms_norm_eps)

        # rope_theta lives in `rope_parameters` in this transformers-5 config, so config.rope_theta
        # is the class default (10000) rather than the checkpoint's 1e7 — get_rope_config reads the
        # right one. rope_scaling is deliberately NOT passed: it carries mrope_section, which would
        # build an MRotaryEmbedding expecting [3, N] positions. On a text-only batch the three mrope
        # axes hold the same position, so interleaved mRoPE is bit-identical to plain RoPE here.
        rope_theta, _ = get_rope_config(c)
        rope_frac = getattr(c, "partial_rotary_factor", 0.25)
        # Sized to the checkpoint's full context. rotary_dim is 0.25 * 256 = 64 (192 dims stay
        # NoPE), so the table is [max_pos, 64] and get_rope memoises it across all 18 attention
        # layers.
        max_pos = int(getattr(c, "max_position_embeddings", 4096))
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_pos,
            base=rope_theta,
            partial_rotary_factor=rope_frac,
            is_neox_style=True,
        )
        # The same table as an MRotaryEmbedding, used ONLY when a batch carries [3, N] image
        # positions (an extend batch with image inputs). Text batches — and every decode step, where
        # the three mrope rows coincide — keep the plain rotary above (MRotaryEmbedding's 1-D path
        # runs the native kernel, a different rounding).
        rp = dict(
            getattr(c, "rope_parameters", None)
            or getattr(c, "rope_scaling", None)
            or {}
        )
        self.mrope_section = list(rp.get("mrope_section") or [])
        self.mrotary_emb = None
        if self.mrope_section:
            self.mrotary_emb = get_rope(
                self.head_dim,
                rotary_dim=self.head_dim,
                max_position=max_pos,
                base=rope_theta,
                partial_rotary_factor=rope_frac,
                is_neox_style=True,
                rope_scaling={
                    "rope_type": "default",
                    "mrope_section": self.mrope_section,
                    "mrope_interleaved": bool(rp.get("mrope_interleaved", False)),
                },
            )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions,
        hidden_states,
        forward_batch: ForwardBatch,
        rope_positions=None,
        **kw,
    ):
        """`positions` is always the 1-D token index (KV / block / n-gram bookkeeping). `rope_positions`
        is the [3, N] MRoPE tensor of a multimodal batch, or None."""
        qg, _ = self.q_proj(hidden_states)
        gate = None
        if self.output_gate:
            qg = qg.view(-1, self.num_heads, 2 * self.head_dim)
            q, gate = torch.chunk(qg, 2, dim=-1)
            q = q.reshape(-1, self.num_heads * self.head_dim)
            gate = gate.reshape(-1, self.num_heads * self.head_dim)
        else:
            q = qg
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)
        q = self.q_norm(q.reshape(-1, self.head_dim)).reshape(
            -1, self.num_heads * self.head_dim
        )
        k = self.k_norm(k.reshape(-1, self.head_dim)).reshape(
            -1, self.num_kv_heads * self.head_dim
        )
        if (
            rope_positions is not None
            and rope_positions.ndim == 2
            and self.mrotary_emb is not None
        ):
            # An extend batch with images. Only tokens whose three MRoPE rows differ (the image
            # spans) need the multimodal kernel; every other token — the text of the same request
            # and any text-only request sharing the batch — goes through the plain rotary kernel on
            # the t-row, so it stays bit-identical to a text-only batch (the two kernels round
            # differently). Both kernels rotate in place, hence the clones.
            mp = rope_positions
            same = ((mp[0] == mp[1]) & (mp[1] == mp[2]))[:, None]
            qa, ka = self.rotary_emb(mp[0], q.clone(), k.clone())
            qb, kb = self.mrotary_emb(mp, q, k)
            q = torch.where(same, qa, qb)
            k = torch.where(same, ka, kb)
        else:
            q, k = self.rotary_emb(positions, q, k)
        if self.bcsa and q.shape[0] > 0 and not forward_batch.forward_mode.is_idle():
            out = self._bcsa_attention(hidden_states, q, k, v, positions, forward_batch)
        else:
            out = self.attn(q, k, v, forward_batch)
        if gate is not None:
            out = out * torch.sigmoid(gate)
        return self.o_proj(out)[0]

    # ------------------------------------------------------------------ BCSA
    def _ensure_state(self, kcache: torch.Tensor, fa):
        """One-time sizing of the indexer-key side cache and the static block count.

        Both need the memory pools, which only exist after the weights are loaded and the KV budget
        has been profiled, so they are sized on the first forward. That is safe: the CUDA-graph
        backend runs two eager warmup forwards before it captures anything (see
        runner_backend/full_cuda_graph_backend.py capture_one), so the allocation never lands inside
        a capture — asserted rather than assumed, because an allocation captured into a graph pool
        would be garbage on replay.
        """
        if self.idx_cache is not None:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "the BCSA indexer-key cache was still unallocated at CUDA-graph capture; it must be "
                "sized by a warmup forward first"
            )
        rows = int(kcache.shape[0])
        # +1 so a clamped/padded index has somewhere harmless to land.
        self.idx_cache = torch.zeros(
            rows + 1, self.idx_dim, dtype=kcache.dtype, device=kcache.device
        )
        self.blk_cache = torch.zeros(
            rows + 1, self.idx_dim, dtype=kcache.dtype, device=kcache.device
        )
        # req_to_token is [max_reqs, max_context_len] and lives for the server's lifetime, so both
        # the handle and the block count are fixed here rather than looked up per forward. The
        # backend's copy is used, not the metadata's page_table: that one is narrowed to the batch's
        # own max_seq_len_k on the eager path and only reaches full width inside a graph.
        self.r2t = fa.req_to_token_pool.req_to_token
        self.max_blocks = int(self.r2t.shape[1]) // self.compress

    def _bcsa_attention(
        self, hidden_states, q, k, v, positions, forward_batch: ForwardBatch
    ):
        mode = forward_batch.forward_mode
        if not (
            mode.is_decode()
            or (
                mode.is_extend()
                and not mode.is_target_verify()
                and not mode.is_draft_extend_v2()
            )
        ):
            raise NotImplementedError(f"BCSA serves extend and decode, not {mode}")
        fa = _full_attn_backend()
        if not _is_flash_attention(fa):
            raise NotImplementedError(
                "the BCSA local branch runs on the FlashAttention-3 backend; launch with "
                "--attention-backend fa3"
            )
        if fa.page_size != 1:
            raise RuntimeError(
                "BCSA needs page_size 1: it reads pooled far-field K/V straight out of the "
                "token-indexed KV cache. Launch with --page-size 1"
            )
        meta = fa.forward_metadata
        pool = fa.token_to_kv_pool
        kcache, vcache = pool.get_kv_buffer(self.attn.layer_id)
        H, KV, D = self.num_heads, self.num_kv_heads, self.head_dim
        T = q.shape[0]
        qh = q.view(T, H, D)

        # ---- write this step's K/V and indexer keys into their caches -------------------------
        pool.set_kv_buffer(
            self.attn,
            forward_batch.out_cache_loc,
            k.view(-1, KV, D),
            v.view(-1, KV, D),
            self.attn.k_scale,
            self.attn.v_scale,
        )
        self._ensure_state(kcache, fa)
        loc = forward_batch.out_cache_loc.long()
        ik = self.idx_norm(self.idx_k(hidden_states)[0])  # [T, Di]
        self.idx_cache.index_copy_(0, loc, ik.to(self.idx_cache.dtype))
        self._write_block_keys(positions, forward_batch)

        # ---- local branch: the sliding window, with its log-sum-exp ----------------------------
        is_decode = mode.is_decode()
        o_loc, lse = self._local_fa(fa, meta, qh, kcache, vcache, is_decode)

        # ---- far branch ------------------------------------------------------------------------
        if not is_decode:
            out = self._far_extend(
                hidden_states, qh, o_loc, lse, forward_batch, kcache, vcache
            )
            return out.reshape(T, H * D).to(q.dtype)
        # The far set is empty for every request while the batch is inside the window. Only an
        # eager forward may skip it: a captured graph replays at lengths its capture never saw
        # (and capture warmups must run the far kernels so they are compiled before recording).
        slc = forward_batch.seq_lens_cpu
        if (
            not get_is_capture_mode()
            and not torch.cuda.is_current_stream_capturing()
            and slc is not None
            and int(slc.max()) <= self.window
        ):
            return o_loc.reshape(T, H * D)
        out = far_sparse_fused(
            self,
            hidden_states,
            qh,
            o_loc,
            lse,
            positions.long(),
            forward_batch.req_pool_indices.long(),
            kcache,
            vcache,
        )
        return out.reshape(T, H * D).to(q.dtype)

    def _local_fa(self, fa, meta, qh, kcache, vcache, is_decode):
        """One flash_attn_with_kvcache over the paged cache, window (W-1, 0), with the LSE."""
        T, H, D = qh.shape
        KV = self.num_kv_heads
        res = flash_attn_with_kvcache(
            q=qh.contiguous(),
            k_cache=kcache.view(-1, 1, KV, D),
            v_cache=vcache.view(-1, 1, KV, D),
            page_table=meta.page_table,
            cache_seqlens=meta.cache_seqlens_int32,
            cu_seqlens_q=meta.cu_seqlens_q,
            cu_seqlens_k_new=None if is_decode else meta.cu_seqlens_k,
            max_seqlen_q=meta.max_seq_len_q,
            softmax_scale=self.scaling,
            causal=True,
            # (left, right) is inclusive on both sides, so W-1 back plus the query itself is the
            # W keys of (t - W, t].
            window_size=(self.window - 1, 0),
            return_softmax_lse=True,
            num_splits=fa.num_splits,
            ver=fa.fa_impl_ver,
        )
        o_loc, lse = res[0].view(T, H, D), res[1]
        # FA3 varlen returns the LSE as [heads, tokens].
        return o_loc, lse.transpose(0, 1).float().contiguous()  # [T, H]

    def _merge(self, sf, lse, o_loc, vb, kv_first: bool):
        """One softmax over [local | selected far] (see merge_local_far)."""
        return merge_local_far(sf, lse, o_loc, vb, self.num_kv_heads, kv_first)

    def _write_block_keys(self, positions, forward_batch: ForwardBatch):
        """Pool the indexer keys of every block completed by this forward into blk_cache.

        Decode (graph-captured): every request pools the C slots of the block its new position
        lies in and writes the result to its own out_cache_loc if that position closes the block
        (pos % C == C-1), else to the trash row. Static shapes, no host sync.
        Extend: per request, the blocks whose last position falls inside this chunk — which covers
        chunked prefill, because a block straddling two chunks has all its positions' keys in
        idx_cache by the time the chunk holding its last position runs.
        """
        C, Di = self.compress, self.idx_dim
        hi = int(self.idx_cache.shape[0]) - 1
        if forward_batch.forward_mode.is_decode():
            bcsa_block_keys_decode(
                positions,
                forward_batch.req_pool_indices,
                forward_batch.out_cache_loc,
                self.r2t,
                self.idx_cache,
                self.blk_cache,
                C,
            )
            return
        seq_lens = forward_batch.seq_lens_cpu.tolist()
        qlens = forward_batch.extend_seq_lens_cpu
        reqs = forward_batch.req_pool_indices.long()
        for i, ql in enumerate(qlens):
            ql = int(ql)
            if ql == 0:
                continue
            slen = int(seq_lens[i])
            n_lo, n_hi = (
                (slen - ql) // C,
                slen // C,
            )  # blocks whose last position is in this chunk
            if n_hi <= n_lo:
                continue
            row = self.r2t[reqs[i]]
            slots = row[n_lo * C : n_hi * C].long().clamp_(0, hi)
            pooled = self.idx_cache[slots].view(n_hi - n_lo, C, Di).mean(1)
            dst = row[n_lo * C + C - 1 : n_hi * C : C].long().clamp_(0, hi)
            self.blk_cache.index_copy_(0, dst, pooled.to(self.blk_cache.dtype))

    def _far_extend(self, hidden_states, qh, o_loc, lse, forward_batch, kcache, vcache):
        """Prefill / chunked prefill: one request at a time (no graphs here), chunked over queries
        because the far scores are [tokens, 24, blocks] and that is gigabytes at 12k tokens."""
        C, W = self.compress, self.window
        H, KV, D = self.num_heads, self.num_kv_heads, self.head_dim
        G = H // KV
        dev = qh.device
        out = o_loc.float()
        seq_lens = forward_batch.seq_lens_cpu.tolist()
        qlens = forward_batch.extend_seq_lens_cpu
        r2t = self.r2t
        reqs = forward_batch.req_pool_indices.long()
        qi_all = self.idx_norm(self.idx_q(hidden_states)[0].reshape(-1, self.idx_dim))
        qi_all = qi_all.view(-1, self.idx_heads, self.idx_dim)
        hi = int(self.idx_cache.shape[0]) - 1
        start = 0
        for i, ql in enumerate(qlens):
            ql = int(ql)
            if ql == 0:
                continue
            slen = int(seq_lens[i])
            # blocks that are far for at least the LAST query of this request; every earlier query
            # sees a prefix of the same set, so this bounds the work exactly.
            nb = (slen - 1 - W - (C - 1)) // C + 1
            if nb <= 0:
                start += ql
                continue
            tk = r2t[reqs[i], : C * nb].long().clamp_(0, hi)
            kbi = self.blk_cache[
                r2t[reqs[i], C - 1 : C * nb : C].long().clamp_(0, hi)
            ]  # [nb, Di]
            kb = kcache[tk].view(nb, C, KV, D).mean(1)  # [nb, KV, D]
            vb = vcache[tk].view(nb, C, KV, D).mean(1)
            n = torch.arange(nb, device=dev)
            ksel = min(self.bcsa_topk, nb)
            # keep the fp32 [chunk, 24, nb] score block near 200 MB whatever nb is
            step = max(64, min(ql, int(2.0e8) // max(1, 4 * H * nb)))
            for a in range(0, ql, step):
                b = min(ql, a + step)
                sl = slice(start + a, start + b)
                qpos = torch.arange(slen - ql + a, slen - ql + b, device=dev)
                valid = ((n[None, :] + 1) * C - 1) <= (qpos[:, None] - W)  # [t, nb]
                s_idx = (
                    torch.einsum("qhd,nd->qhn", qi_all[sl], kbi).mean(1)
                    * self.idx_scale
                )
                s_idx = s_idx.masked_fill(~valid, float("-inf"))
                top = s_idx.topk(ksel, dim=-1).indices
                sel = torch.zeros_like(valid).scatter_(-1, top, True) & valid
                t = b - a
                sf = (
                    torch.einsum("tkgd,nkd->tkgn", qh[sl].view(t, KV, G, D), kb)
                    * self.scaling
                )
                sf = (
                    sf.reshape(t, H, nb)
                    .float()
                    .masked_fill(~sel[:, None, :], float("-inf"))
                )
                out[sl] = self._merge(sf, lse[sl], o_loc[sl], vb, kv_first=True)
            start += ql
        return out


def _sinkhorn(logits: torch.Tensor, iters: int) -> torch.Tensor:
    """Doubly-stochastic projection of exp(logits) over the last two dims, in fp32."""
    m = torch.exp(logits.float() - logits.float().amax(dim=(-1, -2), keepdim=True))
    for _ in range(iters):
        m = m / m.sum(-1, keepdim=True).clamp_min(1e-9)
        m = m / m.sum(-2, keepdim=True).clamp_min(1e-9)
    return m


class VolundrMHC(nn.Module):
    """Manifold hyper-connections for one layer. The residual is n streams H [tokens, n, d]:

        x_in  = pre . H                                   (pre: [tokens, n])
        H_out = res . H + post (x) (layer(x_in) - x_in)    (res doubly stochastic, post: [tokens, n])

    pre / post / res are static parameters plus input-dependent deltas from `dyn` applied to the
    normalised stream mean, as in the reference implementation.
    """

    def __init__(
        self,
        hidden_size: int,
        n: int,
        iters: int,
        eps: float,
        quant_config=None,
        prefix: str = "",
    ):
        super().__init__()
        self.n, self.iters = n, iters
        self.pre_static = nn.Parameter(torch.empty(n))
        self.post_static = nn.Parameter(torch.empty(n))
        self.res_static = nn.Parameter(torch.empty(n, n))
        # torch's nn.RMSNorm in the reference (plain `weight *`), not the Gemma form the
        # attention/MLP norms use.
        self.norm = RMSNorm(hidden_size, eps=eps)
        self.dyn = ReplicatedLinear(
            hidden_size,
            n + n + n * n,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("dyn", prefix),
        )

    def mix(self, H: torch.Tensor):
        n = self.n
        d = self.dyn(self.norm(H.mean(1)))[0].float()  # [tokens, 2n + n*n]
        pre = self.pre_static.float() + d[..., :n]
        post = self.post_static.float() + d[..., n : 2 * n]
        res = _sinkhorn(
            self.res_static.float() + d[..., 2 * n :].view(-1, n, n), self.iters
        )
        x_in = (pre.to(H.dtype).unsqueeze(-1) * H).sum(1)  # [tokens, d]
        return x_in, post, res

    def combine(
        self,
        H: torch.Tensor,
        post: torch.Tensor,
        res: torch.Tensor,
        delta: torch.Tensor,
    ):
        mixed = torch.einsum("tij,tjd->tid", res.to(H.dtype), H)
        return mixed + post.to(H.dtype).unsqueeze(-1) * delta.unsqueeze(1)

    # ---- fused path (kernels/ops/layernorm/volundr_mhc.py): mix / combine, bit for bit
    def mix_fused(self, H: torch.Tensor, h_mean: torch.Tensor):
        """mix() with H.mean(1) supplied (the previous layer's combine_fused computed it). Returns
        (x_in, post_bf16, res_bf16): only the bf16 casts of post / res are ever consumed."""
        d = self.dyn(self.norm(h_mean))[0]
        return mhc_mix(
            d, self.pre_static, self.post_static, self.res_static, H, self.iters
        )

    def combine_fused(
        self,
        H: torch.Tensor,
        post_bf16: torch.Tensor,
        res_bf16: torch.Tensor,
        y: torch.Tensor,
        x_in: torch.Tensor,
        want_mean: bool,
    ):
        """combine(H, post, res, y - x_in), plus H_out.mean(1) for the next layer's mix_fused."""
        mixed = torch.einsum("tij,tjd->tid", res_bf16, H)
        return mhc_combine(mixed, post_bf16, y, x_in, want_mean)


# Hash constants of the reference Engram implementation; the row index must be bit-identical.
_ENGRAM_PRIMES = (1000003, 1000033, 1000037, 1000039, 1000081)
_ENGRAM_MASK = 0x1FFFFFFFFFFFFFFF


def _engram_rows(parts, rows: int) -> torch.Tensor:
    """Row index of the n-gram whose ids are `parts` (oldest first, int64, -1 = no token).

    Ids are shifted +1 so the no-token sentinel becomes 0, then folded with the rolling-prime hash
    and reduced mod the table size (bit-identical to the reference implementation).
    """
    code = torch.zeros_like(parts[0])
    for j, p in enumerate(parts):
        code = (code * _ENGRAM_PRIMES[j % len(_ENGRAM_PRIMES)] + (p + 1)) & _ENGRAM_MASK
    return code % rows


class VolundrEngram(nn.Module):
    """Hashed n-gram conditional memory (DeepSeek-Engram style), on layers 2 and 22.

    For each order n in engram_orders, the trailing n token ids hash into a row of an embedding
    table; the concatenated rows are RMS-normalised (plain torch-style norm), projected by `value`,
    and gated by a sigmoid computed from the layer
    output. Applied additively AFTER the MLP: out = out + engram(out, ids). Row indices are computed
    once per forward in VolundrModel (both engram layers share them) because they need the
    per-request previous-token cache.
    """

    def __init__(self, config, quant_config=None, prefix: str = ""):
        super().__init__()
        c = _text_cfg(config)
        orders = list(c.engram_orders)
        dim = c.engram_dim
        self.tables = nn.ModuleList([nn.Embedding(c.engram_rows, dim) for _ in orders])
        self.norm = RMSNorm(dim * len(orders), eps=c.rms_norm_eps)
        self.gate = ReplicatedLinear(
            c.hidden_size,
            1,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("gate", prefix),
        )
        self.value = ReplicatedLinear(
            dim * len(orders),
            c.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("value", prefix),
        )

    def forward(
        self, hidden_states: torch.Tensor, row_indices
    ) -> Optional[torch.Tensor]:
        if row_indices is None:
            return None
        e = torch.cat([t(r) for t, r in zip(self.tables, row_indices)], dim=-1)
        e = self.norm(e)
        g = torch.sigmoid(self.gate(hidden_states)[0].float()).to(hidden_states.dtype)
        return g * self.value(e)[0]


class VolundrDecoderLayer(nn.Module):
    """Returns the layer's full output (input + mixer + MLP). sglang's usual fused add-RMSNorm
    protocol, which threads a single residual through the stack, cannot be used: under mHC the
    residual is n streams and the layer sees a mixed `x_in` rather than the stream itself."""

    def __init__(self, config, layer_id: int, quant_config=None, prefix: str = ""):
        super().__init__()
        c = _text_cfg(config)
        kind = _mixer_kind(c, layer_id)  # the tensors' own truth, via mixer_types
        self.is_linear = kind == "kda"
        self.layer_id = layer_id
        if self.is_linear:
            self.linear_attn = VolundrKDA(
                config, layer_id, quant_config, add_prefix("linear_attn", prefix)
            )
        else:
            self.self_attn = VolundrAttention(
                config,
                layer_id,
                quant_config,
                add_prefix("self_attn", prefix),
                bcsa=(kind == "bcsa"),
            )
        self.mlp = VolundrMLP(
            c.hidden_size, c.intermediate_size, quant_config, add_prefix("mlp", prefix)
        )
        self.input_layernorm = GemmaRMSNorm(c.hidden_size, eps=c.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(c.hidden_size, eps=c.rms_norm_eps)
        self.engram = (
            VolundrEngram(config, quant_config, add_prefix("engram", prefix))
            if layer_id in c.engram_layers
            else None
        )

    def forward(
        self,
        positions,
        x,
        forward_batch: ForwardBatch,
        engram_rows=None,
        rope_positions=None,
    ):
        h = self.input_layernorm(x)
        h = (
            self.linear_attn(h, forward_batch)
            if self.is_linear
            else self.self_attn(
                positions, h, forward_batch, rope_positions=rope_positions
            )
        )
        h = x + h
        h = h + self.mlp(self.post_attention_layernorm(h))
        if self.engram is not None:
            delta = self.engram(h, engram_rows)
            if delta is not None:
                h = h + delta
        return h


class VolundrModel(nn.Module):
    def __init__(self, config, quant_config=None, prefix: str = ""):
        super().__init__()
        c = _text_cfg(config)
        self.embed_tokens = VocabParallelEmbedding(
            c.vocab_size, c.hidden_size, prefix=add_prefix("embed_tokens", prefix)
        )
        self.layers = nn.ModuleList(
            [
                VolundrDecoderLayer(
                    config, i, quant_config, add_prefix(f"layers.{i}", prefix)
                )
                for i in range(c.num_hidden_layers)
            ]
        )
        self.streams = max(1, int(c.mhc_streams))
        if self.streams > 1:
            self.mhc = nn.ModuleList(
                [
                    VolundrMHC(
                        c.hidden_size,
                        self.streams,
                        c.mhc_sinkhorn_iters,
                        c.rms_norm_eps,
                        quant_config,
                        add_prefix(f"mhc.{i}", prefix),
                    )
                    for i in range(c.num_hidden_layers)
                ]
            )
        self.norm = GemmaRMSNorm(c.hidden_size, eps=c.rms_norm_eps)
        # The fused mHC kernels are written for four streams; other stream counts use the torch
        # path (VolundrMHC.mix / combine), which is the definition the kernels reproduce.
        self.fused_mhc = self.streams == 4
        # Engram bookkeeping: the n-gram hash needs the 1-2 token ids BEFORE each token, which a
        # ForwardBatch does not carry at decode time. engram_prev[slot] holds the last two token
        # ids of each request, keyed by its mamba state slot: the per-request index the hybrid
        # backend already keeps as a static per-bs buffer inside the decode CUDA graph, refreshed
        # in place each replay with pad rows set to -1. The table has one extra trailing row so
        # that -1 wraps onto a trash row. It is sized from the mamba pool on the first forward
        # (a warmup, never inside a capture).
        self.engram_layers = list(c.engram_layers)
        self.engram_rows_total = c.engram_rows
        self.engram_orders = list(c.engram_orders)
        if self.engram_layers and self.engram_orders != [2, 3]:
            raise ValueError(
                f"Volundr Engram row computation supports orders (2, 3), got {self.engram_orders}"
            )
        self.engram_prev: Optional[torch.Tensor] = None

    def _ensure_engram_prev(self, lin_backend, device) -> torch.Tensor:
        if self.engram_prev is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "the Engram previous-token table was unallocated at CUDA-graph capture; a "
                    "warmup forward must run first"
                )
            slots = lin_backend.req_to_token_pool.mamba_pool.size
            self.engram_prev = torch.full(
                (slots + 1, 2), -1, dtype=torch.int32, device=device
            )
        return self.engram_prev

    def _engram_row_indices(self, input_ids, positions, forward_batch: ForwardBatch):
        """(rows_order2, rows_order3) for every token in the batch, plus the persistent prev-token
        cache update. Returns None when this batch carries no tokens.

        Decode: fully CUDA-graph-safe — reads input_ids and the backend's static mamba-slot buffer
        (both refreshed in place at replay), gathers the stored prevs, and rotates the cache.
        Extend: runs eagerly (see the capture guard); prev ids come from in-batch shifts, falling
        back to the stored cache only for tokens whose predecessors are outside this batch (chunked
        prefill).
        """
        mode = forward_batch.forward_mode
        if mode.is_idle() or input_ids.numel() == 0:
            return None
        backend = get_attn_backend()
        lin = getattr(backend, "linear_attn_backend", None)
        meta = getattr(lin, "forward_metadata", None)
        if meta is None or meta.mamba_cache_indices is None:
            raise RuntimeError(
                "Engram needs the linear-attn backend metadata (mamba slot per request)"
            )
        slots = meta.mamba_cache_indices.long()  # [bs]; pad rows are -1 -> trash row
        prev = self._ensure_engram_prev(lin, input_ids.device)
        R = self.engram_rows_total

        if mode.is_decode():
            stored = prev[slots]  # [B, 2]
            p2, p1 = stored[:, 0].long(), stored[:, 1].long()
            cur = input_ids.long()
            rows2 = _engram_rows((p1, cur), R)
            rows3 = _engram_rows((p2, p1, cur), R)
            prev[slots] = torch.stack([stored[:, 1], input_ids.to(prev.dtype)], dim=-1)
            return rows2, rows3

        if (
            mode.is_extend()
            and not mode.is_target_verify()
            and not mode.is_draft_extend_v2()
        ):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "the Engram extend path reads per-batch metadata that is not stable across "
                    "prefill-graph replays; prefill CUDA graphs must be disabled for Volundr"
                )
            ids = input_ids.long()
            pos = positions.long()
            n = ids.numel()
            qsl = meta.query_start_loc.long()  # [bs + 1]
            ar = torch.arange(n, device=ids.device)
            seg = torch.searchsorted(qsl, ar, right=True) - 1  # token -> request row
            off = ar - qsl[seg]  # offset inside this request's segment
            stored_tok = prev[slots[seg]].long()  # [n, 2] this token's request cache
            # in-batch shifts via one left-padded extension, so the lengths are right for any n
            # (a naive cat([pad2, ids[:-2]]) is length 2, not n, when n == 1)
            ext = torch.cat([ids.new_full((2,), -1), ids])  # [n + 2]
            p1 = torch.where(off >= 1, ext[1 : n + 1], stored_tok[:, 1])
            p2 = torch.where(
                off >= 2,
                ext[:n],
                torch.where(off == 1, stored_tok[:, 1], stored_tok[:, 0]),
            )
            # absolute-position validity: nothing precedes the start of the sequence
            p1 = torch.where(pos >= 1, p1, torch.full_like(p1, -1))
            p2 = torch.where(pos >= 2, p2, torch.full_like(p2, -1))
            rows2 = _engram_rows((p1, ids), R)
            rows3 = _engram_rows((p2, p1, ids), R)
            # cache each request's last two ids for the decode steps that follow. For a 1-token
            # segment the second-last id comes from the stored cache (chunked-prefill continuation)
            # — but only if the sequence actually has a token before this one; a brand-new request
            # with a 1-token prompt must store the no-token sentinel, not the slot's stale content.
            last = qsl[1:] - 1
            seg_len = qsl[1:] - qsl[:-1]
            old = prev[slots].long()
            new_p2 = torch.where(
                seg_len >= 2,
                ids[(last - 1).clamp(min=0)],
                torch.where(pos[last] >= 1, old[:, 1], torch.full_like(old[:, 1], -1)),
            )
            prev[slots] = torch.stack(
                [new_p2.to(prev.dtype), ids[last].to(prev.dtype)], dim=-1
            )
            return rows2, rows3

        raise NotImplementedError(
            f"Engram row computation supports extend and decode, not {mode}"
        )

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        input_ids=None,
        positions=None,
        forward_batch: ForwardBatch = None,
        inputs_embeds=None,
        input_embeds=None,
        rope_positions=None,
        token_ids=None,
        **kw,
    ):
        """Two calling conventions: the text class calls (input_ids, positions, forward_batch);
        general_mm_embed_routine calls (input_ids=None, forward_batch=, input_embeds=, **kwargs) with
        `positions`, `rope_positions` and `token_ids` in kwargs — token_ids being the real ids (image
        spans set to image_token_id) that Engram hashes, since the routine's input_ids are None."""
        if inputs_embeds is None:
            inputs_embeds = input_embeds
        if input_ids is None:
            input_ids = token_ids
        if positions is None:
            positions = forward_batch.positions
        hidden_states = (
            self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        )

        engram_rows = (
            self._engram_row_indices(input_ids, positions, forward_batch)
            if self.engram_layers
            else None
        )

        if self.streams > 1:
            hidden_states = self._forward_mhc(
                positions, hidden_states, forward_batch, engram_rows, rope_positions
            )
        else:
            for layer in self.layers:
                hidden_states = layer(
                    positions,
                    hidden_states,
                    forward_batch,
                    engram_rows=engram_rows,
                    rope_positions=rope_positions,
                )
        return self.norm(hidden_states)

    def _forward_mhc(
        self, positions, hidden_states, forward_batch, engram_rows, rope_positions
    ):
        """The layer stack over n residual streams; returns stream 0 (what the final norm reads)."""
        H = hidden_states.unsqueeze(1).expand(-1, self.streams, -1).contiguous()
        h_mean = H.mean(1) if self.fused_mhc else None
        n_layers = len(self.layers)
        for i, layer in enumerate(self.layers):
            mhc = self.mhc[i]
            if self.fused_mhc:
                x_in, post, res = mhc.mix_fused(H, h_mean)
            else:
                x_in, post, res = mhc.mix(H)
            y = layer(
                positions,
                x_in,
                forward_batch,
                engram_rows=engram_rows,
                rope_positions=rope_positions,
            )
            if self.fused_mhc:
                H, h_mean = mhc.combine_fused(
                    H, post, res, y, x_in, want_mean=i + 1 < n_layers
                )
            else:
                H = mhc.combine(H, post, res, y - x_in)
        return H[:, 0, :]


def _suppressed_token_ids(vocab_size: int) -> list:
    """generation_config.suppress_tokens of the served checkpoint, clipped to the vocabulary."""
    from sglang.srt.runtime_context import get_model
    from sglang.srt.utils.hf_transformers.common import get_generation_config

    model = get_model()
    gen = get_generation_config(
        model.model_path, trust_remote_code=False, revision=model.revision
    )
    ids = getattr(gen, "suppress_tokens", None) or []
    return sorted({int(i) for i in ids if 0 <= int(i) < vocab_size})


class _VolundrLM(nn.Module):
    """The language model + head; vision tensors are dropped at load time here. The two public
    classes below select the text-only model (VolundrForCausalLM) or the vision-serving one
    (VolundrForConditionalGeneration)."""

    # Tensors that are deliberately not loaded, and why. Anything else the checkpoint carries that
    # this class has no parameter for is a bug, and load_weights raises rather than serving a model
    # with a silently missing tower.
    _IGNORED = (".visual.", "visual.")

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        c = _text_cfg(config)
        self.quant_config = quant_config
        self.model = VolundrModel(
            config, quant_config, add_prefix("model.language_model", prefix)
        )
        self.lm_head = ParallelLMHead(
            c.vocab_size,
            c.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
        )
        self.logits_processor = LogitsProcessor(c)
        # Ids the checkpoint's generation_config marks as never-sampled (the padding rows of the
        # embedding and lm_head past the tokenizer's vocabulary). Only next-token logits are
        # masked, as transformers' SuppressTokensLogitsProcessor does; input logprobs are untouched.
        # A device buffer, so the mask is CUDA-graph safe.
        self.register_buffer(
            "suppress_idx",
            torch.tensor(_suppressed_token_ids(c.vocab_size), dtype=torch.long),
            persistent=False,
        )

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def _suppress(self, out):
        if (
            self.suppress_idx.numel()
            and isinstance(out, LogitsProcessorOutput)
            and out.next_token_logits is not None
        ):
            out.next_token_logits.index_fill_(1, self.suppress_idx, float("-inf"))
        return out

    @torch.no_grad()
    def forward(
        self, input_ids, positions, forward_batch: ForwardBatch, inputs_embeds=None
    ):
        hidden_states = self.model(input_ids, positions, forward_batch, inputs_embeds)
        return self._suppress(
            self.logits_processor(input_ids, hidden_states, self.lm_head, forward_batch)
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        stacked = [("gate_up_proj", "gate_proj", 0), ("gate_up_proj", "up_proj", 1)]
        params = dict(self.named_parameters())
        filled, skipped = set(), 0
        for name, w in weights:
            if any(tok in name for tok in self._IGNORED):
                skipped += 1
                continue
            n = name.replace("model.language_model.", "model.").replace(
                "language_model.", "model."
            )
            if n.startswith("model.model."):
                n = n[len("model.") :]
            done = False
            for target, src, shard in stacked:
                if src in n:
                    tn = n.replace(src, target)
                    p = params.get(tn)
                    if p is None:
                        break
                    p.weight_loader(p, w, shard)
                    filled.add(tn)
                    done = True
                    break
            if done:
                continue
            p = params.get(n)
            if p is None:
                raise ValueError(
                    f"checkpoint tensor {name!r} has no parameter in the Volundr text model "
                    f"(mapped to {n!r}); add it to the class or to _IGNORED with a reason"
                )
            getattr(p, "weight_loader", default_weight_loader)(p, w)
            filled.add(n)
        # vision parameters (only present on VolundrForConditionalGeneration) are loaded and
        # verified by that class before it hands the text tensors here
        missing = sorted(n for n in set(params) - filled if not n.startswith("visual."))
        if missing:
            raise ValueError(
                f"{len(missing)} parameters were never loaded, e.g. {missing[:8]} — "
                "the checkpoint and this class disagree"
            )
        return {"loaded": len(filled), "skipped": skipped}


class VolundrForCausalLM(_VolundrLM):
    """Text-only class (architectures: ["VolundrForCausalLM"]): the language model without the
    vision tower."""


class VolundrForConditionalGeneration(_VolundrLM):
    """The full checkpoint: vision tower + merger in front of the Volundr language model. Serve with
    --enable-multimodal; QwenVLImageProcessor turns `image_url` parts into
    <|vision_start|><|image_pad|>...<|vision_end|> spans, the tokens the chat template emits.

    Positions: the model is MRoPE (rope_parameters.mrope_section over the rotary dims). An extend
    batch that carries images feeds forward_batch.mrope_positions [3, N] to the BCSA layers' rotary;
    every other batch (text-only extends, all decodes — after an image the three rows coincide) uses
    the t-row through the plain rotary kernel, so text-only requests are bit-identical to
    VolundrForCausalLM. KV slots, BCSA block boundaries and Engram's prev-token cache are indexed by
    the 1-D token position, never by mrope values.

    Engram ids: sglang overwrites image-token ids with per-image hash values (that is how it finds
    the spans to embed). For the n-gram hash those positions are restored to image_token_id, which
    is what the HF path sees, so the Engram contribution on image tokens matches it.
    Deepstack: the vision config has no deepstack layers (deepstack_visual_indexes = []).
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__(config, quant_config, prefix)
        self.pp_group = get_pp_group()
        vc = getattr(config, "vision_config", None)
        if vc is None:
            raise ValueError(
                "VolundrForConditionalGeneration needs config.vision_config; serve the "
                "text-only view (VolundrForCausalLM) if the vision tower is not wanted"
            )
        if getattr(vc, "deepstack_visual_indexes", None):
            raise NotImplementedError(
                "Volundr's vision config is expected to have no deepstack layers"
            )
        from sglang.srt.models.qwen3_vl import Qwen3VLMoeVisionModel

        c = _text_cfg(config)
        self.use_data_parallel = bool(
            getattr(get_server_args(), "mm_enable_dp_encoder", False)
        )
        if self.use_data_parallel:
            raise NotImplementedError("--mm-enable-dp-encoder is not wired for Volundr")
        self.visual = Qwen3VLMoeVisionModel(
            vc,
            quant_config=None,
            norm_eps=c.rms_norm_eps,
            prefix=add_prefix("model.visual", prefix),
            use_data_parallel=False,
        )
        rp = dict(
            getattr(c, "rope_parameters", None)
            or getattr(c, "rope_scaling", None)
            or {}
        )
        self.is_mrope_enabled = "mrope_section" in rp
        self.deepstack_visual_indexes = []
        self.use_deepstack = {}
        self.image_token_id = config.image_token_id
        self.video_token_id = config.video_token_id

    # ---- hooks sglang's multimodal machinery calls
    def pad_input_ids(self, input_ids, mm_inputs: MultimodalInputs):
        return MultiModalityDataPaddingPatternMultimodalTokens().pad_input_tokens(
            input_ids, mm_inputs
        )

    def get_image_feature(self, items: list) -> torch.Tensor:
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            self.visual.dtype
        )
        grid = torch.cat([item.image_grid_thw for item in items], dim=0)
        assert pixel_values.dim() == 2 and grid.dim() == 2, (
            pixel_values.shape,
            grid.shape,
        )
        return self.visual(pixel_values, grid_thw=grid)

    def get_video_feature(self, items: list) -> torch.Tensor:
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            self.visual.dtype
        )
        grid = torch.cat([item.video_grid_thw for item in items], dim=0)
        return self.visual(pixel_values, grid_thw=grid)

    def _engram_token_ids(
        self, input_ids: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        """input_ids with every multimodal span set back to its placeholder token id."""
        if (
            forward_batch.forward_mode.is_decode()
            or not forward_batch.contains_mm_inputs()
        ):
            return input_ids
        pads_img, pads_vid = [], []
        for mm in forward_batch.mm_inputs or []:
            for it in getattr(mm, "mm_items", None) or []:
                (pads_vid if it.modality == Modality.VIDEO else pads_img).append(
                    int(it.pad_value)
                )
        ids = input_ids.clone()
        for pads, tok in (
            (pads_img, self.image_token_id),
            (pads_vid, self.video_token_id),
        ):
            if pads:
                m = torch.isin(
                    ids, torch.tensor(pads, device=ids.device, dtype=ids.dtype)
                )
                ids = torch.where(m, torch.full_like(ids, tok), ids)
        return ids

    @torch.no_grad()
    def forward(
        self, input_ids, positions, forward_batch: ForwardBatch, inputs_embeds=None
    ):
        rope_positions = None
        if self.is_mrope_enabled:
            mp = forward_batch.mrope_positions
            if mp is not None and mp.ndim == 2:
                if (
                    forward_batch.forward_mode.is_extend()
                    and forward_batch.contains_image_inputs()
                ):
                    rope_positions = mp  # [3, N]: the BCSA layers take the MRoPE path
                else:
                    positions = mp[0]  # rows coincide: plain rotary on the t-row
        token_ids = self._engram_token_ids(input_ids, forward_batch)
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.model,
            multimodal_model=self,
            positions=positions,
            rope_positions=rope_positions,
            token_ids=token_ids,
            use_deepstack=self.use_deepstack,
        )
        return self._suppress(
            self.logits_processor(input_ids, hidden_states, self.lm_head, forward_batch)
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Vision tensors go to self.visual (sglang's VisionAttention names the fused projection
        qkv_proj); everything else through the text loader, which then verifies completeness."""
        params = dict(self.named_parameters())
        text, filled = [], set()
        for name, w in weights:
            if ".visual." in name or name.startswith("visual."):
                n = name.replace("model.visual.", "visual.").replace(
                    "attn.qkv.", "attn.qkv_proj."
                )
                p = params.get(n)
                if p is None:
                    raise ValueError(
                        f"vision tensor {name!r} has no parameter (mapped to {n!r})"
                    )
                getattr(p, "weight_loader", default_weight_loader)(p, w)
                filled.add(n)
                continue
            text.append((name, w))
        vis_missing = sorted(
            n for n in params if n.startswith("visual.") and n not in filled
        )
        if vis_missing:
            raise ValueError(
                f"{len(vis_missing)} vision parameters never loaded, e.g. {vis_missing[:6]}"
            )
        out = _VolundrLM.load_weights(self, text)
        out["visual"] = len(filled)
        return out


EntryClass = [VolundrForConditionalGeneration, VolundrForCausalLM]
