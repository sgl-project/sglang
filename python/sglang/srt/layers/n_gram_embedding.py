import torch
from torch import nn
from torch.nn import Parameter

from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import is_npu

_is_npu = is_npu()

if _is_npu:
    import sgl_kernel_npu  # noqa: F401  Registers torch.ops.npu OE operators.
else:
    from sglang.kernels.ops.embeddings.ngram_embedding import compute_n_gram_ids


class NgramEmbedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        over_embedding_m: int,
        over_embedding_k: int,
        over_embedding_n: int,
        eos_token_id: int,
    ):
        super().__init__()
        assert over_embedding_n > 1, (
            f"over_embedding_n must be > 1, got {over_embedding_n}"
        )
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.over_embedding_m = over_embedding_m
        self.over_embedding_k = over_embedding_k
        self.over_embedding_n = over_embedding_n
        self.eos_token_id = eos_token_id

        use_attn_tp_group = is_dp_attention_enabled()
        self.word_embeder = VocabParallelEmbedding(
            num_embeddings,
            embedding_dim,
            use_attn_tp_group=use_attn_tp_group,
        )
        self.n_grams = (over_embedding_n - 1) * over_embedding_k
        oe_hidden_dim = embedding_dim // (over_embedding_k * (over_embedding_n - 1))
        self.exclusive_oe_embedder_size_sums = torch.zeros(
            [over_embedding_k * (over_embedding_n - 1) + 1],
            dtype=torch.int32,
            device="cuda",
        )
        for i in range(over_embedding_k * (over_embedding_n - 1)):
            self.exclusive_oe_embedder_size_sums[i + 1] = (
                self.exclusive_oe_embedder_size_sums[i]
                + int(over_embedding_m + i * 2 + 1)
            )
        self.oe_embeder = VocabParallelEmbedding(
            num_embeddings=int(self.exclusive_oe_embedder_size_sums[-1]),
            embedding_dim=oe_hidden_dim,
            use_attn_tp_group=use_attn_tp_group,
        )

        self.oe_projection = nn.Parameter(
            torch.empty(
                (over_embedding_n - 1) * over_embedding_k, oe_hidden_dim, embedding_dim
            ),
            requires_grad=False,
        )

        self.oe_mods = torch.zeros(
            [self.over_embedding_n - 1, self.over_embedding_k], dtype=torch.int32
        )
        self.oe_weights = torch.zeros(
            [self.over_embedding_n - 1, self.over_embedding_k, self.over_embedding_n],
            dtype=torch.int32,
        )
        for n in range(2, self.over_embedding_n + 1):
            for k in range(self.over_embedding_k):
                mod = (
                    self.over_embedding_m
                    + 2 * ((n - 2) * self.over_embedding_k + k)
                    + 1
                )
                self.oe_mods[n - 2][k] = mod
                for delta in range(self.over_embedding_n):
                    self.oe_weights[n - 2][k][delta] = pow(num_embeddings, delta, mod)

    def init_buffers(
        self, max_running_requests: int, chunked_prefill_size: int, device: str
    ):
        max_tokens = max(chunked_prefill_size, max_running_requests)
        self.oe_n_gram_ids = torch.zeros(
            [max_tokens, self.n_grams],
            dtype=torch.int32,
            device=device,
        )
        self.exclusive_req_len_sums = torch.zeros(
            max_running_requests + 1, dtype=torch.int32, device=device
        )

    def load_weight(
        self, param: Parameter, weight_name: str, loaded_weight: torch.Tensor
    ):
        if ".embed_tokens." in weight_name:
            param.weight_loader(param, loaded_weight)
        elif "model.ngram_embeddings.embedders." in weight_name:
            index = int(
                weight_name.replace("model.ngram_embeddings.embedders.", "").replace(
                    ".weight", ""
                )
            )
            oe_weight_start = self.exclusive_oe_embedder_size_sums[index]
            oe_weight_end = self.exclusive_oe_embedder_size_sums[index + 1]
            assert oe_weight_end - oe_weight_start == loaded_weight.shape[0], (
                f"{oe_weight_end - oe_weight_start=} {loaded_weight.shape[0]=}"
            )
            tp_start = self.oe_embeder.shard_indices.org_vocab_start_index
            tp_end = self.oe_embeder.shard_indices.org_vocab_end_index
            to_load_start = max(oe_weight_start, tp_start)
            to_load_end = min(oe_weight_end, tp_end)
            if to_load_start < to_load_end:
                src_start = to_load_start - oe_weight_start
                src_end = to_load_end - oe_weight_start
                dest_start = to_load_start - tp_start
                dest_end = to_load_end - tp_start
                self.oe_embeder.weight.data[dest_start:dest_end] = loaded_weight[
                    src_start:src_end
                ]
            else:
                return
        elif "model.ngram_embeddings.post_projs." in weight_name:
            index = int(
                weight_name.replace("model.ngram_embeddings.post_projs.", "").replace(
                    ".weight", ""
                )
            )
            self.oe_projection[index].copy_(loaded_weight.data.t())
        else:
            assert False, f"Unknown ngram embedding weight name: {weight_name}"

    def _compute_n_gram_ids_npu(self, input_ids, forward_batch, batch_size):
        output = self.oe_n_gram_ids[: len(input_ids)]
        if input_ids.numel() == 0:
            return
        if batch_size == 0:
            output.zero_()
            return
        info = forward_batch.ngram_embedding_info
        offsets = torch.arange(len(input_ids), device=input_ids.device)
        requests = torch.searchsorted(
            self.exclusive_req_len_sums[1 : batch_size + 1], offsets, right=True
        )
        valid = requests < batch_size
        requests.clamp_max_(batch_size - 1)
        positions = forward_batch.positions
        rows = forward_batch.req_pool_indices[requests]
        lookback = torch.arange(1 - self.over_embedding_n, 1, device=input_ids.device)
        # The native op stops at negative tokens but has no EOS boundary check.
        # Match the GPU rule with a separate N-token history for each query:
        # keep the current token, even if it is EOS, and mask historical EOS to
        # -1. A shared EOS mask would also discard the current EOS's own n-gram.
        # Each history row is a temporary request that computes only its last
        # token; the original request's token table is not modified.
        #
        # 8192 limits the number of temporary requests per call, not the history
        # length. Each core stages all request metadata in UB: 4-byte lengths,
        # 8-byte row indices, and 4-byte column starts per request. For LongCat-2.0
        # (N=5, K=4), the other buffers use 512 bytes after 32-byte alignment.
        # 8192 requests therefore use 128.5 KiB, leaving 63.5 KiB of a 192 KiB UB.
        for start in range(0, len(input_ids), 8192):
            end = min(start + 8192, len(input_ids))
            size = end - start
            chunk_positions = torch.where(valid[start:end], positions[start:end], 0)
            cols = chunk_positions[:, None] + lookback
            context = info.token_table[rows[start:end, None], cols.clamp_min(0)]
            context = context.masked_fill(
                (cols < 0) | ((context == self.eos_token_id) & (lookback != 0)), -1
            ).contiguous()
            result = torch.ops.npu.compute_n_gram_ids(
                self.oe_weights,
                self.oe_mods,
                self.exclusive_oe_embedder_size_sums,
                context[:, -1].contiguous(),
                torch.arange(1, size + 1, dtype=torch.int32, device=context.device),
                context,
                torch.arange(size, dtype=torch.int64, device=context.device),
                torch.full(
                    (size,),
                    self.over_embedding_n - 1,
                    dtype=torch.int32,
                    device=context.device,
                ),
                batch_size=size,
                oe_n=self.over_embedding_n,
                oe_k=self.over_embedding_k,
                max_context_len=self.over_embedding_n,
            )
            output[start:end].copy_(result)

    def forward(self, input_ids: torch.Tensor, forward_batch: ForwardBatch):
        if (
            forward_batch.forward_mode.is_extend()
            or forward_batch.forward_mode.is_decode()
        ):
            ngram_embedding_info = forward_batch.ngram_embedding_info
            # NGRAM_BS_GUARD: the ngram_info arrays can be shorter than
            # forward_batch.batch_size in mixed/overlap batches; drive the req loop
            # off the array length so the kernel never reads column_starts/req_lens
            # out of bounds (cudaErrorIllegalAddress).
            _ng_bs = min(
                forward_batch.batch_size,
                ngram_embedding_info.req_lens.shape[0],
                ngram_embedding_info.column_starts.shape[0],
            )
            torch.cumsum(
                ngram_embedding_info.req_lens[:_ng_bs],
                dim=0,
                dtype=torch.int32,
                out=self.exclusive_req_len_sums[1 : 1 + _ng_bs],
            )
            if _is_npu:
                self._compute_n_gram_ids_npu(input_ids, forward_batch, _ng_bs)
            else:
                compute_n_gram_ids(
                    ne_n=self.over_embedding_n,
                    ne_k=self.over_embedding_k,
                    ne_weights=self.oe_weights,
                    ne_mods=self.oe_mods,
                    tokens=input_ids.to(torch.int32),
                    exclusive_ne_embedder_size_sums=self.exclusive_oe_embedder_size_sums,
                    exclusive_req_len_sums=self.exclusive_req_len_sums[: _ng_bs + 1],
                    ne_token_table=ngram_embedding_info.token_table,
                    row_indices=forward_batch.req_pool_indices[:_ng_bs],
                    column_starts=ngram_embedding_info.column_starts[:_ng_bs],
                    n_gram_ids=self.oe_n_gram_ids[: len(input_ids)],
                    eos_token_id=self.eos_token_id,
                )

        # [13, seq_len, hidden_dim]
        all_hidden_states = torch.empty(
            [self.n_grams + 1, len(input_ids), self.embedding_dim],
            dtype=self.oe_projection.dtype,
            device=input_ids.device,
        )
        all_hidden_states[0] = self.word_embeder(input_ids)
        # oe_hidden_states: [12, seq_len, hidden_dim / 12]
        oe_hidden_states = self.oe_embeder(
            self.oe_n_gram_ids[: len(input_ids)].permute(1, 0).contiguous()
        )
        torch.bmm(oe_hidden_states, self.oe_projection, out=all_hidden_states[1:])
        return all_hidden_states.mean(dim=0)
