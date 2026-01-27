import torch
from torch import nn
from torch.nn import Parameter

from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_kernel import compute_n_gram_ids

from sglang.srt.server_args import get_global_server_args


class NgramEmbedding(torch.nn.Module):

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 over_embedding_m: int,
                 over_embedding_k: int,
                 over_embedding_n: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.over_embedding_m = over_embedding_m
        self.over_embedding_k = over_embedding_k
        self.over_embedding_n = over_embedding_n

        self.word_embeder = VocabParallelEmbedding(
            num_embeddings,
            embedding_dim,
            enable_tp=is_dp_attention_enabled(),
        )
        self.n_grams = (over_embedding_n - 1) * over_embedding_k
        oe_hidden_dim = embedding_dim // (over_embedding_k * (over_embedding_n - 1))
        self.exclusive_oe_embeder_size_sums = torch.zeros([over_embedding_k * (over_embedding_n - 1) + 1],
                                                          dtype=torch.int32,
                                                          device = 'cuda')
        for i in range(over_embedding_k * (over_embedding_n - 1)):
            self.exclusive_oe_embeder_size_sums[i + 1] = self.exclusive_oe_embeder_size_sums[i] + int(over_embedding_m + i * 2 + 1)
        self.oe_embeder = VocabParallelEmbedding(
                num_embeddings=self.exclusive_oe_embeder_size_sums[-1],
                embedding_dim=oe_hidden_dim,
                enable_tp=is_dp_attention_enabled(),
            )

        self.oe_projection = nn.Parameter(
            torch.empty((over_embedding_n - 1) * over_embedding_k, oe_hidden_dim, embedding_dim),
            requires_grad=False
        )

        self.oe_mods = torch.zeros([self.over_embedding_n-1, self.over_embedding_k], dtype=torch.int32)
        self.oe_weights = torch.zeros([self.over_embedding_n-1, self.over_embedding_k, self.over_embedding_n], dtype=torch.int32)
        for n in range(2, self.over_embedding_n + 1):
            for k in range(self.over_embedding_k):
                mod = self.over_embedding_m + 2 * ((n - 2) * self.over_embedding_k + k) + 1
                self.oe_mods[n-2][k] = mod
                for delta in range(self.over_embedding_n):
                    self.oe_weights[n-2][k][delta] = pow(num_embeddings, delta, mod)
        server_args = get_global_server_args()
        device = server_args.device
        self.oe_n_gram_ids = torch.zeros([server_args.chunked_prefill_size, self.n_grams],
                                         dtype=torch.int32, device=device)
        self.exclusive_req_len_sums = torch.zeros(server_args.max_running_requests + 1,
                                                  dtype=torch.int32, device=device)

    def load_weight(self, param: Parameter, weight_name: str, loaded_weight: torch.Tensor):
        if '.embed_tokens.' in weight_name:
            param.weight_loader(param, loaded_weight)
        elif 'model.ngram_embeddings.embedders.' in weight_name:
            index = int(weight_name.replace('model.ngram_embeddings.embedders.', '').replace('.weight', ''))
            oe_weight_start = self.exclusive_oe_embeder_size_sums[index]
            oe_weight_end = self.exclusive_oe_embeder_size_sums[index + 1]
            assert oe_weight_end - oe_weight_start == loaded_weight.shape[0], f'{oe_weight_end - oe_weight_start=} {loaded_weight.shape[0]=}'
            tp_start = self.oe_embeder.shard_indices.org_vocab_start_index
            tp_end = self.oe_embeder.shard_indices.org_vocab_end_index
            to_load_start = max(oe_weight_start, tp_start)
            to_load_end = min(oe_weight_end, tp_end)
            if to_load_start < to_load_end:
                src_start = to_load_start - oe_weight_start
                src_end = to_load_end - oe_weight_start
                dest_start = to_load_start - tp_start
                dest_end = to_load_end - tp_start
                self.oe_embeder.weight.data[dest_start:dest_end] = loaded_weight[src_start:src_end]
            else:
                return
        elif 'model.ngram_embeddings.post_projs.' in weight_name:
            index = int(weight_name.replace('model.ngram_embeddings.post_projs.', '').replace('.weight', ''))
            self.oe_projection[index].copy_(loaded_weight.data.t())
        else:
            assert False, f'Unknown ngram embedding weight name: {weight_name}'

    def forward(self,
                input_ids: torch.Tensor,
                forward_batch: ForwardBatch):
        if forward_batch.forward_mode.is_extend() or forward_batch.forward_mode.is_decode():
            torch.cumsum(forward_batch.ne_req_lens, dim=0, dtype=torch.int32,
                         out=self.exclusive_req_len_sums[1:1 + forward_batch.batch_size])
            compute_n_gram_ids(
                ne_n=self.over_embedding_n,
                ne_k=self.over_embedding_k,
                ne_weights=self.oe_weights,
                ne_mods=self.oe_mods,
                tokens=input_ids.to(torch.int32),
                exclusive_ne_embeder_size_sums=self.exclusive_oe_embeder_size_sums,
                exclusive_req_len_sums=self.exclusive_req_len_sums[:forward_batch.batch_size + 1],
                ne_token_table=forward_batch.ne_token_table,
                row_indices=forward_batch.req_pool_indices,
                column_starts=forward_batch.ne_column_starts,
                n_gram_ids=self.oe_n_gram_ids[:forward_batch.batch_size]
            )

        # [13, seq_len, hidden_dim]
        all_hidden_states = torch.empty([self.n_grams + 1, len(input_ids), self.embedding_dim],
                                        dtype=self.oe_projection.dtype, device=input_ids.device)
        all_hidden_states[0] = self.word_embeder(input_ids)
        # oe_hidden_states: [12, seq_len, hidden_dim / 12]
        oe_hidden_states = self.oe_embeder(self.oe_n_gram_ids[:len(input_ids)].permute(1, 0).contiguous())
        torch.bmm(oe_hidden_states, self.oe_projection, out=all_hidden_states[1:])
        return all_hidden_states.mean(dim=0)
