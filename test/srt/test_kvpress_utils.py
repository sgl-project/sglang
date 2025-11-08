import unittest

import torch
from torch import nn

from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.kvpress_utils import SnapKVPress
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner, ServerArgs
from sglang.test.test_utils import CustomTestCase

BATCH_SIZE = 4
N_CTX = 100
N_HEAD = 8
HEAD_DIM = 128

"""
This test suite includes

1.
the basic test to test KVPress functionality only,
We need to call an attention module, with input of QKV,
where all of QKV are 4D input [batch_size, n_heads, n_ctx, head_dim]
and hidden_dim = n_heads * head_dim

In KVPress, the arguments of `compress` method includes
* `hidden_states`: the attention input before projection, shape [batch_size, n_ctx, hidden_dim]
* `key`: the key after calling the attention, shape [batch_size, n_heads, n_ctx, head_dim]
* `outputs`: includes the hidden states output [batch_size, n_ctx, hidden_dim], and the attention score

The hidden_states is passed to q_proj of the layer

To conclude, to call KVPress compression, we need
* the unsplit input before QKV projections
* the attention module, taking the input, do the projection, split heads, do the computation, and return KV
* the output of the attention, where heads are merged

To implement this attention module for this test,
we use MultiheadAttention from Pytorch with some modification of the projection layers here


2.
TODO: We need to call RadixAttention later


PS: Multiple ratios should be tested
"""


def split_head(tensor, batch_size, n_ctx, num_heads, head_dim):
    # reshape to [batch_size, n_ctx, n_heads, head_dim]
    tensor = tensor.view(batch_size, n_ctx, num_heads, head_dim)

    # then usually we transpose to [batch_size, n_heads, n_ctx, head_dim]
    return tensor.transpose(1, 2)


def merge_head(tensor, batch_size, n_ctx, num_heads, head_dim):
    # tensor: [batch_size, num_heads, n_ctx, head_dim]
    tensor = tensor.transpose(
        1, 2
    ).contiguous()  # -> [batch_size, n_ctx, num_heads, head_dim]
    return tensor.view(batch_size, n_ctx, num_heads * head_dim)


# the MHA based on Pytorch MultiheadAttention, specified for KVPress scenario
class MultiheadAttention4DConfig:
    num_heads = N_HEAD
    num_attention_heads = N_HEAD
    num_key_value_heads = N_HEAD
    head_dim = HEAD_DIM
    hidden_dim = HEAD_DIM * N_HEAD


class MultiheadAttention4D(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = MultiheadAttention4DConfig()
        self.head_dim = self.config.head_dim  # for KVPress API
        self._prepare_qkv()

    def _prepare_qkv(self):
        self.q_proj = nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        self.k_proj = nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        self.v_proj = nn.Linear(self.config.hidden_dim, self.config.hidden_dim)

        # in Pytorch MultiheadAttention, embed_dim is the hidden_dim before split
        self.mha = nn.MultiheadAttention(
            embed_dim=self.config.hidden_dim,
            num_heads=self.config.num_heads,
            batch_first=True,  # to fit our shape here [batch_size, n_ctc, hidden_dim]
        )

    def forward(
        self, hidden_states, batch_size, n_ctx, num_heads, head_dim, attn_mask=None
    ):
        self.q = self.q_proj(hidden_states)
        self.k = self.k_proj(hidden_states)

        self.v = self.v_proj(hidden_states)

        self.q_split = split_head(self.q, batch_size, n_ctx, num_heads, head_dim)
        self.k_split = split_head(self.k, batch_size, n_ctx, num_heads, head_dim)
        self.v_split = split_head(self.v, batch_size, n_ctx, num_heads, head_dim)

        out, attn_weights = self.mha(
            self.q,
            self.k,
            self.v,
            attn_mask=attn_mask,
            average_attn_weights=False,  # KVPress score computing needs attn weights per head
        )
        return out, attn_weights


class TestKVPressUtils(CustomTestCase):
    def setUp(self):
        self._init_args()

        # input hidden states are needed for compression, pipeline should start before qkv projection
        self.hidden_states = torch.rand(
            [self.batch_size, self.n_ctx, self.num_heads * self.head_dim]
        )
        self.attention = MultiheadAttention4D()

        # TODO: fix this in RadixAttention test
        # self.set_up_forward_batch()
        # run reference forward on original kv
        self.output_ref, self.attn_weights_ref = self.attention(
            hidden_states=self.hidden_states,
            batch_size=self.batch_size,
            n_ctx=self.n_ctx,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )

    def _init_args(self):
        self.batch_size = BATCH_SIZE
        self.num_heads = N_HEAD
        self.n_ctx = N_CTX
        self.head_dim = HEAD_DIM

    def _init_model_runner(self, page_size=1):
        # TODO: fix this in RadixAttention test
        self.model_runner = ModelRunner(
            # initialize the model runner
        )
        self.backend = FlashInferAttnBackend(self.model_runner)
        self.model_runner.model_config.num_attention_heads = self.num_heads

    def set_up_forward_batch(self):
        self._init_model_runner()
        # constructing the dummy ForwordBatch for prefill
        self.batch_size = BATCH_SIZE
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # For prefill, we process the entire context length
        total_len = N_CTX

        # Set up cache locations - for prefill we write to the whole range
        out_cache_start = 0
        out_cache_end = total_len

        self.forward_batch = ForwardBatch(
            batch_size=self.batch_size,
            input_ids=torch.randint(
                0, 100, (self.batch_size, total_len), device=self.device
            ),  # changed from decode_len
            out_cache_loc=torch.tensor(
                [out_cache_start, out_cache_end], device=self.device
            ),
            seq_lens_sum=self.batch_size * total_len,
            forward_mode=ForwardMode.EXTEND,  # only prefill phase compresses the KV
            req_pool_indices=torch.arange(self.batch_size, device=self.device),
            seq_lens=torch.tensor([total_len] * self.batch_size, device=self.device),
            seq_lens_cpu=torch.tensor([total_len] * self.batch_size, device="cpu"),
            attn_backend=self.backend,
        )

    def test_snapkvpress(self):
        snap_kv = SnapKVPress(compression_ratio=0.5)

        # compress the kv based on current setting
        k_compressed, v_compressed = snap_kv.compress(
            # need to use the RadixAttention here to simulate
            # we can not construct a fake module during runtime
            module=self.attention,
            hidden_states=self.hidden_states,
            keys=self.attention.k_split,
            values=self.attention.v_split,
            attentions=self.attn_weights_ref,
            kwargs=None,
        )
        # forward the q and compressed kv
        n_ctx_compressed = k_compressed.shape[-2]
        k = merge_head(
            k_compressed,
            self.batch_size,
            n_ctx_compressed,
            self.num_heads,
            self.head_dim,
        )
        v = merge_head(
            v_compressed,
            self.batch_size,
            n_ctx_compressed,
            self.num_heads,
            self.head_dim,
        )
        self.output_compressed, attn_weights = self.attention.mha(
            self.attention.q,  # reuse the query, and use the compressed kv
            k,
            v,
            attn_mask=None,
            average_attn_weights=False,  # KVPress score computing needs attn weights per head
        )
        # compare the difference with reference output
        # this works as a reference of compression result for RadixAttention tests later
        diff = self.output_ref - self.output_compressed
        self.diff_sum = torch.sum(diff)


if __name__ == "__main__":
    unittest.main()
