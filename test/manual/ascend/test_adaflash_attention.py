import math
import unittest
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401
from sglang.srt.hardware_backend.npu.attention.ascend_backend import AscendAttnBackend
from sglang.srt.hardware_backend.npu.utils import init_npu_backend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout

init_npu_backend()


class TestAdaFlashRaggedAttention(unittest.TestCase):
    def test_mixed_query_lengths_across_kv_pages(self):
        torch.manual_seed(23)
        page_size, head_dim, heads = 128, 128, 2
        prefixes = [120, 129, 254]
        lengths = [1, 7, 16]
        total = sum(lengths)
        # Three disjoint pages per request; the final request crosses page 2.
        key = torch.randn(9, page_size, 1, head_dim, dtype=torch.bfloat16)
        value = torch.randn_like(key)
        query = torch.randn(total, heads, head_dim, dtype=torch.bfloat16)
        key_npu, value_npu = key.npu(), value.npu()
        pool = SimpleNamespace(
            get_key_buffer=lambda _: key_npu,
            get_value_buffer=lambda _: value_npu,
        )
        backend = SimpleNamespace(
            graph_mode=False,
            use_mla=False,
            use_paged_mtp=False,
            use_quant_flash_attn=False,
            use_flash_mla=False,
            dense_mla_dcp_target=False,
            use_fias_v2_bsnd=False,
            is_hybrid_swa=False,
            page_size=page_size,
            speculative_num_draft_tokens=16,
            token_to_kv_pool=pool,
            mtp_mask=torch.ones((2048, 2048), dtype=torch.bool).triu(1).npu(),
            forward_metadata=SimpleNamespace(
                block_tables=torch.arange(9, dtype=torch.int32).reshape(3, 3).npu(),
                seq_lens_cpu_int=torch.tensor(
                    [p + q for p, q in zip(prefixes, lengths)], dtype=torch.int32
                ),
            ),
        )
        layer = SimpleNamespace(
            layer_id=0,
            tp_q_head_num=heads,
            tp_k_head_num=1,
            tp_v_head_num=1,
            qk_head_dim=head_dim,
            v_head_dim=head_dim,
            scaling=1 / math.sqrt(head_dim),
            sliding_window_size=-1,
            attn_type=AttentionType.DECODER,
        )
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=lengths, device=torch.device("npu"), grid=[total]
        )
        batch = SimpleNamespace(
            ragged_verify_layout=layout,
            batch_size=3,
            global_num_token_non_padded_cpu=total,
            forward_mode=ForwardMode.TARGET_VERIFY,
        )
        actual = (
            AscendAttnBackend.forward_mtp(
                backend, query.npu(), None, None, layer, batch, save_kv_cache=False
            )
            .cpu()
            .float()
            .reshape(total, heads, head_dim)
        )
        expected, offset = [], 0
        for row, (prefix, length) in enumerate(zip(prefixes, lengths)):
            k = (
                key[row * 3 : row * 3 + 3]
                .reshape(-1, head_dim)[: prefix + length]
                .float()
            )
            v = (
                value[row * 3 : row * 3 + 3]
                .reshape(-1, head_dim)[: prefix + length]
                .float()
            )
            q = query[offset : offset + length].float().transpose(0, 1)
            scores = (q @ k.T) * layer.scaling
            allowed = torch.arange(prefix + length)[None, :] <= (
                prefix + torch.arange(length)[:, None]
            )
            scores.masked_fill_(~allowed.unsqueeze(0), float("-inf"))
            expected.append((scores.softmax(-1) @ v).transpose(0, 1))
            offset += length
        torch.testing.assert_close(actual, torch.cat(expected), rtol=0.025, atol=0.025)


if __name__ == "__main__":
    unittest.main()
