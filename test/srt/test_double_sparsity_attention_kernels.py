import random
import unittest

import torch
import torch.nn.functional as F

from sglang.srt.layers.triton_attention.sparse_decode_attention import decode_sparse_attention_fwd
    

def torch_sparse_fwd(
    q,
    k_buffer,
    v_buffer,
    q_label,
    k_label_buffer,
    Req_to_tokens,
    B_Seqlen,
    max_len_in_batch,
    sm_scale,
    logit_cap,
    heavy_token_num,
):
    """
    PyTorch implementation of the sparse attention mechanism.
    """
    batch_size = q.shape[0]
    num_heads = q.shape[1]
    d_model = q.shape[2]
    d_label = q_label.shape[2]
    device = q.device
    dtype = q.dtype

    # Initialize approximate attention output
    att_out_approx = torch.full(
        (num_heads, batch_size, max_len_in_batch), float('-inf'), device=device, dtype=dtype
    )

    # Step 1: Compute approximate attention
    for b in range(batch_size):
        seq_len = B_Seqlen[b].item()
        req_idx = b
        tokens = Req_to_tokens[req_idx, :seq_len]  # [seq_len]
        k_labels = k_label_buffer[tokens]  # [seq_len, num_heads, d_label]
        q_labels = q_label[b]  # [num_heads, d_label]

        for h in range(num_heads):
            ql = q_labels[h]  # [d_label]
            kl = k_labels[:, h, :]  # [seq_len, d_label]
            att_scores = torch.matmul(kl, ql.unsqueeze(-1)).squeeze(-1)  # [seq_len]
            att_scores *= sm_scale
            if logit_cap > 0.0:
                att_scores = logit_cap * torch.tanh(att_scores / logit_cap)
            att_out_approx[h, b, :seq_len] = att_scores
            
    # Step 2: Top-K token selection
    # topk_values, topk_indices = torch.sort(att_out_approx, dim=-1, descending=True)
    # topk_indices = topk_indices[:, :, :heavy_token_num]  # [num_heads, batch_size, heavy_token_num]
    topk_indices = torch.topk(att_out_approx, k=heavy_token_num, dim=-1).indices
    Req_to_tokens_topk = torch.zeros_like(topk_indices)
    for b in range(batch_size):
        seq_len = B_Seqlen[b].item()
        req_idx = b
        tokens = Req_to_tokens[req_idx, :seq_len]  # [seq_len]
        for h in range(num_heads):
            indices = topk_indices[h, b]  # [heavy_token_num]
            selected_tokens = tokens[indices]
            Req_to_tokens_topk[h, b] = selected_tokens
            
    # Step 3: Compute full attention over selected tokens
    att_out = torch.full(
        (num_heads, batch_size, heavy_token_num), float('-inf'), device=device, dtype=dtype
    )
    for b in range(batch_size):
        seq_len_topk = min(B_Seqlen[b].item(), heavy_token_num)
        q_batch = q[b]  # [num_heads, d_model]
        for h in range(num_heads):
            qh = q_batch[h]  # [d_model]
            selected_tokens = Req_to_tokens_topk[h, b, :seq_len_topk]  # [seq_len_topk]
            k_selected = k_buffer[selected_tokens, h, :]  # [seq_len_topk, d_model]
            att_scores = torch.matmul(k_selected, qh.unsqueeze(-1)).squeeze(-1)  # [seq_len_topk]
            att_scores *= sm_scale
            if logit_cap > 0.0:
                att_scores = logit_cap * torch.tanh(att_scores / logit_cap)
            att_out[h, b, :seq_len_topk] = att_scores
            
    # Step 4: Compute the final output
    o = torch.zeros(batch_size, num_heads, d_model, device=device, dtype=dtype)
    for b in range(batch_size):
        seq_len_topk = min(B_Seqlen[b].item(), heavy_token_num)
        for h in range(num_heads):
            att_scores = att_out[h, b, :seq_len_topk]  # [seq_len_topk]
            att_scores = att_scores - torch.max(att_scores)  # For numerical stability
            att_probs = F.softmax(att_scores, dim=-1)  # [seq_len_topk]
            selected_tokens = Req_to_tokens_topk[h, b, :seq_len_topk]  # [seq_len_topk]
            v_selected = v_buffer[selected_tokens, h, :]  # [seq_len_topk, d_model]
            o[b, h, :] = torch.matmul(att_probs, v_selected)  # [d_model]
            
    return o


class TestDoubleSparseDecodeAttention(unittest.TestCase):
    
    def _set_all_seeds(self, seed):
        """Set all random seeds for reproducibility."""
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    def setUp(self):
        # Set seeds before each test method
        self._set_all_seeds(42)

    def test_decode_sparse_attention_fwd(self):
        # Initialization parameters
        batch_size = 5
        num_heads = 32
        seq_len_list = [1024, 512, 4096, 128, 64]
        d_model = 128
        d_label = 16  # Reduced dimension for labels
        heavy_token_num = 32  # Number of top tokens to select
        max_len_in_batch = max(seq_len_list)
        total_token_num = sum(seq_len_list) + 10

        sm_scale = 1.0 / (d_label ** 0.5)
        logit_cap = 0.0  # No logit cap for simplicity

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = torch.float16  # Use float16 as in your code

        # 1. Initialize k_buffer and v_buffer
        k_buffer = torch.randn(total_token_num, num_heads, d_model, device=device, dtype=dtype)
        v_buffer = torch.randn(total_token_num, num_heads, d_model, device=device, dtype=dtype)

        # 2. Initialize B_Seqlen
        B_Seqlen = torch.tensor(seq_len_list, device=device, dtype=torch.long)  # [batch_size]
        assert B_Seqlen.sum().item() <= total_token_num

        # 3. Initialize Req_to_tokens
        num_requests = batch_size
        Req_to_tokens = torch.zeros(num_requests, max_len_in_batch, device=device, dtype=torch.long)
        token_offset = 0
        for req_idx in range(num_requests):
            seq_len = B_Seqlen[req_idx].item()
            Req_to_tokens[req_idx, :seq_len] = torch.arange(token_offset, token_offset + seq_len, device=device, dtype=torch.long)
            token_offset += seq_len

        # Initialize q
        q = torch.randn(batch_size, num_heads, d_model, device=device, dtype=dtype)

        # Create projection matrices
        projection_matrix = torch.randn(d_model, d_label, device=device, dtype=dtype)

        # Compute q_label and k_label_buffer
        q_label = torch.einsum('bhd,df->bhf', q, projection_matrix)  # [batch_size, num_heads, d_label]
        k_label_buffer = torch.einsum('lhd,df->lhf', k_buffer, projection_matrix)  # [total_token_num, num_heads, d_label]

        # Prepare output tensors
        o_triton = torch.zeros(batch_size, num_heads, d_model, device=device, dtype=dtype)

        # Run your Triton implementation
        decode_sparse_attention_fwd(
            q,
            k_buffer,
            v_buffer,
            o_triton,
            q_label,
            k_label_buffer,
            Req_to_tokens,
            B_Seqlen,
            max_len_in_batch,
            sm_scale,
            logit_cap,
            heavy_token_num,
        )
        
        # Run the PyTorch implementation
        o_torch = torch_sparse_fwd(
            q,
            k_buffer,
            v_buffer,
            q_label,
            k_label_buffer,
            Req_to_tokens,
            B_Seqlen,
            max_len_in_batch,
            sm_scale,
            logit_cap,
            heavy_token_num,
        )

        # Compare the outputs
        diff = torch.norm(o_triton.float() - o_torch.float()) / torch.norm(o_torch.float())
        print("Relative difference between Triton and PyTorch outputs:", diff.item())
        
        #NOTE(Andy): The precision diff may come from online softmax
        assert diff < 0.01

if __name__ == "__main__":
    unittest.main()