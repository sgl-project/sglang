from dataclasses import dataclass
#from sglang.srt.layers.attention import cudnn_backend
import torch
import logging
import math
import cudnn
import time

@dataclass
class InputParameters:
    num_token = 30
    num_heads = 4
    head_size = 128
    max_total_num_tokens = 300
    max_num_reqs = 100
    max_context_lenght = 300
    num_seqs = 10

@dataclass
class InputParametersLarge:
    num_token = 100
    num_heads = 64
    head_size = 128
    max_total_num_tokens = 10000
    max_num_reqs = 500
    max_context_lenght = 10000
    num_seqs = 100


class CuDNNBackend():
    def __init__(self):
        super().__init__()
        self.forward_metadata = None

    def _run_sdpa_forward_extend(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        scaling=None,
        enable_gqa=False,
        causal=False,
    ):
        """Run the extend forward by using torch native sdpa op.

        Args:
            query: [num_tokens, num_heads, head_size]
            output: [num_tokens, num_heads, head_size]
            k_cache: [max_total_num_tokens, num_heads, head_size]
            v_cache: [max_total_num_tokens, num_heads, head_size]
            req_to_token: [max_num_reqs, max_context_len]
            req_pool_indices: [num_seqs]
            seq_lens: [num_seqs]
            extend_prefix_lens: [num_seqs]
            extend_seq_lens: [num_seqs]
            scaling: float or None
            enable_gqa: bool
            causal: bool

        Returns:
            output: [num_tokens, num_heads, head_size]
        """
        

        assert seq_lens.shape[0] == extend_prefix_lens.shape[0]
        assert seq_lens.shape[0] == extend_seq_lens.shape[0]

        start_time =time.perf_counter()

        # B = batch_size
        B = seq_lens.shape[0]
        assert query.shape[0] == extend_seq_lens.sum(), (
            f"query.shape[0] = {query.shape[0]}, but sum(extend_seq_lens) = {extend_seq_lens.sum()}"
        )


        # how many tokens can store in KV cache
        max_seq_len = k_cache.shape[0]

        # 1) Reshape the multi-token query to [B, max_new_tokens, H, D] in a padded fashion
        H = query.shape[1]
        D = query.shape[2]
        max_new_tokens = extend_seq_lens.max()

        padded_query = query.new_zeros((B, max_new_tokens, H, D))

        # Fill in each sequence's slice
        offset = 0
        for i in range(B):
            length_i = extend_seq_lens[i].item()
            padded_query[i, :length_i, :, :] = query[offset : offset + length_i, :, :]
            offset += length_i

        # [B, H, max_new_tokens, D]
        padded_query = padded_query.movedim(2, 1)

        
        # 2) Build CuDNN pygraph and define input tensors
        # TODO: determine data type
        graph = cudnn.pygraph(
            io_data_type=cudnn.data_type.HALF,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
        )
        print(graph)

        # query contains num_tokens queries batched togather
        q_gpu = padded_query
        q = graph.tensor_like(q_gpu)

        # heads, tokens, head size
        # The tokens of queries are indexed by req_to_token
        s, h, d = k_cache.shape
        # Block Size of Paged Cache, 1 since only one token per block
        b = 1

        # 3) Reshape k_cache, v_cache into container shapes for “paged” attention
        # container: num_blocks, num heads, tokens_per_block, dim
        # TODO: permute for correctness
        container_k_gpu = k_cache.view(s,h,b,d)
        print('cache shape: ',container_k_gpu.shape)
        container_v_gpu = v_cache.view(s,h,b,d)

        container_k = graph.tensor_like(container_k_gpu)
        container_v = graph.tensor_like(container_v_gpu)

        # 4) Build the page table
        # only want prefix + the newly added tokens for each sequence
        # Then pad it to the maximum across the batch
        max_ctx_len = (extend_prefix_lens + extend_seq_lens).max().item()
        list_req_tokens = []
        for i in range(B):
            total_len = (extend_prefix_lens[i] + extend_seq_lens[i]).item()
            row_i = req_to_token[req_pool_indices[i], :total_len]
            # Pad up to max_ctx_len
            padded_indices = row_i.new_zeros(max_ctx_len)
            padded_indices[:total_len] = row_i
            list_req_tokens.append(padded_indices)
        
        # Now stack into a single [B, max_ctx_len]
        per_req_tokens = torch.stack(list_req_tokens, dim=0)

        # reshape to [B, 1, max_ctx_len, 1]
        page_table_k_gpu = per_req_tokens.view(per_req_tokens.shape[0],1,per_req_tokens.shape[1],1)
        print("paged table k shape: ",page_table_k_gpu.shape)
        page_table_v_gpu = per_req_tokens.view(per_req_tokens.shape[0],1,per_req_tokens.shape[1],1)
        print("page table v shape: ",page_table_v_gpu.shape)
        page_table_k = graph.tensor_like(page_table_k_gpu)
        page_table_v = graph.tensor_like(page_table_v_gpu)

        # 5) Sequence lengths
        seq_lens_kv = (extend_prefix_lens + extend_seq_lens).view(B, 1, 1, 1)
        seq_lens_q = extend_seq_lens.view(B, 1, 1, 1)

        seq_len_q_tensor_info = graph.tensor_like(seq_lens_q)
        seq_len_kv_tensor_info = graph.tensor_like(seq_lens_kv)

        # 6) Build the SDPA operation
        o, _ = graph.sdpa(
            name="sdpa",
            q=q,
            k=container_k,  # Container K: non contiguous container with K blocks
            v=container_v,  # Container V: non contiguous container with V blocks
            is_inference=True,
            attn_scale=scaling,
            use_causal_mask=causal,
            use_padding_mask=True,
            seq_len_q=seq_len_q_tensor_info,
            seq_len_kv=seq_len_kv_tensor_info,
            paged_attention_k_table=page_table_k,  # Page Table K: Tensor containing offsets to the container with K blocks
            paged_attention_v_table=page_table_v,  # Page Table V: Tensor containing offsets to the container with V blocks
            paged_attention_max_seq_len_kv=max_ctx_len,  # The maximum sequence length for K caches (this is optional, but recommended)
        )
        logging.info(graph)

        # 7) Set output tensor
        # CuDNN output will also be [B, H, max_new_tokens, D]
        # eventually flatten it back to [sum_of_new_tokens_across_batch, H, D]
        
        #output = output.view(*padded_query.shape)
        B_out, H_out, S_out, D_out = padded_query.shape
        output = output.new_zeros((B_out, H_out, S_out, D_out))
        dims = output.shape
        strides = output.stride()


        o.set_output(True).set_dim(dims).set_stride(strides)
        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph.check_support()
        graph.build_plans()

        build_graph_time = time.perf_counter()

        variant_pack = {
            q: q_gpu,
            container_k: container_k_gpu,
            container_v: container_v_gpu,
            page_table_k: page_table_k_gpu,
            page_table_v: page_table_v_gpu,
            seq_len_q_tensor_info: seq_lens_q,
            seq_len_kv_tensor_info: seq_lens_kv,
            o: output,
        }


        workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
        graph.execute(variant_pack, workspace)
        print(output.shape)
        torch.cuda.synchronize()

        # 8) Reshape the output back to [sum_of_new_tokens_across_batch, H, D]
        final_out = []
        offset = 0
        for i in range(B):
            length_i = extend_seq_lens[i].item()
            seq_out = output[i, :, :length_i, :] 
            # permute => [length_i, H, D]
            seq_out = seq_out.movedim(0, 1)
            final_out.append(seq_out)
        final_output = torch.cat(final_out, dim=0)
        end_time = time.perf_counter()

        print(f"Graph Construction Time: {build_graph_time-start_time}")
        print(f"Forward Time: {end_time-build_graph_time}")

        return final_output

      

    def _run_sdpa_forward_decode(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        scaling=None,
        enable_gqa=False,
        causal=False,
    ):
        """Run the decode forward by using torch native sdpa op.

        Args:
            query: [num_tokens, num_heads, head_size]
            output: [num_tokens, num_heads, head_size]
            k_cache: [max_total_num_tokens, num_heads, head_size]
            v_cache: [max_total_num_tokens, num_heads, head_size]
            req_to_token: [max_num_reqs, max_context_len]
            req_pool_indices: [num_seqs]
            seq_lens: [num_seqs]
            scaling: float or None
            enable_gqa: bool
            causal: bool

        Returns:
            output: [num_tokens, num_heads, head_size]
        """

        assert query.shape[0] == seq_lens.shape[0], "batch size must be the same"
        
        start_time =time.perf_counter()

        max_seq_len = k_cache.shape[0]
        # Convert into CuDNN Query format (B, H, S, D)
        # where B is number of queries and S is sequence per query (1 in decoding)
        # [num_tokens, num_heads, head_size] -> [num_token, num_heads, 1,  head_size]
        query = query.unsqueeze(1).movedim(1,2)

        # heads, tokens, head size
        # The tokens of queries are indexed by req_to_token
        s, h, d = k_cache.shape
        # Block Size of Paged Cache, 1 since only one token per block
        b = 1

        # Radix Attention use KVCache of Block Size 1

        # TODO: determine data type
        graph = cudnn.pygraph(
            io_data_type=cudnn.data_type.HALF,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
        )
        print(graph)

        # query contains num_tokens queries batched togather
        q_gpu = query
        q = graph.tensor_like(q_gpu)


        # get the request id of each query up to t
        # per_req_tokens = req_to_token[req_pool_indices, :seq_len_kv]

        # get the token location in kvcache, only up to seq_len_kv is valid
        # cudnn required shape: (num_block, 1, ceil(s/num_block), 1)
        print("req index shape: ",req_pool_indices.shape,"req to token shape: ",req_to_token.shape)
        per_req_tokens = req_to_token[req_pool_indices, :]
        print("per req token shape: ",per_req_tokens.shape)

        # get the kv cache with request id
        # container: num_blocks, num heads, tokens_per_block, dim
        # TODO: permute for correctness
        container_k_gpu = k_cache.view(s,h,b,d)
        print('cache shape: ',container_k_gpu.shape)
        container_v_gpu = v_cache.view(s,h,b,d)


        container_k = graph.tensor_like(container_k_gpu)
        container_v = graph.tensor_like(container_v_gpu)


        page_table_k_gpu = per_req_tokens.view(per_req_tokens.shape[0],1,per_req_tokens.shape[1],1)
        print("paged table k shape: ",page_table_k_gpu.shape)
        page_table_v_gpu = per_req_tokens.view(per_req_tokens.shape[0],1,per_req_tokens.shape[1],1)
        print("page table v shape: ",page_table_v_gpu.shape)
        page_table_k = graph.tensor_like(page_table_k_gpu)
        page_table_v = graph.tensor_like(page_table_v_gpu)

        seq_lens_kv = seq_lens.view(seq_lens.shape[0], 1, 1, 1)

        seq_lens_q = torch.ones_like(seq_lens_kv)

        seq_len_q_tensor_info = graph.tensor_like(seq_lens_q)
        seq_len_kv_tensor_info = graph.tensor_like(seq_lens_kv)

        o, _ = graph.sdpa(
            name="sdpa",
            q=q,
            k=container_k,  # Container K: non contiguous container with K blocks
            v=container_v,  # Container V: non contiguous container with V blocks
            is_inference=True,
            attn_scale=scaling,
            use_causal_mask=causal,
            use_padding_mask=True,
            seq_len_q=seq_len_q_tensor_info,
            seq_len_kv=seq_len_kv_tensor_info,
            paged_attention_k_table=page_table_k,  # Page Table K: Tensor containing offsets to the container with K blocks
            paged_attention_v_table=page_table_v,  # Page Table V: Tensor containing offsets to the container with V blocks
            paged_attention_max_seq_len_kv=max_seq_len,  # The maximum sequence length for K caches (this is optional, but recommended)
        )
        logging.info(graph)

        output = output.view(*query.shape)
        dims = output.shape
        strides = output.stride()


        o.set_output(True).set_dim(dims).set_stride(strides)
        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph.check_support()
        graph.build_plans()

        build_graph_time = time.perf_counter()

        variant_pack = {
            q: q_gpu,
            container_k: container_k_gpu,
            container_v: container_v_gpu,
            page_table_k: page_table_k_gpu,
            page_table_v: page_table_v_gpu,
            seq_len_q_tensor_info: seq_lens_q,
            seq_len_kv_tensor_info: seq_lens_kv,
            o: output,
        }


        workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
        graph.execute(variant_pack, workspace)
        print(output.shape)
        torch.cuda.synchronize()
        end_time = time.perf_counter()

        print(f"Graph Construction Time: {build_graph_time-start_time}")
        print(f"Forward Time: {end_time-build_graph_time}")

        return output

class TorchNativeAttnBackend():
    def __init__(self, ):
        super().__init__()
        self.forward_metadata = None



    def _run_sdpa_forward_extend(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        scaling=None,
        enable_gqa=False,
        causal=False,
    ):
        """Run the extend forward by using torch native sdpa op.

        Args:
            query: [num_tokens, num_heads, head_size]
            output: [num_tokens, num_heads, head_size]
            k_cache: [max_total_num_tokens, num_heads, head_size]
            v_cache: [max_total_num_tokens, num_heads, head_size]
            req_to_token: [max_num_reqs, max_context_len]
            req_pool_indices: [num_seqs]
            seq_lens: [num_seqs]
            extend_prefix_lens: [num_seqs]
            extend_seq_lens: [num_seqs]
            scaling: float or None
            enable_gqa: bool
            causal: bool

        Returns:
            output: [num_tokens, num_heads, head_size]
        """

        assert seq_lens.shape[0] == extend_prefix_lens.shape[0]
        assert seq_lens.shape[0] == extend_seq_lens.shape[0]

        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        query = query.movedim(0, query.dim() - 2)

        start_q, start_kv = 0, 0
        for seq_idx in range(seq_lens.shape[0]):
            # TODO: this loop process a sequence per iter, this is inefficient.
            # Need optimize the performance later.

            extend_seq_len_q = extend_seq_lens[seq_idx]
            prefill_seq_len_q = extend_prefix_lens[seq_idx]

            seq_len_kv = seq_lens[seq_idx]
            end_q = start_q + extend_seq_len_q
            end_kv = start_kv + seq_len_kv

            per_req_query = query[:, start_q:end_q, :]
            per_req_query_redudant = torch.empty(
                (per_req_query.shape[0], seq_len_kv, per_req_query.shape[2]),
                dtype=per_req_query.dtype,
                device=per_req_query.device,
            )

            per_req_query_redudant[:, prefill_seq_len_q:, :] = per_req_query

            # get key and value from cache. per_req_tokens contains the kv cache
            # index for each token in the sequence.
            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

            per_req_out_redudant = (
                torch.nn.functional.scaled_dot_product_attention(
                    per_req_query_redudant.unsqueeze(0),
                    per_req_key.unsqueeze(0),
                    per_req_value.unsqueeze(0),
                    enable_gqa=enable_gqa,
                    scale=scaling,
                    is_causal=causal,
                )
                .squeeze(0)
                .movedim(query.dim() - 2, 0)
            )
            output[start_q:end_q, :, :] = per_req_out_redudant[prefill_seq_len_q:, :, :]
            start_q, start_kv = end_q, end_kv
        return output

    def _run_sdpa_forward_decode(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        scaling=None,
        enable_gqa=False,
        causal=False,
    ):
        """Run the decode forward by using torch native sdpa op.

        Args:
            query: [num_tokens, num_heads, head_size]
            output: [num_tokens, num_heads, head_size]
            k_cache: [max_total_num_tokens, num_heads, head_size]
            v_cache: [max_total_num_tokens, num_heads, head_size]
            req_to_token: [max_num_reqs, max_context_len]
            req_pool_indices: [num_seqs]
            seq_lens: [num_seqs]
            scaling: float or None
            enable_gqa: bool
            causal: bool

        Returns:
            output: [num_tokens, num_heads, head_size]
        """

        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        query = query.movedim(0, query.dim() - 2)

        start_q, start_kv = 0, 0
        for seq_idx in range(seq_lens.shape[0]):
            # TODO: this loop process a sequence per iter, this is inefficient.
            # Need optimize the performance later.

            seq_len_q = 1
            seq_len_kv = seq_lens[seq_idx]
            end_q = start_q + seq_len_q
            end_kv = start_kv + seq_len_kv

            per_req_query = query[:, start_q:end_q, :]

            # get key and value from cache. per_req_tokens contains the kv cache
            # index for each token in the sequence.
            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

            per_req_out = (
                torch.nn.functional.scaled_dot_product_attention(
                    per_req_query.unsqueeze(0),
                    per_req_key.unsqueeze(0),
                    per_req_value.unsqueeze(0),
                    enable_gqa=enable_gqa,
                    scale=scaling,
                    is_causal=causal,
                )
                .squeeze(0)
                .movedim(query.dim() - 2, 0)
            )
            output[start_q:end_q, :, :] = per_req_out
            start_q, start_kv = end_q, end_kv

        return output



def test_correctness():
    input_parem = InputParameters()
    cudnn_bknd = CuDNNBackend()
    torch_native_backend = TorchNativeAttnBackend()
    
    # TODO: dtype
    query = torch.randn([input_parem.num_token, input_parem.num_heads, input_parem.head_size]).half().cuda()
    output = torch.randn([input_parem.num_token, input_parem.num_heads, input_parem.head_size]).half().cuda()
    k_cache = torch.randn([input_parem.max_total_num_tokens, input_parem.num_heads, input_parem.head_size]).half().cuda()
    v_cache = torch.randn([input_parem.max_total_num_tokens, input_parem.num_heads, input_parem.head_size]).half().cuda()

    # the following are int tensors

    # the request index of inputs sequences in req_to_token
    req_pool_indices = torch.randint(low=0,high=input_parem.max_num_reqs,size=[input_parem.num_seqs],dtype=torch.int32).cuda()

    extend_prefix_lens = torch.randint(low=0,high=5,size=[input_parem.num_seqs],dtype=torch.int32).cuda()
    vals = [1, 2, 3, 4, 5, 3, 3, 3, 3, 3]  # sum is 30
    extend_seq_lens = torch.tensor(vals, dtype=torch.int32).cuda()
    #extend_seq_lens = torch.randint(low=1,high=3,size=[input_parem.num_seqs],dtype=torch.int32).cuda()

    # req_to_token[request_index]: list of index of tokens in query and value for that request_index
    # sum(len(tokens_per_request)) = num_tokens in query
    req_to_token = torch.randint(low=0,high=input_parem.num_token,size=[input_parem.max_num_reqs, input_parem.max_context_lenght],dtype=torch.int32).cuda()
    #seq_lens = torch.randint(low=0,high=input_parem.max_total_num_tokens,size=[input_parem.num_seqs]).cuda()
    seq_lens = (extend_prefix_lens + extend_seq_lens).cuda()
    scaling = 1/math.sqrt(input_parem.head_size)


    # logging.info("Start Extend")

    # output = cudnn_bknd._run_sdpa_forward_decode(
    #     query=query,
    #     output=output,
    #     k_cache=k_cache,
    #     v_cache=v_cache,
    #     req_to_token=req_to_token,
    #     req_pool_indices=req_pool_indices,
    #     seq_lens=seq_lens,
    #     scaling=scaling
    # )

    # torch_output = torch.randn([input_parem.num_token, input_parem.num_heads, input_parem.head_size]).half().cuda()
    # torch_output = torch_native_backend._run_sdpa_forward_decode(
    #     query=query,
    #     output=torch_output,
    #     k_cache=k_cache,
    #     v_cache=v_cache,
    #     req_to_token=req_to_token,
    #     req_pool_indices=req_pool_indices,
    #     seq_lens=seq_lens,
    #     scaling=scaling
    # )

    # print("Decode output shapes:", output.shape, torch_output.shape)
    # output = output.squeeze()
    # torch_output = torch_output.squeeze()
    # torch.testing.assert_close(output,torch_output)
    # print("Decode Result Same")


    logging.info("Start Extend Test")

    # Force sum(extend_seq_lens) = input_parem.num_token
    current_sum = extend_seq_lens.sum()
    diff = input_parem.num_token - current_sum
    extend_seq_lens[0] += diff
    assert extend_seq_lens.sum() == input_parem.num_token, \
           "extend_seq_lens sum doesn't match input_parem.num_token."

    output = cudnn_bknd._run_sdpa_forward_extend(
        query=query,
        output=output,
        k_cache=k_cache,
        v_cache=v_cache,
        req_to_token=req_to_token,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        extend_prefix_lens=extend_prefix_lens,
        extend_seq_lens=extend_seq_lens,
        scaling=scaling
    )

    torch_output = torch.randn([input_parem.num_token, input_parem.num_heads, input_parem.head_size]).half().cuda()
    torch_output = torch_native_backend._run_sdpa_forward_extend(
        query=query,
        output=torch_output,
        k_cache=k_cache,
        v_cache=v_cache,
        req_to_token=req_to_token,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        extend_prefix_lens=extend_prefix_lens,
        extend_seq_lens=extend_seq_lens,
        scaling=scaling
    )

    print("Extend output shapes:", output.shape, torch_output.shape)
    output = output.squeeze()
    torch_output = torch_output.squeeze()
    torch.testing.assert_close(output,torch_output)
    print("Extend Result Same")

if __name__=='__main__':
    assert torch.cuda.is_available()
    assert (
        torch.cuda.get_device_capability()[0] >= 8
    ), f"SDPA operation is only supported on SM80 architecture (Ampere) or above, got {torch.cuda.get_device_capability()[0]}"

    # assert (
    #     cudnn.backend_version() >= 90500
    # ), f"SDPA operation is only supported cuDNN version 9.5.0 or above, got {cudnn.backend_version()}"
    test_correctness()



    