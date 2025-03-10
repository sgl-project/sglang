from dataclasses import dataclass
from sglang.srt.layers.attention import cudnn_backend
import torch
import logging
import math
import cudnn

@dataclass
class InputParameters:
    num_token = 10
    num_heads = 4
    head_size = 128
    max_total_num_tokens = 300
    max_num_reqs = 100
    max_context_lenght = 300
    num_seqs = 10

def test_correctness():
    input_parem = InputParameters()
    cudnn_bknd = cudnn_backend.CuDNNBackend()
    # TODO: dtype
    query = torch.randn([input_parem.num_token, input_parem.num_heads, input_parem.head_size]).cuda()
    output = torch.randn([input_parem.num_token, input_parem.num_heads, input_parem.head_size]).cuda()
    k_cache = torch.randn([input_parem.max_total_num_tokens, input_parem.num_heads, input_parem.head_size]).cuda()
    v_cache = torch.randn([input_parem.max_total_num_tokens, input_parem.num_heads, input_parem.head_size]).cuda()

    # the following are int tensors

    # the request index of inputs sequences in req_to_token
    req_pool_indices = torch.randint(low=0,high=input_parem.max_num_reqs,size=[input_parem.num_seqs],dtype=torch.int32).cuda()

    # req_to_token[request_index]: list of index of tokens in query and value for that request_index
    # sum(len(tokens_per_request)) = num_tokens in query
    req_to_token = torch.randint(low=0,high=input_parem.num_token,size=[input_parem.max_num_reqs, input_parem.max_context_lenght],dtype=torch.int32).cuda()

    seq_lens = torch.randint(low=0,high=input_parem.num_token,size=[input_parem.num_seqs]).cuda()
    # extend_prefix_lens = torch.randint(low=0,high=1,size=[input_parem.num_seqs],dtype=torch.int32)
    # extend_seq_lens = torch.randint(low=0,high=1,size=[input_parem.num_seqs],dtype=torch.int32)
    scaling = 1/math.sqrt(input_parem.head_size)

    logging.info("Start Extend")


    output = cudnn_bknd._run_sdpa_forward_decode(
        query=query,
        output=output,
        k_cache=k_cache,
        v_cache=v_cache,
        req_to_token=req_to_token,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        scaling=scaling
    )

    # TODO correctness
    print(output)

if __name__=='__main__':
    assert torch.cuda.is_available()
    assert (
        torch.cuda.get_device_capability()[0] >= 8
    ), f"SDPA operation is only supported on SM80 architecture (Ampere) or above, got {torch.cuda.get_device_capability()[0]}"

    assert (
        cudnn.backend_version() >= 90500
    ), f"SDPA operation is only supported cuDNN version 9.5.0 or above, got {cudnn.backend_version()}"
    test_correctness()



    