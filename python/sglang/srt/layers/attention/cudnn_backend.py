from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import cudnn
import torch
import tqdm

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.model_executor.forward_batch_info import ForwardBatch,ForwardMode

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.speculative.eagle_utils import EagleDraftInput, EagleVerifyInput



class _CuDNNInputParameters:
    def __init__(
        self, num_heads_q, num_heads_kv, head_size, q_head_size, max_total_num_tokens, max_batch_size
    ):
        self.num_heads_q = num_heads_q
        self.num_heads_kv = num_heads_kv
        self.head_size = head_size
        self.q_head_size = head_size
        self.max_total_num_tokens = max_total_num_tokens
        self.max_batch_size = max_batch_size


# If True, the tensors will be validated against the CuDNN graph for dims and strides
VALIDATE_PARAMS = True


class CuDNNBackend(AttentionBackend):
    @dataclass
    class _ArgMapKeys:
        q = "q"
        k_container = "k_container"
        v_container = "v_container"
        k_page_table = "k_page_table"
        v_page_table = "v_page_table"
        seq_len_q_tensor_info = "seq_len_q_tensor_info"
        seq_len_kv_tensor_info = "seq_len_kv_tensor_info"
        o = "o"
        q_ragged_offset = "q_ragged_offset"

    def __init__(self, model_runner: ModelRunner, extend_seq_len_interval = 1000):
        super().__init__()
        self.forward_metadata = None

        self._model_runner = model_runner
        # should the number of requests be max_request or max_batch_size
        tensor_parallel_level = get_attention_tp_size()
        print("Tensor Parallel Level: ", tensor_parallel_level)
        kv_head_num = model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )
        q_head_num = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.input_size_params = _CuDNNInputParameters(
            num_heads_q=q_head_num,
            num_heads_kv=kv_head_num,
            head_size=model_runner.model_config.head_dim,
            q_head_size=model_runner.model_config.head_dim,
            max_total_num_tokens=model_runner.max_total_num_tokens,
            # use cuda_graph_max_bs for max batch size of cudnn decode graph cache for now
            max_batch_size=model_runner.server_args.cuda_graph_max_bs,
        )

        print("Max Tokens: ", model_runner.max_total_num_tokens)
        print("Max Batch Size: ", model_runner.server_args.cuda_graph_max_bs)

        # the step length when creating the cudnn graph cache
        # one extend graph is created for each step
        self._extend_seq_len_interval = extend_seq_len_interval

        # create the cudnn graph cache
        # self._prefill_graphs[i][j] is the graph for seq_len=(i+1)*step_size, prefix_len=j*step_size

        # TODO: better ranges for graph cache
        max_prefill_len = self._model_runner.server_args.max_prefill_tokens // 8 + 100
        self._prefill_graphs = self._init_prefill_graphs(
            max_prefill_len, step_size=self._extend_seq_len_interval
        )

        # create one decode graph per batch size
        self._decode_graphs = self._init_decode_graphs(
            self.input_size_params.max_batch_size
        )
        self._decode_seq_len = torch.ones(
            (self.input_size_params.max_batch_size, 1, 1, 1),
            dtype=torch.int32,
            device="cuda",
        )

    def _create_cudnn_graph(
        self,
        batch_size: int,
        query_shape,
        query_strides,
        kv_container_shape,
        kv_page_table_shape,
        seq_len_shape,
        diagonal_band_right_bound=None,
        ragged_query=False,
        causal=True,
    ):
        graph = cudnn.pygraph(
            io_data_type=cudnn.data_type.HALF,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
        )
        # sglang query shape is [bs, s, h, d]
        # so the input query tensor is not compact
        q_cudnn = graph.tensor(
            name="q",
            dim=query_shape,
            stride=query_strides,
            data_type=cudnn.data_type.HALF,
        )

        if ragged_query:
            ragged_offset_shape = [batch_size + 1, 1, 1, 1]
            q_ragged_offset = graph.tensor(
                name="q_ragged_offset",
                dim=ragged_offset_shape,
                stride=self._make_compact_strides(ragged_offset_shape),
                data_type=cudnn.data_type.INT32,
            )
            q_cudnn.set_ragged_offset(q_ragged_offset)

        # container: num_blocks, num heads, tokens_per_block, dim
        # container: max_tokens, num_heads, 1, head_dim since sglang block size is 1
        k_container_cudnn = graph.tensor(
            name="k_container",
            dim=kv_container_shape,
            stride=self._make_compact_strides(kv_container_shape),
            data_type=cudnn.data_type.HALF,
        )
        v_container_cudnn = graph.tensor(
            name="v_container",
            dim=kv_container_shape,
            stride=self._make_compact_strides(kv_container_shape),
            data_type=cudnn.data_type.HALF,
        )

        # page tables should be int
        k_page_table = graph.tensor(
            name="k_page_table",
            dim=kv_page_table_shape,
            stride=self._make_compact_strides(kv_page_table_shape),
            data_type=cudnn.data_type.INT32,
        )
        v_page_table = graph.tensor(
            name="v_page_table",
            dim=kv_page_table_shape,
            stride=self._make_compact_strides(kv_page_table_shape),
            data_type=cudnn.data_type.INT32,
        )

        kv_seq_len = graph.tensor(
            name="kv_seq_len",
            dim=seq_len_shape,
            stride=self._make_compact_strides(seq_len_shape),
            data_type=cudnn.data_type.INT32,
        )
        q_seq_len = graph.tensor(
            name="q_seq_len",
            dim=seq_len_shape,
            stride=self._make_compact_strides(seq_len_shape),
            data_type=cudnn.data_type.INT32,
        )

        if diagonal_band_right_bound is not None:
            o, _ = graph.sdpa(
                name="sdpa",
                q=q_cudnn,
                k=k_container_cudnn,  # Container K: non contiguous container with K blocks
                v=v_container_cudnn,  # Container V: non contiguous container with V blocks
                is_inference=True,
                # TODO: passing atten_scale as arg
                attn_scale=1 / math.sqrt(self.input_size_params.head_size),
                # TODO: enable passing casual argument in graph or cache graph for both casual_mask = True and False
                use_causal_mask=causal,
                use_padding_mask=True,
                diagonal_band_right_bound=diagonal_band_right_bound,
                seq_len_q=q_seq_len,
                seq_len_kv=kv_seq_len,
                paged_attention_k_table=k_page_table,  # Page Table K: Tensor containing offsets to the container with K blocks
                paged_attention_v_table=v_page_table,  # Page Table V: Tensor containing offsets to the container with V blocks
                paged_attention_max_seq_len_kv=kv_container_shape[
                    0
                ],  # The maximum sequence length for K caches (this is optional, but recommended)
            )
        else:
            o, _ = graph.sdpa(
                name="sdpa",
                q=q_cudnn,
                k=k_container_cudnn,  # Container K: non contiguous container with K blocks
                v=v_container_cudnn,  # Container V: non contiguous container with V blocks
                is_inference=True,
                attn_scale=1 / math.sqrt(self.input_size_params.head_size),
                use_causal_mask=causal,
                use_padding_mask=True,
                seq_len_q=q_seq_len,
                seq_len_kv=kv_seq_len,
                paged_attention_k_table=k_page_table,  # Page Table K: Tensor containing offsets to the container with K blocks
                paged_attention_v_table=v_page_table,  # Page Table V: Tensor containing offsets to the container with V blocks
                paged_attention_max_seq_len_kv=kv_container_shape[
                    0
                ],  # The maximum sequence length for K caches (this is optional, but recommended)
            )

        o.set_output(True).set_dim(query_shape).set_stride(
            self._make_compact_strides(query_shape)
        )
        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph.check_support()
        graph.build_plans()

        args_map = {
            self._ArgMapKeys.q: q_cudnn,
            self._ArgMapKeys.k_container: k_container_cudnn,
            self._ArgMapKeys.v_container: v_container_cudnn,
            self._ArgMapKeys.k_page_table: k_page_table,
            self._ArgMapKeys.v_page_table: v_page_table,
            self._ArgMapKeys.seq_len_q_tensor_info: q_seq_len,
            self._ArgMapKeys.seq_len_kv_tensor_info: kv_seq_len,
            self._ArgMapKeys.o: o,
        }
        return args_map, graph

    def init_cuda_graph_state(self, max_bs):
        pass

    def get_cuda_graph_seq_len_fill_value(self):
        return 0

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        if forward_mode.is_decode_or_idle():
            batch_size = bs
            args_and_graph_decode = self._decode_graphs[batch_size - 1]
            self.forward_metadata = args_and_graph_decode
            

       
        
    

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        if forward_mode.is_decode_or_idle():
            batch_size = bs
            args_and_graph_decode = self._decode_graphs[batch_size - 1]
            self.forward_metadata = args_and_graph_decode
            
        


    def _create_cudnn_graph_extend(self, seq_len: int, diagonal_band_right_bound: int):
        batch_size = 1
        # see memory_pool.py: kv cache size has max_total_num_tokens+1 tokens (with 1 dummy token)
        kv_cache_num_tokens = self.input_size_params.max_total_num_tokens + 1
        # see model_runner.py
        # paged_table_size = self._model_runner.model_config.context_len + 4
        q_shape = [
            batch_size,
            self.input_size_params.num_heads_q,
            seq_len,
            self.input_size_params.head_size,
        ]
        query_strides = self._make_compact_strides(q_shape)
        kv_container_shape = [
            kv_cache_num_tokens,
            self.input_size_params.num_heads_kv,
            1,
            self.input_size_params.head_size,
        ]
        kv_page_table_shape = [batch_size, 1, kv_cache_num_tokens, 1]
        seq_len_shape = [batch_size, 1, 1, 1]
        return self._create_cudnn_graph(
            batch_size,
            q_shape,
            query_strides,
            kv_container_shape,
            kv_page_table_shape,
            seq_len_shape,
            diagonal_band_right_bound,
        )

    def init_forward_metadata(
        self,
        forward_batch,
        mock=False,
        decode_batch_size=None,
        prefix_lens=None,
        extend_seq_lens=None,
    ):
        """Fetch CuDNN graph from graph cache, create if not exist"""

        if forward_batch.forward_mode.is_decode_or_idle():
            batch_size = forward_batch.batch_size
            args_and_graph_decode = self._decode_graphs[batch_size - 1]
            padded_k_table, padded_v_table = self._init_decode_page_table(
                forward_batch.req_to_token_pool.req_to_token,
                forward_batch.req_pool_indices,
            )
            self.forward_metadata = args_and_graph_decode, (padded_k_table, padded_v_table)
            

        else:
            prefix_lens = forward_batch.extend_prefix_lens
            extend_seq_lens = forward_batch.extend_seq_lens

            extend_args_and_graphs = []

            for i in range(len(prefix_lens)):
                prefix_len = prefix_lens[i]
                extend_seq_len = extend_seq_lens[i]
                (extend_seq_len_index, prefix_len_index), _ = (
                    self._get_extend_graph_index(extend_seq_len, prefix_len)
                )

                if (
                    len(self._prefill_graphs) <= extend_seq_len_index
                    or len(self._prefill_graphs[extend_seq_len_index])
                    <= prefix_len_index
                ):
                    new_diagonal_band = prefix_len_index * self._extend_seq_len_interval
                    new_seq_len = (
                        extend_seq_len_index + 1
                    ) * self._extend_seq_len_interval
                    new_graph = self._create_cudnn_graph_extend(
                        new_seq_len, new_diagonal_band
                    )
                    print(
                        f"Warning: Graph not found for seq_len {extend_seq_len} and prefix_len {prefix_len}, creating new graph"
                    )
                    extend_args_and_graphs.append(new_graph)

                # Since the minimal seq_len is 64 rather than 0, the index need to -1
                extend_args_and_graphs.append(
                    self._prefill_graphs[extend_seq_len_index][prefix_len_index]
                )

            self.forward_metadata = extend_args_and_graphs

    def _make_compact_strides(self, tensor_shape):
        """Make compact strides for a tensor shape."""
        strides = []
        stride = 1
        for dim in reversed(tensor_shape):
            strides.append(stride)
            stride *= dim
        return list(reversed(strides))

    def _init_decode_graphs(self, max_batch_size):

        decode_graphs = []
        print("Create decode graphs for batch sizes: ", max_batch_size)
        for batch_size in range(1, max_batch_size + 1):
            # Radix Attention use KVCache of Block Size 1

            q_shape = [
                batch_size,
                self.input_size_params.num_heads_q,
                1,
                self.input_size_params.head_size,
            ]
            q_strides = self._make_compact_strides(q_shape)
            # see memory_pool.py: kv cache size has max_total_num_tokens+1 tokens (with 1 dummy token)
            kv_cache_num_tokens = self.input_size_params.max_total_num_tokens + 1
            # see model_runner.py
            # paged_table_size = self._model_runner.model_config.context_len + 4
            kv_container_shape = [
                kv_cache_num_tokens,
                self.input_size_params.num_heads_kv,
                1,
                self.input_size_params.head_size,
            ]
            kv_page_table_shape = [batch_size, 1, kv_cache_num_tokens, 1]
            seq_len_shape = [batch_size, 1, 1, 1]

            tensor_args, graph = self._create_cudnn_graph(
                batch_size,
                q_shape,
                q_strides,
                kv_container_shape,
                kv_page_table_shape,
                seq_len_shape,
                causal=False,
            )
            decode_graphs.append((tensor_args, graph))
            assert batch_size == len(
                decode_graphs
            ), f"batch size {batch_size} does not match the number of graphs {len(decode_graphs)}"

        return decode_graphs

    def _get_extend_graph_index(self, exnted_seq_len: int, prefix_len: int):
        """Get the graph index for the given seq_len and prefix_len"""
        # find the first prefix_len in the cache smaller than prefix_len
        prefix_len_index = math.floor(prefix_len / self._extend_seq_len_interval)
        # to maintain causality, the sequence need to be padded in the front
        front_padding = prefix_len % self._extend_seq_len_interval
        padded_seq_length = exnted_seq_len + front_padding
        # find the first seq_len in the cache larger than padded_seq_length
        seq_len_index = math.ceil(padded_seq_length / self._extend_seq_len_interval)
        back_padding = seq_len_index * self._extend_seq_len_interval - padded_seq_length
        # front padding is for causality, back padding is for the rest of the sequence

        return (seq_len_index - 1, prefix_len_index), (front_padding, back_padding)

    def _init_prefill_graphs(self, max_sequence_len, step_size=2000, parallelism=1):
        # TODO: use python multiprocessing to create the graphs in parallel
        prefill_graphs = []
        print(
            f"Creating Prefill Graphs for max seq len {max_sequence_len} with step size {step_size}:"
        )
        for seq_len in tqdm.tqdm(
            range(step_size, max_sequence_len + step_size, step_size)
        ):
            extend_graph_seq_len = []
            for diagonal_band_offset in range(0, max_sequence_len, step_size):
                extend_graph_seq_len.append(
                    self._create_cudnn_graph_extend(seq_len, diagonal_band_offset)
                )
            prefill_graphs.append(extend_graph_seq_len)

        return prefill_graphs

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
        assert seq_lens.shape[0] == extend_prefix_lens.shape[0]
        assert seq_lens.shape[0] == extend_seq_lens.shape[0]

        # B = batch_size
        B = seq_lens.shape[0]
        # assert (
        #     query.shape[0] == extend_seq_lens.sum()
        # ), f"query.shape[0] = {query.shape[0]}, but sum(extend_seq_lens) = {extend_seq_lens.sum()}"

        # 1) Reshape the multi-token query to [B, max_new_tokens, H, D] in a padded fashion
        H = query.shape[1]
        D = query.shape[2]

        # cudnn graphs should be populated in self.init_forward_metadata()
        cudnn_extend_args_and_graphs = self.forward_metadata

        # covert tensor to python list to minimize the cuda kernel launch
        extend_seq_len_list = extend_seq_lens.tolist()
        extend_prefix_len_list = extend_prefix_lens.tolist()
        # create the shape-related tensor ahead to avoid cuda kernel launch
        seq_front_paddings = []
        for i in range(B):
            _, (front_padding, back_padding) = (
                self._get_extend_graph_index(extend_seq_len_list[i], extend_prefix_len_list[i])
            )
            seq_front_paddings.append(front_padding)
        seq_front_paddings = torch.tensor(seq_front_paddings, device=query.device,dtype=torch.int32)
        padded_extend_seq_lens = seq_front_paddings + extend_seq_lens
        torch.cuda.synchronize()

        # Fill in each sequence's slice
        offset = 0
        padded_k_page_table = torch.empty(
            (1, 1, self.input_size_params.max_total_num_tokens + 1, 1),
            dtype=torch.int32,
            device=query.device,
        )
        padded_v_page_table = torch.empty(
            (1, 1, self.input_size_params.max_total_num_tokens + 1, 1),
            dtype=torch.int32,
            device=query.device,
        )
        for i in range(B):
            args_i, graph_i = cudnn_extend_args_and_graphs[i]
            length_i = extend_seq_len_list[i]
            prefix_len_i = extend_prefix_len_list[i]
            (seq_len_index, prefix_len_index), (front_padding, back_padding) = (
                self._get_extend_graph_index(length_i, prefix_len_i)
            )
            padding_total_size = front_padding + back_padding + length_i

            if not (front_padding == 0 and back_padding == 0):
                # pad the query with front and back padding
                query_i = torch.empty(
                    (1, H, padding_total_size, D),
                    dtype=query.dtype,
                    device=query.device,
                )
                query_i[0, :, front_padding : front_padding + length_i, :] = query[
                    offset : offset + length_i, :, :
                ].movedim(0, 1)
            else:
                query_i = query[offset : offset + length_i, :, :]
                query_i = query_i.unsqueeze(0).movedim(1,2).contiguous()



            # heads, tokens, head size
            # The tokens of queries are indexed by req_to_token
            s, h, d = k_cache.shape
            # Block Size of Paged Cache, 1 since only one token per block
            b = 1

            # Reshape k_cache, v_cache into container shapes for “paged” attention
            # container: num_blocks, num heads, tokens_per_block, dim
            container_k_gpu = k_cache.view(s, h, b, d)
            container_v_gpu = v_cache.view(s, h, b, d)

            # Sequence lengths
            seq_lens_kv = seq_lens[i].view(1, 1, 1, 1)
            seq_lens_q = padded_extend_seq_lens[i].view(1, 1, 1, 1)

            # Build the page table
            # only want prefix + the newly added tokens for each sequence
            # Then pad it to the maximum across the batch
            per_req_tokens = req_to_token[req_pool_indices[i], :]

            # reshape to [1, 1, max_ctx_len, 1]
            padded_k_page_table[:, :, : per_req_tokens.shape[0], :] = (
                per_req_tokens.view(1, 1, per_req_tokens.shape[0], 1)
            )
            padded_v_page_table[:, :, : per_req_tokens.shape[0], :] = (
                per_req_tokens.view(1, 1, per_req_tokens.shape[0], 1)
            )

            # 7) Set output tensor
            # CuDNN output will also be [B, H, max_new_tokens, D]
            # eventually flatten it back to [sum_of_new_tokens_across_batch, H, D]
            B_out, H_out, S_out, D_out = query_i.shape
            output_i = output.new_zeros((B_out, H_out, S_out, D_out))

            variant_pack = {
                args_i[self._ArgMapKeys.q]: query_i,
                args_i[self._ArgMapKeys.k_container]: container_k_gpu,
                args_i[self._ArgMapKeys.v_container]: container_v_gpu,
                args_i[self._ArgMapKeys.k_page_table]: padded_k_page_table,
                args_i[self._ArgMapKeys.v_page_table]: padded_v_page_table,
                args_i[self._ArgMapKeys.seq_len_q_tensor_info]: seq_lens_q,
                args_i[self._ArgMapKeys.seq_len_kv_tensor_info]: seq_lens_kv,
                args_i[self._ArgMapKeys.o]: output_i,
            }

            if VALIDATE_PARAMS:
                # Validate the shape and strides of cudnn args are same as torch inputs
                for key in args_i:
                    tensor_attr = args_i[key]
                    torch_tensor = variant_pack[tensor_attr]
                    if tensor_attr.get_dim() != list(torch_tensor.shape):
                        raise ValueError(
                            f"Invalid tensor shape {key}: cudnn expect {tensor_attr.get_dim()} but got {torch_tensor.shape}"
                        )
                    if tensor_attr.get_stride() != list(torch_tensor.stride()):
                        raise ValueError(
                            f"Invalid tensor stride {key}: cudnn expect {tensor_attr.get_stride()} (shape: {tensor_attr.get_dim()}) but got {torch_tensor.stride()} (shape: {torch_tensor.shape})"
                        )

            workspace = torch.empty(
                graph_i.get_workspace_size(), device="cuda", dtype=torch.uint8
            )
            graph_i.execute(variant_pack, workspace)
            # move the true value out from the padded output
            result_i = output_i[:, :, front_padding : front_padding + length_i, :]
            output[
                offset : offset + length_i,
                :,
            ] = result_i.squeeze(
                0
            ).movedim(1, 0)
            offset += length_i
                
        return output
    
    def _init_decode_page_table(self, req_to_token, req_pool_indices):
        # Build the page table
        # only want prefix + the newly added tokens for each sequence
        # Then pad it to the maximum across the batch
        per_req_tokens = req_to_token[req_pool_indices, :]

        # reshape to [bs, 1, max_ctx_len, 1]
        padded_k_page_table = torch.empty(
            (per_req_tokens.shape[0], 1, self.input_size_params.max_total_num_tokens + 1, 1),
            dtype=torch.int32,
            device=req_to_token.device,
        )
        padded_v_page_table = torch.empty(
            (per_req_tokens.shape[0], 1, self.input_size_params.max_total_num_tokens + 1, 1),
            dtype=torch.int32,
            device=req_to_token.device,
        )
        padded_k_page_table[:, :, : per_req_tokens.shape[1], :] = (
            per_req_tokens.view(per_req_tokens.shape[0], 1, per_req_tokens.shape[1], 1)
        )
        padded_v_page_table[:, :, : per_req_tokens.shape[1], :] = per_req_tokens.view(
            per_req_tokens.shape[0], 1, per_req_tokens.shape[1], 1
        )

        return padded_k_page_table, padded_v_page_table

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
        causal=True,
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
        batch_size_query = query.shape[0]

        (tensor_key_map, cudnn_decode_graph) = self.forward_metadata
        padded_k_page_table, padded_v_page_table = self._init_decode_page_table(
                req_to_token,
                req_pool_indices,
        )
        # Convert into CuDNN Query format (B, H, S, D)
        # where B is number of queries and S is sequence per query (1 in decoding)
        # [num_tokens, num_heads, head_size] -> [num_token, num_heads, 1,  head_size]
        # TODO: change stride in cudnn to avoid contiguous call
        query = query.unsqueeze(2).contiguous()

        # heads, tokens, head size
        # The tokens of queries are indexed by req_to_token
        s, h, d = k_cache.shape
        # Block Size of Paged Cache, 1 since only one token per block
        b = 1
        # get the kv cache with request id
        # container: num_blocks, num heads, tokens_per_block, dim
        container_k_gpu = k_cache.view(s, h, b, d)
        container_v_gpu = v_cache.view(s, h, b, d)


        seq_lens_kv = seq_lens.view(seq_lens.shape[0], 1, 1, 1)
        seq_lens_q = self._decode_seq_len[:batch_size_query]

        output = output.view(*query.shape)
        variant_pack = {
            tensor_key_map[self._ArgMapKeys.q]: query,
            tensor_key_map[self._ArgMapKeys.k_container]: container_k_gpu,
            tensor_key_map[self._ArgMapKeys.v_container]: container_v_gpu,
            tensor_key_map[self._ArgMapKeys.k_page_table]: padded_k_page_table,
            tensor_key_map[self._ArgMapKeys.v_page_table]: padded_v_page_table,
            tensor_key_map[self._ArgMapKeys.seq_len_q_tensor_info]: seq_lens_q,
            tensor_key_map[self._ArgMapKeys.seq_len_kv_tensor_info]: seq_lens_kv,
            tensor_key_map[self._ArgMapKeys.o]: output,
        }

        if VALIDATE_PARAMS:
            # Validate the shape and strides of cudnn args are same as torch inputs
            for key in tensor_key_map:
                tensor_attr = tensor_key_map[key]
                torch_tensor = variant_pack[tensor_attr]
                if tensor_attr.get_dim() != list(torch_tensor.shape):
                    raise ValueError(
                        f"Invalid tensor shape {key}: cudnn expect {tensor_attr.get_dim()} but got {torch_tensor.shape}"
                    )
                if tensor_attr.get_stride() != list(torch_tensor.stride()):
                    raise ValueError(
                        f"Invalid tensor stride {key}: cudnn expect {tensor_attr.get_stride()} but got {torch_tensor.stride()}"
                    )
        workspace = torch.empty(
            cudnn_decode_graph.get_workspace_size(), device="cuda", dtype=torch.uint8
        )
        cudnn_decode_graph.execute(variant_pack, workspace)

        return output

    def forward_extend(
        self,
        q,
        k,
        v,
        layer,
        forward_batch,
        save_kv_cache=True,
    ):
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        use_gqa = layer.tp_q_head_num != layer.tp_k_head_num

        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        self._run_sdpa_forward_extend(
            q_,
            o_,
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch.extend_prefix_lens,
            forward_batch.extend_seq_lens,
            scaling=layer.scaling,
            enable_gqa=use_gqa,
            causal=not layer.is_cross_attention,
        )
        return o

    def forward_decode(
        self,
        q,
        k,
        v,
        layer,
        forward_batch,
        save_kv_cache=True,
    ):
        # During torch.compile, there is a bug in rotary_emb that causes the
        # output value to have a 3D tensor shape. This reshapes the output correctly.
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        use_gqa = layer.tp_q_head_num != layer.tp_k_head_num

        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        self._run_sdpa_forward_decode(
            q_,
            o_,
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            scaling=layer.scaling,
            enable_gqa=use_gqa,
            causal=False,
        )

        return o
