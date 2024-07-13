
import torch
from vllm.distributed.parallel_state import graph_capture

from sglang.srt.managers.controller.infer_batch import Batch, ForwardMode, InputMetadata, init_flashinfer_args


MAX_BATCH_SIZE = 256


class CudaGraphRunner:
    def __init__(self, model_runner):
        self.model_runner = model_runner
        self.graphs = {}
        self.input_buffers = {}
        self.output_buffers = {}
        self.graph_memory_pool = None

        max_bs = MAX_BATCH_SIZE
        self.input_ids = torch.zeros((max_bs,), dtype=torch.int32, device="cuda")
        self.req_pool_indices = torch.zeros((max_bs,), dtype=torch.int32, device="cuda")
        self.seq_lens = torch.zeros((max_bs,), dtype=torch.int32, device="cuda")
        self.position_ids_offsets = torch.zeros((max_bs,), dtype=torch.int32, device="cuda")
        self.out_cache_loc = torch.zeros((max_bs,), dtype=torch.int32, device="cuda")

    def capture(self, batch_size_list):
        with graph_capture() as graph_capture_context:
            self.stream = graph_capture_context.stream
            for bs in batch_size_list:
                graph, input_buffers, output_buffers = self.capture_one_batch_size(bs)
                self.graphs[bs] = graph
                self.input_buffers[bs] = input_buffers
                self.output_buffers[bs] = output_buffers

    def capture_one_batch_size(self, bs):
        graph = torch.cuda.CUDAGraph()
        stream = self.stream

        input_ids = self.input_ids[:bs]
        req_pool_indices = self.req_pool_indices[:bs]
        seq_lens = self.seq_lens[:bs]
        position_ids_offsets = self.position_ids_offsets[:bs]
        out_cache_loc = self.out_cache_loc[:bs]

        init_flashinfer_args(
            ForwardMode.DECODE,
            self.model_runner,
            req_pool_indices,
            seq_lens,
            None,
            self.model_runner.flashinfer_decode_wrapper,
        )

        def run_once():
            input_metadata = InputMetadata.create(
                self.model_runner,
                forward_mode=ForwardMode.DECODE,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                prefix_lens=None,
                position_ids_offsets=position_ids_offsets,
                out_cache_loc=out_cache_loc,
                out_cache_cont_start=None,
                out_cache_cont_end=None,
                return_logprob=False,
                top_logprobs_nums=0,
                skip_flashinfer_init=True,
            )
            return self.model_runner.model.forward(
                input_ids, input_metadata.positions, input_metadata
            )

        for _ in range(4):
            run_once()

        torch.cuda.synchronize()
        with torch.cuda.graph(graph, pool=self.graph_memory_pool, stream=stream):
            out = run_once()
        torch.cuda.synchronize()
        self.graph_memory_pool = graph.pool()
        return graph, None, out

    def replay(self, batch: Batch):
        assert batch.out_cache_loc is not None
        assert not batch.return_logprob
        bs = len(batch.reqs)

        self.input_ids[:bs] = batch.input_ids
        self.req_pool_indices[:bs] = batch.req_pool_indices
        self.seq_lens[:bs] = batch.seq_lens
        self.position_ids_offsets[:bs] = batch.position_ids_offsets
        self.out_cache_loc[:bs] = batch.out_cache_loc

        init_flashinfer_args(
            ForwardMode.DECODE,
            self.model_runner,
            self.req_pool_indices[:bs],
            self.seq_lens[:bs],
            None,
            self.model_runner.flashinfer_decode_wrapper,
        )

        self.graphs[bs].replay()

        return self.output_buffers[bs]
