import torch
import time
import threading
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable
from sglang.srt.model_executor.graph_runner import get_global_graph_memory_pool

class ManagerConfig:
    def __init__(self,
                 keys: List[torch.Tensor],
                 values: List[torch.Tensor],
                 num_layers: int,
                 num_q_heads: int,
                 q_dtype: torch.dtype,
                 max_bs: int,
                 page_size: int,
                 retrive_budget_per_seq: int,
                 device: torch.device,
                 async_retrive: bool,
                 req_to_token: torch.Tensor,
                 max_seq_len: int,
                 stream_budget: Tuple[int, int],
                 is_cuda_graph: bool,
                 decode_cuda_graph_metadata: Optional[dict] = None
                 ):
        self.keys = keys
        self.values = values
        self.num_layers = num_layers
        self.num_q_heads = num_q_heads
        self.q_dtype = q_dtype
        self.max_bs = max_bs
        self.page_size = page_size
        self.retrive_budget_per_seq = retrive_budget_per_seq
        self.device = device
    
        self.async_retrive = async_retrive
        self.req_to_token = req_to_token
        self.max_seq_len = max_seq_len
        self.stream_budget = stream_budget
        self.is_cuda_graph = is_cuda_graph
        
        self.top_k = (self.retrive_budget_per_seq - self.stream_budget[0] - self.stream_budget[1]) // self.page_size
        self.head_num = self.keys[0].shape[1]
        self.head_dim = self.keys[0].shape[2]
        
class RetriveQuery:
    def __init__(self, config: ManagerConfig, layer_id: int):
        self.query = torch.empty(config.max_bs, config.num_q_heads*config.head_dim, device=config.device, dtype=config.q_dtype)
        self.req_pool_indices = torch.full(
            size=(config.max_bs,),
            fill_value=-1,
            dtype=torch.int32,
            device=config.device,
        )
        self.seq_lens = torch.empty(config.max_bs, device=config.device, dtype=torch.int32)
        # for quest
        # self.proxy_k_tensor = torch.zeros(
        #         size=(config.keys[0].shape[0]//config.page_size, 2, config.head_num, config.head_dim),
        #         device=config.device,
        #         dtype=config.q_dtype
        #     )
        
        # for average
        self.proxy_k_tensor = torch.zeros(
            size=(config.keys[0].shape[0]//config.page_size, 1, config.head_num, config.head_dim),
            device=config.device,
            dtype=config.q_dtype
        )
        
        self.count_steps = torch.zeros(
            config.max_bs,
            device=config.device,
            dtype=torch.int32,
        )

        self.score = torch.zeros(
            config.max_bs, # bs
            config.head_num,  # head_num
            (config.max_seq_len + config.page_size - 1) // config.page_size,
            dtype=torch.float32, 
            device=config.device)

        self.selected_page_indices = torch.full(
            (config.max_bs, config.head_num, config.top_k), 
            -1, 
            dtype=torch.int32, 
            device=config.device
        )
        self.updated = False
        self.layer_id = layer_id
        
class RetriveResult:
    def __init__(self, config: ManagerConfig, layer_id: int):
        self.retrive_budget_per_seq = config.retrive_budget_per_seq
        self.page_table = torch.zeros(
            (config.max_bs, config.head_num, config.max_seq_len // config.page_size),
            dtype=torch.int32,
            device=config.device,
        )
        self.retrived_cache_indices = torch.zeros(
            config.max_bs,
            config.retrive_budget_per_seq,
            dtype=torch.int32,
            device=config.device,
        )
        self.req_pool_indices = torch.full(
            size=(config.max_bs,),
            fill_value=-1,
            dtype=torch.int32,
            device=config.device,
        )
        self.seq_lens = torch.zeros(
            config.max_bs,
            dtype=torch.int32,
            device=config.device,
        )
        
        self.retrived_cache_indices_page = torch.full(
            (config.max_bs, config.head_num, config.top_k), 
            -1, 
            dtype=torch.int32, 
            device=config.device
        )
        self.updated = False
        self.layer_id = layer_id
        
    def copy_from(self, 
                  req_pool_indices: torch.Tensor, 
                  seq_lens: torch.Tensor, 
                  retrived_cache_indices_page: torch.Tensor
                ):
        self.req_pool_indices.copy_(req_pool_indices)
        self.seq_lens.copy_(seq_lens)
        self.retrived_cache_indices_page.copy_(retrived_cache_indices_page)
        self.updated = True
        
class RetriveCudaGraphRunner:
    def __init__(self, config: ManagerConfig, queries: List[RetriveQuery], results: List[RetriveResult], 
                retrive_function: Callable, stream: torch.cuda.Stream, device: torch.device):
        self.config = config
        self.queries = queries
        self.results = results
        self.stream = stream
        self.retrive_function = retrive_function
        self.graphs = [None for i in range(self.config.num_layers)]
        self.device_module = torch.get_device_module(device)
        self.capture_graph()
        
    def _capture_one_layer(self, layer_id: int):
        graph = torch.cuda.CUDAGraph()
        memory_pool = get_global_graph_memory_pool()
        with self.device_module.graph(graph, pool=memory_pool, stream=self.stream):
            self.retrive_function(self.queries[layer_id], self.results[layer_id])
        return graph
        
    def capture_graph(self):
        for layer_id in range(self.config.num_layers):
            self.graphs[layer_id] = self._capture_one_layer(layer_id)
    
    def replay(self, layer_id: int):
        with self.stream:
            self.graphs[layer_id].replay()
        
class CacheManager:
    def __init__(self, config: ManagerConfig):
        self.config = config
        
        self.stream = torch.cuda.Stream()
        self.start_retrive_event = [torch.cuda.Event(external=True) for _ in range(self.config.num_layers)]
        self.end_retrive_event = [torch.cuda.Event(external=True) for _ in range(self.config.num_layers)]

        self.retrived_result = [RetriveResult(self.config, layer_id) for layer_id in range(self.config.num_layers)]
        self.retrived_query = [RetriveQuery(self.config, layer_id) for layer_id in range(self.config.num_layers)]
        
        self.accumlation_step = self.config.page_size
        self._retrive_cache_indices = None
        
    def init_cuda_graph(self):
        self.graph_runner = RetriveCudaGraphRunner(self.config, self.retrived_query, self.retrived_result, 
                                                   self._retrive_one_layer, self.stream, self.config.device)
        
    
    def update_query(self, query: torch.Tensor, req_pool_indices: torch.Tensor, seq_lens: torch.Tensor, 
                    layer_id: int):

        bs = req_pool_indices.shape[0]
        self.retrived_query[layer_id].query[:bs] = query[:bs]
        self.retrived_query[layer_id].req_pool_indices[:bs] = req_pool_indices
        self.retrived_query[layer_id].seq_lens[:bs] = seq_lens
        self.retrived_query[layer_id].updated = True
        
        self._call_after_update_query(
            key_cache=self.config.keys[layer_id],
            req_to_token=self.config.req_to_token,
            req_pool_indices=req_pool_indices, 
            page_size=self.config.page_size,
            seq_lens=seq_lens,
            count_steps=self.retrived_query[layer_id].count_steps, # [bs]
            accumlation_step=self.accumlation_step,
            proxy_k_tensor=self.retrived_query[layer_id].proxy_k_tensor,
        )
        
    def call_after_init_cuda_graph(self):
        for layer_id in range(self.config.num_layers):
            self.retrived_query[layer_id].req_pool_indices.fill_(-1)
            self.retrived_result[layer_id].req_pool_indices.fill_(-1)

    def get_result(self, layer_id: int):
        return self.retrived_result[layer_id]
    
    def start_retrive_loop(self):
        if not self.config.async_retrive:
            return
        self.loop = threading.Thread(target=self._retrive_loop, daemon=True)
        self.loop.start()
        
    def _retrive_one_layer(self, query: RetriveQuery, result: RetriveResult):
        with self.stream:
            self._retrive_cache_indices(
                query=query.query, 
                proxy_k_tensor=query.proxy_k_tensor, 
                req_to_token=self.config.req_to_token,
                req_pool_indices=query.req_pool_indices,
                seq_lens=query.seq_lens, 
                top_k=self.config.top_k,
                score=query.score,
                selected_page_indices=query.selected_page_indices,
                layer_id=query.layer_id,
            )
            result.copy_from(
                req_pool_indices=query.req_pool_indices, 
                seq_lens=query.seq_lens, 
                retrived_cache_indices_page=query.selected_page_indices,
            )
    
    def _retrive_loop(self):
        while True:
            for layer_id in range(self.config.num_layers):
                if self.retrived_query[layer_id].updated:
                    if self.config.is_cuda_graph:
                        self.graph_runner.graphs[layer_id].replay()
                    else:
                        self._retrive_one_layer(self.retrived_query[layer_id], self.retrived_result[layer_id])
                time.sleep(0.001)
