import torch
import time
import threading
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ManagerConfig:
    # kv_cache config
    keys: List[torch.Tensor]
    values: List[torch.Tensor]
    req_to_token: torch.Tensor
    
    use_cuda_graph: bool
    max_bs: int
    max_seq_len: int

    # page config
    page_size: int
    retrive_budget_per_seq: int
    device: torch.device
    async_retrive: bool
    num_layers: int
    num_q_heads: int
    q_dtype: torch.dtype
    top_k: int
    stream_budget: Tuple[int, int]
    
    # cuda graph
    is_cuda_graph: bool
    decode_cuda_graph_metadata: Optional[dict] = None
    
class RetriveQuery:
    def __init__(self, config: ManagerConfig):
        self.query = torch.empty(config.max_bs, config.num_q_heads*config.keys[0].shape[2], device=config.device, dtype=config.q_dtype)
        self.req_pool_indices = torch.full(
            size=(config.max_bs,),
            fill_value=-1,
            dtype=torch.int32,
            device=config.device,
        )
        self.seq_lens = torch.empty(config.max_bs, device=config.device, dtype=torch.int32)
        # for quest
        # self.proxy_k_tensor = torch.zeros(
        #         size=(config.keys[0].shape[0]//config.page_size, 2, config.keys[0].shape[1], config.keys[0].shape[2]),
        #         device=config.device,
        #         dtype=config.keys[0].dtype
        #     )
        
        # for average
        self.proxy_k_tensor = torch.zeros(
            size=(config.keys[0].shape[0]//config.page_size, 1, config.keys[0].shape[1], config.keys[0].shape[2]),
            device=config.device,
            dtype=config.keys[0].dtype
        )
        
        self.count_steps = torch.zeros(
            config.max_bs,
            device=config.device,
            dtype=torch.int32,
        )

        self.score = torch.zeros(
            config.max_bs, # bs
            config.keys[0].shape[1],  # head_num
            config.keys[0].shape[0] // config.page_size,  # max_num_pages
            dtype=config.keys[0].dtype, 
            device=config.device)

        self.selected_page_indices = torch.full(
            (config.max_bs, config.keys[0].shape[1], config.top_k), 
            -1, 
            dtype=torch.int32, 
            device=config.device
        )
        self.updated = False
        
class RetriveResult:
    def __init__(self, config: ManagerConfig):
        self.retrive_budget_per_seq = config.retrive_budget_per_seq
        self.page_table = torch.zeros(
            (config.max_bs, config.keys[0].shape[1], config.max_seq_len // config.page_size),
            dtype=torch.int32,
            device=config.device,
        )
        self.retrived_cache_indices = torch.zeros(
            config.max_bs,
            config.retrive_budget_per_seq,
            dtype=torch.int32,
            device=config.device,
        )
        self.scores = torch.zeros(
            config.max_bs,
            (config.max_seq_len + config.page_size - 1) // config.page_size,
            dtype=torch.float32,
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
            (config.max_bs, config.keys[0].shape[1], config.top_k), 
            -1, 
            dtype=torch.int32, 
            device=config.device
        )
        self.updated = False
        
    def copy_from(self, 
                  req_pool_indices: torch.Tensor, 
                  seq_lens: torch.Tensor, 
                  retrived_cache_indices_page: torch.Tensor
                ):
        self.req_pool_indices.copy_(req_pool_indices)
        self.seq_lens.copy_(seq_lens)
        self.retrived_cache_indices_page.copy_(retrived_cache_indices_page)
        self.updated = True

class CacheManager:
    def __init__(self, config: ManagerConfig):
        self.config = config
        
        self.stream = torch.cuda.Stream()
        self.start_retrive_event = [torch.cuda.Event(external=True) for _ in range(self.config.num_layers)]
        self.end_retrive_event = [torch.cuda.Event(external=True) for _ in range(self.config.num_layers)]

        self.retrived_result = [RetriveResult(self.config) for _ in range(self.config.num_layers)]
        self.retrived_query = [RetriveQuery(self.config) for _ in range(self.config.num_layers)]
        
        self.accumlation_step = self.config.page_size
        
    def update_query(self, query: torch.Tensor, req_pool_indices: torch.Tensor, seq_lens: torch.Tensor, 
                    layer_id: int):

        bs = req_pool_indices.shape[0]
        self.retrived_query[layer_id].query[:bs] = query[:bs]
        self.retrived_query[layer_id].req_pool_indices[:bs] = req_pool_indices
        self.retrived_query[layer_id].seq_lens[:bs] = seq_lens
        self.retrived_query[layer_id].updated = True
        self.retrived_query[layer_id].bs = bs
        
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
    
    def _retrive_loop(self):
        with self.stream:
            while True:
                for layer_id in range(self.config.num_layers):
                    if self.retrived_query[layer_id].updated:
                        query = self.retrived_query[layer_id]
                        self._retrive_cache_indices(
                            query=query.query, 
                            proxy_k_tensor=query.proxy_k_tensor, 
                            req_to_token=self.config.req_to_token,
                            req_pool_indices=query.req_pool_indices,
                            seq_lens=query.seq_lens, 
                            top_k=self.config.top_k,
                            score=query.score,
                            selected_page_indices=query.selected_page_indices,
                        )
                        self.retrived_result[layer_id].copy_from(
                            req_pool_indices=query.req_pool_indices, 
                            seq_lens=query.seq_lens, 
                            retrived_cache_indices_page=query.selected_page_indices,
                        )
                        #self.retrived_query[layer_id].updated = False
                    time.sleep(0.02)
