import torch
import torch_npu
from typing import Tuple


def qk_mul_LBHD(query,key):
    L_q,B,H_q,D = query.shape
    L_kv,B,H_kv,D = key.shape
    scale = D**(-0.5)
    query_reshape = query.view(L_q, B, H_kv, H_q//H_kv,  D)
    attn_weights = torch.einsum('qbhzd,kbhd->bhzqk', query_reshape, key) * scale
    attn_weights = attn_weights.view(B, H_q, L_q, L_kv)
    return attn_weights

def qk_mul(query,key):
    B,H_q,L_q,D = query.shape
    B,H_kv,L_kv,D = key.shape
    scale = D**(-0.5)
    if H_q == H_kv:
        attn_weights = torch.matmul(query, key.transpose(2, 3)) * scale
    else:
        query_reshape = query.view(B, H_kv, H_q//H_kv, L_q, D)
        attn_weights = torch.einsum('bhzqd,bhkd->bhzqk', query_reshape, key) * scale
        attn_weights = attn_weights.view(B, H_q, L_q, L_kv)
    return attn_weights

def kv_cache_update_(
        kv_cache_stat, #[B,HN_kv,1,L] FP32
        kv_cache_pos, #[B,HN_kv,1,L] FP32 (better to be int32 but big overhead, operations on AI_CPU)
        kv_cache_size_old_guard, #int
        kv_cache_size_young_guard, #int
        key_cache, #[B,HN_kv,L,D] FP16/BF16
        value_cache, #[B,HN_kv,L,D] FP16/BF16
        query, #[B,HN_q,1,D] FP16/BF16
        key, #[B,HN_kv,1,D] FP16/BF16
        value #[B,HN_kv,1,D] FP16/BF16
    ):
    stat_dim_L = 3
    stat_dim_HN = 1
    
    #guard young positions
    kv_cache_pos_last = kv_cache_pos.max(dim=3,keepdim=True).values
    current_total_length = kv_cache_pos_last + 1
    pos_young_guard = current_total_length - kv_cache_size_young_guard

    guard_mask = (kv_cache_pos >= pos_young_guard).float()
    kv_cache_mask_by_stat = (kv_cache_stat.narrow(stat_dim_HN,0,1) >= 0).float()
    guard_mask = guard_mask*kv_cache_mask_by_stat
    
    #set max statistics for guarded positions to avoid to be pruned
    mask_value_max = torch.finfo(kv_cache_stat.dtype).max
    kv_cache_stat_masked = guard_mask*mask_value_max + (1-guard_mask)*kv_cache_stat
    kv_cache_stat_masked = kv_cache_stat_masked.narrow(stat_dim_L, kv_cache_size_old_guard, kv_cache_stat_masked.shape[stat_dim_L] - kv_cache_size_old_guard)
    
    #find positions to prune on this iter
    kv_cache_prune_idx = torch.min(kv_cache_stat_masked, dim=stat_dim_L, keepdim=True).indices
    kv_cache_prune_idx = kv_cache_prune_idx + kv_cache_size_old_guard

    #update kv_cache
    L,B,HN,D = key_cache.shape
    kv_idx0 = torch.arange(0, B*HN, 1, dtype=kv_cache_prune_idx.dtype, device=kv_cache_prune_idx.device )
    kv_idx1 = kv_cache_prune_idx.view(-1) * B*HN + kv_idx0
    torch_npu.npu_scatter_nd_update_(key_cache.view(L*B*HN,-1), kv_idx1.view(-1,1), key.reshape(B*HN,-1))
    torch_npu.npu_scatter_nd_update_(value_cache.view(L*B*HN,-1), kv_idx1.view(-1,1), value.reshape(B*HN,-1))

    #update kv_cache_stat and kv_cache_pos
    idx0 = torch.arange(0, B*HN*L, L, dtype=kv_cache_prune_idx.dtype, device=kv_cache_prune_idx.device )
    idx1 = idx0 + kv_cache_prune_idx.view(-1)
    stat0 = torch.zeros([B*HN,1], dtype=kv_cache_stat.dtype, device=kv_cache_stat.device)
    torch_npu.npu_scatter_nd_update_(kv_cache_stat.view(B*HN*L,-1), idx1.view(-1,1), stat0)
    torch_npu.npu_scatter_nd_update_(kv_cache_pos.view(B*HN*L,-1), idx1.view(-1,1), current_total_length.view(B*HN,-1))        
        
    #calc new weights
    attn_weights_qi = qk_mul_LBHD(query, key_cache) # output is B,Hq,L_q,L_kv
    #set unmasked values into float.min
    kv_cache_mask_by_stat = (kv_cache_stat.narrow(stat_dim_HN,0,1) >= 0).float()    
    attn_weights_qi = kv_cache_mask_by_stat * attn_weights_qi + (1-kv_cache_mask_by_stat) * torch.finfo(attn_weights_qi.dtype).min
    attn_weights_qi = torch.nn.functional.softmax(attn_weights_qi, dim=stat_dim_L, dtype=torch.float32).float()
    
    #mean values for single KV head
    B, q_hn, q_len, kv_len = attn_weights_qi.shape
    num_kv_heads = key.shape[2]
    num_q_heads = query.shape[2]
    repeat_num = num_q_heads//num_kv_heads
    attn_weights_q_reshaped = attn_weights_qi.view(B, num_kv_heads, repeat_num, 1, kv_len).float()
    attn_weights_qi = attn_weights_q_reshaped.mean(dim=2, keepdim=False)
    
    #update statistics w/inplace add 
    kv_cache_stat.add_(attn_weights_qi.float())

def get_layout_as_list(l):
    if isinstance(l,str):
        l = l.split(',')
    return l

def get_layout_dim_index(l, dim_name):
    l_list = get_layout_as_list(l)
    return l_list.index(dim_name)

def permute(x, layout_in, layout_out):
    if layout_in == layout_out:
        return x
    layout_in = get_layout_as_list(layout_in)
    layout_out = get_layout_as_list(layout_out)
    permute_args = [layout_in.index(l) for l in layout_out]
    y = x.permute(*permute_args)
    return y

MASK_TENSORS = {}
def get_type_value(w, type, dtype=None):
    if dtype == None:
        dtype = w.dtype
    id = (dtype, w.device, type)
    if id not in MASK_TENSORS:
        val = getattr(torch.finfo(w.dtype),type)
        MASK_TENSORS[id] = torch.tensor(val, device=w.device, dtype = w.dtype)
    return MASK_TENSORS[id]

def get_type_min(attn_weights, dtype=None):
    return get_type_value(attn_weights,'min')


CROP_MASK_DICT = {}

def get_casual_mask_crop(q_len, kv_len, B,HN, device):
    name = "x".join(str(t) for t in [q_len, kv_len, B, HN, device])
    if name not in CROP_MASK_DICT:
        causal_mask_internal = torch.tril(
            torch.ones((q_len, kv_len), dtype=torch.bool),
            diagonal=kv_len-q_len,
        )
        causal_mask_internal = causal_mask_internal.to(device)
        causal_mask_internal = causal_mask_internal.view(1,1,q_len, kv_len)
        causal_mask_internal = causal_mask_internal.expand(B,HN,q_len, kv_len)
        CROP_MASK_DICT[name] = causal_mask_internal
    return  CROP_MASK_DICT[name]

def calc_guard_sizes(kv_cache_pruning_config, kv_cache_size):
    #forms real guard
    kv_cache_size_young_guard = kv_cache_pruning_config['kv_cache_prune_size_young_guard']
    kv_cache_size_old_guard = kv_cache_pruning_config['kv_cache_prune_size_old_guard']

    #calc old and young guards
    if kv_cache_size_young_guard + kv_cache_size_old_guard > kv_cache_size:
        if kv_cache_size_old_guard*2 > kv_cache_size:
            kv_cache_size_old_guard = kv_cache_size//2
        kv_cache_size_young_guard = kv_cache_size - kv_cache_size_old_guard
    return kv_cache_size_old_guard, kv_cache_size_young_guard

def real_prune_prefill_get_default_state(kv_cache_pruning_config, batch_size, device):
    head_split_list = kv_cache_pruning_config["head_split_list"]
    head_size_list  = kv_cache_pruning_config["head_size_list"]
    kv_cache_state_list = []
    for head_num, kv_cache_size in zip(head_split_list, head_size_list):

        kv_cache_pos_hg = torch.arange(kv_cache_size, dtype=torch.float, device=device)
        kv_cache_pos_hg = kv_cache_pos_hg.view(1,1,1,kv_cache_size).expand(batch_size, head_num, 1, kv_cache_size).contiguous()
        kv_cache_stat_hg = kv_cache_pos_hg - kv_cache_size

        kv_cache_size_old_guard, kv_cache_size_young_guard = calc_guard_sizes(kv_cache_pruning_config, kv_cache_size)

        #form default empty state
        kv_cache_state_list.append({
            'kv_cache_size': kv_cache_size,
            'kv_cache_size_young_guard': kv_cache_size_young_guard,
            'kv_cache_size_old_guard': kv_cache_size_old_guard,
            "kv_cache_stat": kv_cache_stat_hg,
            "kv_cache_pos": kv_cache_pos_hg,
            "kv_cache_mask": None
        })

    return kv_cache_state_list


def real_prune_prefill_pack(q_head_num, k_head_num, 
                            kv_cache_pruning_config,
                            cache_position,
                            query_states, key_states, value_states,
                            keys, values,
                            layout,
                            page_padding = 0):
    q_len,_, _ = query_states.shape
    kv_len =  q_len # NB:use of q_len
    
    B=1  # only batch_size==1 is supported now
    query_states = permute(query_states.reshape(1, q_len, q_head_num, -1), "B,S,N,D", "B,N,S,D")
    key_states = permute(key_states.reshape(1, kv_len, k_head_num, -1), "B,S,N,D", "B,N,S,D")
    value_states = permute(value_states.reshape(1, kv_len, k_head_num, -1), "B,S,N,D", "B,N,S,D")

    keys = permute(keys.narrow(0, page_padding, keys.shape[0]-page_padding), layout, "B,N,S,D")  # some pages can be reserved (e.g. please see _create_buffers for Ascend)
    kv_cache_size = keys.shape[-2]
    values = permute(values.narrow(0, page_padding, values.shape[0]-page_padding), layout, "B,N,S,D")   # some pages can be reserved (e.g. please see _create_buffers for Ascend)
    HN_q = q_head_num
    HN_kv = k_head_num
    prefill_stat_size = kv_cache_pruning_config["prefill_stat_size"]
    #calc small number of weights for initial statistics
    stat_size = min(prefill_stat_size, q_len)
    query_states_crop = query_states[:,:,-stat_size:]
    attn_weights = qk_mul(query_states_crop, key_states)

    causal_mask_crop_stat_size = get_casual_mask_crop(stat_size, stat_size, B, HN_q, attn_weights.device)
    causal_mask_crop_stat_size = torch.logical_not(causal_mask_crop_stat_size)
    attn_weights[:,:,-stat_size:,-stat_size:].masked_fill_(causal_mask_crop_stat_size, get_type_min(attn_weights))    
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).float()

    if HN_kv < HN_q:
        #merge statistics from multiple query heads !!!!!!!!!!!!!!!!!!!!!
        attn_weights_reshaped = attn_weights.view(B, HN_kv, HN_q//HN_kv, -1, kv_len)
        attn_weights = attn_weights_reshaped.mean(dim=2)   
    kv_cache_stat_join = attn_weights.sum(dim=2, keepdim=True)
    
    head_split_list = kv_cache_pruning_config["head_split_list"]
    head_size_list  = kv_cache_pruning_config["head_size_list"]
    #split along heads
    kv_cache_stat_list = torch.split(kv_cache_stat_join, head_split_list, dim=1)
    keys = list(torch.split(keys, head_split_list, dim=1))
    values = list(torch.split(values, head_split_list, dim=1))
    key_states_list = list(torch.split(key_states, head_split_list, dim=1))
    value_states_list = list(torch.split(value_states, head_split_list, dim=1))
    
    kv_cache_state_list = []
    kv_cache_pos_hg_base = cache_position.view(1,1,1,kv_len).float() - page_padding  # e.g. first page is dumb in Ascend code-path
   
    for hg,kv_cache_stat_hg in enumerate(kv_cache_stat_list):
        Lkv_origin = kv_cache_stat_hg.shape[-1]
        kv_cache_pos_hg = kv_cache_pos_hg_base.expand_as(kv_cache_stat_hg)
        keys_hg = key_states_list[hg]
        values_hg = value_states_list[hg]

        #forms real guard
        kv_cache_size_young_guard = kv_cache_pruning_config['kv_cache_prune_size_young_guard']
        kv_cache_size_old_guard = kv_cache_pruning_config['kv_cache_prune_size_old_guard']
        kv_cache_size = head_size_list[hg]


        if kv_cache_size_young_guard + kv_cache_size_old_guard > kv_cache_size:
            if kv_cache_size_old_guard*2 > kv_cache_size:
                kv_cache_size_old_guard = kv_cache_size//2
            kv_cache_size_young_guard = kv_cache_size - kv_cache_size_old_guard

        #save as state for next iterations
        prune_state_hg = {
            'kv_cache_size': kv_cache_size,
            'kv_cache_size_young_guard': kv_cache_size_young_guard,
            'kv_cache_size_old_guard': kv_cache_size_old_guard
        }

        if Lkv_origin  > kv_cache_size:
            #query is too big for the kv-cache. lets find what to discard
            kv_cache_size_guard = kv_cache_size_young_guard + kv_cache_size_old_guard
            prune_num_selective = kv_cache_size - kv_cache_size_guard #extra 1 to have prune position on next iter before atten
            #prune stat and pos and key and values
            if prune_num_selective>1: #this is not sliding window
                kv_cache_stat_head_size = Lkv_origin - kv_cache_size_guard
                kv_cache_stat_head_masked = kv_cache_stat_hg.narrow(3, kv_cache_size_old_guard, kv_cache_stat_head_size)
                select_val, select_idx = torch.topk(kv_cache_stat_head_masked, prune_num_selective, dim=3, largest=True, sorted=False)
                select_idx = select_idx + kv_cache_size_old_guard
                def gather(src, index, dim):
                    dst_list = [ 
                        src.narrow(dim, 0, kv_cache_size_old_guard),
                        src.gather(dim=dim, index=index),
                        src.narrow(dim, -kv_cache_size_young_guard, kv_cache_size_young_guard)
                    ]
                    return torch.cat(dst_list, dim=dim)

                kv_cache_stat_hg = gather(kv_cache_stat_hg, select_idx, dim=-1)
                kv_cache_pos_hg = gather(kv_cache_pos_hg, select_idx, dim=-1)

                B,HN,_,L_dst = select_idx.shape
                B,HN,L_src,D = keys_hg.shape
                index = select_idx.view(B,HN,L_dst,1)
                index = index.expand(-1,-1,-1,D)
                keys_hg = gather(keys_hg, index, dim=-2)
                values_hg = gather(values_hg, index, dim=-2)
            else:
                #sliding window head.
                def gather_SW(src, nums, dim):
                    dst_list = [ 
                        src.narrow(dim, 0, nums[0]),
                        src.narrow(dim, -nums[1], nums[1])
                    ]
                    return torch.cat(dst_list, dim=dim)
                
                young_max = Lkv_origin - kv_cache_size_old_guard
                nums = (kv_cache_size_old_guard, min(young_max, kv_cache_size_young_guard+1))

                kv_cache_stat_hg = gather_SW(kv_cache_stat_hg, nums, dim=-1)
                kv_cache_pos_hg = gather_SW(kv_cache_pos_hg, nums, dim=-1)
                keys_hg = gather_SW(keys_hg, nums,  dim=-2)
                values_hg = gather_SW(values_hg, nums, dim=-2)
            
        Lkv = kv_cache_stat_hg.shape[-1]

        #forms fixed sized buffer
        pad = kv_cache_size - Lkv
        if pad > 0:
            kv_cache_stat_pad = torch.arange(pad).to(kv_cache_stat_hg)-pad
            kv_cache_stat_pad = kv_cache_stat_pad.view(1,1,1,pad).expand(*kv_cache_stat_hg.shape[:-1],pad)
            kv_cache_stat_hg = torch.cat([kv_cache_stat_hg, kv_cache_stat_pad], dim=-1)
            kv_cache_pos_hg = torch.cat([kv_cache_pos_hg, kv_cache_stat_pad], dim=-1)
            #pad values and keys to have fullsized buffers
            keys_hg = torch.nn.functional.pad(keys_hg, (0,0,0,pad))
            values_hg = torch.nn.functional.pad(values_hg, (0,0,0,pad))

        #save resulting tensor dense to kv-cache 
        keys[hg].copy_(keys_hg.contiguous())
        values[hg].copy_(values_hg.contiguous())
        assert kv_cache_stat_hg.shape[0] == 1, "only batch_size==1 is supported now"
        prune_state_hg["kv_cache_stat"] = kv_cache_stat_hg.contiguous()
        prune_state_hg["kv_cache_pos"] = kv_cache_pos_hg.contiguous()
        
        kv_cache_state_list.append(prune_state_hg)

    return kv_cache_state_list
    
def real_prune_update(
    prune_state,
    kv_cache_pruning_config,
    q_head_num, k_head_num,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    query_states: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    page_padding = 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    dim_HN = 2
    q_len,_ = query_states.shape
    query_states = query_states.reshape(1, q_len, q_head_num, -1)
    key_states = key_states.reshape(key_states.shape[0], 1, key_states.shape[1],key_states.shape[2])
    value_states = value_states.reshape(value_states.shape[0], 1, value_states.shape[1],value_states.shape[2])
  
    keys = keys.narrow(0, page_padding, keys.shape[0]-page_padding)  # some pages can be reserved (e.g. please see _create_buffers for Ascend)
    values = values.narrow(0, page_padding, values.shape[0]-page_padding)   # some pages can be reserved (e.g. please see _create_buffers for Ascend)
    kv_head_num_list = kv_cache_pruning_config["head_split_list"]
           
    assert k_head_num==keys.shape[dim_HN]
    assert k_head_num==values.shape[dim_HN]
    #split along heads
    keys = list(torch.split(keys, kv_head_num_list, dim=dim_HN))
    values = list(torch.split(values, kv_head_num_list, dim=dim_HN))
    key_states = list(torch.split(key_states, kv_head_num_list, dim=dim_HN))
    value_states = list(torch.split(value_states, kv_head_num_list, dim=dim_HN))
    q_head_num_list = [(n*q_head_num)//sum(kv_head_num_list) for n in kv_head_num_list]
    query_states = list(torch.split(query_states, q_head_num_list, dim=dim_HN))   
    
    for hg,state in enumerate(prune_state):
        kv_cache_update_(
            state['kv_cache_stat'], #[B,HN_kv,1,L] FP32
            state['kv_cache_pos'], #[B,HN_kv,1,L] FP32 (better to be int32 but big overhead, operations on AI_CPU)
            state['kv_cache_size_old_guard'], #int
            state['kv_cache_size_young_guard'], #int
            keys[hg],   # "L,B,HN,D"
            values[hg], # "L,B,HN,D"
            query_states[hg], 
            key_states[hg],
            value_states[hg]
        )
        
    return key_states, value_states, query_states
