import torch


import os
import json
import logging

from sglang.srt.utils import get_bool_env_var

#default config for prune.
_kv_cache_pruning_config = {
    "kv_cache_prune_size_scale": 1,
    "zero_attention_output": 0,
    "prefill_chunk_stat_scale": "max_no_old,no_old_poolmaxil6",
    "prefill_chunk_stat_head_merge": "mean",
    "prefill_chunk_stat_new": 1,
    "kv_cache_prune_size_young_guard": 408,
    "kv_cache_prune_size_old_guard": 16,
    "prefill_stat_size" : 128,
    # for chunked prefill:
    "prefill_chunk_stat_separate": 0,
    "prefill_chunk_stat_separate_merge_prefill": 0.5,            
    "prefill_chunk_stat_separate_merge_decode": 0.5,
    "prefill_chunk_tail": 128,
    "prefill_chunk_tail2":0,
}

def get_default_kv_cache_pruning_config():
    return dict(_kv_cache_pruning_config)

def read_kv_cache_pruning_config(kv_cache_pruning_config_param, layer_num, page_size, num_kv_heads, attn_tp_rank, attn_tp_size):
    #init kv_cache_pruning_config
    if kv_cache_pruning_config_param is None:
        return None
    
    if not get_bool_env_var("ASCEND_USE_FIA", "False"):  # initial impl is only for FIA
        raise RuntimeError(f'Scheduler {get_bool_env_var("ASCEND_USE_FIA", "False")=} KV-CACHE PRUNING is currently implemented  only for Ascend FIA')
    
    _kv_cache_pruning_config = get_default_kv_cache_pruning_config()

    kv_cache_size = None
    if isinstance(kv_cache_pruning_config_param, int):
        kv_cache_size = kv_cache_pruning_config_param
    elif kv_cache_pruning_config_param.isdigit():
        kv_cache_size = int(kv_cache_pruning_config_param)

    if kv_cache_size is not None:
        if kv_cache_size <=0:
            logging.warning(f" {kv_cache_pruning_config_param=} disables kv_cache_pruning")
            return None
        else:                
            # pad 
            kv_cache_size_in_pages = (int(kv_cache_size-1) // page_size) + 1
            kv_cache_size = kv_cache_size_in_pages*page_size - 1# without 1 because real occupation is kv_cahcE_size+1
            #signle groups with all heads in them
            head_split_list = [num_kv_heads]
            head_size_list = [kv_cache_size]
            _kv_cache_pruning_config['head_layer_groups'] = [(head_split_list, head_size_list)]*layer_num
            return _kv_cache_pruning_config

    #kv_cache_pruning_config_param is not a digit. lets try to read as config json file
    if not os.path.isfile(kv_cache_pruning_config_param):
        return None

    with open(kv_cache_pruning_config_param, "rt") as inp:
        _kv_cache_pruning_config.update(json.load(inp))

    #calc and split heads sizes
    kv_cache_prune_size_layer_shift = _kv_cache_pruning_config['kv_cache_prune_size_layer_shift']
    
    head_layer_groups = []
    for ls in kv_cache_prune_size_layer_shift:
        def page_align(s):
            return ((s+page_size-1)//page_size)*page_size
        def size_scale(s):
            scale = _kv_cache_pruning_config['kv_cache_prune_size_scale']
            if scale==1:
                return s
            guard = _kv_cache_pruning_config['kv_cache_prune_size_young_guard'] + _kv_cache_pruning_config['kv_cache_prune_size_old_guard']
            return (s-guard)*scale+guard
        
        
        if ls:
            per_rank = len(ls)//attn_tp_size
            ls = ls[attn_tp_rank*per_rank:(attn_tp_rank+1)*per_rank]
            kv_cache_size_list = [page_align(1+size_scale(s+_kv_cache_pruning_config['kv_cache_prune_size']))-1 for s in ls]
        else:
            kv_cache_size_list = [page_align(1+size_scale(_kv_cache_pruning_config['kv_cache_prune_size']))-1]*num_kv_heads
        
        def _kv_cache_size_groups(kv_cache_size_list):
            assert isinstance(kv_cache_size_list, list)
            head_split_list = []
            head_size_list = []
            for s in kv_cache_size_list:
                if not head_size_list or s != head_size_list[-1]:
                    head_split_list.append(0)
                    head_size_list.append(s)
                head_split_list[-1] += 1
            return head_split_list, head_size_list
        
        groups = _kv_cache_size_groups(kv_cache_size_list)
        head_layer_groups.append(groups)

    hnhs_acc = 0
    hn_acc = 0
    for head_split_list, head_size_list in head_layer_groups:
        for hn,hs in zip(head_split_list, head_size_list):
            hnhs_acc += hn*hs
            hn_acc += hn
    
    _kv_cache_pruning_config['head_layer_groups'] = head_layer_groups
    tail_size = _kv_cache_pruning_config['prefill_chunk_tail'] + _kv_cache_pruning_config['prefill_chunk_tail2']
    tail_size_align = page_align(tail_size)
    if tail_size != tail_size_align:
        tail_size1 = (_kv_cache_pruning_config['prefill_chunk_tail'] * tail_size_align) // tail_size
        tail_size2 = tail_size_align - tail_size1
        _kv_cache_pruning_config['prefill_chunk_tail'] = tail_size1
        _kv_cache_pruning_config['prefill_chunk_tail2'] = tail_size2


    return _kv_cache_pruning_config



#functions to multiply QxK with L,B,H,D layout L=seq length, B is batch_size, H is head num, D is head dimention size
def qk_mul_LBHD(query, key):
    L_q,B,H_q,D = query.shape
    L_kv,B,H_kv,D = key.shape
    scale = D**(-0.5)
    query_reshape = query.view(L_q, B, H_kv, H_q//H_kv,  D)
    attn_weights = torch.einsum('qbhzd,kbhd->bhzqk', query_reshape, key) * scale
    attn_weights = attn_weights.view(B, H_q, L_q, L_kv)
    return attn_weights

def get_layout_as_list(layout):
    if isinstance(layout,str):
        layout = layout.split(',')
    return layout

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

#dict with some const tensors
CROP_MASK_DICT = {}

def get_casual_mask_crop(q_len, kv_len, B, HN, device):
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

def real_prune_prefill_pack_empty_state(kv_cache_config, layer_idx, batch_size, device):
    head_split_list, head_size_list = kv_cache_config['head_layer_groups'][layer_idx]

    #create empty statistics for each head group (heads with same number of kv cache size)
    kv_cache_state_list = []
    for HN_kv, kv_cache_size in zip(head_split_list, head_size_list):

        ch_num = 2 if kv_cache_config['prefill_chunk_stat_separate'] else 1
        kv_cache_stat_hg_2ch = torch.zeros([batch_size, HN_kv, ch_num, 0], dtype=torch.float32, device=device)        
        kv_cache_pos_hg      = torch.zeros([batch_size, HN_kv, 1, 0],      dtype=torch.float32, device=device)
        
        kv_cache_size_old_guard, kv_cache_size_young_guard = calc_guard_sizes(kv_cache_config, kv_cache_size)

        #save as state for next iterations
        prune_state_hg = {
            'kv_cache_head_num': HN_kv,
            'kv_cache_size': kv_cache_size,
            'kv_cache_size_young_guard': kv_cache_size_young_guard,
            'kv_cache_size_old_guard': kv_cache_size_old_guard,
            "kv_cache_stat_2ch": kv_cache_stat_hg_2ch,
            "kv_cache_pos": kv_cache_pos_hg,
            "kv_cache_pruned_num": 0,
        }
            
        kv_cache_state_list.append(prune_state_hg)
    
    return kv_cache_state_list

#function that detects most useless KV positions and eliminate them with repack for each head group separatly
def real_prune_prefill_pack_(
    kv_cache_pruning_config,
    q_head_num,
    k_head_num, 
    query_states,
    keys,
    values,
    layout,
    current_prune_state,
    is_first_chunk,
    is_last_chunk,
    layer_idx,
    page_padding = 0
):
    
    q_len,_, _ = query_states.shape
    query_states = query_states.unsqueeze(1) #L,H,D -> Lq,1,H,D
    
    if is_last_chunk:
        #last chunk has no guiding tail
        prefill_chunk_tail = 0
    else:
        prefill_chunk_tail = kv_cache_pruning_config['prefill_chunk_tail'] + kv_cache_pruning_config['prefill_chunk_tail2']

    HN_q = q_head_num
    HN_kv = k_head_num
    prefill_stat_size = kv_cache_pruning_config["prefill_stat_size"]

    stat_size = min(prefill_stat_size, q_len)

    #get dimention indexes for seq length (L) and head num (H)
    dim_L = get_layout_dim_index(layout, 'L')
    dim_HN = get_layout_dim_index(layout, 'HN')

    #get heads groups (all heads in signle groups has same Kv cache size )
    head_split_list, head_size_list = kv_cache_pruning_config['head_layer_groups'][layer_idx]
    #head_split_list = kv_cache_pruning_config["head_split_list"]
    q_head_split_list = [(query_states.shape[dim_HN] * h) // sum(head_split_list) for h in head_split_list]
    query_new_list = torch.split(query_states, q_head_split_list, dim=dim_HN)
    if False:
        keys_list = list(torch.split(keys, head_split_list, dim=dim_HN))
        values_list = list(torch.split(values, head_split_list, dim=dim_HN))    
    else:
        keys_list  = []
        values_list = []
        hs = 0
        for hn in head_split_list:
            keys_list.append( keys.narrow(dim_HN, hs, hn) )
            values_list.append( values.narrow(dim_HN, hs, hn) )
            hs += hn
   
    #process each heads group separatly
    for hg, state_hg in enumerate(current_prune_state):
        kv_cache_pos_old = state_hg["kv_cache_pos"]
        kv_len_add = q_len-prefill_chunk_tail
        kv_len_old = kv_cache_pos_old.shape[-1]
        kv_len_new = kv_len_old + kv_len_add

        keys_hg = keys_list[hg].narrow(dim_L, page_padding, kv_len_new)
        values_hg = values_list[hg].narrow(dim_L, page_padding, kv_len_new)

        kv_cache_pos_add = torch.arange(kv_len_add)
        kv_cache_pos_add = kv_cache_pos_add.to(kv_cache_pos_old)
        kv_cache_pos_add = kv_cache_pos_add.view(1,1,1,kv_len_add)
        kv_cache_pos_add = kv_cache_pos_add.expand(*kv_cache_pos_old.shape[:-1],kv_len_add)

        if is_first_chunk:
            #init pos tensor
            kv_cache_pos_new = kv_cache_pos_add
        else:
            #update pos tensor
            pos_next = kv_cache_pos_old.max(dim=-1, keepdim=True).values + 1
            kv_cache_pos_add = kv_cache_pos_add + pos_next
            kv_cache_pos_new = torch.cat([kv_cache_pos_old, kv_cache_pos_add], dim=-1)
        
        kv_cache_size_young_guard = state_hg['kv_cache_size_young_guard']
        kv_cache_size_old_guard = state_hg['kv_cache_size_old_guard']
        kv_cache_size = state_hg['kv_cache_size']

        #calc new statistics
        query_states_crop = query_new_list[hg].narrow(dim_L,-stat_size, stat_size) #take the tail for stat
        attn_weights = qk_mul_LBHD(query_states_crop, keys_hg)
        #mask future 
        B,HN_q,Q,L = attn_weights.shape
        causal_mask_crop_stat_size = get_casual_mask_crop(stat_size, stat_size, B, HN_q, attn_weights.device)
        if prefill_chunk_tail>0:
            #crop guiding tail from mask
            causal_mask_crop_stat_size = causal_mask_crop_stat_size[:,:,:,:-prefill_chunk_tail]
        causal_mask_crop_stat_size = torch.logical_not(causal_mask_crop_stat_size)
        min_val = get_type_min(attn_weights)        
        mask_len = causal_mask_crop_stat_size.shape[-1]
        if mask_len>0:
            attn_weights[:,:,:,-mask_len:].masked_fill_(causal_mask_crop_stat_size, min_val)
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).float()
        
        #process statistics by one of the method
        prefill_chunk_stat_scale_list = kv_cache_pruning_config['prefill_chunk_stat_scale'].split(',')
        pool_flag = False
        
        if prefill_chunk_stat_scale_list and prefill_chunk_stat_scale_list[0] == "max":
            attn_weights = attn_weights * attn_weights.max(dim=-1, keepdim=True).values
            del prefill_chunk_stat_scale_list[0]
        
        if prefill_chunk_stat_scale_list and prefill_chunk_stat_scale_list[0] == "max_no_old":
            attn_weight_no_old = attn_weights.narrow(-1, kv_cache_size_old_guard, kv_len_new-kv_cache_size_old_guard)
            attn_weights = attn_weights * attn_weight_no_old.max(dim=-1, keepdim=True).values
            del prefill_chunk_stat_scale_list[0]
            del attn_weight_no_old
        
        if prefill_chunk_stat_scale_list and prefill_chunk_stat_scale_list[0] == "rand":
            attn_weights = torch.rand_like(attn_weights)
            del prefill_chunk_stat_scale_list[0]
        
        if prefill_chunk_stat_scale_list and prefill_chunk_stat_scale_list[0].startswith('pool'):
            pool_flag = True
            w_view = attn_weights.view(B*HN_q, stat_size, kv_len_new)

            if prefill_chunk_stat_scale_list[0].startswith("poolmax"):
                r = int(prefill_chunk_stat_scale_list[0][7:])
                w_view = torch.nn.functional.max_pool1d(w_view, r*2+1, padding=r, stride=1)

            elif prefill_chunk_stat_scale_list[0].startswith("poolavgmax"):
                r = int(prefill_chunk_stat_scale_list[0][10:])
                w_view_avg = torch.nn.functional.avg_pool1d(w_view, r*2+1, padding=r, stride=1)
                w_view = torch.max(w_view, w_view_avg)
            elif prefill_chunk_stat_scale_list[0].startswith("poolavg"):
                r = int(prefill_chunk_stat_scale_list[0][7:])
                w_view = torch.nn.functional.avg_pool1d(w_view, r*2+1, padding=r, stride=1)
            del prefill_chunk_stat_scale_list[0]
            
            attn_weights = w_view.view(B, HN_q, stat_size, kv_len_new)

        if prefill_chunk_stat_scale_list and prefill_chunk_stat_scale_list[0].startswith('no_old_pool'):
            pool_flag = True
            w_view = attn_weights.view(B*HN_q, stat_size, kv_len_new)
            w_view = w_view.narrow(-1, kv_cache_size_old_guard, kv_len_new-kv_cache_size_old_guard )
            w_view = torch.nn.functional.pad(w_view, (kv_cache_size_old_guard,0))

            if prefill_chunk_stat_scale_list[0].startswith("no_old_poolmax"):

                if prefill_chunk_stat_scale_list[0].startswith("no_old_poolmaxi"):
                    if prefill_chunk_stat_scale_list[0].startswith("no_old_poolmaxil"):
                        iter_num = int(prefill_chunk_stat_scale_list[0][16:])
                        offset = 0
                    elif prefill_chunk_stat_scale_list[0].startswith("no_old_poolmaxir"):
                        iter_num = int(prefill_chunk_stat_scale_list[0][16:])
                        offset = (2**iter_num)-1
                    else:
                        iter_num = int(prefill_chunk_stat_scale_list[0][15:])
                        offset = (2**iter_num+1)//2
                    assert iter_num>0
                    if iter_num==1:
                        shift = 1
                        w_view_0 = w_view.narrow(-1, 0, 1).clone()
                        w_view_1 = w_view.narrow(-1, -1, 1).clone()
                        w_view = torch.max(
                            w_view.narrow(-1, 1, w_view.shape[-1]-1),
                            w_view.narrow(-1, 0, w_view.shape[-1]-1)
                        )
                        if offset == 0:
                            w_view = torch.cat([w_view_0, w_view], dim=-1)
                        else:
                            w_view = torch.cat([w_view, w_view_1], dim=-1)
                    else:
                        for i in range(iter_num):
                            shift = 2**i
                            w_view_0 = w_view.narrow(-1, 0, shift).clone()
                            w_view_1 = w_view.narrow(-1, -shift, shift).clone()
                            w_view = torch.max(
                                w_view.narrow(-1, shift, w_view.shape[-1]-shift),
                                w_view.narrow(-1, 0, w_view.shape[-1]-shift)
                            )
                            w_view = torch.cat([w_view_0, w_view, w_view_1], dim=-1)
                        w_view = w_view.narrow(-1, offset, kv_len_new)
                else:
                    #!!!single call max_pool1d requires a lot of memory.
                    r = int(prefill_chunk_stat_scale_list[0][14:])
                    w_view = torch.nn.functional.max_pool1d(w_view, r*2+1, padding=r, stride=1)
            elif prefill_chunk_stat_scale_list[0].startswith("no_old_poolavg"):
                r = int(prefill_chunk_stat_scale_list[0][14:])
                w_view = torch.nn.functional.avg_pool1d(w_view, r*2+1, padding=r, stride=1)
            del prefill_chunk_stat_scale_list[0]
            
            attn_weights = w_view.view(B, HN_q, stat_size, kv_len_new)

        #merge statistics from multiple query heads
        B = attn_weights.shape[0]
        HN_kv = keys_hg.shape[dim_HN]
        if kv_cache_pruning_config['prefill_chunk_stat_head_merge'] == "mean":
            attn_weights = attn_weights.view(B, HN_kv, HN_q//HN_kv, stat_size, kv_len_new).mean(dim=2)
        if kv_cache_pruning_config['prefill_chunk_stat_head_merge'] == "max":
            attn_weights = attn_weights.view(B, HN_kv, HN_q//HN_kv, stat_size, kv_len_new).max(dim=2).values


        #agregate over all query positions
        if kv_cache_pruning_config['prefill_chunk_stat_separate']:
            assert attn_weights.shape[2] > kv_cache_pruning_config['prefill_chunk_tail']
            #2 separate statistics for last chunk and next chunk
            s_last = kv_cache_pruning_config['prefill_chunk_tail']
            s_next = attn_weights.shape[2] - s_last
            kv_cache_stat_new_next = attn_weights.narrow(2,0,s_next).sum(dim=2, keepdim=True)
            kv_cache_stat_new_last = attn_weights.narrow(2,s_next,s_last).sum(dim=2, keepdim=True)
            kv_cache_stat_new_2ch = torch.cat([kv_cache_stat_new_next,kv_cache_stat_new_last], dim=2)

            weight_decay = torch.Tensor(kv_cache_pruning_config['prefill_chunk_stat_separate_weight_decay']).to(kv_cache_stat_new_2ch)
            weight_decay = weight_decay.view(1,1,2,1)

            kv_cache_stat_old_2ch = state_hg["kv_cache_stat_2ch"]
            assert torch.all(kv_cache_stat_old_2ch >= 0), "there are invalid positions"
            pad = kv_cache_stat_new_2ch.shape[-1] - kv_cache_stat_old_2ch.shape[-1]
            kv_cache_stat_old_2ch = torch.nn.functional.pad(kv_cache_stat_old_2ch, (0,pad))

            kv_cache_stat_new_2ch = kv_cache_stat_new_2ch + kv_cache_stat_old_2ch * weight_decay

            weight = kv_cache_pruning_config['prefill_chunk_stat_separate_merge_prefill']
            kv_cache_stat_new_topk = kv_cache_stat_new_2ch.narrow(2,0,1)*(1-weight) + kv_cache_stat_new_2ch.narrow(2,1,1)*weight
            
        else:
            kv_cache_stat_new_2ch = attn_weights.sum(dim=2, keepdim=True)

            if kv_len_old>0 and not kv_cache_pruning_config['prefill_chunk_stat_new']:
                #merge with old statistics
                kv_cache_stat_old_2ch = state_hg["kv_cache_stat_2ch"]
                assert torch.all(kv_cache_stat_old_2ch >= 0), "there are invalid positions"
                pad = kv_cache_stat_new_2ch.shape[-1] - kv_cache_stat_old_2ch.shape[-1]
                kv_cache_stat_old_2ch = torch.nn.functional.pad(kv_cache_stat_old_2ch, (0,pad))
                kv_cache_stat_new_2ch = kv_cache_stat_new_2ch + kv_cache_stat_old_2ch

            kv_cache_stat_new_topk = kv_cache_stat_new_2ch

        kv_cache_pruned_num = 0
        #check if prune is reuired for the heads group
        if kv_len_new > kv_cache_size + 1:
            kv_cache_pruned_num = kv_len_new - (kv_cache_size + 1)
            
            current_total_length = kv_cache_pos_new.max(dim=-1, keepdim=True).values + 1

            if kv_cache_size_young_guard+kv_cache_size_old_guard == kv_cache_size:
                pos_young_guard = current_total_length - (kv_cache_size_young_guard+1)
            else:
                pos_young_guard = current_total_length - (kv_cache_size_young_guard+0)

            kv_cache_stat_new_masked = kv_cache_stat_new_topk

            #young guard
            guard_mask_1 = (kv_cache_pos_new >= pos_young_guard)
            max_val = kv_cache_stat_new_topk.max(dim=-1, keepdim=True).values
            max_val = max_val + kv_cache_pos_new
            kv_cache_stat_new_masked = torch.where(guard_mask_1, max_val, kv_cache_stat_new_masked)

            #calc size without sink (old_guard)
            kv_len_new_crop = kv_len_new - kv_cache_size_old_guard
            Ltop_crop = (kv_cache_size+1) - kv_cache_size_old_guard

            #crop old from stat
            kv_cache_stat_new_masked_crop = kv_cache_stat_new_masked.narrow(-1,kv_cache_size_old_guard,kv_len_new_crop)
            #crop old from gatherd source
            kv_cache_stat_new_crop_2ch = kv_cache_stat_new_2ch.narrow(-1,kv_cache_size_old_guard,kv_len_new_crop)
            kv_cache_pos_new_crop = kv_cache_pos_new.narrow(-1,kv_cache_size_old_guard,kv_len_new_crop)
            #crop old from KV
            keys_hg_crop = keys_hg.narrow(dim_L,kv_cache_size_old_guard,kv_len_new_crop)
            values_hg_crop = values_hg.narrow(dim_L,kv_cache_size_old_guard,kv_len_new_crop)

            select_val, select_idx = torch.topk(kv_cache_stat_new_masked_crop, Ltop_crop, dim=-1, largest=True, sorted=False)

            if pool_flag:
                #need to keep index order for pool operation
                select_idx = torch.sort(select_idx, dim=-1).values

            #prune stat and pos and key and values
            select_idx_2ch = select_idx.expand(-1,-1,kv_cache_stat_new_crop_2ch.shape[2],-1)
            kv_cache_stat_new_crop_2ch = kv_cache_stat_new_crop_2ch.gather(dim=-1, index=select_idx_2ch)
            kv_cache_pos_new_crop = kv_cache_pos_new_crop.gather(dim=-1, index=select_idx)
            #merge back the cropped sink positions
            kv_cache_stat_new_2ch = torch.cat([kv_cache_stat_new_2ch.narrow(-1,0,kv_cache_size_old_guard), kv_cache_stat_new_crop_2ch], dim =-1)
            kv_cache_pos_new = torch.cat([kv_cache_pos_new.narrow(-1,0,kv_cache_size_old_guard), kv_cache_pos_new_crop], dim =-1)

            kv_index = select_idx
            _,_,_,D = keys_hg_crop.shape
            kv_index = select_idx.expand(-1,-1,D,-1)#B,HN,1->D,L
            #permute index to original KV layout
            kv_index = permute(kv_index, "B,HN,D,L", layout)
            keys_hg_crop = keys_hg_crop.gather(dim=dim_L, index=kv_index)
            values_hg_crop = values_hg_crop.gather(dim=dim_L, index=kv_index)
            #"L,B,HN,D"
            keys_hg.narrow(dim_L, kv_cache_size_old_guard, keys_hg_crop.shape[0]).copy_(keys_hg_crop)
            values_hg.narrow(dim_L, kv_cache_size_old_guard, values_hg_crop.shape[0]).copy_(values_hg_crop)

        
        state_hg["kv_cache_stat_2ch"] = kv_cache_stat_new_2ch
        state_hg["kv_cache_pos"] = kv_cache_pos_new
        state_hg["kv_cache_pruned_num"] += kv_cache_pruned_num
