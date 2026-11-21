"""Actual SG prefill consumer with reused VA FIA/GBSA dispatch, opt-in on NPU."""
import os
from types import MethodType, SimpleNamespace

import pytest

if os.getenv('SGLANG_TEST_CANN930_QSA_NATIVE') != '1':
    pytest.skip('Requires ops-transformer9.3 and an idle NPU', allow_module_level=True)

from test_qsa_compact_prefill_npu import (
    QSAIndexerMetadata, QwenSparseAttnBackend, QwenSparseAttnMetadata,
    ForwardMode, make_case, torch,
)
from sglang.kernels.ops.attention.qsa_prefill_dispatch_va import (
    QSARequest, prepare_sg_prefill_inputs,
)


def backend_and_batch(c):
    layer_cache = SimpleNamespace(key=lambda: c.k.unsqueeze(1), value=lambda: c.v.unsqueeze(1))
    pool = SimpleNamespace(qsa_block_topk=512, qsa_compressed_page_size=32,
                           layer_router=SimpleNamespace(read=lambda _: layer_cache))
    backend = SimpleNamespace(token_to_kv_pool=pool, compress_ratio=4,
                              qsa_profile=SimpleNamespace(block_topk=512, budget=2048),
                              _mtp_shared_sparse_indices=None,
                              _is_speculative_paged_mode=lambda _: False,
                              _pad_extend_output=QwenSparseAttnBackend._pad_extend_output)
    for name in ('_uses_block_indices', '_can_defer_prefill_block_expansion', '_make_npu_prefill_inputs'):
        setattr(backend,name,MethodType(getattr(QwenSparseAttnBackend,name),backend))
    batch=SimpleNamespace(forward_mode=ForwardMode.EXTEND, positions=c.positions,
                          extend_seq_lens=c.extends, extend_seq_lens_cpu=c.extends.cpu().tolist(),
                          seq_lens_cpu=c.lengths.cpu(), attn_cp_metadata=None)
    return backend,batch


def run(backend,batch,c,dispatch,padded=0,have_chunk=True):
    inputs=backend._make_npu_prefill_inputs(batch,c.lengths,c.table,c.q.shape[0]) if dispatch else None
    im=QSAIndexerMetadata(sequence_lengths=c.lengths, token_to_batch_idx=c.sequence_ids,
                          token_slot_table=c.table,out_cache_loc=torch.zeros(c.q.shape[0],device='npu',dtype=torch.int32),
                          token_to_kv_pool=backend.token_to_kv_pool,compress_ratio=4,block_topk=512,
                          defer_prefill_block_expansion=True)
    meta=QwenSparseAttnMetadata(sequence_lengths=c.lengths,token_to_batch_idx=c.sequence_ids,
                               token_slot_table=c.table,indexer_metadata=im,npu_prefill_inputs=inputs)
    backend._resolve_metadata=lambda _:meta
    slots=c.table[c.sequence_ids.long(),c.positions.long()].long()
    k,v=(c.k.index_select(0,slots),c.v.index_select(0,slots)) if have_chunk else (None,None)
    q=torch.cat((c.q,c.q.new_zeros(padded,3,256))) if padded else c.q
    layer=SimpleNamespace(layer_id=0,tp_q_head_num=3,head_dim=256,scaling=1/16)
    output=QwenSparseAttnBackend.forward_extend(backend,q,k,v,layer,batch,save_kv_cache=False,topk_indices=c.groups)
    return output,inputs


@pytest.mark.parametrize('extends,prefixes,padded,have_chunk',[
    ([128],[0],0,True),([128],[0],0,False),([128],[2049],4,True),
    ([128],[11001],0,True),([128],[24575],0,True),([128],[24576],0,True),
    ([32,64,32],[0,24576,25000],0,True),([17,111],[0,3001],4,True),
    ([16]*8,[0,4,128,1024,2048,7000,9000,11001],0,True),
])
def test_actual_prefill_dispatch(extends,prefixes,padded,have_chunk,monkeypatch):
    monkeypatch.setenv('SGLANG_NPU_QSA_PREFILL_GBSA','1')
    monkeypatch.setenv('SGLANG_NPU_QSA_VA_PREFILL_DISPATCH','1')
    c=make_case(extends,prefixes)
    backend,batch=backend_and_batch(c)
    for step in range(3):
        if step:
            c.q.mul_(-0.875);c.k.mul_(0.75);c.v.add_(0.125)
            c.table.add_(128).remainder_(c.k.shape[0])
            c.groups.copy_(c.groups.roll(7,dims=-1))
        expected,_=run(backend,batch,c,False,padded,have_chunk)
        actual,inputs=run(backend,batch,c,True,padded,have_chunk)
        torch.npu.synchronize()
        torch.testing.assert_close(actual,expected,atol=0.004,rtol=0.02)
        assert bool(torch.isfinite(actual).all())
        if padded:assert not actual[-padded:].count_nonzero().item()
        if all(prefix>=24576 for prefix in prefixes):
            assert inputs is None
        else:
            assert inputs is not None
            assert [r.request_index for r in inputs.plan.fia_requests]==[i for i,p in enumerate(prefixes) if p<24576]
            if len(prefixes)==3:assert inputs.plan.gbsa_spans==((32,128),)


def test_full512_plus_tail_marker(monkeypatch):
    monkeypatch.setenv('SGLANG_NPU_QSA_PREFILL_GBSA','1')
    monkeypatch.setenv('SGLANG_NPU_QSA_VA_PREFILL_DISPATCH','1')
    c=make_case([128],[4096]);backend,batch=backend_and_batch(c)
    c.q.zero_();c.k.zero_();c.v.zero_()
    c.groups.copy_(torch.arange(512,device='npu',dtype=torch.int32).expand(128,-1))
    c.v[c.table[0,2044:2048].long(),0,0]=1024
    c.v[c.table[0,4096:4224].long(),0,1]=1024
    actual,_=run(backend,batch,c,True)
    tails=(c.positions+1)%4
    oracle=torch.stack((4096/(2048+tails),1024*tails/(2048+tails)),-1)
    torch.testing.assert_close(actual.view(128,3,256)[:,0,:2].float(),oracle.float(),atol=0.008,rtol=0.01)


def test_dispatch_eligibility_and_request_threshold(monkeypatch):
    monkeypatch.setenv('SGLANG_NPU_QSA_PREFILL_GBSA','1')
    monkeypatch.setenv('SGLANG_NPU_QSA_VA_PREFILL_DISPATCH','1')
    c=make_case([128],[0]);backend,batch=backend_and_batch(c)
    assert backend._make_npu_prefill_inputs(batch,c.lengths,c.table,128) is not None
    assert backend._make_npu_prefill_inputs(batch,c.lengths,c.table,127) is None
    backend._mtp_shared_sparse_indices=object()
    assert backend._make_npu_prefill_inputs(batch,c.lengths,c.table,128) is None
    backend._mtp_shared_sparse_indices=None
    for mode in (ForwardMode.DECODE,ForwardMode.TARGET_VERIFY,ForwardMode.DRAFT_EXTEND_V2,ForwardMode.MIXED):
        batch.forward_mode=mode
        assert backend._make_npu_prefill_inputs(batch,c.lengths,c.table,128) is None
    batch.forward_mode=ForwardMode.EXTEND
    batch.seq_lens_cpu=None
    assert backend._make_npu_prefill_inputs(batch,c.lengths,c.table,128) is None
    assert QSARequest(0,0,4096,24576,True).use_fia
    assert QSARequest(0,0,4096,24576+4095,True).use_fia
    assert not QSARequest(0,0,4096,24576+4096,True).use_fia
    assert not QSARequest(0,0,4,128,False).use_fia
