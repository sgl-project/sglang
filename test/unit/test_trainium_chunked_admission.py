"""Execute the actual scheduler method with bounded CPU admission states."""
import ast
from pathlib import Path
from types import SimpleNamespace as S
import pytest
p=Path(__file__).resolve().parents[2] / 'python/sglang/srt/managers/scheduler.py'
cls=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='Scheduler')
method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='get_num_allocatable_reqs')
ns={'get_parallel':lambda:S(pp_max_micro_batch_size=16)}
exec(compile(ast.Module(body=[method],type_ignores=[]),str(p),'exec'),ns)
f=ns['get_num_allocatable_reqs']
@pytest.mark.parametrize('running,free,reused,new,blocked',[(14,1,1,0,False),(14,1,1,1,True),(15,0,1,0,True),(15,1,0,0,False),(15,1,0,1,True),(14,0,1,0,True),(0,16,0,15,False),(0,16,0,16,True),(16,8,0,0,True),(14,5,2,0,True),(12,2,2,1,False),(12,2,2,2,True)])
def test_slot_and_batch_bounds(running,free,reused,new,blocked):
    pool=S(available_size=lambda:free)
    requests=[S(req_pool_idx=i) for i in range(reused)]+[S(req_pool_idx=None) for _ in range(new)]
    assert (len(requests)>=f(S(req_to_token_pool=pool),running,requests)) is blocked

def test_pre_adder_query_remains_unchanged():
    assert f(S(req_to_token_pool=S(available_size=lambda:1)),14)==1
