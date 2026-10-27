"""Unit check for the DSV4 C128 match validator (no model, no NPU).

Run with the sglang tree on PYTHONPATH, e.g.
    PYTHONPATH=python python3 test/manual/npu/test_c128_match_validator.py
"""
import sys
from types import SimpleNamespace

if len(sys.argv) > 1:
    sys.path.insert(0, sys.argv[1])
from sglang.srt.hardware_backend.npu.dsv4.c128_sidecar_component import (  # noqa: E402
    C128SidecarComponent,
)

PAGE = 128          # tree page_size
KERNEL = 16         # c128_page_size  -> one C128 block = 2048 tokens
CT = C128SidecarComponent.component_type


def make_component():
    comp = C128SidecarComponent.__new__(C128SidecarComponent)
    comp.tree_core = SimpleNamespace(page_size=PAGE)
    comp.cache = SimpleNamespace(
        token_to_kv_pool_allocator=SimpleNamespace(
            c128_attn_allocator=SimpleNamespace(page_size=KERNEL)
        )
    )
    return comp


def nodes(n_pages, page_with_value=None):
    out = []
    for i in range(n_pages):
        has = page_with_value is not None and (i + 1) == page_with_value
        out.append(
            SimpleNamespace(
                key=[0] * PAGE,
                component_data={
                    CT: SimpleNamespace(
                        value=object() if has else None,
                        host_value=None,
                    )
                },
            )
        )
    return out


def run(label, seq, validator, expect):
    got = [validator(n) for n in seq]
    ok = got == expect
    print(f"{'PASS' if ok else 'FAIL'} {label}")
    if not ok:
        print(f"     expected {expect}\n     got      {got}")
    return ok


comp = make_component()
group = comp._group_tokens()
print(f"group_tokens = {group} (page_size {PAGE} x kernel {KERNEL})")

results = []
# 1) span shorter than one block, no C128 page anywhere (the reported bug)
v = comp.create_match_validator()
results.append(run("short span (7 pages / 896 tok), no C128 page -> all matchable",
                   nodes(7), v, [True] * 7))

# 2) span >= 1 block with the page on the block endpoint (2048 tok = page 16)
v = comp.create_match_validator()
exp2 = [True] * 31 + [False] + [True] * 8   # endpoint 32 still has no page
results.append(run("long span, page on endpoint 16 -> only endpoint 32 rejected",
                   nodes(40, page_with_value=16), v, exp2))

# 3) span >= 1 block, endpoint page missing -> match must stop at that endpoint
v = comp.create_match_validator()
exp = ([True] * 15 + [False] + [True] * 15 + [False] + [True] * 8)  # 16 & 32
results.append(run("long span, no pages -> rejected only at endpoints 16/32",
                   nodes(40), v, exp))

# 4) device-only mode on a short span must stay permissive as well
v = comp.create_match_validator(match_device_only=True)
results.append(run("short span, device-only -> all matchable", nodes(4), v, [True] * 4))

print("OVERALL", "PASS" if all(results) else "FAIL")
sys.exit(0 if all(results) else 1)
