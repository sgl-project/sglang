import json, torch, torch_npu  # noqa: F401 - initializes the NPU extension
from pathlib import Path
import sglang.srt.hardware_backend.npu.attention.fia_decode_graph  # noqa: F401 - registers capture handler
from sglang.srt.hardware_backend.npu.attention.fia_decode import fia_decode_bsh
import argparse

p = argparse.ArgumentParser()
p.add_argument("--output", required=True)
p.add_argument("--device", default="npu:0")
a = p.parse_args()
torch.npu.set_device(a.device)
q = torch.zeros((1, 1, 128), dtype=torch.bfloat16, device="npu")
k = torch.zeros((1, 128, 128), dtype=q.dtype, device=q.device)
v = k.clone()
v[:, 0, :] = 2
v[:, 1, :] = 6
bt = torch.zeros((1, 1), dtype=torch.int32, device=q.device)


def call(fn, n):
    return fn(q, k, v, bt, 128, 1, 1, 1.0, [n], None, 0, 128)


compiled = torch.compile(
    fia_decode_bsh, fullgraph=True, mode="max-autotune-no-cudagraphs"
)
report = {
    "synthetic": True,
    "expected": {"1": 2.0, "2": 4.0},
    "eager": {},
    "compiled": {},
    "replay": [],
}
for n in [1, 2]:
    e = call(fia_decode_bsh, n)
    c = call(compiled, n)
    torch.npu.synchronize()
    report["eager"][n] = float(e.float().mean())
    report["compiled"][n] = float(c.float().mean())
    assert torch.equal(e, c)
for _ in range(3):
    call(compiled, 1)
torch.npu.synchronize()
g = torch.npu.NPUGraph()
with torch.npu.graph(g, auto_dispatch_capture=True):
    out = call(compiled, 1)
for n in [1, 2, 1]:
    g.update(cpu_update_input=[{"actual_seq_lengths_kv": [n]}])
    g.replay()
    torch.npu.synchronize()
    actual = float(out.float().mean())
    report["replay"].append({"length": n, "actual": actual})
    assert actual == report["expected"][str(n)], report
Path(a.output).write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
