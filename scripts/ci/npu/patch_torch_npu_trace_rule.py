"""Temporary workaround for the CANN 9.2 nightly images.

torch_npu registers torch.npu.random.manual_seed(_all) as
TorchInGraphFunctionVariable, which conflicts with the SkipFunctionVariable rule
torch._dynamo already holds for the same object. get_torch_obj_rule_map() then
raises "Duplicate torch object ... with different rules" and every
sglang.srt.models import fails. Moving both names to skip_functions_npu makes
the rules consistent.

Remove this script and its workflow step once the image is fixed.
"""

import pathlib

import torch_npu
import torch_npu.dynamo.trace_rule as trace_rule

if "torch.npu.random.manual_seed_all" in trace_rule.skip_functions_npu:
    print("[patch] already applied")
    raise SystemExit(0)

path = pathlib.Path(torch_npu.__file__).parent / "dynamo" / "trace_rule.py"
source = path.read_text()
remove = [
    '        "torch.npu.random.manual_seed_all",\n',
    '        "torch.npu.random.manual_seed",\n',
]
anchor = '        "torch.npu.set_device",\n'

assert all(source.count(line) == 1 for line in remove), f"unexpected layout in {path}"
assert source.count(anchor) == 1, f"unexpected layout in {path}"

for line in remove:
    source = source.replace(line, "")

path.write_text(source.replace(anchor, "".join(remove) + anchor))
print(f"[patch] applied to {path}")
