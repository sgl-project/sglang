"""Original and plain-dict counterexample plus fresh OrderedDict-subclass candidate.
CPU eager/Dynamo-eager scope only; does not establish NPU compiler acceptance.
"""

import collections
import json
import torch


class StateMap(collections.OrderedDict):
    def __init__(self):
        super().__init__()
        self._metadata = collections.OrderedDict()


m = torch.nn.Sequential(torch.nn.Linear(2, 2))


def hook(module, state, prefix, local_metadata):
    state._metadata[prefix[:-1]]["hook_marker"] = "preserved"


m[0].register_state_dict_post_hook(hook)
old = m.state_dict()
assert old._metadata["0"]["hook_marker"] == "preserved"
report = {
    "original": "pass",
    "original_keys": list(old),
    "plain_dict": "unexpected_pass",
    "scope": "CPU eager and Dynamo fullgraph backend=eager only; NPU unverified",
}
try:
    m.state_dict(destination={})
except AttributeError as error:
    report.update(plain_dict="regression", error=str(error))
assert report["plain_dict"] == "regression"
first = m.state_dict(destination=StateMap())
second = m.state_dict(destination=StateMap())
assert isinstance(first, collections.OrderedDict)
assert list(first) == list(old) and first._metadata == old._metadata
assert first is not second and first._metadata is not second._metadata
for k in old:
    assert torch.equal(first[k], old[k]) and first[k].data_ptr() == old[k].data_ptr()
for k in first._metadata:
    assert first._metadata[k] is not second._metadata[k]
first._metadata["0"]["sentinel"] = "only_first"
assert "sentinel" not in second._metadata["0"]


def forward(x):
    state = m.state_dict(destination=StateMap())
    # Retain original functional-call semantics; no frozen parameter cache.
    return torch.func.functional_call(m, state, (x,))


x = torch.ones(1, 2)
compiled = torch.compile(forward, backend="eager", fullgraph=True)
y = compiled(x)
assert torch.equal(y, forward(x))
with torch.no_grad():
    m[0].weight.add_(1)
z = compiled(x)
assert torch.equal(z, forward(x)) and not torch.equal(y, z)
report["state_map"] = {
    "eager_metadata_parity": True,
    "fresh_state_and_metadata": True,
    "key_order_and_tensor_storage_parity": True,
    "cpu_dynamo_fullgraph_eager": True,
    "updated_weights_observed": True,
}
print(json.dumps(report, indent=2))
