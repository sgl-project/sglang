# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Model-free positive/negative control of real CUDA-graph lifetime inspection."""

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.cuda_graph_lifetime import tracer

register_cuda_ci(est_time=20, stage="extra-a", runner_config="1-gpu-small")


def main():
    if not torch.cuda.is_available():
        raise SystemExit("This check requires a CUDA GPU.")

    tracer.patch_torch()
    audit = tracer.get()
    audit.check_every = 1
    stream = torch.cuda.Stream()
    static = torch.ones(1024, device="cuda")
    table, retained, graphs, outputs = {}, [], [], []
    for variant in range(2):
        table["metadata"] = torch.full((1024,), float(variant + 2), device="cuda")
        retained.append(table["metadata"])
        with audit.labelled(f"variant-{variant}"):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = static + table["metadata"]
        graphs.append(graph)
        outputs.append(output)

    # Both captures remain valid while their metadata has an owner.
    for graph in graphs:
        graph.replay()
    torch.cuda.synchronize()
    for variant, output in enumerate(outputs):
        torch.testing.assert_close(output, torch.full_like(output, variant + 3))
    assert audit.effective_status() == "clean", audit.last_report
    assert audit.checks > 0
    print("PASS: both graphs replay with live metadata.")

    # The table now owns only variant 1; drop variant 0's last metadata owner.
    retained[0] = None
    try:
        graphs[0].replay()
    except RuntimeError as error:
        assert audit.effective_status() == "dangling", str(error)
        assert "variant-0" in str(error), str(error)
    else:
        raise AssertionError("The tracer missed the first graph's freed metadata.")
    print("PASS: freed metadata is rejected before graph replay.")


if __name__ == "__main__":
    main()
