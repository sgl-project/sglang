# Copyright 2023-2025 SGLang Team
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

import logging
from types import SimpleNamespace

import torch
from torch._inductor.utils import run_and_get_code

from sglang.srt.compilation.inductor_pass import SGLangPatternMatcherInductorPass

# FusionConfig.enable_torch_compile_graph_trace_logs requires
# log level debug to print the pre and post graph changes
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s Fusion Example] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)


class ExampleModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, gating_output: torch.Tensor, topk: int):
        softmax_output = torch.softmax(gating_output, dim=-1)
        topk_weights_ref, topk_indices_ref = torch.topk(softmax_output, topk, dim=-1)
        return topk_weights_ref, topk_indices_ref


"""
Fake op registration, dynamo uses this while tracing to avoid
running the actual kernel and slow down the compilation process.

This registration should be part of sgl-kernel and can be done
in python or C++
"""


@torch.library.register_fake("sgl_kernel::topk_softmax")
def topk_softmax_fake(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    gating_output: float,
    renormalize: bool = False,
):
    pass


class ExampleFusionPass(SGLangPatternMatcherInductorPass):
    def build_pass(self):
        """graph trace of Example Model obtained using TORCH_LOGS="post_grad_graphs"
        def forward(self, arg0_1: "f32[1, 4][4, 1]cuda:0"):
            prepare_softmax_online_default = torch.ops.prims.prepare_softmax_online.default(arg0_1, -1)
            getitem_2: "f32[1, 1][1, 1]cuda:0" = prepare_softmax_online_default[0]
            getitem_3: "f32[1, 1][1, 1]cuda:0" = prepare_softmax_online_default[1];  prepare_softmax_online_default = None
            sub_tensor: "f32[1, 4][4, 1]cuda:0" = torch.ops.aten.sub.Tensor(arg0_1, getitem_2);  arg0_1 = getitem_2 = None
            exp_default: "f32[1, 4][4, 1]cuda:0" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None

            div: "f32[1, 4][4, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_default, getitem_3);  exp_default = getitem_3 = None

            topk = torch.ops.aten.topk.default(div, 1);  div = None
            getitem: "f32[1, 1][1, 1]cuda:0" = topk[0]
            getitem_1: "i64[1, 1][1, 1]cuda:0" = topk[1];  topk = None
            return (getitem, getitem_1)
        """

        def pattern(gating_output, topk):
            prepare_softmax_online_default = (
                torch.ops.prims.prepare_softmax_online.default(gating_output, -1)
            )
            sub_tensor = torch.ops.aten.sub.Tensor(
                gating_output, prepare_softmax_online_default[0]
            )
            exp_default = torch.ops.aten.exp.default(sub_tensor)
            div = torch.ops.aten.div.Tensor(
                exp_default, prepare_softmax_online_default[1]
            )
            topk_op = torch.ops.aten.topk.default(div, topk)
            return topk_op[0], topk_op[1]

        """ Replacement graph obtained by running topk_softmax_kernel_compiled_run with TORCH_LOGS="post_grad_graphs"
        def forward(self, arg0_1: "f32[1, 4][4, 1]cuda:0"):
            empty: "f32[1, 1][1, 1]cuda:0" = torch.ops.aten.empty.memory_format([1, 1], dtype = torch.float32, device = device(type='cuda'), pin_memory = False)

            empty_1: "i32[1, 1][1, 1]cuda:0" = torch.ops.aten.empty.memory_format([1, 1], dtype = torch.int32, device = device(type='cuda'), pin_memory = False)

            auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.sgl_kernel.topk_softmax.default, gating_output = arg0_1, renormalize = False, _topk_weights_base_index = 0, _topk_indices_base_index = 1, _all_bases = [empty, empty_1]);  arg0_1 = empty = empty_1 = None
            getitem_1: "f32[1, 1][1, 1]cuda:0" = auto_functionalized_v2[1]
            getitem_2: "i32[1, 1][1, 1]cuda:0" = auto_functionalized_v2[2];  auto_functionalized_v2 = None
            return (getitem_1, getitem_2)
        """

        def replacement(gating_output, topk):
            empty = torch.empty(
                (gating_output.shape[0], topk), dtype=torch.float32, device="cuda"
            )
            empty_1 = torch.empty(
                (gating_output.shape[0], topk), dtype=torch.int32, device="cuda"
            )
            topk_softmax = torch.ops.higher_order.auto_functionalized_v2(
                torch.ops.sgl_kernel.topk_softmax.default,
                gating_output=gating_output,
                renormalize=False,
                _topk_weights_base_index=0,
                _topk_indices_base_index=1,
                _all_bases=[empty, empty_1],
            )
            return topk_softmax[1], topk_softmax[2]

        """
        Input used by graph for tracing, this is passed to the fake op
        The absolute shape don't matter as much as the relative shapes of the input,
        In this case since there is only 1 input we only need to match no. of dims
        """
        example_inputs = [torch.empty(16, 16).cuda()]

        self.register_replacement_pattern(
            pattern=pattern,
            replacement=replacement,
            example_inputs=example_inputs,
            # Handling scalars is not the cleanest I feel, this essentially requires
            # passes to be registered for each expected scalar value
            scalar_workaround={"topk": 2},
        )


def mock_fusion_manager(graph: torch.fx.graph):
    ExampleFusionPass(
        fusion_config=SimpleNamespace(enable_torch_compile_graph_trace_logs=True)
    )(graph)


def test_fusion_example_pass(num_experts, num_tokens, topk):
    model, model_ref = ExampleModel(), ExampleModel()

    torch._inductor.config.post_grad_custom_post_pass = mock_fusion_manager

    model.compile()

    gating_output = torch.randn(
        (num_tokens, num_experts), dtype=torch.float32, device="cuda"
    )

    topk_weights_ref, topk_indices_ref = model_ref(gating_output, topk)

    res, source_codes = run_and_get_code(model, gating_output, topk)
    code = "\n".join(source_codes)

    assert "sgl_kernel.topk_softmax" in code

    torch.testing.assert_close(res[0], topk_weights_ref, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(res[1].int(), topk_indices_ref.int(), atol=0, rtol=0)


# Use this with TORCH_LOGS="post_grad_graphs" to print the fx graph
# of torch compiled kernel to figure out the replacement
def topk_softmax_kernel_compiled_run(num_experts, num_tokens, topk):
    @torch.compile()
    def fwd(gating_output, topk):
        topk_weights = torch.empty(
            (gating_output.shape[0], topk), dtype=torch.float32, device="cuda"
        )
        topk_indices = torch.empty(
            (gating_output.shape[0], topk), dtype=torch.int32, device="cuda"
        )
        torch.ops.sgl_kernel.topk_softmax.default(
            topk_weights, topk_indices, gating_output, False
        )
        return topk_weights, topk_indices

    gating_output = torch.randn(
        (num_tokens, num_experts), dtype=torch.float32, device="cuda"
    )

    _, _ = fwd(gating_output, topk)


if __name__ == "__main__":
    # topk_softmax_kernel_compiled_run(8, 16, 2)
    test_fusion_example_pass(8, 16, 2)
