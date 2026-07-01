# SPDX-License-Identifier: Apache-2.0
"""Tests for conv1d tree traversal optimization:
- Phase A: pre-computed parent_map correctness
- Phase B: token-parallel kernel correctness vs serial kernel and reference
- Performance comparison between serial and token-parallel kernels
"""

import time

import pytest
import torch
import triton

from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
    PAD_SLOT_ID,
    _causal_conv1d_update_kernel,
    _causal_conv1d_verify_token_parallel_kernel,
    causal_conv1d_update,
)
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")


def _get_eagle_utils_jit_module():
    """JIT compile eagle_utils.cu and return the module.

    This allows testing the CUDA kernel without requiring a full sgl-kernel rebuild.
    The compiled module is cached across calls.
    """
    import os
    import tempfile

    from torch.utils.cpp_extension import load

    sgl_kernel_dir = os.path.join(os.path.dirname(__file__), "../../../../sgl-kernel")
    sgl_kernel_dir = os.path.abspath(sgl_kernel_dir)
    eagle_cu = os.path.join(sgl_kernel_dir, "csrc/speculative/eagle_utils.cu")

    # Create stub header and wrapper in a temp directory
    build_dir = os.path.join(tempfile.gettempdir(), "sgl_eagle_test_build")
    os.makedirs(build_dir, exist_ok=True)

    header_path = os.path.join(build_dir, "pytorch_extension_utils.h")
    if not os.path.exists(header_path):
        with open(header_path, "w") as f:
            f.write(
                "#include <torch/library.h>\n"
                '#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")\n'
                '#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")\n'
                "#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)\n"
                '#define CHECK_DIM(d, x) TORCH_CHECK(x.dim() == d, #x " must be a " #d "D tensor")\n'
                '#define CHECK_EQ(a, b) TORCH_CHECK((a) == (b), "CHECK_EQ(" #a ", " #b ") failed.")\n'
                '#define CHECK_GE(a, b) TORCH_CHECK((a) >= (b), "CHECK_GE(" #a ", " #b ") failed.")\n'
            )

    wrapper_path = os.path.join(build_dir, "wrapper.cu")
    if not os.path.exists(wrapper_path):
        with open(wrapper_path, "w") as f:
            f.write(
                "#include <torch/extension.h>\n"
                "void build_tree_kernel_efficient(\n"
                "    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,\n"
                "    at::Tensor, at::Tensor, at::Tensor, at::Tensor,\n"
                "    int64_t, int64_t, int64_t, int64_t);\n"
                "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n"
                '    m.def("build_tree_kernel_efficient", &build_tree_kernel_efficient);\n'
                "}\n"
            )

    out_dir = os.path.join(build_dir, "out")
    os.makedirs(out_dir, exist_ok=True)

    module = load(
        name="eagle_utils_test_ext",
        sources=[eagle_cu, wrapper_path],
        extra_include_paths=[build_dir],
        extra_cuda_cflags=["-O2"],
        build_directory=out_dir,
        verbose=False,
    )
    return module


def _call_build_tree_jit(topk, depth, bs, device):
    """Call the CUDA build_tree kernel via JIT and return parent map.

    Constructs a complete k-ary tree where parent[i] = (i-1) // topk.

    Returns:
        retrieve_parent_token: (bs, draft_token_num) int64
        retrieve_next_token: (bs, draft_token_num) int64
        retrieve_next_sibling: (bs, draft_token_num) int64
        draft_token_num: int
    """
    module = _get_eagle_utils_jit_module()

    draft_token_num = topk * depth + 1
    # Build parent_list for a complete k-ary tree: parent[i] = (i-1) // topk
    parents = [(i - 1) // topk for i in range(1, draft_token_num)]
    parent_list = torch.tensor([parents] * bs, device=device, dtype=torch.int64)
    selected_index = (
        torch.arange(draft_token_num - 1, device=device, dtype=torch.int64)
        .unsqueeze(0)
        .expand(bs, -1)
        .contiguous()
    )
    seq_lens = torch.full((bs,), 50, device=device, dtype=torch.int64)
    tree_mask = torch.full(
        (draft_token_num * bs * draft_token_num,),
        True,
        dtype=torch.bool,
        device=device,
    )
    positions = torch.empty((bs * draft_token_num,), device=device, dtype=torch.long)
    retrieve_buf = torch.full(
        (3, bs, draft_token_num), -1, device=device, dtype=torch.long
    )
    retrieve_index, retrieve_next_token, retrieve_next_sibling = retrieve_buf
    retrieve_parent_token = torch.full(
        (bs, draft_token_num), -1, device=device, dtype=torch.long
    )

    module.build_tree_kernel_efficient(
        parent_list,
        selected_index,
        seq_lens,
        tree_mask,
        positions,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        retrieve_parent_token,
        topk,
        depth,
        draft_token_num,
        1,  # tree_mask_mode = QLEN_ONLY
    )
    torch.cuda.synchronize()

    return (
        retrieve_parent_token,
        retrieve_next_token,
        retrieve_next_sibling,
        draft_token_num,
    )


def _verify_parent_map_consistency(
    retrieve_parent_token, retrieve_next_token, retrieve_next_sibling
):
    """Verify parent_map is consistent with next_token/next_sibling encoding.

    For every token t > 0:
      if retrieve_next_token[b, p] == t → parent[t] should be p
      if retrieve_next_sibling[b, s] == t → parent[t] == parent[s]
    """
    bs, seqlen = retrieve_parent_token.shape
    for b in range(bs):
        parent = retrieve_parent_token[b].cpu().tolist()
        next_token = retrieve_next_token[b].cpu().tolist()
        next_sibling = retrieve_next_sibling[b].cpu().tolist()

        # Root has no parent
        assert parent[0] == -1, f"Root parent should be -1, got {parent[0]}"

        # Check: if next_token[p] == t, then parent[t] == p
        for p in range(seqlen):
            t = next_token[p]
            if t != -1 and t < seqlen:
                assert (
                    parent[t] == p
                ), f"batch {b}: next_token[{p}]={t}, but parent[{t}]={parent[t]} != {p}"

        # Check: if next_sibling[s] == t, then parent[t] == parent[s]
        for s in range(seqlen):
            t = next_sibling[s]
            if t != -1 and t < seqlen:
                assert (
                    parent[t] == parent[s]
                ), f"batch {b}: next_sibling[{s}]={t}, parent[{t}]={parent[t]} != parent[{s}]={parent[s]}"


def _reference_conv1d_tree(
    x,
    weight,
    bias,
    conv_state,
    conv_state_indices,
    retrieve_parent_token,
    activation="silu",
):
    """Pure PyTorch reference for tree-traversal conv1d.

    Args:
        x: (batch, dim, seqlen)
        weight: (dim, width)
        bias: (dim,) or None
        conv_state: (num_cache_lines, dim, state_len)  state_len = width-1
        conv_state_indices: (batch,) int32
        retrieve_parent_token: (batch, seqlen) int64
        activation: "silu" or None

    Returns:
        out: (batch, dim, seqlen)
    """
    batch, dim, seqlen = x.shape
    _, width = weight.shape
    state_len = width - 1

    out = torch.zeros_like(x, dtype=torch.float32)

    for b in range(batch):
        cs_idx = conv_state_indices[b].item()
        for t in range(seqlen):
            # Walk ancestor chain: self, parent, grandparent, ...
            current = t
            for j in range(width):
                # Weight index: w[:, width-1-j] (self gets last column, parent gets second-to-last, etc)
                w_col = weight[:, width - 1 - j].float()

                if current >= 0:
                    # Load from input x
                    val = x[b, :, current].float()
                else:
                    # Load from conv_state history
                    # current=-1 → slot state_len-1 (most recent), -2 → state_len-2, ...
                    history_idx = state_len + current  # -1→state_len-1, -2→state_len-2
                    if history_idx >= 0:
                        val = conv_state[cs_idx, :, history_idx].float()
                    else:
                        val = torch.zeros(dim, device=x.device, dtype=torch.float32)

                out[b, :, t] += val * w_col

                # Move to parent
                if current > 0:
                    current = retrieve_parent_token[b, current].item()
                else:
                    current = current - 1

            # Add bias
            if bias is not None:
                out[b, :, t] += bias.float()

    # Apply activation
    if activation in ("silu", "swish"):
        out = out * torch.sigmoid(out)

    return out.to(x.dtype)


class TestParentMapCorrectness:
    """Test that build_tree_kernel_efficient outputs correct parent_map.

    Uses JIT compilation of eagle_utils.cu to test the CUDA kernel directly,
    without requiring a full sgl-kernel package rebuild.
    """

    @pytest.mark.parametrize("topk", [2, 4])
    @pytest.mark.parametrize("depth", [2, 3, 4])
    @pytest.mark.parametrize("bs", [1, 2])
    def test_parent_map_consistent_with_tree(self, topk, depth, bs):
        device = get_device()
        (
            retrieve_parent_token,
            retrieve_next_token,
            retrieve_next_sibling,
            draft_token_num,
        ) = _call_build_tree_jit(topk, depth, bs, device)

        _verify_parent_map_consistency(
            retrieve_parent_token, retrieve_next_token, retrieve_next_sibling
        )

    @pytest.mark.parametrize("topk", [2, 4])
    def test_parent_map_expected_values(self, topk):
        """Verify parent_map matches the expected k-ary tree structure."""
        device = get_device()
        depth = 2
        bs = 1
        (
            retrieve_parent_token,
            retrieve_next_token,
            retrieve_next_sibling,
            draft_token_num,
        ) = _call_build_tree_jit(topk, depth, bs, device)

        # For a complete k-ary tree: parent[i] = (i-1) // topk
        parent = retrieve_parent_token[0].cpu().tolist()
        assert parent[0] == -1
        for i in range(1, draft_token_num):
            expected_parent = (i - 1) // topk
            assert (
                parent[i] == expected_parent
            ), f"topk={topk}: parent[{i}] should be {expected_parent}, got {parent[i]}"

    def test_nullptr_no_crash(self):
        """Passing empty tensor (nullptr) should not crash the kernel."""
        module = _get_eagle_utils_jit_module()
        device = get_device()
        topk, depth, bs = 2, 2, 1
        draft_token_num = topk * depth + 1

        parents = [(i - 1) // topk for i in range(1, draft_token_num)]
        parent_list = torch.tensor([parents], device=device, dtype=torch.int64)
        selected_index = (
            torch.arange(draft_token_num - 1, device=device, dtype=torch.int64)
            .unsqueeze(0)
            .contiguous()
        )
        seq_lens = torch.tensor([50], device=device, dtype=torch.int64)
        tree_mask = torch.full(
            (draft_token_num * bs * draft_token_num,),
            True,
            dtype=torch.bool,
            device=device,
        )
        positions = torch.empty(
            (bs * draft_token_num,), device=device, dtype=torch.long
        )
        retrieve_buf = torch.full(
            (3, bs, draft_token_num), -1, device=device, dtype=torch.long
        )
        ri, rnt, rns = retrieve_buf
        # Empty tensor → nullptr in kernel
        rpt_empty = torch.empty((0,), device=device, dtype=torch.long)

        module.build_tree_kernel_efficient(
            parent_list,
            selected_index,
            seq_lens,
            tree_mask,
            positions,
            ri,
            rnt,
            rns,
            rpt_empty,
            topk,
            depth,
            draft_token_num,
            1,
        )
        torch.cuda.synchronize()
        # Should complete without crash


def _make_conv1d_inputs(batch, dim, seqlen, width, dtype, device, topk=2):
    """Create all inputs needed for conv1d tree-verify path."""
    torch.manual_seed(0)

    # Input x: (batch, dim, seqlen)
    x = torch.randn(batch, dim, seqlen, device=device, dtype=dtype)

    # Weight: (dim, width)
    weight = torch.randn(dim, width, device=device, dtype=dtype)
    weight_contiguous = weight.contiguous()
    assert weight_contiguous.stride(1) == 1

    # Bias: (dim,)
    bias = torch.randn(dim, device=device, dtype=dtype)

    # Conv state: (num_cache_lines, dim, state_len) where state_len = width - 1
    num_cache_lines = batch + 5
    state_len = width - 1
    conv_state = torch.randn(
        num_cache_lines, dim, state_len, device=device, dtype=dtype
    )

    # Conv state indices: (batch,)
    conv_state_indices = torch.arange(batch, device=device, dtype=torch.int32)

    # Intermediate conv window: (batch, seqlen, dim, width-1)
    intermediate_conv_window = torch.zeros(
        batch, seqlen, dim, width - 1, device=device, dtype=dtype
    )
    intermediate_state_indices = torch.arange(batch, device=device, dtype=torch.int32)

    # Build tree structure for parent map
    # Simple tree: token 0 is root, tokens 1..topk are children of 0,
    # tokens topk+1..2*topk are children of 1, etc.
    retrieve_parent_token = torch.full(
        (batch, seqlen), -1, device=device, dtype=torch.int64
    )
    # Build a simple tree
    for b in range(batch):
        # Root: parent = -1 (already set)
        for i in range(1, seqlen):
            # Parent of token i is (i-1) // topk
            retrieve_parent_token[b, i] = (i - 1) // topk

    retrieve_next_token = torch.full(
        (batch, seqlen), -1, device=device, dtype=torch.int64
    )
    retrieve_next_sibling = torch.full(
        (batch, seqlen), -1, device=device, dtype=torch.int64
    )
    # Fill next_token and next_sibling from parent_map
    for b in range(batch):
        for i in range(1, seqlen):
            p = retrieve_parent_token[b, i].item()
            if retrieve_next_token[b, p] == -1:
                retrieve_next_token[b, p] = i
            else:
                # Find the last sibling
                existing = retrieve_next_token[b, p].item()
                retrieve_next_token[b, p] = i
                retrieve_next_sibling[b, i] = existing

    return (
        x,
        weight_contiguous,
        bias,
        conv_state,
        conv_state_indices,
        intermediate_conv_window,
        intermediate_state_indices,
        retrieve_parent_token,
        retrieve_next_token,
        retrieve_next_sibling,
    )


def _call_serial_kernel(
    x,
    weight,
    bias,
    conv_state,
    conv_state_indices,
    intermediate_conv_window,
    intermediate_state_indices,
    retrieve_parent_token,
    retrieve_next_token,
    retrieve_next_sibling,
    activation="silu",
):
    """Call the serial kernel directly (bypassing dispatch logic)."""
    batch, dim, seqlen = x.shape
    _, width = weight.shape
    num_cache_lines, _, state_len = conv_state.size()

    out = torch.empty_like(x)
    stride_x_seq, stride_x_dim, stride_x_token = x.stride()
    stride_o_seq, stride_o_dim, stride_o_token = out.stride()
    stride_w_dim, stride_w_width = weight.stride()
    stride_istate_seq, stride_istate_dim, stride_istate_token = conv_state.stride()
    stride_state_indices = conv_state_indices.stride(0)
    stride_intermediate_state_indices = intermediate_state_indices.stride(0)
    stride_inter_seq, stride_inter_step, stride_inter_dim, stride_inter_win = (
        intermediate_conv_window.stride()
    )
    stride_retrieve_next_token_seq = retrieve_next_token.stride(0)
    stride_retrieve_next_token_token = retrieve_next_token.stride(1)
    stride_retrieve_next_sibling_seq = retrieve_next_sibling.stride(0)
    stride_retrieve_next_sibling_token = retrieve_next_sibling.stride(1)
    stride_retrieve_parent_token_seq = retrieve_parent_token.stride(0)
    stride_retrieve_parent_token_token = retrieve_parent_token.stride(1)

    state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)
    np2_seqlen = triton.next_power_of_2(seqlen)

    def grid(META):
        return (batch, triton.cdiv(dim, META["BLOCK_N"]))

    _causal_conv1d_update_kernel[grid](
        x,
        weight,
        bias,
        conv_state,
        None,  # cache_seqlens
        conv_state_indices,
        None,  # num_accept_tokens
        intermediate_conv_window,
        intermediate_state_indices,
        retrieve_next_token,
        retrieve_next_sibling,
        retrieve_parent_token,
        out,
        batch,
        dim,
        seqlen,
        state_len,
        num_cache_lines,
        stride_x_seq,
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_state_indices,
        stride_inter_seq,
        stride_inter_step,
        stride_inter_dim,
        stride_inter_win,
        stride_intermediate_state_indices,
        stride_retrieve_next_token_seq,
        stride_retrieve_next_token_token,
        stride_retrieve_next_sibling_seq,
        stride_retrieve_next_sibling_token,
        stride_retrieve_parent_token_seq,
        stride_retrieve_parent_token_token,
        stride_o_seq,
        stride_o_dim,
        stride_o_token,
        PAD_SLOT_ID,
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_CONTINUOUS_BATCHING=True,
        IS_SPEC_DECODING=False,
        NP2_STATELEN=np2_statelen,
        NP2_SEQLEN=np2_seqlen,
        USE_PAD_SLOT=True,
        BLOCK_N=256,
        SAVE_INTERMEDIATE=True,
        HAS_EAGLE_TREE_CUSTOM_ATTN_MASK=True,
    )
    return out


def _call_token_parallel_kernel(
    x,
    weight,
    bias,
    conv_state,
    conv_state_indices,
    intermediate_conv_window,
    intermediate_state_indices,
    retrieve_parent_token,
    activation="silu",
):
    """Call the token-parallel kernel directly."""
    batch, dim, seqlen = x.shape
    _, width = weight.shape
    num_cache_lines, _, state_len = conv_state.size()

    out = torch.empty_like(x)
    stride_x_seq, stride_x_dim, stride_x_token = x.stride()
    stride_o_seq, stride_o_dim, stride_o_token = out.stride()
    stride_w_dim, stride_w_width = weight.stride()
    stride_istate_seq, stride_istate_dim, stride_istate_token = conv_state.stride()
    stride_state_indices = conv_state_indices.stride(0)
    stride_intermediate_state_indices = intermediate_state_indices.stride(0)
    stride_inter_seq, stride_inter_step, stride_inter_dim, stride_inter_win = (
        intermediate_conv_window.stride()
    )
    stride_retrieve_parent_token_seq = retrieve_parent_token.stride(0)
    stride_retrieve_parent_token_token = retrieve_parent_token.stride(1)

    state_len = width - 1

    def grid(META):
        return (batch, triton.cdiv(dim, META["BLOCK_N"]), seqlen)

    _causal_conv1d_verify_token_parallel_kernel[grid](
        x,
        weight,
        bias,
        conv_state,
        conv_state_indices,
        retrieve_parent_token,
        intermediate_conv_window,
        intermediate_state_indices,
        out,
        batch,
        dim,
        seqlen,
        state_len,
        num_cache_lines,
        stride_x_seq,
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_state_indices,
        stride_inter_seq,
        stride_inter_step,
        stride_inter_dim,
        stride_inter_win,
        stride_intermediate_state_indices,
        stride_retrieve_parent_token_seq,
        stride_retrieve_parent_token_token,
        stride_o_seq,
        stride_o_dim,
        stride_o_token,
        PAD_SLOT_ID,
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        BLOCK_N=256,
        SAVE_INTERMEDIATE=True,
    )
    return out


class TestTokenParallelCorrectness:
    """Test that token-parallel kernel produces same output as serial kernel."""

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("width", [3, 4])
    @pytest.mark.parametrize("seqlen", [5, 7, 9, 13])
    @pytest.mark.parametrize("dim", [256, 512, 2048])
    @pytest.mark.parametrize("batch", [1, 2, 4])
    def test_token_parallel_matches_serial(self, batch, dim, seqlen, width, dtype):
        """Token-parallel kernel should produce same outputs as serial kernel."""
        device = get_device()
        topk = 2

        (
            x,
            weight,
            bias,
            conv_state,
            conv_state_indices,
            intermediate_conv_window,
            intermediate_state_indices,
            retrieve_parent_token,
            retrieve_next_token,
            retrieve_next_sibling,
        ) = _make_conv1d_inputs(batch, dim, seqlen, width, dtype, device, topk)

        # Run serial kernel directly
        intermediate_serial = intermediate_conv_window.clone()
        out_serial = _call_serial_kernel(
            x.clone(),
            weight,
            bias,
            conv_state.clone(),
            conv_state_indices.clone(),
            intermediate_serial,
            intermediate_state_indices.clone(),
            retrieve_parent_token,
            retrieve_next_token,
            retrieve_next_sibling,
        )

        # Run token-parallel kernel directly
        intermediate_tp = intermediate_conv_window.clone()
        out_tp = _call_token_parallel_kernel(
            x.clone(),
            weight,
            bias,
            conv_state.clone(),
            conv_state_indices.clone(),
            intermediate_tp,
            intermediate_state_indices.clone(),
            retrieve_parent_token,
        )

        # Check output matches
        if dtype == torch.float32:
            rtol, atol = 1e-5, 1e-5
        else:
            rtol, atol = 1e-2, 5e-2

        assert (
            out_serial.shape == out_tp.shape
        ), f"Shape mismatch: serial={out_serial.shape}, tp={out_tp.shape}"
        assert torch.allclose(
            out_serial, out_tp, rtol=rtol, atol=atol
        ), f"Output mismatch! Max diff: {(out_serial - out_tp).abs().max().item()}"

        # Check intermediate conv window matches
        assert torch.allclose(
            intermediate_serial, intermediate_tp, rtol=rtol, atol=atol
        ), (
            f"Intermediate window mismatch! Max diff: "
            f"{(intermediate_serial - intermediate_tp).abs().max().item()}"
        )

    @pytest.mark.parametrize("width", [3, 4])
    def test_token_parallel_no_bias(self, width):
        """Test without bias."""
        device = get_device()
        batch, dim, seqlen = 2, 512, 7
        dtype = torch.bfloat16

        (
            x,
            weight,
            bias,
            conv_state,
            conv_state_indices,
            intermediate_conv_window,
            intermediate_state_indices,
            retrieve_parent_token,
            retrieve_next_token,
            retrieve_next_sibling,
        ) = _make_conv1d_inputs(batch, dim, seqlen, width, dtype, device, topk=2)

        # Serial path
        out_serial = _call_serial_kernel(
            x.clone(),
            weight,
            None,
            conv_state.clone(),
            conv_state_indices.clone(),
            intermediate_conv_window.clone(),
            intermediate_state_indices.clone(),
            retrieve_parent_token,
            retrieve_next_token,
            retrieve_next_sibling,
            activation="silu",
        )

        # Token-parallel path
        out_tp = _call_token_parallel_kernel(
            x.clone(),
            weight,
            None,
            conv_state.clone(),
            conv_state_indices.clone(),
            intermediate_conv_window.clone(),
            intermediate_state_indices.clone(),
            retrieve_parent_token,
            activation="silu",
        )

        rtol, atol = 1e-2, 5e-2
        assert torch.allclose(out_serial, out_tp, rtol=rtol, atol=atol)

    def test_token_parallel_linear_chain(self):
        """Test with topk=1 (linear chain) where parent[i] = i-1."""
        device = get_device()
        batch, dim, seqlen, width = 2, 512, 5, 4
        dtype = torch.float32

        (
            x,
            weight,
            bias,
            conv_state,
            conv_state_indices,
            intermediate_conv_window,
            intermediate_state_indices,
            retrieve_parent_token,
            retrieve_next_token,
            retrieve_next_sibling,
        ) = _make_conv1d_inputs(batch, dim, seqlen, width, dtype, device, topk=1)

        # Serial path
        out_serial = _call_serial_kernel(
            x.clone(),
            weight,
            bias,
            conv_state.clone(),
            conv_state_indices.clone(),
            intermediate_conv_window.clone(),
            intermediate_state_indices.clone(),
            retrieve_parent_token,
            retrieve_next_token,
            retrieve_next_sibling,
        )

        # Token-parallel path
        out_tp = _call_token_parallel_kernel(
            x.clone(),
            weight,
            bias,
            conv_state.clone(),
            conv_state_indices.clone(),
            intermediate_conv_window.clone(),
            intermediate_state_indices.clone(),
            retrieve_parent_token,
        )

        rtol, atol = 1e-5, 1e-5
        assert torch.allclose(
            out_serial, out_tp, rtol=rtol, atol=atol
        ), f"Linear chain mismatch! Max diff: {(out_serial - out_tp).abs().max().item()}"

    @pytest.mark.parametrize("width", [3, 4])
    @pytest.mark.parametrize("seqlen", [3, 5])
    def test_against_reference(self, width, seqlen):
        """Test both kernels against pure Python reference implementation."""
        device = get_device()
        batch, dim = 2, 64  # small dim for reference speed
        dtype = torch.float32

        (
            x,
            weight,
            bias,
            conv_state,
            conv_state_indices,
            intermediate_conv_window,
            intermediate_state_indices,
            retrieve_parent_token,
            retrieve_next_token,
            retrieve_next_sibling,
        ) = _make_conv1d_inputs(batch, dim, seqlen, width, dtype, device, topk=2)

        # Reference
        out_ref = _reference_conv1d_tree(
            x,
            weight,
            bias,
            conv_state,
            conv_state_indices,
            retrieve_parent_token,
            activation="silu",
        )

        # Token-parallel kernel
        out_tp = _call_token_parallel_kernel(
            x.clone(),
            weight,
            bias,
            conv_state.clone(),
            conv_state_indices.clone(),
            intermediate_conv_window.clone(),
            intermediate_state_indices.clone(),
            retrieve_parent_token,
        )

        rtol, atol = 1e-4, 1e-4
        assert torch.allclose(out_ref, out_tp, rtol=rtol, atol=atol), (
            f"Token-parallel vs reference mismatch! "
            f"Max diff: {(out_ref - out_tp).abs().max().item()}"
        )

    def test_dispatch_uses_token_parallel(self):
        """Test that causal_conv1d_update dispatches to token-parallel when appropriate."""
        device = get_device()
        batch, dim, seqlen, width = 2, 256, 5, 4
        dtype = torch.float32

        (
            x,
            weight,
            bias,
            conv_state,
            conv_state_indices,
            intermediate_conv_window,
            intermediate_state_indices,
            retrieve_parent_token,
            retrieve_next_token,
            retrieve_next_sibling,
        ) = _make_conv1d_inputs(batch, dim, seqlen, width, dtype, device, topk=2)

        # Call via top-level wrapper with retrieve_parent_token → should use token-parallel
        out_dispatch = causal_conv1d_update(
            x.clone(),
            conv_state.clone(),
            weight,
            bias,
            activation="silu",
            conv_state_indices=conv_state_indices.clone(),
            intermediate_conv_window=intermediate_conv_window.clone(),
            intermediate_state_indices=intermediate_state_indices.clone(),
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            retrieve_parent_token=retrieve_parent_token,
        )

        # Call token-parallel kernel directly for comparison
        out_tp = _call_token_parallel_kernel(
            x.clone(),
            weight,
            bias,
            conv_state.clone(),
            conv_state_indices.clone(),
            intermediate_conv_window.clone(),
            intermediate_state_indices.clone(),
            retrieve_parent_token,
        )

        rtol, atol = 1e-5, 1e-5
        assert torch.allclose(
            out_dispatch, out_tp, rtol=rtol, atol=atol
        ), f"Dispatch mismatch! Max diff: {(out_dispatch - out_tp).abs().max().item()}"


class TestTokenParallelPerformance:
    """Benchmark token-parallel vs serial conv1d kernels."""

    def _benchmark_kernel(self, fn, warmup=50, repeats=200):
        """Benchmark a callable, returns median time in ms."""
        # Warmup
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        # Timed runs
        times = []
        for _ in range(repeats):
            torch.cuda.synchronize()
            start = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)

        times.sort()
        return times[len(times) // 2]  # median

    @pytest.mark.parametrize(
        "batch,dim,seqlen,width",
        [
            (4, 2048, 9, 4),  # typical EAGLE topk=2 depth=3
            (8, 2048, 9, 4),  # larger batch
            (4, 4096, 13, 4),  # larger dim + more tokens
            (8, 4096, 17, 4),  # stress test
            (16, 2048, 9, 4),  # large batch
        ],
    )
    def test_performance(self, batch, dim, seqlen, width):
        """Compare token-parallel vs serial kernel performance."""
        device = get_device()
        dtype = torch.bfloat16

        (
            x,
            weight,
            bias,
            conv_state,
            conv_state_indices,
            intermediate_conv_window,
            intermediate_state_indices,
            retrieve_parent_token,
            retrieve_next_token,
            retrieve_next_sibling,
        ) = _make_conv1d_inputs(batch, dim, seqlen, width, dtype, device, topk=2)

        def run_serial():
            return _call_serial_kernel(
                x.clone(),
                weight,
                bias,
                conv_state.clone(),
                conv_state_indices,
                intermediate_conv_window.clone(),
                intermediate_state_indices,
                retrieve_parent_token,
                retrieve_next_token,
                retrieve_next_sibling,
            )

        def run_token_parallel():
            return _call_token_parallel_kernel(
                x.clone(),
                weight,
                bias,
                conv_state.clone(),
                conv_state_indices,
                intermediate_conv_window.clone(),
                intermediate_state_indices,
                retrieve_parent_token,
            )

        serial_ms = self._benchmark_kernel(run_serial)
        tp_ms = self._benchmark_kernel(run_token_parallel)
        speedup = serial_ms / tp_ms if tp_ms > 0 else float("inf")

        print(
            f"\n[batch={batch}, dim={dim}, seqlen={seqlen}, width={width}] "
            f"Serial: {serial_ms:.3f}ms | Token-Parallel: {tp_ms:.3f}ms | "
            f"Speedup: {speedup:.2f}x"
        )

        # Token-parallel should not be significantly slower (allow 20% regression)
        assert (
            tp_ms < serial_ms * 1.2
        ), f"Token-parallel is too slow: {tp_ms:.3f}ms vs serial {serial_ms:.3f}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
