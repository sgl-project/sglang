# SPDX-License-Identifier: Apache-2.0
"""AOT-only Rubin routed engine for M8/M12/M16; physical intermediate192."""
from functools import cache
import hashlib
import logging
import os
from pathlib import Path

import torch
from flashinfer.tllm_enums import ActivationType

from sglang.kernels.ops.moe import q38_router_jit as jit
from sglang.kernels.ops.moe.q38_mk_moe import tensor_map
from sglang.kernels.ops.moe.q38_mk_q35 import Params, _bundle, _finalize, _identity
from sglang.kernels.ops.moe.q38_moe_aot import bundle_manifest

H, I, E, TOPK = 2560, 192, 512, 10
SHARED, THREADS = 215040, 256
logger = logging.getLogger(__name__)
_seen = set()


@cache
def bundle(device_index, m8=False):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('wide MoE AOT must load during eager warmup')
    variable = ('SGLANG_Q38C_MKM_WIDE_M8_AOT'
                if m8 and os.environ.get('SGLANG_Q38C_MKM_WIDE_M8_AOT')
                else 'SGLANG_Q38C_MKM_WIDE_AOT')
    root, manifest = bundle_manifest(variable, torch.cuda.get_device_capability(device_index))
    source = Path(__file__).resolve().parents[2] / 'jit/csrc/moe/q38_mkm_wide.cu'
    assert manifest['source_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert manifest['shape'] == [H, I, E, TOPK, 16]
    assert manifest['shared'] == SHARED and manifest['threads'] == THREADS
    cubin = root / 'wide.cubin'
    assert manifest['cubin_sha256'] == hashlib.sha256(cubin.read_bytes()).hexdigest()
    props = torch.cuda.get_device_properties(device_index)
    assert SHARED <= props.shared_memory_per_block_optin
    expert = jit.load(cubin.read_bytes(), 'q38_mkm_wide', (THREADS, 1, 1), SHARED)
    _, finalize, metadata = _bundle(device_index)
    return expert, finalize, dict(metadata, expert_spec_stages=manifest.get('spec_stages', 4))


def maps(w13, w2, xq):
    swizzle = jit.driver().CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B
    assert tuple(w13.shape) == (E, 2 * I, H // 2)
    assert tuple(w2.shape) == (E, H, I // 2)
    return (tensor_map(w13, E * 2 * I, H // 2, 128, 128, swizzle),
            tensor_map(w2, E * H, I // 2, 128, 256, swizzle),
            tensor_map(xq, xq.shape[0], H // 2, 128, 16, swizzle))


def run(kernel, tensor_maps, kwargs, rows, weights, ids):
    m = kwargs['hidden_states'].shape[0]
    assert m in (8, 12, 16)
    tensors = (kwargs['routing_logits'], weights, ids, kwargs['hidden_states_scale'],
               kwargs['gemm1_weights_scale'], kwargs['gemm2_weights_scale'],
               kwargs['output1_scale_gate_scalar'], kwargs['output1_scale_scalar'],
               kwargs['output2_scale_scalar'], rows)
    params = Params(*(tensor.data_ptr() for tensor in tensors), 0, m)
    jit.launch(kernel, (m * TOPK, 1, 1), [*tensor_maps, params],
               torch.cuda.current_stream().cuda_stream)
    return rows


def maybe_run_service(kwargs):
    xq = kwargs['hidden_states']
    m = xq.shape[0]
    if m not in (8, 12, 16):
        return None
    batch34 = os.environ.get('SGLANG_Q38C_MKM_WIDE_BATCH34', '0') == '1'
    if batch34:
        from sglang.srt.model_executor.forward_context import (
            get_forward_context,
            has_forward_context,
        )

        if m not in (12, 16) or not has_forward_context():
            return None
        context = get_forward_context()
        mode = context.moe_forward_mode
        batch = context.moe_batch_size
        if (mode is None or not mode.is_target_verify()
                or batch not in (3, 4) or m != 4 * batch):
            return None
    if m == 8 and os.environ.get('SGLANG_Q38C_MKM_WIDE_M8', '1') == '0':
        return None
    if (kwargs['intermediate_size'] != I or xq.shape[-1] != H // 2
            or kwargs['num_experts'] != E or kwargs['local_num_experts'] != E
            or kwargs['local_expert_offset'] != 0 or kwargs['top_k'] != TOPK
            or int(kwargs['routing_method_type']) != 4
            or kwargs['activation_type'] != ActivationType.Swiglu
            or kwargs['routing_bias'] is not None or kwargs['per_token_scale'] is not None
            or kwargs['routed_scaling_factor'] not in (None, 1.0)
            or kwargs['gemm1_alpha'] is not None or kwargs['gemm1_beta'] is not None
            or kwargs['gemm1_clamp_limit'] is not None):
        return None
    kernel, finalize, metadata = bundle(xq.device.index, m == 8)
    rows = torch.empty((m, TOPK, H), dtype=torch.bfloat16, device=xq.device)
    weights = torch.empty((m, TOPK), dtype=torch.bfloat16, device=xq.device)
    ids = torch.empty((m, TOPK), dtype=torch.int32, device=xq.device)
    permutation = _identity(xq.device.index, m) if not kwargs['do_finalize'] else None
    run(kernel, maps(kwargs['gemm1_weights'], kwargs['gemm2_weights'], xq),
        kwargs, rows, weights, ids)
    signature = m, torch.cuda.is_current_stream_capturing(), kwargs['do_finalize']
    if signature not in _seen:
        _seen.add(signature)
        if batch34:
            logger.info('Q38C_MKM_BATCH34_ACTIVE M=%d batch=%d mode=%s capture=%s',
                        m, batch, mode.name, signature[1])
        logger.info('Q38C_MKM_WIDE_ACTIVE M=%d I=192 capture=%s finalize=%s grid=%d cluster=1 spec_stages=%d',
                    *signature, m * TOPK, metadata['expert_spec_stages'])
    if not kwargs['do_finalize']:
        return rows.view(m * TOPK, H), weights, permutation
    output = kwargs.get('output')
    if output is None:
        output = torch.empty((m, H), dtype=torch.bfloat16, device=xq.device)
    _finalize(finalize, metadata, rows, weights, output)
    return (output,)
