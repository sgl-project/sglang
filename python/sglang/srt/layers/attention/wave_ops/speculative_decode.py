# Copyright 2023-2024 SGLang Team
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

import functools
import os

import iree.turbine.kernel as tk
import torch
import torch.nn.functional as F
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile
from iree.turbine.kernel.wave.templates.speculative_decoding import (
    get_speculative_decoding_kernel,
    get_speculative_sampling_kernel,
)
from iree.turbine.kernel.wave.utils.general_utils import get_default_scheduling_params
from iree.turbine.kernel.wave.utils.run_utils import set_default_run_config

dump_generated_mlir = int(os.environ.get("WAVE_DUMP_MLIR", 0))
enable_scheduling_barriers = int(os.environ.get("WAVE_USE_SCHED_BARRIERS", 0))


@functools.lru_cache
def get_wave_speculative_sampling_kernel(
    batch_size,
    num_speculative_tokens,
    threshold_acc,
    threshold_single,
    num_draft_tokens,
    d,
    seq_len,
):
    speculative_sampling, symbols, _, _ = get_speculative_sampling_kernel(
        batch_size,
        num_speculative_tokens,
        threshold_acc,
        threshold_single,
        num_draft_tokens,
        d,
        seq_len,
    )
    symbols.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=symbols,
        canonicalize=True,
        run_bench=False,
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
        wave_runtime=True,
    )
    options = set_default_run_config(options)
    speculative_sampling = wave_compile(options, speculative_sampling)
    return speculative_sampling


@functools.lru_cache
def get_wave_speculative_decoding_kernel(batch_size, num_draft_tokens, d, seq_len):
    speculative_decoding, symbols, _, _ = get_speculative_decoding_kernel(
        batch_size,
        num_draft_tokens,
        d,
        seq_len,
    )
    symbols.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=symbols,
        canonicalize=True,
        run_bench=False,
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
        schedule=False,
        wave_runtime=True,
        use_scheduling_barriers=enable_scheduling_barriers,
    )
    options = set_default_run_config(options)
    speculative_decoding = wave_compile(options, speculative_decoding)
    return speculative_decoding


def get_wave_kernel(
    seq_len,
    threshold_acc,
    threshold_single,
    batch_size,
    num_speculative_tokens,
    num_draft_tokens,
    d,
):

    sampling_kernel = get_wave_speculative_sampling_kernel(
        batch_size,
        num_speculative_tokens,
        threshold_acc,
        threshold_single,
        num_draft_tokens,
        d,
        seq_len,
    )

    decode_kernel = get_wave_speculative_decoding_kernel(
        batch_size, num_draft_tokens, d, seq_len
    )

    return sampling_kernel, decode_kernel


def speculative_decode_wave(
    predicts,  # [seq_len], mutable
    accept_index,  # [batch_size, num_speculative_tokens], mutable
    accept_token_num,  # [batch_size], mutable
    candidates,  # [batch_size, num_draft_tokens]
    retrive_index,  # [batch_size, num_draft_tokens]
    retrive_next_token,  # [batch_size, num_draft_tokens]
    retrive_next_sibling,  # [batch_size, num_draft_tokens]
    uniform_samples,  # [batch_size, num_draft_tokens]
    target_probs,  # [batch_size, num_draft_tokens, vocab_size]
    draft_probs,  # [batch_size, num_draft_tokens, vocab_size]
    batch_size,
    num_speculative_tokens,
    num_draft_tokens,
    d,
    threshold_single=1.0,
    threshold_acc=1.0,
    deterministic=True,
):
    threshold_acc = max(threshold_acc, 1e-9)
    seq_len = predicts.shape[0]
    cur_prob_offset_vec = torch.empty(
        [batch_size], dtype=torch.int32, device=draft_probs.device
    )
    last_accepted_retrive_idx_vec = torch.empty(
        [batch_size], dtype=torch.int32, device=draft_probs.device
    )

    sampling_kernel, decode_kernel = get_wave_kernel(
        seq_len,
        threshold_acc,
        threshold_single,
        batch_size,
        num_speculative_tokens,
        num_draft_tokens,
        d,
    )

    sampling_kernel(
        uniform_samples,
        target_probs,
        draft_probs,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        predicts,
        accept_token_num,
        accept_index,
        cur_prob_offset_vec,
        last_accepted_retrive_idx_vec,
    )

    decode_kernel(
        target_probs,
        draft_probs,
        cur_prob_offset_vec,
        uniform_samples,
        last_accepted_retrive_idx_vec,
        predicts,
    )
