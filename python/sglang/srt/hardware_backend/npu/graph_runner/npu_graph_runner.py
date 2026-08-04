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
"""Run the model with NPU graph and torch.compile.

NPUGraphRunner is a thin subclass of DecodeCudaGraphRunner: the
factory returns NPUCudaGraphBackend for NPU devices, so all
capture/replay mechanics live in the backend. This class adds:
  - NPU-specific patch_model monkey-patch for the decode-Full +
    torch.compile path.
  - Profile context override (NPU profiler emits to disk, not in-mem).
  - Replay override that issues an async NPUGraph.update for
    seq_lens before replay (skipped for deepseek-nsa).
  - Smaller cache_loc dtype (int32 instead of int64).
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Union

import numpy as np
import torch

from sglang.srt.configs.model_config import (
    AttentionArch,
    is_deepseek_dsa,
    is_deepseek_v4,
)
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.environ import envs
from sglang.srt.model_executor.runner import DecodeCudaGraphRunner
from sglang.srt.utils import (
    empty_context,
    get_bool_env_var,
    get_compiler_backend,
    is_npu,
)

is_npu = is_npu()

if is_npu:
    import torch_npu
    from torch_npu.profiler import ProfilerActivity, profile

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors


@contextmanager
def patch_model_npu(
    model: torch.nn.Module,
    enable_compile: bool,
    num_tokens: int,
    tp_group: GroupCoordinator,
):
    if enable_compile:
        backend = get_compiler_backend("npugraph_ex")
        yield torch.compile(
            torch.no_grad()(model.forward),
            fullgraph=True,
            dynamic=False,
            backend=backend,
        )
    else:
        yield model.forward


class NPUGraphRunner(DecodeCudaGraphRunner):
    """A NPUGraphRunner runs the forward pass of a model with NPU graph and torch.compile."""

    def __init__(
        self,
        model_runner: ModelRunner,
        *,
        attn_backend=None,
        speculative_num_steps: Optional[int] = None,
        speculative_num_draft_tokens: Optional[int] = None,
    ):
        # NPU patch_model override: monkey-patch torch_compile_decoration's
        # patch_model with the NPU-specific version.
        from sglang.srt.compilation import torch_compile_decoration

        torch_compile_decoration.patch_model = patch_model_npu
        super().__init__(
            model_runner,
            attn_backend=attn_backend,
            speculative_num_steps=speculative_num_steps,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
        )
        self.update_attr_name = None
        self.update_attr_type = None
        self.model_runner = model_runner
        self._init_arch_map()
        self.use_fia = get_bool_env_var("ASCEND_USE_FIA", "False")
        self.if_use_v2 = any(
            arch
            in ("MiMoV2ForCausalLM", "MiMoV2FlashForCausalLM", "Step3p5ForCausalLM")
            for arch in (model_runner.model_config.hf_config.architectures or [])
        )

    def _init_arch_map(self):
        if self.is_dllm:
            self.attr_name: Dict[str, str] = {
                AttentionArch.MLA: "actual_seq_lengths_kv",
                AttentionArch.MHA: "actual_seq_lengths_kv",
                "TARGET_VERIFY": "actual_seq_kvlen",
            }
        else:
            self.attr_name: Dict[str, str] = {
                AttentionArch.MLA: "actual_seq_lengths_kv",
                AttentionArch.MHA: "context_lens",
                "TARGET_VERIFY": "actual_seq_kvlen",
            }
        self.attr_type: Dict[str, Union[list, torch.Tensor]] = {
            AttentionArch.MLA: [],
            AttentionArch.MHA: torch.Tensor(),
            "TARGET_VERIFY": [],
        }

    def _create_device_graph(self):
        return torch.npu.NPUGraph()

    def _capture_graph(self, graph, pool, stream, run_once_fn):
        if self.enable_torch_compile:
            skip_guard_context = torch.compiler.set_stance(skip_guard_eval_unsafe=True)
        else:
            skip_guard_context = empty_context()

        with (
            skip_guard_context,
            torch.npu.graph(
                graph,
                pool=pool,
                stream=stream,
                auto_dispatch_capture=True,
            ),
        ):
            out = run_once_fn()
        return out

    def _get_update_attr_name(self):
        if self.if_use_v2:
            return self.attr_name["TARGET_VERIFY"]
        return self.attr_name[AttentionArch.MLA]

    def _get_update_attr_type(self):
        if self.if_use_v2:
            return self.attr_type["TARGET_VERIFY"]
        return self.attr_type[AttentionArch.MLA]

    def _update_inputs(self, seq_lens):
        if isinstance(self.update_attr_type, torch.Tensor):
            seq_lens = torch.from_numpy(np.array(seq_lens).astype(np.int32))

        self.graphs[self.bs].update(
            cpu_update_input=[{self.update_attr_name: seq_lens}]
        )

    def _cache_loc_dtype(self):
        return torch.int32

    def _init_profile_context_and_memory_record(self):
        output_dir = os.path.join(
            os.getenv("SGLANG_TORCH_PROFILER_DIR", "/tmp"), "graph_capture_profile"
        )
        if not Path(output_dir).exists():
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Profiling starts for graph capture for NPU. Traces will be saved to: {output_dir}"
        )
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            export_type=[torch_npu.profiler.ExportType.Text],
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        )
        profile_context = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
            record_shapes=True,
            profile_memory=True,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                output_dir, async_mode=True
            ),
            experimental_config=experimental_config,
        )
        return profile_context

    def _post_process_after_profile(self, prof_context):
        # for NPU, profile data will be saved to disk for further analysis.
        pass

    def _trace_mamba_graph_boundary(
        self, forward_batch: ForwardBatch, stage: str
    ) -> None:
        """Trace persistent Mamba caches at the Python graph-replay boundary."""
        marker = os.environ.get("SGLANG_K3_TRACE_STATE_FILE") or os.environ.get(
            "SGLANG_K3_TRACE_HIDDEN_FILE"
        )
        if not marker or not os.path.exists(marker):
            return
        try:
            backend = self.attn_backend
            select_backend = getattr(backend, "_select_backend", None)
            if select_backend is not None:
                backend = select_backend(forward_batch.forward_mode)
            linear_backend = getattr(backend, "linear_attn_backend", None)
            if linear_backend is None:
                return

            from sglang.srt.hardware_backend.npu.attention.ascend_hybrid_linear_attn_backend import (
                _state_trace_enabled,
                _trace_selected_state,
            )

            if not _state_trace_enabled():
                return
            req_pool_indices = forward_batch.req_pool_indices[
                : forward_batch.batch_size
            ]
            mamba_indices = linear_backend.req_to_token_pool.get_mamba_indices(
                req_pool_indices
            )
            caches = (
                linear_backend.req_to_token_pool.get_speculative_mamba2_params_all_layers()
            )
            mode = forward_batch.forward_mode.name.lower()
            logger.warning(
                "K3_GRAPH_STATE stage=%s mode=%s req_pool_indices=%s "
                "mamba_indices=%s seq_lens=%s",
                stage,
                mode,
                req_pool_indices.detach().cpu().tolist(),
                mamba_indices.detach().cpu().tolist(),
                forward_batch.seq_lens.detach().cpu().tolist(),
            )
            _trace_selected_state(
                stage=f"{stage}_{mode}",
                tensor_name="ssm_persistent",
                tensor=caches.temporal,
                slot_indices=mamba_indices,
            )
            _trace_selected_state(
                stage=f"{stage}_{mode}",
                tensor_name="conv_persistent",
                tensor=caches.conv[0],
                slot_indices=mamba_indices,
            )
        except Exception:
            logger.warning("K3 graph-boundary state trace failed", exc_info=True)

    def _trace_k3_verify_rows(self, forward_batch: ForwardBatch) -> None:
        """Dump live verify-row metadata and captured per-layer row buffers."""
        if not forward_batch.forward_mode.is_target_verify():
            return
        try:
            from sglang.srt.hardware_backend.npu.k3_graph_row_trace import (
                dump_graph_row_traces,
                graph_row_trace_marker_enabled,
            )

            if not graph_row_trace_marker_enabled():
                return

            replay_id = int(getattr(self, "_k3_graph_trace_replay_id", 0))
            self._k3_graph_trace_replay_id = replay_id + 1
            rank = (
                int(torch.distributed.get_rank())
                if torch.distributed.is_available()
                and torch.distributed.is_initialized()
                else 0
            )

            raw_bs = int(self.raw_bs)
            raw_num_tokens = int(self.raw_num_token)
            spec_info = forward_batch.spec_info
            draft_token_num = int(spec_info.draft_token_num)
            ragged_layout = getattr(spec_info, "ragged_verify_layout", None)
            if ragged_layout is None:
                verify_lens = [draft_token_num] * raw_bs
                capture_num_tokens = int(self.bs) * draft_token_num
            else:
                verify_lens = (
                    ragged_layout.verify_lens[:raw_bs].detach().cpu().tolist()
                )
                capture_num_tokens = int(ragged_layout.graph_num_tokens)

            input_ids = (
                forward_batch.input_ids.reshape(-1)[:raw_num_tokens]
                .detach()
                .cpu()
                .tolist()
            )
            positions = (
                forward_batch.positions.reshape(-1)[:raw_num_tokens]
                .detach()
                .cpu()
                .tolist()
            )
            seq_lens = (
                forward_batch.seq_lens[:raw_bs].detach().cpu().tolist()
            )
            req_pool_indices = (
                forward_batch.req_pool_indices[:raw_bs].detach().cpu().tolist()
            )
            rows = []
            compact_row = 0
            for req_row, verify_len in enumerate(verify_lens):
                for verify_step in range(int(verify_len)):
                    if compact_row >= raw_num_tokens:
                        break
                    rows.append(
                        {
                            "compact_row": compact_row,
                            "dense_row": req_row * draft_token_num + verify_step,
                            "req_row": req_row,
                            "req_pool_index": req_pool_indices[req_row],
                            "verify_step": verify_step,
                            "input_id": input_ids[compact_row],
                            "position": positions[compact_row],
                            "seq_len": seq_lens[req_row],
                        }
                    )
                    compact_row += 1

            logger.warning(
                "K3_GRAPH_VERIFY_ROW_MAP rank=%d replay_id=%d mode=%s "
                "capture_bs=%d "
                "capture_num_tokens=%d raw_bs=%d "
                "raw_num_tokens=%d draft_token_num=%d verify_lens=%s rows=%s",
                rank,
                replay_id,
                getattr(
                    forward_batch.forward_mode,
                    "name",
                    str(forward_batch.forward_mode),
                ).lower(),
                int(self.bs),
                capture_num_tokens,
                raw_bs,
                raw_num_tokens,
                draft_token_num,
                verify_lens,
                rows,
            )
            dump_graph_row_traces(
                replay_id=replay_id,
                mode=getattr(
                    forward_batch.forward_mode,
                    "name",
                    str(forward_batch.forward_mode),
                ).lower(),
                capture_bs=int(self.bs),
                capture_num_tokens=capture_num_tokens,
                raw_bs=raw_bs,
                raw_num_tokens=raw_num_tokens,
                dense_verify_tokens=raw_bs * draft_token_num,
            )
        except Exception:
            logger.warning("K3 graph verify-row trace failed", exc_info=True)

    def execute(
        self,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        if forward_batch.needs_forward_metadata_init():
            self.load_batch(forward_batch, pp_proxy_tensors)
        else:
            # In speculative decoding, these two fields are still needed.
            self.buffers.input_ids[: self.raw_num_token].copy_(forward_batch.input_ids)
            self.buffers.positions[: self.raw_num_token].copy_(forward_batch.positions)
            if (
                self.model_runner.spec_algorithm.is_dflash()
                and self.model_runner.is_draft_worker
                and forward_batch.input_embeds is not None
            ):
                self.buffers.input_embeds[: self.raw_num_token].copy_(
                    forward_batch.input_embeds
                )
            if (
                envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get()
                and forward_batch.mrope_positions is not None
            ):
                self.buffers.mrope_positions[:, : self.raw_num_token].copy_(
                    forward_batch.mrope_positions
                )

        graph_key = self._make_graph_key(self.bs)

        self._trace_mamba_graph_boundary(forward_batch, "graph_replay_before")

        if not (
            is_deepseek_dsa(self.model_runner.model_config.hf_config)
            or is_deepseek_v4(self.model_runner.model_config.hf_config)
        ):
            if forward_batch.forward_mode.is_target_verify():
                seq_lens_cpu = forward_batch.seq_lens.cpu() + self.captured_req_width
                seq_lens = seq_lens_cpu.tolist() + [0] * (self.bs - self.raw_bs)
            else:
                seq_lens = forward_batch.seq_lens.cpu().tolist() + [0] * (
                    self.bs - self.raw_bs
                )
            output = self.backend.replay_with_input_update(
                graph_key,
                seq_lens=seq_lens,
                attr_name=self._get_update_attr_name(),
                attr_type=self._get_update_attr_type(),
            )
        else:
            output = self.backend.replay(graph_key, forward_batch)

        self._trace_mamba_graph_boundary(forward_batch, "graph_replay_after")
        self._trace_k3_verify_rows(forward_batch)

        if isinstance(output, LogitsProcessorOutput):
            if self.is_dllm:
                next_token_logits = None
                full_logits = (
                    output.full_logits[: self.raw_num_token]
                    if output.full_logits is not None
                    else None
                )
            else:
                full_logits = None
                next_token_logits = (
                    output.next_token_logits[: self.raw_num_token]
                    if output.next_token_logits is not None
                    else None
                )
            return LogitsProcessorOutput(
                next_token_logits=next_token_logits,
                full_logits=full_logits,
                hidden_states=(
                    output.hidden_states[: self.raw_num_token]
                    if output.hidden_states is not None
                    else None
                ),
            )
        else:
            assert isinstance(output, PPProxyTensors)
            return PPProxyTensors({k: v[: self.bs] for k, v in output.tensors.items()})
