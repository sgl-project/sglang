# SPDX-License-Identifier: Apache-2.0
"""Manual live smoke for BAGEL Qwen2-MoT native SRT U-forward.

Usage:
CUDA_VISIBLE_DEVICES=0 \
SGLANG_TEST_BAGEL_QWEN2_MOT_MODEL=/data/models/BAGEL-7B-MoT \
python3 test/registered/scheduler/test_bagel_qwen2_mot_native_live.py
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(
    est_time=420,
    suite="stage-b-test-1-gpu-large",
    disabled=(
        "Manual BAGEL Qwen2-MoT native live smoke; requires "
        "SGLANG_TEST_BAGEL_QWEN2_MOT_MODEL"
    ),
)

_MODEL_ENV = "SGLANG_TEST_BAGEL_QWEN2_MOT_MODEL"


class _NoopSender:
    def send_output(self, *args, **kwargs):
        del args, kwargs


class _RecordingUGModelRunner:
    def __init__(self, inner):
        self.inner = inner
        self.srt_request_views = []

    def observe_srt_u_forward(self, *, record, request, messages):
        del record, messages
        self.srt_request_views.append(request)

    def prefill_interleaved(self, *, record, messages):
        return self.inner.prefill_interleaved(record=record, messages=messages)

    def decode_next_segment(self, *, record):
        return self.inner.decode_next_segment(record=record)

    def predict_velocity_from_session(self, *, request, record):
        return self.inner.predict_velocity_from_session(request=request, record=record)

    def prepare_latents_from_session(self, *, request, record):
        return self.inner.prepare_latents_from_session(request=request, record=record)

    def append_generated_image(self, *, record, image):
        return self.inner.append_generated_image(record=record, image=image)

    def decode_latents_to_image(self, *, request, record):
        return self.inner.decode_latents_to_image(request=request, record=record)

    def close_session(self, *, session_id):
        self.inner.close_session(session_id=session_id)


def _replace_sender_with_noop(scheduler, name: str) -> None:
    sender = getattr(scheduler, name, None)
    socket = getattr(sender, "socket", None)
    if socket is not None:
        socket.close(linger=0)
    setattr(scheduler, name, _NoopSender())


def _write_language_model_view(checkpoint_dir: Path, output_dir: Path) -> Path:
    config_path = checkpoint_dir / "llm_config.json"
    weight_path = checkpoint_dir / "ema.safetensors"
    if not config_path.exists() or not weight_path.exists():
        raise FileNotFoundError(
            "BAGEL Qwen2-MoT live smoke requires llm_config.json and "
            f"ema.safetensors under {checkpoint_dir}"
        )

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    config.update(
        {
            "architectures": ["BAGELQwen2MoTForCausalLM"],
            "layer_module": "Qwen2MoTDecoderLayer",
            "qk_norm": True,
            "tie_word_embeddings": False,
        }
    )
    (output_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    os.symlink(weight_path, output_dir / "model.safetensors")
    return output_dir


@unittest.skipUnless(os.getenv(_MODEL_ENV), f"Set {_MODEL_ENV} for live smoke")
class TestBAGELQwen2MoTNativeLive(CustomTestCase):
    def test_real_bagel_language_weights_run_srt_u_forward(self):
        import torch
        import torch.distributed as dist

        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for the BAGEL native live smoke")

        from sglang.srt.managers.scheduler import Scheduler
        from sglang.srt.server_args import (
            PortArgs,
            ServerArgs,
            set_global_server_args_for_scheduler,
        )
        from sglang.srt.ug.runtime import (
            FakeUGModelRunner,
            UGInterleavedMessage,
            UGSessionRuntime,
        )
        from sglang.srt.ug.srt_executor import UGSRTSchedulerExecutor
        from sglang.srt.ug.bagel_checkpoint import (
            load_bagel_checkpoint_keys,
            summarize_bagel_checkpoint_keys,
        )
        from sglang.srt.models.bagel_qwen2_mot import (
            _iter_bagel_language_model_weights,
        )

        checkpoint_dir = Path(os.environ[_MODEL_ENV])
        keys = load_bagel_checkpoint_keys(checkpoint_dir)
        summary = summarize_bagel_checkpoint_keys(keys)
        language_keys = list(
            _iter_bagel_language_model_weights((key, None) for key in keys)
        )
        self.assertEqual(
            len(language_keys),
            summary.counts["qwen2_shared"] + summary.counts["mot_gen_branch"],
        )
        self.assertGreater(summary.counts["mot_gen_branch"], 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = _write_language_model_view(checkpoint_dir, Path(tmpdir))
            server_args = ServerArgs(
                model_path=str(model_path),
                tokenizer_path=str(checkpoint_dir),
                trust_remote_code=True,
                dtype="bfloat16",
                tp_size=1,
                pp_size=1,
                dp_size=1,
                disable_cuda_graph=True,
                disable_piecewise_cuda_graph=True,
                disable_overlap_schedule=True,
                skip_server_warmup=True,
                mem_fraction_static=float(
                    os.getenv("SGLANG_TEST_BAGEL_QWEN2_MOT_MEM_FRACTION", "0.35")
                ),
                chunked_prefill_size=int(
                    os.getenv("SGLANG_TEST_BAGEL_QWEN2_MOT_CHUNKED_PREFILL", "256")
                ),
                log_level="error",
            )
            server_args.check_server_args()
            set_global_server_args_for_scheduler(server_args)

            scheduler = Scheduler(
                server_args,
                PortArgs.init_new(server_args),
                gpu_id=int(os.getenv("SGLANG_TEST_BAGEL_QWEN2_MOT_GPU_ID", "0")),
                tp_rank=0,
                moe_ep_rank=0,
                pp_rank=0,
                attn_cp_rank=0,
                moe_dp_rank=0,
                dp_rank=None,
            )
            _replace_sender_with_noop(scheduler, "send_to_tokenizer")
            _replace_sender_with_noop(scheduler, "send_to_detokenizer")

            try:
                executor = UGSRTSchedulerExecutor(scheduler, max_sync_steps=16)
                model_runner = _RecordingUGModelRunner(FakeUGModelRunner())
                runtime = UGSessionRuntime(
                    model_runner=model_runner,
                    session_controller=scheduler.session_controller,
                    srt_request_executor=executor,
                    tokenizer=scheduler.tokenizer,
                    vocab_size=scheduler.model_config.vocab_size,
                    srt_u_decode_max_new_tokens=1,
                )

                handle = runtime.prefill_interleaved(
                    [UGInterleavedMessage(type="text", content="hello from bagel")],
                    session_id="bagel-native-srt-u-forward",
                )
                marker = runtime.decode_next_segment(handle)
                counters = runtime.get_debug_counters(handle)

                self.assertEqual(marker.type, "image_marker")
                self.assertEqual(counters["prefill_count"], 1)
                self.assertEqual(counters["decode_count"], 1)
                self.assertEqual(counters["srt_request_count"], 2)
                self.assertEqual(counters["srt_executed_request_count"], 2)
                self.assertEqual(counters["srt_u_decode_request_count"], 1)
                self.assertEqual(len(counters["srt_last_u_decode_output_ids"]), 1)
                self.assertEqual(
                    counters["srt_model_runner_forward_request_ids"],
                    ["bagel-native-srt-u-forward:u1"],
                )
                self.assertEqual(executor.sync_step_count, 2)
                self.assertTrue(scheduler.is_fully_idle())

                token_bindings = [
                    request.metadata.get("srt_kv_token_binding")
                    for request in model_runner.srt_request_views
                ]
                self.assertTrue(token_bindings)
                self.assertTrue(all(binding is not None for binding in token_bindings))
                self.assertTrue(
                    all(binding.token_count > 0 for binding in token_bindings)
                )

                runtime.close_session(handle)
                self.assertNotIn(
                    handle.session_id, scheduler.session_controller.sessions
                )
            finally:
                if dist.is_available() and dist.is_initialized():
                    dist.destroy_process_group()
                torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
