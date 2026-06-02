"""SANA-WM realtime engine — in-process backend for the Live Web UI (S3).

Loads the streaming pipeline in-process (no scheduler subprocess), bootstraps the
incremental SanaWMRealtimeSession (encode prompt + first frame -> reset), and
exposes step(keys) that takes a WASD/IJKL keypress, extends the camera trajectory
(pose-continuous), generates ONE chunk via forward_long with the carried KV cache,
decodes it through the causal VAE, and returns RGB frames. Both the websocket
handler and the scripted validator below call engine.step(keys).
"""
from __future__ import annotations
import os
import numpy as np
import torch

MODEL = "/data/yihao/sana-wm-streaming-model"
ASSET = "/data/yihao/sana-wm-streaming-tree/assets/sana_wm"
STRIDE = 8  # LTX-2 VAE temporal stride


class SanaWMRealtimeEngine:
    def __init__(self, model_path=MODEL, height=704, width=1280, seed=42, port="29699"):
        from sglang.multimodal_gen.runtime.server_args import ServerArgs, set_global_server_args
        from sglang.multimodal_gen.runtime.pipelines_core import build_pipeline
        from sglang.multimodal_gen.runtime.distributed import (
            maybe_init_distributed_environment_and_model_parallel,
        )
        sa = ServerArgs.from_kwargs(
            model_path=model_path, pipeline_config={"streaming": True},
            pipeline_class_name="SanaWMPipeline",
            dit_cpu_offload=False, dit_layerwise_offload=False)
        for k, v in dict(MASTER_ADDR="localhost", MASTER_PORT=port, LOCAL_RANK="0",
                         RANK="0", WORLD_SIZE="1").items():
            os.environ.setdefault(k, v)
        set_global_server_args(sa)
        torch.cuda.set_device(0)
        maybe_init_distributed_environment_and_model_parallel(
            tp_size=1, cfg_degree=1, ulysses_degree=1, ring_degree=1, sp_size=1, dp_size=1,
            distributed_init_method=f"tcp://127.0.0.1:{port}")
        self.sa = sa
        self.pcfg = sa.pipeline_config
        self.pipeline = build_pipeline(sa)
        self.device = torch.device("cuda:0")
        self.transformer = self.pipeline.get_module("transformer").to(self.device)
        self.vae = self.pipeline.get_module("vae")
        self.height, self.width, self.seed = height, width, seed
        self.nfpb = int(getattr(self.pcfg, "num_frame_per_block", 3))
        # locate the before-denoising stage (camera builder) + prefix stages
        self._before = next(s for s in self.pipeline.stages
                            if s.__class__.__name__ == "SanaWMBeforeDenoisingStage")
        self._decstage = next(s for s in self.pipeline.stages
                             if hasattr(s, "scale_and_shift") and hasattr(s, "decode"))
        self._vae_dtype = self.vae.dtype if hasattr(self.vae, "dtype") else torch.bfloat16
        self._prefix = []
        for s in self.pipeline.stages:
            self._prefix.append(s)
            if s.__class__.__name__ == "SanaWMBeforeDenoisingStage":
                break
        self.session = None
        self._segments = []
        self._T_lat = 1   # condition frame
        self._dtype = None

    def reset(self, prompt, image_path, init_keys="w"):
        from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm import _first_tensor
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm_realtime import (
            SanaWMRealtimeSession,
        )
        req = Req(prompt=prompt, image_path=image_path, num_frames=49, num_inference_steps=4,
                  height=self.height, width=self.width, seed=self.seed, save_output=False)
        req.extra = {"action": f"{init_keys}-8"}  # placeholder; per-step camera rebuilt
        full = list(self.pipeline.stages)
        self.pipeline._stages = self._prefix
        try:
            batch = self.pipeline.forward(req, self.sa)
        finally:
            self.pipeline._stages = full
        self._dtype = batch.latents.dtype
        first_latent = batch.latents[:, :, :1].clone()
        pos = _first_tensor(self.pcfg.get_pos_prompt_embeds(batch)).clone()
        pmask = _first_tensor(batch.prompt_attention_mask)
        pmask = pmask.clone() if pmask is not None else None
        self._prompt, self._image = prompt, image_path
        self.session = SanaWMRealtimeSession(
            self.transformer,
            denoising_step_list=tuple(getattr(self.pcfg, "denoising_step_list", (1000, 960, 889, 727, 0))),
            num_frame_per_block=self.nfpb,
            num_cached_blocks=int(getattr(self.pcfg, "num_cached_blocks", 2)),
            sink_token=bool(getattr(self.pcfg, "sink_token", True)),
            cfg_scale=float(getattr(self.pcfg, "streaming_cfg_scale", 1.0)),
            device=self.device, dtype=self._dtype, vae=self.vae)
        self.session.reset(first_latent, pos, pos_mask=pmask)
        self._segments = []
        self._T_lat = 1
        self._H, self._W = first_latent.shape[3], first_latent.shape[4]
        # Engine-owned causal-VAE decode (scale_and_shift + decode_chunk), mirroring
        # SanaWMStreamingDecodingStage. decode_end tracks the last decoded latent frame.
        self._conv = self.vae.reset_decoder_cache()
        self._decode_end = 0
        return None

    def _build_camera(self, total_T_lat):
        from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm import _to_device_dtype
        num_pixel = (total_T_lat - 1) * STRIDE + 1
        action = ",".join(self._segments)
        req = Req(prompt=self._prompt, num_frames=num_pixel, num_inference_steps=4,
                  height=self.height, width=self.width, seed=self.seed)
        req.extra = {"action": action}
        cc, cp, src = self._before._build_camera_conditioning(
            req, batch_size=1, num_frames=num_pixel,
            latent_shape=(1, 128, total_T_lat, self._H, self._W),
            device=self.device, dtype=self._dtype)
        return cc, cp

    @torch.no_grad()
    def step(self, keys):
        if self.session is None:
            raise RuntimeError("call reset() first")
        keys = "".join(c for c in keys.lower() if c in "wasdijkl") or "w"
        self._segments.append(f"{keys}-{self.nfpb * STRIDE}")
        from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
        total_T_lat = self._T_lat + self.nfpb
        cc, cp = self._build_camera(total_T_lat)
        with set_forward_context(current_timestep=0, attn_metadata=None, forward_batch=None):
            self.session.step(camera_conditions=cc, chunk_plucker=cp,
                              n_frames=self.nfpb, decode=False)
        new_end = self.session.latents.shape[2]
        seg = self.session.latents[:, :, self._decode_end:new_end].to(self._vae_dtype)
        z = self._decstage.scale_and_shift(seg, self.sa)
        px = self.vae.decode_chunk(z, self._conv)
        self._decode_end = new_end
        self._T_lat = new_end
        return self._to_rgb(px)

    @staticmethod
    def _to_rgb(px):
        px = (px / 2 + 0.5).clamp(0, 1).float().cpu()  # (B,C,T,H,W)
        v = (px[0].permute(1, 2, 3, 0).numpy() * 255).round().astype(np.uint8)  # (T,H,W,C)
        return v


def main():
    import imageio.v3 as iio
    eng = SanaWMRealtimeEngine()
    prompt = open(f"{ASSET}/demo_0.txt").read().strip()
    eng.reset(prompt, f"{ASSET}/demo_0.png")
    frames = []
    print("RESET ok", flush=True)
    for keys in ["w", "wl", "l"]:
        f = eng.step(keys)
        frames.append(f)
        print(f"STEP {keys!r}: {f.shape} latent_T={eng.session.latents.shape[2]}", flush=True)
    vid = np.concatenate(frames, axis=0)
    out = "/data/yihao/sana-wm-streaming-tree/outputs/sana_wm_realtime_scripted.mp4"
    iio.imwrite(out, vid, fps=16, codec="libx264")
    print("SAVED", out, vid.shape, flush=True)


if __name__ == "__main__":
    main()
