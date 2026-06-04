"""SANA-WM realtime engine — in-process backend for the Live Web UI (S3).

Loads the streaming TWO-STAGE pipeline in-process (no scheduler subprocess) and
bootstraps the incremental SanaWMRealtimeSession (encode prompt + first frame ->
reset). step(keys) takes a WASD/IJKL keypress, extends the camera trajectory
(pose-continuous), generates ONE stage-1 chunk via forward_long with the carried
KV cache, REFINES it with the chunked LTX-2 refiner (RefinerChunkRunner carrying a
sink/history KV cache), and decodes through the causal VAE. The DMD stage-1 is
coarse by design, so the refiner is required for sharp output.
"""
from __future__ import annotations
import os
import numpy as np
import torch

MODEL = "/data/yihao/sana-wm-streaming-model"
ASSET = "/data/yihao/sana-wm-streaming-tree/assets/sana_wm"
STRIDE = 8  # LTX-2 VAE temporal stride


class SanaWMRealtimeEngine:
    def __init__(self, model_path=MODEL, height=704, width=1280, seed=42, port="29699",
                 use_refiner=True):
        from sglang.multimodal_gen.runtime.server_args import ServerArgs, set_global_server_args
        from sglang.multimodal_gen.runtime.pipelines_core import build_pipeline
        from sglang.multimodal_gen.runtime.distributed import (
            maybe_init_distributed_environment_and_model_parallel,
        )
        self.use_refiner = use_refiner
        sa = ServerArgs.from_kwargs(
            model_path=model_path, pipeline_config={"streaming": True},
            pipeline_class_name="SanaWMTwoStagePipeline" if use_refiner else "SanaWMPipeline",
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
        self._refiner_stage = None
        if use_refiner:
            self._refiner_stage = next(
                (s for s in self.pipeline.stages
                 if s.__class__.__name__ == "SanaWMStreamingRefinerStage"), None)
            # refiner modules resident on GPU (no residency manager in-engine)
            for m in ("transformer_2", "connectors", "text_encoder_2"):
                mod = self.pipeline.get_module(m)
                if mod is not None and hasattr(mod, "to"):
                    mod.to(self.device)
        self.session = None
        self._segments = []
        self._T_lat = 1
        self._dtype = None

    def reset(self, prompt, image_path, intrinsics=None, init_keys="w"):
        from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
        from .sana_wm_base import (
            _first_tensor, _SANA_WM_CONDITION_IMAGE_PREPROCESS_KEY,
        )
        from .realtime import (
            SanaWMRealtimeSession,
        )
        self._intrinsics = intrinsics  # path / array / None (-> heuristic centered)
        req = Req(prompt=prompt, image_path=image_path, num_frames=49, num_inference_steps=4,
                  height=self.height, width=self.width, seed=self.seed, save_output=False)
        req.extra = {"action": f"{init_keys}-8"}
        if intrinsics is not None:
            req.extra["intrinsics"] = intrinsics
        full = list(self.pipeline.stages)
        self.pipeline._stages = self._prefix
        try:
            batch = self.pipeline.forward(req, self.sa)
        finally:
            self.pipeline._stages = full
        self._dtype = batch.latents.dtype
        # condition-image resize/crop metadata — needed to map source intrinsics
        # into the cropped pixel grid for the per-step camera rebuild.
        self._preprocess = (batch.extra or {}).get(_SANA_WM_CONDITION_IMAGE_PREPROCESS_KEY)
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
        self._conv = self.vae.reset_decoder_cache()
        self._decode_end = 0
        # build the chunked refiner runner (carries sink/history KV across steps)
        self._runner = None
        if self.use_refiner and self._refiner_stage is not None:
            self._build_refiner_runner(prompt, first_latent)

    def _build_refiner_runner(self, prompt, first_latent):
        from .refiner import (
            STAGE_2_DISTILLED_SIGMA_VALUES, _unwrap_diffusers_ltx2_refiner,
        )
        from .streaming_refiner import (
            RefinerChunkRunner, _RefinerCore,
        )
        rs = self._refiner_stage
        embeds, mask = rs._encode_prompt(prompt, self.device)
        unwrapped = _unwrap_diffusers_ltx2_refiner(rs.transformer)
        core = _RefinerCore(unwrapped, self.device, rs.dtype)
        sigmas = torch.tensor(STAGE_2_DISTILLED_SIGMA_VALUES, dtype=torch.float32, device=self.device)
        self._sink = int(rs.sink_size)
        self._runner = RefinerChunkRunner(
            core, prompt_embeds=embeds, prompt_attention_mask=mask, fps=16.0, sigmas=sigmas,
            source_sink_frames=self._sink, block_size=int(rs.block_size),
            kv_max_frames=int(rs.kv_max_frames), seed=int(rs.seed), spatial_shape=(self._H, self._W))
        # refined buffer: [0:sink] = stage-1 sink frame(s) (unrefined, as the official does)
        self._refined = first_latent.clone()

    def _build_camera(self, total_T_lat):
        from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
        from .sana_wm_base import (
            _SANA_WM_CONDITION_IMAGE_PREPROCESS_KEY,
        )
        num_pixel = (total_T_lat - 1) * STRIDE + 1
        req = Req(prompt=self._prompt, num_frames=num_pixel, num_inference_steps=4,
                  height=self.height, width=self.width, seed=self.seed)
        req.extra = {"action": ",".join(self._segments)}
        if self._intrinsics is not None:
            req.extra["intrinsics"] = self._intrinsics
            if self._preprocess is not None:
                req.extra[_SANA_WM_CONDITION_IMAGE_PREPROCESS_KEY] = self._preprocess
        cc, cp, _ = self._before._build_camera_conditioning(
            req, batch_size=1, num_frames=num_pixel,
            latent_shape=(1, 128, total_T_lat, self._H, self._W),
            device=self.device, dtype=self._dtype)
        return cc, cp

    @torch.no_grad()
    def step(self, keys):
        from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
        if self.session is None:
            raise RuntimeError("call reset() first")
        keys = "".join(c for c in keys.lower() if c in "wasdijkl") or "w"
        self._segments.append(f"{keys}-{self.nfpb * STRIDE}")
        total_T_lat = self._T_lat + self.nfpb
        cc, cp = self._build_camera(total_T_lat)
        with set_forward_context(current_timestep=0, attn_metadata=None, forward_batch=None):
            self.session.step(camera_conditions=cc, chunk_plucker=cp,
                              n_frames=self.nfpb, decode=False)
        new_end = self.session.latents.shape[2]

        if self._runner is not None:
            # refine the freshly generated stage-1 chunk, carrying sink/history
            start_f = self._refined.shape[2]            # = sink + already-refined frames
            clean = self.session.latents[:, :, start_f:new_end].contiguous()
            sink_seed = self.session.latents[:, :, :self._sink] if start_f == self._sink else None
            refined = self._runner.refine_block(
                block_idx=start_f, clean_block=clean,
                block_start=start_f, block_end=new_end, sink_seed_frames=sink_seed)
            self._refined = torch.cat([self._refined, refined.to(self._refined.dtype)], dim=2)
            src = self._refined
        else:
            src = self.session.latents

        seg = src[:, :, self._decode_end:new_end].to(self._vae_dtype)
        z = self._decstage.scale_and_shift(seg, self.sa)
        px = self.vae.decode_chunk(z, self._conv)
        self._decode_end = new_end
        self._T_lat = new_end
        return self._to_rgb(px)

    @staticmethod
    def _to_rgb(px):
        px = (px / 2 + 0.5).clamp(0, 1).float().cpu()
        return (px[0].permute(1, 2, 3, 0).numpy() * 255).round().astype(np.uint8)


def main():
    import imageio.v3 as iio
    eng = SanaWMRealtimeEngine()
    prompt = open(f"{ASSET}/demo_0.txt").read().strip()
    eng.reset(prompt, f"{ASSET}/demo_0.png", intrinsics=f"{ASSET}/demo_0_intrinsics.npy")
    frames = []
    print("RESET ok (refiner=%s, intrinsics=yes)" % bool(eng._runner), flush=True)
    for keys in ["w", "wl", "l"]:
        f = eng.step(keys)
        frames.append(f)
        print(f"STEP {keys!r}: {f.shape} latent_T={eng.session.latents.shape[2]}", flush=True)
    vid = np.concatenate(frames, axis=0)
    out = "/data/yihao/sana-wm-streaming-tree/outputs/sana_wm_realtime_refined.mp4"
    iio.imwrite(out, vid, fps=16, codec="libx264")
    print("SAVED", out, vid.shape, flush=True)


if __name__ == "__main__":
    main()
