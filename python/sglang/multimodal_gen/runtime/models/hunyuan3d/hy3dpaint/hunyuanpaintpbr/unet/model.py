# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import os

# import ipdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm
from torchvision.transforms import v2
from torchvision.utils import make_grid, save_image
from einops import rearrange

from diffusers import (
    DiffusionPipeline,
    EulerAncestralDiscreteScheduler,
    DDPMScheduler,
    UNet2DConditionModel,
    ControlNetModel,
)

from .modules import Dino_v2, UNet2p5DConditionModel
import math


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class HunyuanPaint(pl.LightningModule):
    def __init__(
        self,
        stable_diffusion_config,
        control_net_config=None,
        num_view=6,
        view_size=320,
        drop_cond_prob=0.1,
        with_normal_map=None,
        with_position_map=None,
        pbr_settings=["albedo", "mr"],
        **kwargs,
    ):
        """Initializes the HunyuanPaint Lightning Module.

        Args:
            stable_diffusion_config: Configuration for loading the Stable Diffusion pipeline
            control_net_config: Configuration for ControlNet (optional)
            num_view: Number of views to process
            view_size: Size of input views (height/width)
            drop_cond_prob: Probability of dropping conditioning input during training
            with_normal_map: Flag indicating whether normal maps are used
            with_position_map: Flag indicating whether position maps are used
            pbr_settings: List of PBR materials to generate (e.g., albedo, metallic-roughness)
            **kwargs: Additional keyword arguments
        """
        super(HunyuanPaint, self).__init__()

        self.num_view = num_view
        self.view_size = view_size
        self.drop_cond_prob = drop_cond_prob
        self.pbr_settings = pbr_settings

        # init modules
        pipeline = DiffusionPipeline.from_pretrained(**stable_diffusion_config)
        pipeline.set_pbr_settings(self.pbr_settings)
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipeline.scheduler.config, timestep_spacing="trailing"
        )

        self.with_normal_map = with_normal_map
        self.with_position_map = with_position_map

        self.pipeline = pipeline

        self.pipeline.vae.use_slicing = True

        train_sched = DDPMScheduler.from_config(self.pipeline.scheduler.config)

        if isinstance(self.pipeline.unet, UNet2DConditionModel):
            self.pipeline.unet = UNet2p5DConditionModel(
                self.pipeline.unet, train_sched, self.pipeline.scheduler, self.pbr_settings
            )
        self.train_scheduler = train_sched  # use ddpm scheduler during training

        self.register_schedule()

        pipeline.set_learned_parameters()

        if control_net_config is not None:
            pipeline.unet = pipeline.unet.bfloat16().requires_grad_(control_net_config.train_unet)
            self.pipeline.add_controlnet(
                ControlNetModel.from_pretrained(control_net_config.pretrained_model_name_or_path),
                conditioning_scale=0.75,
            )

        self.unet = pipeline.unet

        self.pipeline.set_progress_bar_config(disable=True)
        self.pipeline.vae = self.pipeline.vae.bfloat16()
        self.pipeline.text_encoder = self.pipeline.text_encoder.bfloat16()

        if self.unet.use_dino:
            self.dino_v2 = Dino_v2("facebook/dinov2-giant")
            self.dino_v2 = self.dino_v2.bfloat16()

        self.validation_step_outputs = []

    def register_schedule(self):

        self.num_timesteps = self.train_scheduler.config.num_train_timesteps

        betas = self.train_scheduler.betas.detach().cpu()

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float64), alphas_cumprod[:-1]], 0)

        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas_cumprod", alphas_cumprod.float())
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev.float())

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod).float())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod).float())

        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod).float())
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1).float())

    def on_fit_start(self):
        device = torch.device(f"cuda:{self.local_rank}")
        self.pipeline.to(device)
        if self.global_rank == 0:
            os.makedirs(os.path.join(self.logdir, "images_val"), exist_ok=True)

    def prepare_batch_data(self, batch):
        """Preprocesses a batch of input data for training/inference.

        Args:
            batch: Raw input batch dictionary

        Returns:
            tuple: Contains:
                - cond_imgs: Primary conditioning images (B, 1, C, H, W)
                - cond_imgs_another: Secondary conditioning images (B, 1, C, H, W)
                - target_imgs: Dictionary of target PBR images resized and clamped
                - images_normal: Preprocessed normal maps (if available)
                - images_position: Preprocessed position maps (if available)
        """

        images_cond = batch["images_cond"].to(self.device)  # (B, M, C, H, W), where M is the number of reference images
        cond_imgs, cond_imgs_another = images_cond[:, 0:1, ...], images_cond[:, 1:2, ...]

        cond_size = self.view_size
        cond_imgs = v2.functional.resize(cond_imgs, cond_size, interpolation=3, antialias=True).clamp(0, 1)
        cond_imgs_another = v2.functional.resize(cond_imgs_another, cond_size, interpolation=3, antialias=True).clamp(
            0, 1
        )

        target_imgs = {}
        for pbr_token in self.pbr_settings:
            target_imgs[pbr_token] = batch[f"images_{pbr_token}"].to(self.device)
            target_imgs[pbr_token] = v2.functional.resize(
                target_imgs[pbr_token], self.view_size, interpolation=3, antialias=True
            ).clamp(0, 1)

        images_normal = None
        if "images_normal" in batch:
            images_normal = batch["images_normal"]  # (B, N, C, H, W)
            images_normal = v2.functional.resize(images_normal, self.view_size, interpolation=3, antialias=True).clamp(
                0, 1
            )
            images_normal = [images_normal]

        images_position = None
        if "images_position" in batch:
            images_position = batch["images_position"]  # (B, N, C, H, W)
            images_position = v2.functional.resize(
                images_position, self.view_size, interpolation=3, antialias=True
            ).clamp(0, 1)
            images_position = [images_position]

        return cond_imgs, cond_imgs_another, target_imgs, images_normal, images_position

    @torch.no_grad()
    def forward_text_encoder(self, prompts):
        device = next(self.pipeline.vae.parameters()).device
        text_embeds = self.pipeline.encode_prompt(prompts, device, 1, False)[0]
        return text_embeds

    @torch.no_grad()
    def encode_images(self, images):
        """Encodes input images into latent representations using the VAE.

        Handles both standard input (B, N, C, H, W) and PBR input (B, N_pbrs, N, C, H, W)
        Maintains original batch structure in output latents.

        Args:
            images: Input images tensor

        Returns:
            torch.Tensor: Latent representations with original batch dimensions preserved
        """

        B = images.shape[0]
        image_ndims = images.ndim
        if image_ndims != 5:
            N_pbrs, N = images.shape[1:3]
        images = (
            rearrange(images, "b n c h w -> (b n) c h w")
            if image_ndims == 5
            else rearrange(images, "b n_pbrs n c h w -> (b n_pbrs n) c h w")
        )
        dtype = next(self.pipeline.vae.parameters()).dtype

        images = (images - 0.5) * 2.0
        posterior = self.pipeline.vae.encode(images.to(dtype)).latent_dist
        latents = posterior.sample() * self.pipeline.vae.config.scaling_factor

        latents = (
            rearrange(latents, "(b n) c h w -> b n c h w", b=B)
            if image_ndims == 5
            else rearrange(latents, "(b n_pbrs n) c h w -> b n_pbrs n c h w", b=B, n_pbrs=N_pbrs)
        )

        return latents

    def forward_unet(self, latents, t, **cached_condition):
        """Runs the UNet model to predict noise/latent residuals.

        Args:
            latents: Noisy latent representations (B, C, H, W)
            t: Timestep tensor (B,)
            **cached_condition: Dictionary of conditioning inputs (text embeds, reference images, etc)

        Returns:
            torch.Tensor: UNet output (predicted noise or velocity)
        """

        dtype = next(self.unet.parameters()).dtype
        latents = latents.to(dtype)
        shading_embeds = cached_condition["shading_embeds"]
        pred_noise = self.pipeline.unet(latents, t, encoder_hidden_states=shading_embeds, **cached_condition)
        return pred_noise[0]

    def predict_start_from_z_and_v(self, x_t, t, v):
        """
        Predicts clean image (x0) from noisy latents (x_t) and
        velocity prediction (v) using the v-prediction formula.

        Args:
            x_t: Noisy latents at timestep t
            t: Current timestep
            v: Predicted velocity (v) from UNet

        Returns:
            torch.Tensor: Predicted clean image (x0)
        """

        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def get_v(self, x, noise, t):
        """Computes the target velocity (v) for v-prediction training.

        Args:
            x: Clean latents (x0)
            noise: Added noise
            t: Current timestep

        Returns:
            torch.Tensor: Target velocity
        """

        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) * noise
            - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * x
        )

    def training_step(self, batch, batch_idx):
        """Performs a single training step with both conditioning paths.

        Implements:
        1. Dual-conditioning path training (main ref + secondary ref)
        2. Velocity-prediction with consistency loss
        3. Conditional dropout for robust learning
        4. PBR-specific losses (albedo/metallic-roughness)

        Args:
            batch: Input batch from dataloader
            batch_idx: Index of current batch

        Returns:
            torch.Tensor: Combined loss value
        """

        cond_imgs, cond_imgs_another, target_imgs, normal_imgs, position_imgs = self.prepare_batch_data(batch)

        B, N_ref = cond_imgs.shape[:2]
        _, N_gen, _, H, W = target_imgs["albedo"].shape
        N_pbrs = len(self.pbr_settings)
        t = torch.randint(0, self.num_timesteps, size=(B,)).long().to(self.device)
        t = t.unsqueeze(-1).repeat(1, N_pbrs, N_gen)
        t = rearrange(t, "b n_pbrs n -> (b n_pbrs n)")

        all_target_pbrs = []
        for pbr_token in self.pbr_settings:
            all_target_pbrs.append(target_imgs[pbr_token])
        all_target_pbrs = torch.stack(all_target_pbrs, dim=0).transpose(1, 0)
        gen_latents = self.encode_images(all_target_pbrs)  #! B, N_pbrs N C H W
        ref_latents = self.encode_images(cond_imgs)  #! B, M, C, H, W
        ref_latents_another = self.encode_images(cond_imgs_another)  #! B, M, C, H, W

        all_shading_tokens = []
        for token in self.pbr_settings:
            if token in ["albedo", "mr"]:
                all_shading_tokens.append(
                    getattr(self.unet, f"learned_text_clip_{token}").unsqueeze(dim=0).repeat(B, 1, 1)
                )
        shading_embeds = torch.stack(all_shading_tokens, dim=1)

        if self.unet.use_dino:
            dino_hidden_states = self.dino_v2(cond_imgs[:, :1, ...])
            dino_hidden_states_another = self.dino_v2(cond_imgs_another[:, :1, ...])

        gen_latents = rearrange(gen_latents, "b n_pbrs n c h w -> (b n_pbrs n) c h w")
        noise = torch.randn_like(gen_latents).to(self.device)
        latents_noisy = self.train_scheduler.add_noise(gen_latents, noise, t).to(self.device)
        latents_noisy = rearrange(latents_noisy, "(b n_pbrs n) c h w -> b n_pbrs n c h w", b=B, n_pbrs=N_pbrs)

        cached_condition = {}

        if normal_imgs is not None:
            normal_embeds = self.encode_images(normal_imgs[0])
            cached_condition["embeds_normal"] = normal_embeds  #! B, N, C, H, W

        if position_imgs is not None:
            position_embeds = self.encode_images(position_imgs[0])
            cached_condition["embeds_position"] = position_embeds  #! B, N, C, H, W
            cached_condition["position_maps"] = position_imgs[0]  #! B, N, C, H, W

        for b in range(B):
            prob = np.random.rand()
            if prob < self.drop_cond_prob:
                if "normal_imgs" in cached_condition:
                    cached_condition["embeds_normal"][b, ...] = torch.zeros_like(
                        cached_condition["embeds_normal"][b, ...]
                    )
                if "position_imgs" in cached_condition:
                    cached_condition["embeds_position"][b, ...] = torch.zeros_like(
                        cached_condition["embeds_position"][b, ...]
                    )

            prob = np.random.rand()
            if prob < self.drop_cond_prob:
                if "position_maps" in cached_condition:
                    cached_condition["position_maps"][b, ...] = torch.zeros_like(
                        cached_condition["position_maps"][b, ...]
                    )

            prob = np.random.rand()
            if prob < self.drop_cond_prob:
                dino_hidden_states[b, ...] = torch.zeros_like(dino_hidden_states[b, ...])
            prob = np.random.rand()
            if prob < self.drop_cond_prob:
                dino_hidden_states_another[b, ...] = torch.zeros_like(dino_hidden_states_another[b, ...])

        # MVA & Ref Attention
        prob = np.random.rand()
        cached_condition["mva_scale"] = 1.0
        cached_condition["ref_scale"] = 1.0
        if prob < self.drop_cond_prob:
            cached_condition["mva_scale"] = 0.0
            cached_condition["ref_scale"] = 0.0
        elif prob > 1.0 - self.drop_cond_prob:
            prob = np.random.rand()
            if prob < 0.5:
                cached_condition["mva_scale"] = 0.0
            else:
                cached_condition["ref_scale"] = 0.0
        else:
            pass

        if self.train_scheduler.config.prediction_type == "v_prediction":

            cached_condition["shading_embeds"] = shading_embeds
            cached_condition["ref_latents"] = ref_latents
            cached_condition["dino_hidden_states"] = dino_hidden_states
            v_pred = self.forward_unet(latents_noisy, t, **cached_condition)
            v_pred_albedo, v_pred_mr = torch.split(
                rearrange(
                    v_pred, "(b n_pbr n) c h w -> b n_pbr n c h w", n_pbr=len(self.pbr_settings), n=self.num_view
                ),
                1,
                dim=1,
            )
            v_target = self.get_v(gen_latents, noise, t)
            v_target_albedo, v_target_mr = torch.split(
                rearrange(
                    v_target, "(b n_pbr n) c h w -> b n_pbr n c h w", n_pbr=len(self.pbr_settings), n=self.num_view
                ),
                1,
                dim=1,
            )

            albedo_loss_1, _ = self.compute_loss(v_pred_albedo, v_target_albedo)
            mr_loss_1, _ = self.compute_loss(v_pred_mr, v_target_mr)

            cached_condition["ref_latents"] = ref_latents_another
            cached_condition["dino_hidden_states"] = dino_hidden_states_another
            v_pred_another = self.forward_unet(latents_noisy, t, **cached_condition)
            v_pred_another_albedo, v_pred_another_mr = torch.split(
                rearrange(
                    v_pred_another,
                    "(b n_pbr n) c h w -> b n_pbr n c h w",
                    n_pbr=len(self.pbr_settings),
                    n=self.num_view,
                ),
                1,
                dim=1,
            )

            albedo_loss_2, _ = self.compute_loss(v_pred_another_albedo, v_target_albedo)
            mr_loss_2, _ = self.compute_loss(v_pred_another_mr, v_target_mr)

            consistency_loss, _ = self.compute_loss(v_pred_another, v_pred)

            albedo_loss = (albedo_loss_1 + albedo_loss_2) * 0.5
            mr_loss = (mr_loss_1 + mr_loss_2) * 0.5

            log_loss_dict = {}
            log_loss_dict.update({f"train/albedo_loss": albedo_loss})
            log_loss_dict.update({f"train/mr_loss": mr_loss})
            log_loss_dict.update({f"train/cons_loss": consistency_loss})

            loss_dict = log_loss_dict

        elif self.train_scheduler.config.prediction_type == "epsilon":
            e_pred = self.forward_unet(latents_noisy, t, **cached_condition)
            loss, loss_dict = self.compute_loss(e_pred, noise)
        else:
            raise f"No {self.train_scheduler.config.prediction_type}"

        # logging
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("global_step", self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        lr = self.optimizers().param_groups[0]["lr"]
        self.log("lr_abs", lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return 0.85 * (albedo_loss + mr_loss) + 0.15 * consistency_loss

    def compute_loss(self, noise_pred, noise_gt):
        loss = F.mse_loss(noise_pred, noise_gt)
        prefix = "train"
        loss_dict = {}
        loss_dict.update({f"{prefix}/loss": loss})
        return loss, loss_dict

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        """Performs validation on a single batch.

        Generates predicted images using:
        1. Reference conditioning images
        2. Optional normal/position maps
        3. Frozen DINO features (if enabled)
        4. Text prompt conditioning

        Compares predictions against ground truth targets and prepares visualization.
        Stores results for epoch-level aggregation.

        Args:
            batch: Input batch from validation dataloader
            batch_idx: Index of current batch
        """
        # [Validation image generation and comparison logic...]
        # Key steps:
        # 1. Preprocess conditioning images to PIL format
        # 2. Set up conditioning inputs (normal maps, position maps, DINO features)
        # 3. Run pipeline inference with fixed prompt ("high quality")
        # 4. Decode latent outputs to image space
        # 5. Arrange predictions and ground truths for visualization

        cond_imgs_tensor, _, target_imgs, normal_imgs, position_imgs = self.prepare_batch_data(batch)
        resolution = self.view_size
        image_pils = []
        for i in range(cond_imgs_tensor.shape[0]):
            image_pils.append([])
            for j in range(cond_imgs_tensor.shape[1]):
                image_pils[-1].append(v2.functional.to_pil_image(cond_imgs_tensor[i, j, ...]))

        outputs, gts = [], []
        for idx in range(len(image_pils)):
            cond_imgs = image_pils[idx]

            cached_condition = dict(num_in_batch=self.num_view, N_pbrs=len(self.pbr_settings))
            if normal_imgs is not None:
                cached_condition["images_normal"] = normal_imgs[0][idx, ...].unsqueeze(0)
            if position_imgs is not None:
                cached_condition["images_position"] = position_imgs[0][idx, ...].unsqueeze(0)
            if self.pipeline.unet.use_dino:
                dino_hidden_states = self.dino_v2([cond_imgs][0])
                cached_condition["dino_hidden_states"] = dino_hidden_states

            latent = self.pipeline(
                cond_imgs,
                prompt="high quality",
                num_inference_steps=30,
                output_type="latent",
                height=resolution,
                width=resolution,
                **cached_condition,
            ).images

            image = self.pipeline.vae.decode(latent / self.pipeline.vae.config.scaling_factor, return_dict=False)[
                0
            ]  # [-1, 1]
            image = (image * 0.5 + 0.5).clamp(0, 1)

            image = rearrange(
                image, "(b n_pbr n) c h w -> b n_pbr n c h w", n_pbr=len(self.pbr_settings), n=self.num_view
            )
            image = torch.cat((torch.ones_like(image[:, :, :1, ...]) * 0.5, image), dim=2)
            image = rearrange(image, "b n_pbr n c h w -> (b n_pbr n) c h w")
            image = rearrange(
                image,
                "(b n_pbr n) c h w -> b c (n_pbr h) (n w)",
                b=1,
                n_pbr=len(self.pbr_settings),
                n=self.num_view + 1,
            )
            outputs.append(image)

        all_target_pbrs = []
        for pbr_token in self.pbr_settings:
            all_target_pbrs.append(target_imgs[pbr_token])
        all_target_pbrs = torch.stack(all_target_pbrs, dim=0).transpose(1, 0)
        all_target_pbrs = torch.cat(
            (cond_imgs_tensor.unsqueeze(1).repeat(1, len(self.pbr_settings), 1, 1, 1, 1), all_target_pbrs), dim=2
        )
        all_target_pbrs = rearrange(all_target_pbrs, "b n_pbrs n c h w -> b c (n_pbrs h) (n w)")
        gts = all_target_pbrs
        outputs = torch.cat(outputs, dim=0).to(self.device)
        images = torch.cat([gts, outputs], dim=-2)
        self.validation_step_outputs.append(images)

    @torch.no_grad()
    def on_validation_epoch_end(self):
        """Aggregates validation results at epoch end.

        Gathers outputs from all GPUs (if distributed training),
        creates a unified visualization grid, and saves to disk.
        Only rank 0 process performs saving.
        """
        # [Result aggregation and visualization...]
        # Key steps:
        # 1. Gather validation outputs from all processes
        # 2. Create image grid combining ground truths and predictions
        # 3. Save visualization with step-numbered filename
        # 4. Clear memory for next validation cycle

        images = torch.cat(self.validation_step_outputs, dim=0)
        all_images = self.all_gather(images)
        all_images = rearrange(all_images, "r b c h w -> (r b) c h w")

        if self.global_rank == 0:
            grid = make_grid(all_images, nrow=8, normalize=True, value_range=(0, 1))
            save_image(grid, os.path.join(self.logdir, "images_val", f"val_{self.global_step:07d}.png"))

        self.validation_step_outputs.clear()  # free memory

    def configure_optimizers(self):
        lr = self.learning_rate
        optimizer = torch.optim.AdamW(self.unet.parameters(), lr=lr)

        def lr_lambda(step):
            warm_up_step = 1000
            T_step = 9000
            gamma = 0.9
            min_lr = 0.1 if step >= warm_up_step else 0.0
            max_lr = 1.0
            normalized_step = step % (warm_up_step + T_step)
            current_max_lr = max_lr * gamma ** (step // (warm_up_step + T_step))
            if current_max_lr < min_lr:
                current_max_lr = min_lr
            if normalized_step < warm_up_step:
                lr_step = min_lr + (normalized_step / warm_up_step) * (current_max_lr - min_lr)
            else:
                step_wc_wp = normalized_step - warm_up_step
                ratio = step_wc_wp / T_step
                lr_step = min_lr + 0.5 * (current_max_lr - min_lr) * (1 + math.cos(math.pi * ratio))
            return lr_step

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
            "monitor": "val_loss",
            "strict": False,
            "name": None,
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
