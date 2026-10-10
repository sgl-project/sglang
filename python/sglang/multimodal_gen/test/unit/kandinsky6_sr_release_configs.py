# SPDX-License-Identifier: Apache-2.0
"""The real component configs (no weights) of the two official Kandinsky 6 VSR Diffusers repos, verbatim from the Hub.

* flow-matching ``kandinskylab/Kandinsky-6.0-VSR-5s-Diffusers``: ``FlowMatchEulerDiscreteScheduler``, a 64-wide DiT head;
* 2-step pi-Flow ``kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers``: ``PiflowScheduler`` (``nfe`` 2,
  ``n_grid`` 10), a 640-wide DiT head.

Both repos have the same DiT tensor names (only ``out_layer`` differs in width) and byte-identical ``vae/`` and
``latent_upscaler/`` files.
"""

import json

# --- kandinskylab/Kandinsky-6.0-VSR-5s-Diffusers (flow-matching) ---
# The dict-valued ``_kandinsky6_sr`` entry of its model_index.json and the root ``sr_config.json`` exist only in this repo.
FLOW_MODEL_INDEX = json.loads(
    r"""{"_class_name":"Kandinsky6SRPipeline","_diffusers_version":"0.39.0","_kandinsky6_sr":{"format_version":1,"latent_upscaler":"latent_upscaler","portable_components":true,"source_checkpoint":"9454578e88f86dd9c40df6a8a79529a7e5a5ecc9","uses_patched_diffusers":true,"vae_name":"video-kvae"},"latent_upscaler":["diffusers","Kandinsky6SRLatentUpscalerBank"],"scheduler":["diffusers","FlowMatchEulerDiscreteScheduler"],"transformer":["diffusers","Kandinsky6SRTransformer3DModel"],"vae":["diffusers","Kandinsky6SRVAE"]}"""
)
FLOW_SR_CONFIG = json.loads(
    r"""{"component_loading":"Diffusers ModelMixin wrappers","default_resolution_scale":2.25,"description":"Kandinsky 6 SR Diffusers bundle"}"""
)
FLOW_TRANSFORMER_CONFIG = json.loads(
    r"""{"attention_params":{"1024":{"causal":false,"chunk":false,"glob":false,"local":false,"type":"flash","window":3},"512":{"P":0.8,"P_warmup_start":0.95,"P_warmup_steps":200,"add_sta":true,"causal":false,"chunk":false,"glob":false,"local":false,"method":"topcdf","type":"nabla","wH":7,"wT":11,"wW":7,"window":3}},"attribute_overrides":{},"axes_dims":[16,24,24],"ff_dim":7168,"in_text_dim":3584,"in_text_dim2":768,"in_visual_dim":64,"instruct_type":"hybrid_anchor","model_dim":1792,"num_text_blocks":2,"num_visual_blocks":32,"out_visual_dim":64,"patch_size":[1,1,1],"sr_params":{"cap_noise_timestep":false,"fps":24,"lq_channel_noise_scale":0.0,"lq_noise_scale":0.7,"lq_noise_type":"ddpm","scale_factor":{"512":[1.0,2.0,2.0]},"scheduler_scale":5.0,"visual_size":[512]},"time_dim":512,"use_text":false,"visual_cond":true}"""
)
FLOW_SCHEDULER_CONFIG = json.loads(
    r"""{"_class_name":"FlowMatchEulerDiscreteScheduler","_diffusers_version":"0.39.0","base_image_seq_len":256,"base_shift":0.5,"invert_sigmas":false,"max_image_seq_len":4096,"max_shift":1.15,"num_train_timesteps":1000,"shift":5.0,"shift_terminal":null,"stochastic_sampling":false,"time_shift_type":"exponential","use_beta_sigmas":false,"use_dynamic_shifting":false,"use_exponential_sigmas":false,"use_karras_sigmas":false}"""
)

# --- kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers (2-step pi-Flow) ---
DISTILLED_MODEL_INDEX = json.loads(
    r"""{"_class_name":"Kandinsky6SRPipeline","_diffusers_version":"0.41.0.dev0","latent_upscaler":["diffusers","Kandinsky6SRLatentUpscalerBank"],"scheduler":["diffusers","PiflowScheduler"],"transformer":["diffusers","Kandinsky6SRTransformer3DModel"],"vae":["diffusers","Kandinsky6SRVAE"]}"""
)
DISTILLED_TRANSFORMER_CONFIG = json.loads(
    r"""{"attention_params":{"1024":{"causal":false,"chunk":false,"glob":false,"local":false,"type":"flash","window":3},"512":{"P":0.8,"P_warmup_start":0.95,"P_warmup_steps":0,"add_sta":true,"causal":false,"chunk":false,"glob":false,"local":false,"method":"topcdf","type":"nabla","wH":7,"wT":11,"wW":7,"window":3}},"attribute_overrides":{},"axes_dims":[16,24,24],"ff_dim":7168,"in_text_dim":3584,"in_text_dim2":768,"in_visual_dim":64,"instruct_type":"hybrid_anchor","model_dim":1792,"num_text_blocks":2,"num_visual_blocks":32,"out_visual_dim":640,"patch_size":[1,1,1],"sr_params":{"cap_noise_timestep":false,"fps":24,"lq_channel_noise_scale":0.0,"lq_noise_scale":0.7,"lq_noise_type":"ddpm","scale_factor":{"512":[1.0,2.0,2.0]},"scheduler_scale":3.5,"visual_size":[512]},"time_dim":512,"use_text":false,"visual_cond":true}"""
)
DISTILLED_SCHEDULER_CONFIG = json.loads(
    r"""{"_class_name":"PiflowScheduler","_diffusers_version":"0.41.0.dev0","eps":1e-06,"final_step_size_scale":0.5,"n_grid":10,"nfe":2,"num_policy_substeps":128,"num_train_timesteps":1000,"shift":3.5}"""
)

# --- identical in both repos (the VAE config stores the *string* "None" for unused channel fields) ---
VAE_CONFIG = json.loads(
    r"""{"decoder_config":{"ch":256,"ch_mult":[0.0625,1,2,4,8],"checkpoint_list":[true,true,true,true,true,true],"in_channels":"None","norm_type":"rms_norm","num_res_blocks":2,"out_ch":3,"padding_mode":"zeros","resolution":0,"temporal_compress_start_level":1,"temporal_compress_times":4,"z_channels":64},"encoder_config":{"ch":128,"ch_mult":[0.125,1,2,4,8],"checkpoint_list":[true,true,true,true,true,true],"double_z":true,"downsample_version":2,"fix_pxs":true,"in_channels":3,"norm_type":"rms_norm","num_res_blocks":2,"out_ch":"None","padding_mode":"zeros","resolution":0,"temporal_compress_start_level":1,"temporal_compress_times":4,"z_channels":64},"scaling_factor":0.910344004631042,"spatial_factor":16,"temporal_factor":4,"vae_type":"video-kvae"}"""
)
LATENT_UPSCALER_CONFIG = json.loads(
    r"""{"models":[{"model":{"architecture":"multi_scale","bare_stem":true,"bottleneck_channels":null,"depthwise":false,"dims":3,"enable_x2_entry":false,"expand_ratio":1,"gradient_checkpointing":false,"grn":false,"hidden_channels":2048,"in_channels":64,"input_skip":false,"kernel_size":3,"layer_scale_init":null,"loss_weight_2x":0.0,"loss_weight_4x":1.0,"modulated_norm":true,"modulated_output_proj":true,"num_mid_blocks":3,"num_post_blocks":3,"num_pre_blocks":5,"stage_channels":[2048,1024,512],"stochastic_depth_rate":0.0,"temporal_mix":false,"temporal_padding":"replicate","upsample_mode":"pxs_v2","upsample_padding_mode":"zeros","upscale_factor":4},"target_scale":"4x"},{"model":{"architecture":"multi_scale","bare_stem":true,"bottleneck_channels":null,"depthwise":false,"dims":3,"enable_x2_entry":true,"expand_ratio":1,"gradient_checkpointing":false,"grn":false,"hidden_channels":2048,"in_channels":64,"input_skip":false,"kernel_size":3,"layer_scale_init":null,"loss_weight_2x":0.0,"loss_weight_4x":1.0,"modulated_norm":true,"modulated_output_proj":true,"num_mid_blocks":3,"num_post_blocks":3,"num_pre_blocks":5,"stage_channels":[2048,1024,512],"stochastic_depth_rate":0.0,"temporal_mix":false,"temporal_padding":"replicate","upsample_mode":"pxs_v2","upsample_padding_mode":"zeros","upscale_factor":4,"x2_adapter_blocks":2,"x2_adapter_sources":[0,4],"x2_finisher":"pxs_residual","x2_tail_mode":"private_full"},"target_scale":"2x"}],"scaling_factor":0.910344004631042}"""
)
