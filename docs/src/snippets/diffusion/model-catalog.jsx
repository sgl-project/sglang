// Keep explicit model IDs aligned with multimodal_gen/registry.py. The unit
// test for this component reports registry entries missing from the catalog.
export const DiffusionModelCatalog = ({ category }) => {
  const MODEL_CATALOG = {
  image: [
    {
      name: "FLUX",
      modelIds: [
        "black-forest-labs/FLUX.1-dev",
        "black-forest-labs/FLUX.2-dev",
        "black-forest-labs/FLUX.2-dev-NVFP4",
        "black-forest-labs/FLUX.2-klein-4B",
        "black-forest-labs/FLUX.2-klein-9B",
        "black-forest-labs/FLUX.2-klein-base-4B",
        "black-forest-labs/FLUX.2-klein-base-9B",
      ],
      cookbook: "/cookbook/diffusion/FLUX/FLUX",
    },
    {
      name: "Qwen-Image",
      modelIds: [
        "Qwen/Qwen-Image",
        "nvidia/Qwen-Image-NVFP4",
        "Qwen/Qwen-Image-2512",
      ],
      cookbook: "/cookbook/diffusion/Qwen-Image/Qwen-Image",
    },
    {
      name: "Qwen-Image Edit / Layered",
      modelIds: [
        "Qwen/Qwen-Image-Edit",
        "Qwen/Qwen-Image-Edit-2509",
        "Qwen/Qwen-Image-Edit-2511",
        "Qwen/Qwen-Image-Layered",
      ],
      cookbook: "/cookbook/diffusion/Qwen-Image/Qwen-Image-Edit",
    },
    {
      name: "Z-Image",
      modelIds: ["Tongyi-MAI/Z-Image", "Tongyi-MAI/Z-Image-Turbo"],
      cookbook: "/cookbook/diffusion/Z-Image/Z-Image-Turbo",
    },
    {
      name: "Krea-2",
      modelIds: ["krea/Krea-2"],
      cookbook: "/cookbook/diffusion/Krea/Krea-2",
    },
    {
      name: "LongCat-Image",
      modelIds: [
        "meituan-longcat/LongCat-Image",
        "meituan-longcat/LongCat-Image-Edit",
        "meituan-longcat/LongCat-Image-Edit-Turbo",
      ],
      cookbook: "/cookbook/diffusion/LongCat/LongCat-Image",
    },
    {
      name: "Stable Diffusion 3 / 3.5",
      modelIds: [
        "stabilityai/stable-diffusion-3-medium",
        "stabilityai/stable-diffusion-3-medium-diffusers",
        "stabilityai/stable-diffusion-3.5-medium",
        "stabilityai/stable-diffusion-3.5-medium-diffusers",
        "stabilityai/stable-diffusion-3.5-large",
        "stabilityai/stable-diffusion-3.5-large-diffusers",
      ],
    },
    {
      name: "SANA",
      modelIds: [
        "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers",
        "Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers",
        "Efficient-Large-Model/Sana_1600M_1024px_diffusers",
        "Efficient-Large-Model/Sana_600M_1024px_diffusers",
        "Efficient-Large-Model/Sana_1600M_512px_diffusers",
        "Efficient-Large-Model/Sana_600M_512px_diffusers",
      ],
    },
    {
      name: "Ideogram 4",
      modelIds: [
        "ideogram-ai/ideogram-4-fp8",
        "ideogram-ai/ideogram-4-nf4",
        "Comfy-Org/Ideogram-4",
        "fal/ideogram-v4-fast",
        "fal/ideogram-v4-instant",
      ],
      cookbook: "/cookbook/diffusion/Ideogram/Ideogram4",
    },
    {
      name: "ERNIE-Image",
      modelIds: ["baidu/ERNIE-Image", "baidu/ERNIE-Image-Turbo"],
      cookbook: "/cookbook/diffusion/Ernie-Image/Ernie-Image",
    },
    {
      name: "FireRed-Image",
      modelIds: [
        "FireRedTeam/FireRed-Image-Edit-1.0",
        "FireRedTeam/FireRed-Image-Edit-1.1",
      ],
    },
    {
      name: "JoyAI-Image",
      modelIds: ["jdopensource/JoyAI-Image-Edit-Diffusers"],
    },
    {
      name: "GLM-Image",
      modelIds: ["zai-org/GLM-Image"],
      note: "Resolved by the GLM-Image family detector.",
    },
    {
      name: "Hunyuan3D 2",
      modelIds: ["tencent/Hunyuan3D-2"],
    },
  ],
  video: [
    {
      name: "Wan 2.1",
      modelIds: [
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
        "weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers",
      ],
      cookbook: "/cookbook/diffusion/Wan/Wan2.1",
    },
    {
      name: "Wan 2.2",
      modelIds: [
        "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "nvidia/Wan2.2-T2V-A14B-Diffusers-NVFP4",
        "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
      ],
      cookbook: "/cookbook/diffusion/Wan/Wan2.2",
    },
    {
      name: "FastWan / TurboWan",
      modelIds: [
        "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
        "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers",
        "FastVideo/FastWan2.2-TI2V-5B-Diffusers",
        "IPostYellow/TurboWan2.1-T2V-1.3B-Diffusers",
        "IPostYellow/TurboWan2.1-T2V-14B-Diffusers",
        "IPostYellow/TurboWan2.1-T2V-14B-720P-Diffusers",
        "IPostYellow/TurboWan2.2-I2V-A14B-Diffusers",
      ],
      cookbook: "/cookbook/diffusion/Wan/Wan2.2",
    },
    {
      name: "LTX 2 / 2.3",
      modelIds: ["Lightricks/LTX-2", "Lightricks/LTX-2.3"],
      cookbook: "/cookbook/diffusion/LTX/LTX2 & LTX2.3",
    },
    {
      name: "LTX 2.5",
      modelIds: ["Lightricks/LTX-2.5-Diffusers"],
      cookbook: "/cookbook/diffusion/LTX/LTX2.5",
    },
    {
      name: "HunyuanVideo",
      modelIds: [
        "hunyuanvideo-community/HunyuanVideo",
        "FastVideo/FastHunyuan-diffusers",
      ],
    },
    {
      name: "LongLive 2.0",
      modelIds: [
        "Rabinovich/LongLive-2.0-5B-Diffusers",
        "Efficient-Large-Model/LongLive-2.0-5B",
      ],
      cookbook: "/cookbook/diffusion/LongLive/LongLive-2.0",
    },
    {
      name: "MiniMax-H3",
      modelIds: ["MiniMaxAI/MiniMax-H3", "MiniMax/MiniMax-H3"],
      cookbook: "/cookbook/diffusion/MiniMax/MiniMax-H3",
    },
    {
      name: "FastH3",
      modelIds: [
        "FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree",
      ],
      cookbook: "/cookbook/diffusion/MiniMax/MiniMax-H3#6-fasth3-4-step-distilled-preview",
    },
    {
      name: "MOVA",
      modelIds: ["OpenMOSS-Team/MOVA-360p", "OpenMOSS-Team/MOVA-720p"],
      note: "Resolved by the MOVA resolution detector.",
      cookbook: "/cookbook/diffusion/MOVA/MOVA",
    },
    {
      name: "JoyAI-Echo",
      modelIds: ["jdopensource/JoyAI-Echo"],
      cookbook: "/cookbook/diffusion/JoyEcho/JoyEcho",
    },
    {
      name: "SANA-Video",
      modelIds: ["Efficient-Large-Model/SANA-Video_2B_480p_diffusers"],
      cookbook: "/cookbook/diffusion/SANA-Video/SANA-Video",
    },
    {
      name: "LingBot Video MoE",
      modelIds: ["robbyant/lingbot-video-moe-30b-a3b"],
      note: "Resolved by the LingBot Video MoE family detector.",
      cookbook: "/cookbook/diffusion/LingBot-Video/LingBot-Video-MoE",
    },
    {
      name: "Helios",
      modelIds: [
        "BestWishYsh/Helios-Base",
        "BestWishYsh/Helios-Mid",
        "BestWishYsh/Helios-Distilled",
      ],
    },
  ],
  world: [
    {
      name: "Cosmos 3",
      modelIds: [
        "nvidia/Cosmos3-Nano",
        "nvidia/Cosmos3-Nano-Policy-DROID",
        "nvidia/Cosmos3-Super",
        "nvidia/Cosmos3-Super-Text2Image",
        "nvidia/Cosmos3-Super-Image2Video",
        "nvidia/Cosmos3-Edge",
      ],
      cookbook: "/cookbook/diffusion/Cosmos/Cosmos3",
    },
    {
      name: "LingBotWorld",
      modelIds: [
        "IPostYellow/lingbot-world-fast-diffusers",
        "robbyant/lingbot-world-fast-diffusers",
      ],
      cookbook: "/cookbook/diffusion/LingBot-World/LingBot-World",
    },
    {
      name: "LingBotWorld 2.0",
      modelIds: ["robbyant/lingbot-world-v2-14b-causal-fast-diffusers"],
      cookbook: "/cookbook/diffusion/LingBot-World/LingBot-World-2.0",
    },
    {
      name: "SANA-WM",
      modelIds: [
        "Efficient-Large-Model/SANA-WM_bidirectional",
        "Efficient-Large-Model/SANA-WM_streaming",
      ],
      cookbook: "/cookbook/diffusion/SANA-WM/SANA-WM",
    },
    {
      name: "Pi0.5",
      modelIds: ["lerobot/pi05_base", "lerobot/pi05_libero_base"],
      cookbook: "/cookbook/vla/OpenPI/Pi0.5",
    },
  ],
  };

  const models = MODEL_CATALOG[category] || [];

  return (
    <div className="not-prose sgd-model-catalog" role="list">
      {models.map((model) => (
        <article key={model.name} className="sgd-model-entry" role="listitem">
          <div className="sgd-model-entry-meta">
            <h3>{model.name}</h3>
            {model.cookbook && (
              <a
                className="sgd-model-entry-link"
                href={model.cookbook}
                aria-label={`${model.name} cookbook`}
              >
                Cookbook <span aria-hidden="true">&rarr;</span>
              </a>
            )}
          </div>
          <div className="sgd-model-entry-ids">
            {model.modelIds.map((modelId) => (
              <div key={modelId} className="sgd-model-id">
                <code>{modelId}</code>
              </div>
            ))}
            {model.note && <p className="sgd-model-entry-note">{model.note}</p>}
          </div>
        </article>
      ))}
    </div>
  );
};
