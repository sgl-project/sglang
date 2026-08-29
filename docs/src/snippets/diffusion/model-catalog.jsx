// Keep explicit model IDs aligned with multimodal_gen/registry.py. The unit
// test for this component reports registry entries missing from the catalog.
const MODEL_CATALOG = {
  image: [
    {
      name: "FLUX",
      tasks: ["T2I", "image editing", "multi-reference"],
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
      tasks: ["T2I", "image editing", "layered image"],
      modelIds: [
        "Qwen/Qwen-Image",
        "nvidia/Qwen-Image-NVFP4",
        "Qwen/Qwen-Image-2512",
        "Qwen/Qwen-Image-Edit",
        "Qwen/Qwen-Image-Edit-2509",
        "Qwen/Qwen-Image-Edit-2511",
        "Qwen/Qwen-Image-Layered",
      ],
      cookbook: "/cookbook/diffusion/Qwen-Image/Qwen-Image",
    },
    {
      name: "Z-Image",
      tasks: ["T2I"],
      modelIds: ["Tongyi-MAI/Z-Image", "Tongyi-MAI/Z-Image-Turbo"],
      cookbook: "/cookbook/diffusion/Z-Image/Z-Image-Turbo",
    },
    {
      name: "Krea-2",
      tasks: ["T2I"],
      modelIds: ["krea/Krea-2"],
      cookbook: "/cookbook/diffusion/Krea/Krea-2",
    },
    {
      name: "LongCat-Image",
      tasks: ["T2I", "image editing"],
      modelIds: [
        "meituan-longcat/LongCat-Image",
        "meituan-longcat/LongCat-Image-Edit",
        "meituan-longcat/LongCat-Image-Edit-Turbo",
      ],
      cookbook: "/cookbook/diffusion/LongCat/LongCat-Image",
    },
    {
      name: "Stable Diffusion 3 / 3.5",
      tasks: ["T2I"],
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
      tasks: ["T2I", "512px / 1024px"],
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
      tasks: ["T2I", "typography"],
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
      tasks: ["T2I", "prompt enhancement"],
      modelIds: ["baidu/ERNIE-Image", "baidu/ERNIE-Image-Turbo"],
      cookbook: "/cookbook/diffusion/Ernie-Image/Ernie-Image",
    },
    {
      name: "FireRed-Image",
      tasks: ["image editing"],
      modelIds: [
        "FireRedTeam/FireRed-Image-Edit-1.0",
        "FireRedTeam/FireRed-Image-Edit-1.1",
      ],
    },
    {
      name: "JoyAI-Image",
      tasks: ["image editing"],
      modelIds: ["jdopensource/JoyAI-Image-Edit-Diffusers"],
    },
    {
      name: "GLM-Image",
      tasks: ["T2I", "AR + diffusion"],
      modelIds: ["zai-org/GLM-Image"],
      note: "Resolved by the GLM-Image family detector.",
    },
    {
      name: "Hunyuan3D 2",
      tasks: ["text / image to 3D"],
      modelIds: ["tencent/Hunyuan3D-2"],
    },
  ],
  video: [
    {
      name: "Wan 2.1",
      tasks: ["T2V", "I2V", "inpainting"],
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
      tasks: ["T2V", "I2V", "TI2V"],
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
      tasks: ["T2V", "I2V", "TI2V", "distilled"],
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
      name: "LTX",
      tasks: ["T2V", "I2V", "audio", "one / two-stage"],
      modelIds: [
        "Lightricks/LTX-2",
        "Lightricks/LTX-2.3",
        "Lightricks/LTX-2.5-Diffusers",
      ],
      cookbook: "/cookbook/diffusion/LTX/LTX2 & LTX2.3",
    },
    {
      name: "HunyuanVideo",
      tasks: ["T2V"],
      modelIds: [
        "hunyuanvideo-community/HunyuanVideo",
        "FastVideo/FastHunyuan-diffusers",
      ],
    },
    {
      name: "LongLive 2.0",
      tasks: ["T2V", "I2V", "long video"],
      modelIds: [
        "Rabinovich/LongLive-2.0-5B-Diffusers",
        "Efficient-Large-Model/LongLive-2.0-5B",
      ],
      cookbook: "/cookbook/diffusion/LongLive/LongLive-2.0",
    },
    {
      name: "MiniMax-H3",
      tasks: ["T2VA", "FL2VA", "Ref2VA"],
      modelIds: ["MiniMaxAI/MiniMax-H3", "MiniMax/MiniMax-H3"],
      cookbook: "/cookbook/diffusion/MiniMax/MiniMax-H3",
    },
    {
      name: "MOVA",
      tasks: ["video + audio", "360p / 720p"],
      modelIds: ["OpenMOSS-Team/MOVA-360p", "OpenMOSS-Team/MOVA-720p"],
      note: "Resolved by the MOVA resolution detector.",
      cookbook: "/cookbook/diffusion/MOVA/MOVA",
    },
    {
      name: "JoyAI-Echo",
      tasks: ["video + audio", "multi-shot"],
      modelIds: ["jdopensource/JoyAI-Echo"],
      cookbook: "/cookbook/diffusion/JoyEcho/JoyEcho",
    },
    {
      name: "SANA-Video",
      tasks: ["T2V", "480p"],
      modelIds: ["Efficient-Large-Model/SANA-Video_2B_480p_diffusers"],
      cookbook: "/cookbook/diffusion/SANA-Video/SANA-Video",
    },
    {
      name: "LingBot Video MoE",
      tasks: ["T2V", "30B-A3B"],
      modelIds: ["robbyant/lingbot-video-moe-30b-a3b"],
      note: "Resolved by the LingBot Video MoE family detector.",
      cookbook: "/cookbook/diffusion/LingBot-Video/LingBot-Video-MoE",
    },
    {
      name: "Helios",
      tasks: ["T2V", "720p"],
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
      tasks: ["T2I", "T2V", "I2V", "robot policy"],
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
      tasks: ["realtime", "camera control", "causal state"],
      modelIds: [
        "IPostYellow/lingbot-world-fast-diffusers",
        "robbyant/lingbot-world-fast-diffusers",
        "robbyant/lingbot-world-v2-14b-causal-fast-diffusers",
      ],
      cookbook: "/cookbook/diffusion/LingBot-World/LingBot-World-2.0",
    },
    {
      name: "SANA-WM",
      tasks: ["world model", "bidirectional", "streaming"],
      modelIds: [
        "Efficient-Large-Model/SANA-WM_bidirectional",
        "Efficient-Large-Model/SANA-WM_streaming",
      ],
      cookbook: "/cookbook/diffusion/SANA-WM/SANA-WM",
    },
    {
      name: "Pi0.5",
      tasks: ["robot action", "OpenPI / LeRobot"],
      modelIds: ["lerobot/pi05_base", "lerobot/pi05_libero_base"],
      cookbook: "/cookbook/vla/OpenPI/Pi0.5",
    },
  ],
};

export const DiffusionModelCatalog = ({ category }) => {
  const models = MODEL_CATALOG[category] || [];

  return (
    <div className="not-prose sgd-model-catalog">
      {models.map((model) => (
        <article key={model.name} className="sgd-model-entry">
          <header className="sgd-model-entry-header">
            <h3>{model.name}</h3>
            <div className="sgd-model-entry-tasks">
              {model.tasks.map((task) => (
                <span key={task}>{task}</span>
              ))}
            </div>
          </header>
          <div className="sgd-model-entry-ids">
            {model.modelIds.map((modelId) => (
              <code key={modelId}>{modelId}</code>
            ))}
          </div>
          {model.note && <p className="sgd-model-entry-note">{model.note}</p>}
          {model.cookbook && (
            <a className="sgd-model-entry-link" href={model.cookbook}>
              Open cookbook <span aria-hidden="true">&rarr;</span>
            </a>
          )}
        </article>
      ))}
    </div>
  );
};
