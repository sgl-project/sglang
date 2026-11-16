# SGLang-Diffusion adoption assets

The diffusion README uses `adoption.png`, following the single-image layout of
SGLang's main README. Edit `adoption.html` and the individual files in `logos/`,
then regenerate the image. No external assets are loaded when rendering.

From the repository root, with Node.js and Chrome installed:

```bash
npx --yes --package=playwright playwright screenshot --channel chrome \
  --viewport-size="1600,412" --full-page \
  "file://$PWD/assets/sglang-diffusion/adoption.html" \
  assets/sglang-diffusion/adoption.png
```

## Logo sources

| Logo | Source |
| --- | --- |
| Modal | [Official logo](https://modal-cdn.com/modal-logo-icon.png) |
| Black Forest Labs | [Brand assets](https://bfl.ai/brand) |
| Krea | [Press kit](https://www.krea.ai/press) |
| Baseten, Ant Group | [SGLang's logo source deck](https://github.com/sgl-project/sgl-learning-materials/blob/160433e8779cb38c4a894674474f67a85bc8915d/slides/sglang_adoption_logo.pptx), `image25.png` and `image24.png` |
| Skywork | [Official website asset](https://static-s3.skyworkcdn.com/fe/skywork-site-assets/images/logo-pure.png) |
| Google | [Official wordmark](https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png) |
| TikTok | [Developer logo pack](https://developers.tiktok.com/docs/en/getting-started-design-guidelines), RGB horizontal black variant |
| Tencent Hunyuan | [Official website](https://hunyuan.tencent.com/) |
| Infinigence | [Official wordmark](https://www.infinigence-ai.com/static/home/svg/logo.svg) |
| fal | [Official blog wordmark](https://storage.ghost.io/c/0e/15/0e15ee8a-bd95-4b71-9258-950f77f4d196/content/images/2024/09/fal_logo.svg) |
| MOSS (image and video) | [Official repository](https://github.com/OpenMOSS/MOSS-Image/blob/main/assets/logo.svg) |
| Sol-Engine | [Official repository](https://github.com/NVlabs/Sana/blob/sol-engine/assets/sol-engine-logo.png) |

Keep this list specific to SGLang-Diffusion adoption or implementation reuse.
Model support alone does not establish adoption. In particular, the fal caption
describes a port to Falcon, not deployment of the SGLang-Diffusion runtime.
Logo ownership remains with the respective organizations and projects.
