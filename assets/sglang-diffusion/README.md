# SGLang-Diffusion README assets

The diffusion README uses `adoption.png`, following the single-image layout of
SGLang's main README. Edit `adoption.html` and the individual files in `logos/`,
then regenerate the image. Keep logos alphabetized by English display name,
ignoring case. No external assets are loaded when rendering.

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
| Ant Group | [SGLang's logo source deck](https://github.com/sgl-project/sgl-learning-materials/blob/160433e8779cb38c4a894674474f67a85bc8915d/slides/sglang_adoption_logo.pptx), `image24.png` |
| Baseten | [SGLang's logo source deck](https://github.com/sgl-project/sgl-learning-materials/blob/160433e8779cb38c4a894674474f67a85bc8915d/slides/sglang_adoption_logo.pptx), `image25.png` |
| Black Forest Labs | [Brand assets](https://bfl.ai/brand) |
| Boson AI | [Official website icon](https://www.boson.ai/logo.svg) |
| fal | [Official blog wordmark](https://storage.ghost.io/c/0e/15/0e15ee8a-bd95-4b71-9258-950f77f4d196/content/images/2024/09/fal_logo.svg) |
| Google | [Official wordmark](https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png) |
| Infinigence | [Official wordmark](https://www.infinigence-ai.com/static/home/svg/logo.svg) |
| Krea | [Press kit](https://www.krea.ai/press) |
| Modal | [Official logo](https://modal-cdn.com/modal-logo-icon.png) |
| MOSS (image and video) | [Official repository](https://github.com/OpenMOSS/MOSS-Image/blob/main/assets/logo.svg) |
| Skywork | [Official website asset](https://static-s3.skyworkcdn.com/fe/skywork-site-assets/images/logo-pure.png) |
| Sol-Engine | [Official repository](https://github.com/NVlabs/Sana/blob/sol-engine/assets/sol-engine-logo.png) |
| Tencent Hunyuan | [Official website](https://hunyuan.tencent.com/) |
| TikTok | [Developer logo pack](https://developers.tiktok.com/docs/en/getting-started-design-guidelines), RGB horizontal black variant |

Keep this list specific to SGLang-Diffusion adoption or implementation reuse.
Model support alone does not establish adoption. In particular, the fal caption
describes a port to Falcon, not deployment of the SGLang-Diffusion runtime.
Logo ownership remains with the respective organizations and projects.

## SGLang-Diffusion logo

`logo.svg` and `logo-dark.svg` head the diffusion README on light and dark
themes. They differ only in the colors of the dots and how strong their glow is.

- The SGL mark is `assets/logo.svg`, unchanged.
- The word DIFFUSION is set in Open Sans ExtraBold Italic
  ([SIL Open Font License 1.1](https://github.com/googlefonts/opensans)) and
  stored as outlines, so no font is loaded.
- Each dot is a piece of a letter with noise added to it, more the further
  right it sits. Most stay with their letter; the rest drift off to the right
  and leave a hole behind.
- The dots past the last letter have a soft glow: one blurred layer under the
  word.
- The drifting dots move slowly on a 12 second loop. The motion is CSS inside
  the SVG, with no script, and it stops under `prefers-reduced-motion`.

The two files are generated. The generator is not in this repository.
