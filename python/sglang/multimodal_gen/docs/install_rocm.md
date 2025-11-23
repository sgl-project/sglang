# ROCm quickstart for sgl-diffusion

```bash
docker run --device=/dev/kfd --device=/dev/dri --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env HF_TOKEN=<secret> \
  <IMAGE_NOT_UPLOADED_YET> \
  sglang generate --model-path black-forest-labs/FLUX.1-dev --prompt "A logo With Bold Large text: SGL Diffusion" --save-output
```
