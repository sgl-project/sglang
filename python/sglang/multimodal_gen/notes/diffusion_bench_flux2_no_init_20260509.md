# diffusion bench flux2 no-init 2026-05-09

- commit: 55c9a9721
- 背景: H200 benchmark 中 FLUX.2 text_encoder 启动 9 分钟，py-spy 显示卡在 transformers `init_weights`
- 处理: `skip_init_modules()` 额外进入 transformers `no_init_weights()`，避免 checkpoint load 前重复随机初始化大 text encoder
- 语义: 只跳过会被 checkpoint 覆盖的初始化，不改加载后的权重
- 精度对齐: 100%，理论 bit-exact
