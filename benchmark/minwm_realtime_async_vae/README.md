# MinWM Realtime Async VAE Benchmark

该目录提供同步 VAE 与异步 VAE 的同合同 A/B 压测。每个并发档位会建立独立 WebSocket Session，先运行 warmup chunk，再交替发送完整 Action 状态并记录 action 到首批帧到达、chunk 总耗时、阶段 Trace、FPS 和错误率。

```bash
python benchmark/minwm_realtime_async_vae/load_test.py \
  --ws-url ws://HOST/v1/realtime_video/generate \
  --profile async \
  --concurrency 1,2,4,8 \
  --output artifacts/async.json

python benchmark/minwm_realtime_async_vae/summarize.py \
  --baseline artifacts/sync.json \
  --async-profile artifacts/async.json \
  --output-json artifacts/report.json \
  --output-md artifacts/report.zh-CN.md
```

最高稳定并发的默认门槛为：P95 action 到首批帧 `< 1000 ms`、每会话生成速度 `>= 16 FPS`、错误率 `0`。最终人工浏览器验证另行记录 action 到 canvas 首帧的真实耗时。
