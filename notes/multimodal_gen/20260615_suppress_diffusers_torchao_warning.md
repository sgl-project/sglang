# 2026-06-15 清理 diffusers torchao warning

- 基准 git hash: `1e5b1aade1`
- 问题: server log 里反复出现 `Unable to import `torchao` Tensor objects...`，来源是 `diffusers.quantizers.torchao.torchao_quantizer` 导入 torchao tensor safe globals 失败时的 warning。
- 处理: 在 SGLang import-time patch 中给该 diffusers logger 添加精确消息过滤器，只过滤这条非关键噪音，不影响其它 torchao/diffusers 日志。
- 注意: 按本地开发约定未跑测试；推 PR 前需要跑 pre-commit。
