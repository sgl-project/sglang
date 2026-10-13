# PR `run-ci` 标签触发设计

## 目标

让 PPU Full CI 遵循 SGLang 社区的标签触发模式：维护者通过 `run-ci` 标签决定是否消耗 PPU 资源；标签添加后立即触发，后续 PR push 保留标签并自动重跑。Human Review 继续作为独立合并门禁，不阻塞 Full CI。

## 范围

本次仅调整 PR Full CI 的触发编排和 `/tag-run-ci-label` 授权。保留现有 PPU Stage A/B/C、check 名称、`workflow_call` 结构及其他 slash command 行为；不新增 `/run-ci` 命令，不改为 `workflow_dispatch` 直接触发。

## 触发架构

`.github/workflows/ci.yaml` 继续作为 PR 编排入口，并增加 `pull_request.labeled` 事件。Full CI：

- 依赖 `registration-validation`，不再依赖 `human-review`。
- 仅在事件为 `pull_request` 且当前 PR 标签包含 `run-ci` 时调用 `.github/workflows/pr-test-ppu.yml`。
- 对 `labeled` 事件额外要求本次新增的标签就是 `run-ci`，避免 PR 已有 `run-ci` 时添加其他标签导致重复运行。
- 不再向 reusable workflow 传入 `require_run_ci_label: false`，避免绕过标签门禁。

Human Review job 保持现状，继续产生独立 required check。PPU workflow 内部的测试选择、Stage A/B/C 串行链、失败聚合和 secrets 继承保持不变。

## 事件语义

| PR 事件 | 是否有 `run-ci` | Full CI 行为 |
| --- | --- | --- |
| opened/reopened/ready_for_review | 否 | 跳过 |
| opened/reopened/ready_for_review | 是 | Registration Validation 成功后运行 |
| labeled：添加 `run-ci` | 是 | Registration Validation 成功后运行 |
| labeled：添加其他标签 | 任意 | 跳过 |
| synchronize | 是 | 自动重跑 |
| synchronize | 否 | 跳过 |
| pull_request_review | 任意 | 只更新 Human Review，不启动 Full CI |

标签添加后保留在 PR 上，不自动移除。现有 concurrency 继续取消同一 PR 的过期 run，确保只保留最新提交对应的执行。

## Slash Command 与授权

继续使用 `/tag-run-ci-label` 添加 `run-ci` 标签。授权来源合并为：

1. GitHub repository permission 为 `write`、`maintain` 或 `admin`；或
2. `.github/CI_PERMISSIONS.json` 中显式设置 `can_tag_run_ci_label: true`。

`read`、`triage`、无仓库权限且无显式配置的用户被拒绝。现有 `/tag-and-rerun-ci`、rerun stage/group/test 等命令保持原授权和行为，避免扩大权限变更范围。

`issue_comment` workflow 必须始终从默认分支 checkout 并执行可信的 slash command handler。它不得执行 PR head 中可被贡献者修改的高权限脚本；fork PR 也遵循这一规则。

## 数据流

1. 授权用户在 PR 留言 `/tag-run-ci-label`。
2. `slash-command-handler.yml` 从默认分支运行 handler。
3. handler 查询评论者仓库权限，并与显式权限配置合并。
4. 授权通过后，为 PR 添加 `run-ci`。
5. `pull_request.labeled` 启动 `ci.yaml`。
6. Registration Validation 成功后，Full CI 调用 `pr-test-ppu.yml`。
7. 后续 push 产生 `synchronize`；因为标签仍在，Full CI 自动重跑。

## 错误处理

- 权限查询失败时默认拒绝，不添加标签。
- 权限不足时保留现有失败反馈，不触发 PPU。
- Registration Validation 失败时 Full CI 不运行。
- 非 `run-ci` 标签产生的 `labeled` run 只执行适用的轻量门禁，Full CI 明确跳过。
- PPU workflow 的失败继续由现有 finish job 汇总，不改变失败可见性。

## 测试策略

- Python 单元测试覆盖 repository permission 与显式权限配置的组合，包括 `write`、`maintain`、`admin` 放行以及低权限拒绝。
- handler 测试验证 `/tag-run-ci-label` 只添加预期标签，且权限查询异常时 fail closed。
- Workflow 静态测试验证 `labeled` 触发器、Full CI 的 `needs` 和 `if` 条件，以及不再传入绕过标签的输入。
- 对三个 workflow 执行 YAML 解析，并对 Python handler 执行格式、lint/语法检查和相关测试。

## 兼容性与非目标

- 不改变 Full CI 的显示名称与 reusable workflow check 名称。
- 不改变 PPU runner、secrets、Stage A/B/C 或测试选择逻辑。
- 不自动删除 `run-ci` 标签。
- 不新增 slash command alias。
- 不让 Human Review 成为 Full CI 前置条件。
