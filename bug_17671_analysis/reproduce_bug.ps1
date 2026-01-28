# Issue #17671 复现脚本 (PowerShell版本)
# 用于验证 SGLang Docker 镜像缺少 diffusion 依赖

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Issue #17671 复现脚本" -ForegroundColor Cyan
Write-Host "验证 lmsysorg/sglang:dev 镜像缺少 diffusion 依赖" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Step 0: 拉取最新 dev 镜像
Write-Host "Step 0: 拉取最新 dev 镜像..." -ForegroundColor Yellow
docker pull lmsysorg/sglang:dev
Write-Host "✓ 镜像拉取完成" -ForegroundColor Green
Write-Host ""

# Step 1: 验证 diffusers 模块缺失
Write-Host "Step 1: 验证 diffusers 模块缺失..." -ForegroundColor Yellow
Write-Host "执行: docker run --rm -it lmsysorg/sglang:dev python -c `"import diffusers`"" -ForegroundColor Gray
Write-Host ""

$result = docker run --rm -it lmsysorg/sglang:dev python -c "import diffusers" 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "❌ 意外：diffusers 模块存在！" -ForegroundColor Red
    exit 1
} else {
    Write-Host "✓ 确认：diffusers 模块缺失（ModuleNotFoundError）" -ForegroundColor Green
    Write-Host $result -ForegroundColor Red
}
Write-Host ""

# Step 2: 检查镜像信息
Write-Host "Step 2: 检查镜像信息..." -ForegroundColor Yellow
Write-Host "执行: docker image inspect lmsysorg/sglang:dev | Select-Object -First 40" -ForegroundColor Gray
docker image inspect lmsysorg/sglang:dev | Select-Object -First 40
Write-Host ""

# Step 3: 使用 tiny 模型测试（可选，需要GPU）
Write-Host "Step 3: 使用 tiny 模型测试（需要GPU）..." -ForegroundColor Yellow
Write-Host "注意：这一步需要GPU，如果失败说明缺少diffusion依赖" -ForegroundColor Gray
Write-Host ""
Write-Host "手动执行以下命令：" -ForegroundColor Yellow
Write-Host "  docker run --gpus all --rm -it -v ~/.cache/huggingface:/root/.cache/huggingface lmsysorg/sglang:dev sglang generate --model-path hf-internal-testing/tiny-stable-diffusion-pipe-variants-right-format --prompt `"test`" --save-output --backend diffusers" -ForegroundColor Cyan
Write-Host ""

# Step 4: 验证安装后是否恢复
Write-Host "Step 4: 验证安装后是否恢复..." -ForegroundColor Yellow
Write-Host "进入容器并安装 diffusion extras..." -ForegroundColor Gray
Write-Host ""
Write-Host "执行以下命令：" -ForegroundColor Yellow
Write-Host "  docker run --gpus all --rm -it -v ~/.cache/huggingface:/root/.cache/huggingface lmsysorg/sglang:dev bash" -ForegroundColor Cyan
Write-Host ""
Write-Host "在容器内执行：" -ForegroundColor Yellow
Write-Host "  uv pip install 'sglang[diffusion]' --prerelease=allow" -ForegroundColor Cyan
Write-Host "  python -c 'import diffusers; print(\"✓ diffusers installed\")'" -ForegroundColor Cyan
Write-Host "  sglang generate --model-path hf-internal-testing/tiny-stable-diffusion-pipe-variants-right-format --prompt `"test`" --save-output --backend diffusers" -ForegroundColor Cyan
Write-Host ""

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "复现完成！" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
