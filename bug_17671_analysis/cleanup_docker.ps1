# Docker 清理脚本
# 安全清理未使用的 Docker 资源

Write-Host "=== Docker 清理脚本 ===" -ForegroundColor Green
Write-Host ""

# 1. 显示当前状态
Write-Host "1. 当前 Docker 系统使用情况：" -ForegroundColor Yellow
docker system df
Write-Host ""

# 2. 删除悬空镜像（dangling images）
Write-Host "2. 删除悬空镜像（<none>:<none>，约 46GB）..." -ForegroundColor Yellow
docker image prune -f
Write-Host ""

# 3. 删除旧容器（除了最近使用的）
Write-Host "3. 删除已停止的旧容器..." -ForegroundColor Yellow
docker container prune -f
Write-Host ""

# 4. 询问是否删除 lmsysorg/sglang:latest
Write-Host "4. 是否删除 lmsysorg/sglang:latest（56.4GB）？" -ForegroundColor Yellow
Write-Host "   当前保留: lmsysorg/sglang:dev（57GB）" -ForegroundColor Cyan
$deleteLatest = Read-Host "   删除 latest 版本？(y/N)"
if ($deleteLatest -eq "y" -or $deleteLatest -eq "Y") {
    Write-Host "   正在删除 lmsysorg/sglang:latest..." -ForegroundColor Yellow
    docker rmi lmsysorg/sglang:latest
    Write-Host "   ✓ 已删除" -ForegroundColor Green
} else {
    Write-Host "   保留 lmsysorg/sglang:latest" -ForegroundColor Cyan
}
Write-Host ""

# 5. 显示清理后的状态
Write-Host "5. 清理后的 Docker 系统使用情况：" -ForegroundColor Yellow
docker system df
Write-Host ""

Write-Host "=== 清理完成 ===" -ForegroundColor Green
