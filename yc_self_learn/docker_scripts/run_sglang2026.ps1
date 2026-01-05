# SGLang 2026 容器启动脚本
# 使用方法: .\run_sglang2026.ps1 [start|stop|restart|status|logs]

param(
    [Parameter(Position=0)]
    [ValidateSet("start", "stop", "restart", "status", "logs", "shell")]
    [string]$Action = "start"
)

$ComposeFile = "docker-compose-sglang2026.yml"
$ServiceName = "sglang2026"

# 切换到脚本所在目录
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

switch ($Action) {
    "start" {
        Write-Host "🚀 启动 SGLang 2026 容器..." -ForegroundColor Green
        docker-compose -f $ComposeFile up -d
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ SGLang 2026 容器已启动" -ForegroundColor Green
            Write-Host "📊 查看状态: .\run_sglang2026.ps1 status" -ForegroundColor Yellow
            Write-Host "📝 查看日志: .\run_sglang2026.ps1 logs" -ForegroundColor Yellow
        }
    }
    "stop" {
        Write-Host "🛑 停止 SGLang 2026 容器..." -ForegroundColor Yellow
        docker-compose -f $ComposeFile down
    }
    "restart" {
        Write-Host "🔄 重启 SGLang 2026 容器..." -ForegroundColor Cyan
        docker-compose -f $ComposeFile restart
    }
    "status" {
        Write-Host "📊 SGLang 2026 容器状态:" -ForegroundColor Cyan
        docker-compose -f $ComposeFile ps
        Write-Host "`n💡 健康检查:" -ForegroundColor Cyan
        $healthStatus = docker inspect --format="{{.State.Health.Status}}" $ServiceName 2>$null
        if ($healthStatus) {
            Write-Host "   状态: $healthStatus" -ForegroundColor $(if ($healthStatus -eq "healthy") { "Green" } else { "Yellow" })
        } else {
            Write-Host "   未配置健康检查或容器未运行" -ForegroundColor Yellow
        }
    }
    "logs" {
        Write-Host "📝 SGLang 2026 容器日志 (按 Ctrl+C 退出):" -ForegroundColor Cyan
        docker-compose -f $ComposeFile logs -f
    }
    "shell" {
        Write-Host "🐚 进入 SGLang 2026 容器 shell..." -ForegroundColor Cyan
        docker exec -it $ServiceName /bin/bash
    }
}

