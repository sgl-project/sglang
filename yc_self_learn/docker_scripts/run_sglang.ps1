# SGLang Docker 容器启动脚本
# 使用方法: .\run_sglang.ps1 -Model "qwen/qwen2.5-0.5b-instruct" -Port 30000

param(
    [string]$Model = "qwen/qwen2.5-0.5b-instruct",
    [int]$Port = 30000,
    [string]$ContainerName = "sglang_server",
    [string]$Image = "lmsysorg/sglang:latest",
    [switch]$Dev = $false,  # 使用开发镜像
    [switch]$Stop = $false,  # 停止容器
    [switch]$Logs = $false,  # 查看日志
    [switch]$Shell = $false  # 进入容器 shell
)

# 设置路径（根据你的实际路径修改）
$BaseDir = "I:\yc_research"
$HFCache = "$BaseDir\docker_data\huggingface"
$ModelsDir = "$BaseDir\docker_data\models"
$DataDir = "$BaseDir\docker_data\data"
$OutputsDir = "$BaseDir\docker_data\outputs"
$SglangDir = "$BaseDir\sglang"

# 获取 HuggingFace Token
$HF_TOKEN = $env:HF_TOKEN
if (-not $HF_TOKEN) {
    Write-Host "警告: 未设置 HF_TOKEN 环境变量" -ForegroundColor Yellow
    Write-Host "某些模型可能需要 HuggingFace token" -ForegroundColor Yellow
    $HF_TOKEN = ""
}

# 如果使用开发镜像
if ($Dev) {
    $Image = "lmsysorg/sglang:dev"
}

# 停止容器
if ($Stop) {
    Write-Host "停止容器: $ContainerName" -ForegroundColor Yellow
    docker stop $ContainerName 2>$null
    docker rm $ContainerName 2>$null
    Write-Host "容器已停止" -ForegroundColor Green
    exit 0
}

# 查看日志
if ($Logs) {
    Write-Host "查看容器日志: $ContainerName" -ForegroundColor Green
    docker logs -f $ContainerName
    exit 0
}

# 进入容器 shell
if ($Shell) {
    Write-Host "进入容器 shell: $ContainerName" -ForegroundColor Green
    docker exec -it $ContainerName /bin/bash
    exit 0
}

# 检查容器是否已存在
$existing = docker ps -a --filter "name=$ContainerName" --format "{{.Names}}"
if ($existing -eq $ContainerName) {
    Write-Host "容器 $ContainerName 已存在，正在停止并删除..." -ForegroundColor Yellow
    docker stop $ContainerName 2>$null
    docker rm $ContainerName 2>$null
}

# 创建目录（如果不存在）
Write-Host "检查并创建必要的目录..." -ForegroundColor Green
@($HFCache, $ModelsDir, $DataDir, $OutputsDir) | ForEach-Object {
    if (-not (Test-Path $_)) {
        New-Item -ItemType Directory -Force -Path $_ | Out-Null
        Write-Host "  创建目录: $_" -ForegroundColor Gray
    }
}

# 构建 Docker 运行命令
Write-Host "`n启动 SGLang 容器..." -ForegroundColor Green
Write-Host "  容器名: $ContainerName" -ForegroundColor Yellow
Write-Host "  镜像: $Image" -ForegroundColor Yellow
Write-Host "  模型: $Model" -ForegroundColor Yellow
Write-Host "  端口: $Port" -ForegroundColor Yellow
Write-Host ""

$dockerArgs = @(
    "run", "-d",
    "--name", $ContainerName,
    "--gpus", "all",
    "--shm-size", "32g",
    "-p", "${Port}:30000",
    "-v", "${HFCache}:/root/.cache/huggingface",
    "-v", "${ModelsDir}:/models",
    "-v", "${DataDir}:/data",
    "-v", "${OutputsDir}:/outputs"
)

# 如果是开发模式，挂载源码目录
if ($Dev) {
    $dockerArgs += "-v", "${SglangDir}:/sgl-workspace/sglang"
}

# 添加环境变量
if ($HF_TOKEN) {
    $dockerArgs += "--env", "HF_TOKEN=$HF_TOKEN"
}

$dockerArgs += @(
    "--ipc", "host",
    $Image,
    "python3", "-m", "sglang.launch_server",
    "--model-path", $Model,
    "--host", "0.0.0.0",
    "--port", "30000"
)

# 运行 Docker 命令
docker @dockerArgs

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n容器启动成功！" -ForegroundColor Green
    Write-Host "  查看日志: docker logs -f $ContainerName" -ForegroundColor Cyan
    Write-Host "  或运行: .\run_sglang.ps1 -Logs" -ForegroundColor Cyan
    Write-Host "  进入容器: docker exec -it $ContainerName /bin/bash" -ForegroundColor Cyan
    Write-Host "  或运行: .\run_sglang.ps1 -Shell" -ForegroundColor Cyan
    Write-Host "  停止容器: .\run_sglang.ps1 -Stop" -ForegroundColor Cyan
    Write-Host "`n等待服务启动（约 30-60 秒）..." -ForegroundColor Yellow
    
    # 等待并检查健康状态
    $maxWait = 120  # 最多等待 120 秒
    $waited = 0
    while ($waited -lt $maxWait) {
        Start-Sleep -Seconds 5
        $waited += 5
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:$Port/health" -TimeoutSec 2 -ErrorAction SilentlyContinue
            if ($response.StatusCode -eq 200) {
                Write-Host "`n✅ 服务已就绪！" -ForegroundColor Green
                Write-Host "  API 地址: http://localhost:$Port" -ForegroundColor Cyan
                break
            }
        } catch {
            Write-Host "." -NoNewline -ForegroundColor Gray
        }
    }
    
    if ($waited -ge $maxWait) {
        Write-Host "`n⚠️  服务可能还在启动中，请稍后检查日志" -ForegroundColor Yellow
    }
} else {
    Write-Host "`n❌ 容器启动失败！" -ForegroundColor Red
    Write-Host "请检查错误信息并确保:" -ForegroundColor Yellow
    Write-Host "  1. Docker 正在运行" -ForegroundColor Yellow
    Write-Host "  2. 有足够的磁盘空间" -ForegroundColor Yellow
    Write-Host "  3. 端口 $Port 未被占用" -ForegroundColor Yellow
    Write-Host "  4. GPU 驱动已正确安装（如果使用 GPU）" -ForegroundColor Yellow
}

