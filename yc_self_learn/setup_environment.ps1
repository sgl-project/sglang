# SGLang 环境配置检查脚本
# 用于检查并配置 SGLang 开发环境

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "SGLang 环境配置检查" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$allGood = $true

# 检查 1: Docker
Write-Host "1. 检查 Docker..." -ForegroundColor Yellow
try {
    $dockerVersion = docker --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   ✅ Docker 已安装: $dockerVersion" -ForegroundColor Green
        
        # 检查 Docker 是否运行
        try {
            docker ps 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) {
                Write-Host "   ✅ Docker 服务正在运行" -ForegroundColor Green
            } else {
                Write-Host "   ⚠️  Docker 服务未运行，请启动 Docker Desktop" -ForegroundColor Yellow
                $allGood = $false
            }
        } catch {
            Write-Host "   ⚠️  无法连接到 Docker 服务" -ForegroundColor Yellow
            $allGood = $false
        }
    } else {
        Write-Host "   ❌ Docker 未安装" -ForegroundColor Red
        Write-Host "      请访问: https://www.docker.com/products/docker-desktop/" -ForegroundColor Gray
        $allGood = $false
    }
} catch {
    Write-Host "   ❌ Docker 未安装或不在 PATH 中" -ForegroundColor Red
    $allGood = $false
}

Write-Host ""

# 检查 2: Python
Write-Host "2. 检查 Python..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   ✅ Python 已安装: $pythonVersion" -ForegroundColor Green
        
        # 检查 Python 版本
        $versionStr = $pythonVersion -replace "Python ", ""
        $versionParts = $versionStr -split "\."
        $majorVersion = [int]$versionParts[0]
        $minorVersion = [int]$versionParts[1]
        
        if ($majorVersion -ge 3 -and $minorVersion -ge 8) {
            Write-Host "   ✅ Python 版本符合要求 (>= 3.8)" -ForegroundColor Green
        } else {
            Write-Host "   ⚠️  Python 版本可能过低，推荐 3.8+" -ForegroundColor Yellow
        }
    } else {
        Write-Host "   ❌ Python 未安装" -ForegroundColor Red
        Write-Host "      请访问: https://www.python.org/downloads/" -ForegroundColor Gray
        $allGood = $false
    }
} catch {
    Write-Host "   ❌ Python 未安装或不在 PATH 中" -ForegroundColor Red
    $allGood = $false
}

Write-Host ""

# 检查 3: pip
Write-Host "3. 检查 pip..." -ForegroundColor Yellow
try {
    $pipVersion = pip --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   ✅ pip 已安装: $pipVersion" -ForegroundColor Green
    } else {
        Write-Host "   ❌ pip 未安装" -ForegroundColor Red
        $allGood = $false
    }
} catch {
    Write-Host "   ❌ pip 未安装" -ForegroundColor Red
    $allGood = $false
}

Write-Host ""

# 检查 4: Git
Write-Host "4. 检查 Git..." -ForegroundColor Yellow
try {
    $gitVersion = git --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   ✅ Git 已安装: $gitVersion" -ForegroundColor Green
    } else {
        Write-Host "   ⚠️  Git 未安装（可选，但推荐）" -ForegroundColor Yellow
    }
} catch {
    Write-Host "   ⚠️  Git 未安装（可选，但推荐）" -ForegroundColor Yellow
}

Write-Host ""

# 检查 5: NVIDIA GPU (可选)
Write-Host "5. 检查 NVIDIA GPU..." -ForegroundColor Yellow
try {
    $nvidiaSmi = nvidia-smi 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   ✅ 检测到 NVIDIA GPU" -ForegroundColor Green
        $gpuInfo = ($nvidiaSmi | Select-String "NVIDIA-SMI" | Select-Object -First 1).Line
        Write-Host "      $gpuInfo" -ForegroundColor Gray
        
        # 检查 NVIDIA Docker 支持
        try {
            docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) {
                Write-Host "   ✅ NVIDIA Docker 支持已配置" -ForegroundColor Green
            } else {
                Write-Host "   ⚠️  NVIDIA Docker 支持未配置" -ForegroundColor Yellow
                Write-Host "      请参考: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html" -ForegroundColor Gray
            }
        } catch {
            Write-Host "   ⚠️  无法测试 NVIDIA Docker 支持" -ForegroundColor Yellow
        }
    } else {
        Write-Host "   ℹ️  未检测到 NVIDIA GPU（CPU 模式仍可使用）" -ForegroundColor Cyan
    }
} catch {
    Write-Host "   ℹ️  未检测到 NVIDIA GPU（CPU 模式仍可使用）" -ForegroundColor Cyan
}

Write-Host ""

# 检查 6: 目录结构
Write-Host "6. 检查目录结构..." -ForegroundColor Yellow
$baseDir = "I:\yc_research"
$requiredDirs = @(
    "$baseDir\sglang",
    "$baseDir\docker_data\huggingface",
    "$baseDir\docker_data\models",
    "$baseDir\docker_data\data",
    "$baseDir\docker_data\outputs"
)

$dirsOk = $true
foreach ($dir in $requiredDirs) {
    if (Test-Path $dir) {
        Write-Host "   ✅ $dir" -ForegroundColor Green
    } else {
        Write-Host "   ❌ $dir (不存在)" -ForegroundColor Red
        $dirsOk = $false
        $allGood = $false
    }
}

if (-not $dirsOk) {
    Write-Host "`n   正在创建缺失的目录..." -ForegroundColor Yellow
    foreach ($dir in $requiredDirs) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Force -Path $dir | Out-Null
            Write-Host "   ✅ 已创建: $dir" -ForegroundColor Green
        }
    }
}

Write-Host ""

# 检查 7: HuggingFace Token
Write-Host "7. 检查 HuggingFace Token..." -ForegroundColor Yellow
$hfToken = $env:HF_TOKEN
if ($hfToken) {
    Write-Host "   ✅ HF_TOKEN 环境变量已设置" -ForegroundColor Green
} else {
    Write-Host "   ⚠️  HF_TOKEN 环境变量未设置" -ForegroundColor Yellow
    Write-Host "      某些模型可能需要 HuggingFace token" -ForegroundColor Gray
    Write-Host "      获取 token: https://huggingface.co/settings/tokens" -ForegroundColor Gray
    Write-Host "      设置方法: `$env:HF_TOKEN = 'your-token-here'" -ForegroundColor Gray
}

Write-Host ""

# 检查 8: Docker 镜像
Write-Host "8. 检查 SGLang Docker 镜像..." -ForegroundColor Yellow
try {
    $images = docker images lmsysorg/sglang --format "{{.Tag}}" 2>&1
    if ($images) {
        Write-Host "   ✅ 已找到 SGLang 镜像:" -ForegroundColor Green
        foreach ($tag in $images) {
            Write-Host "      - lmsysorg/sglang:$tag" -ForegroundColor Gray
        }
    } else {
        Write-Host "   ⚠️  SGLang 镜像未下载" -ForegroundColor Yellow
        Write-Host "      运行: docker pull lmsysorg/sglang:latest" -ForegroundColor Gray
    }
} catch {
    Write-Host "   ⚠️  无法检查 Docker 镜像" -ForegroundColor Yellow
}

Write-Host ""

# 总结
Write-Host "========================================" -ForegroundColor Cyan
if ($allGood) {
    Write-Host "✅ 环境检查完成！基本环境已就绪。" -ForegroundColor Green
    Write-Host ""
    Write-Host "下一步:" -ForegroundColor Yellow
    Write-Host "  1. 如果 Docker 镜像未下载，运行: docker pull lmsysorg/sglang:latest" -ForegroundColor Cyan
    Write-Host "  2. 设置 HuggingFace token: `$env:HF_TOKEN = 'your-token'" -ForegroundColor Cyan
    Write-Host "  3. 启动 SGLang: cd yc_self_learn\docker_scripts; .\run_sglang.ps1" -ForegroundColor Cyan
} else {
    Write-Host "⚠️  环境检查发现问题，请先解决上述问题" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "建议的安装顺序:" -ForegroundColor Yellow
    Write-Host "  1. 安装 Docker Desktop: https://www.docker.com/products/docker-desktop/" -ForegroundColor Cyan
    Write-Host "  2. 安装 Python 3.8+: https://www.python.org/downloads/" -ForegroundColor Cyan
    Write-Host "  3. 配置 NVIDIA Docker（如果有 GPU）" -ForegroundColor Cyan
    Write-Host "  4. 重新运行此脚本检查" -ForegroundColor Cyan
}
Write-Host "========================================" -ForegroundColor Cyan

