# 创建和配置 SGLang Conda 环境脚本
# 环境名称: sglang2026

$envName = "sglang2026"
$pythonVersion = "3.10"  # 推荐使用 Python 3.10

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "创建 SGLang Conda 环境" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 检查 conda 是否安装
Write-Host "1. 检查 Conda..." -ForegroundColor Yellow
try {
    $condaVersion = conda --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   ✅ Conda 已安装: $condaVersion" -ForegroundColor Green
    } else {
        Write-Host "   ❌ Conda 未安装或不在 PATH 中" -ForegroundColor Red
        Write-Host "      请先安装 Anaconda 或 Miniconda" -ForegroundColor Yellow
        Write-Host "      Anaconda: https://www.anaconda.com/download" -ForegroundColor Gray
        Write-Host "      Miniconda: https://docs.conda.io/en/latest/miniconda.html" -ForegroundColor Gray
        exit 1
    }
} catch {
    Write-Host "   ❌ Conda 未安装" -ForegroundColor Red
    Write-Host "      请先安装 Anaconda 或 Miniconda" -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# 检查环境是否已存在
Write-Host "2. 检查环境是否已存在..." -ForegroundColor Yellow
$envExists = conda env list | Select-String $envName
if ($envExists) {
    Write-Host "   ⚠️  环境 '$envName' 已存在" -ForegroundColor Yellow
    $response = Read-Host "   是否删除并重新创建? (y/N)"
    if ($response -eq "y" -or $response -eq "Y") {
        Write-Host "   正在删除旧环境..." -ForegroundColor Yellow
        conda env remove -n $envName -y
        Write-Host "   ✅ 旧环境已删除" -ForegroundColor Green
    } else {
        Write-Host "   跳过创建，使用现有环境" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "激活环境: conda activate $envName" -ForegroundColor Cyan
        exit 0
    }
} else {
    Write-Host "   ✅ 环境不存在，将创建新环境" -ForegroundColor Green
}

Write-Host ""

# 创建环境
Write-Host "3. 创建 Conda 环境 '$envName' (Python $pythonVersion)..." -ForegroundColor Yellow
Write-Host "   这可能需要几分钟..." -ForegroundColor Gray

conda create -n $envName python=$pythonVersion -y

if ($LASTEXITCODE -ne 0) {
    Write-Host "   ❌ 环境创建失败" -ForegroundColor Red
    exit 1
}

Write-Host "   ✅ 环境创建成功" -ForegroundColor Green
Write-Host ""

# 激活环境并安装基础包
Write-Host "4. 安装基础依赖包..." -ForegroundColor Yellow

# 使用 conda run 在环境中执行命令
conda run -n $envName pip install --upgrade pip setuptools wheel

# 安装常用包
Write-Host "   安装 pip、setuptools、wheel..." -ForegroundColor Gray
conda run -n $envName pip install --upgrade pip setuptools wheel

Write-Host "   安装常用开发工具..." -ForegroundColor Gray
conda run -n $envName pip install requests openai

Write-Host "   ✅ 基础依赖安装完成" -ForegroundColor Green
Write-Host ""

# 询问是否安装 SGLang
Write-Host "5. 安装 SGLang..." -ForegroundColor Yellow
$installSglang = Read-Host "   是否现在安装 SGLang? (y/N)"
if ($installSglang -eq "y" -or $installSglang -eq "Y") {
    Write-Host "   正在安装 SGLang（这可能需要较长时间）..." -ForegroundColor Gray
    Write-Host "   注意: 如果使用 Docker，可能不需要在本地安装 SGLang" -ForegroundColor Yellow
    
    # 安装 SGLang（仅前端，用于调用 API）
    conda run -n $envName pip install "sglang[openai]"
    
    # 或者安装完整版本（如果需要本地运行）
    # conda run -n $envName pip install "sglang[all]>=0.5.3rc0"
    
    Write-Host "   ✅ SGLang 安装完成" -ForegroundColor Green
} else {
    Write-Host "   跳过 SGLang 安装（可以在需要时手动安装）" -ForegroundColor Cyan
}

Write-Host ""

# 创建激活脚本
Write-Host "6. 创建便捷脚本..." -ForegroundColor Yellow
$activateScript = @"
# 激活 SGLang 环境
# 使用方法: . .\activate_sglang2026.ps1

conda activate $envName
Write-Host "✅ 已激活环境: $envName" -ForegroundColor Green
Write-Host "Python 版本:" -ForegroundColor Cyan
python --version
Write-Host ""
Write-Host "已安装的包:" -ForegroundColor Cyan
pip list
"@

$activateScript | Out-File -FilePath "activate_sglang2026.ps1" -Encoding UTF8
Write-Host "   ✅ 已创建激活脚本: activate_sglang2026.ps1" -ForegroundColor Green

Write-Host ""

# 总结
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "✅ Conda 环境配置完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "使用方法:" -ForegroundColor Yellow
Write-Host "  1. 激活环境:" -ForegroundColor Cyan
Write-Host "     conda activate $envName" -ForegroundColor White
Write-Host ""
Write-Host "  2. 或使用便捷脚本:" -ForegroundColor Cyan
Write-Host "     . .\activate_sglang2026.ps1" -ForegroundColor White
Write-Host ""
Write-Host "  3. 查看已安装的包:" -ForegroundColor Cyan
Write-Host "     pip list" -ForegroundColor White
Write-Host ""
Write-Host "  4. 安装额外包:" -ForegroundColor Cyan
Write-Host "     pip install <package-name>" -ForegroundColor White
Write-Host ""
Write-Host "  5. 退出环境:" -ForegroundColor Cyan
Write-Host "     conda deactivate" -ForegroundColor White
Write-Host ""
Write-Host "注意:" -ForegroundColor Yellow
Write-Host "  - 如果使用 Docker 运行 SGLang，可能只需要安装客户端库" -ForegroundColor Gray
Write-Host "  - 运行 'pip install sglang[openai]' 即可调用 Docker 中的 SGLang API" -ForegroundColor Gray
Write-Host ""

