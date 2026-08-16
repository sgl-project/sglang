param(
    [string]$Python = "python",
    [string]$Wheelhouse = "",
    [string]$CudaArchitectures = "121a",
    [int]$BuildJobs = 2,
    [int]$NvccThreads = 2,
    [string]$BuildDir = "",
    [string]$OutDir = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$AotRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $OutDir) {
    $OutDir = Join-Path $AotRoot "dist"
}
$OutDir = [IO.Path]::GetFullPath($OutDir)
if (-not $BuildDir) {
    $BuildDir = Join-Path $env:TEMP "sgl-kernel-winarm64-build"
}
$BuildDir = [IO.Path]::GetFullPath($BuildDir)

$PythonCommand = Get-Command $Python -ErrorAction Stop
$PythonExe = $PythonCommand.Source
if (-not $PythonExe) {
    $PythonExe = $PythonCommand.Path
}

& $PythonExe -c "import platform, sysconfig; assert platform.machine().lower() == 'arm64', platform.machine(); assert sysconfig.get_platform() == 'win-arm64', sysconfig.get_platform()"
if ($LASTEXITCODE -ne 0) {
    throw "A native Windows ARM64 Python interpreter is required."
}

& $PythonExe -m pip --version *> $null
if ($LASTEXITCODE -ne 0) {
    & $PythonExe -m ensurepip --upgrade
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to bootstrap pip in the selected Python environment."
    }
}

try {
    & $PythonExe -c "import torch; assert torch.version.cuda is not None; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
    if ($LASTEXITCODE -ne 0) {
        throw "PyTorch validation failed."
    }
} catch {
    if (-not $Wheelhouse) {
        throw "CUDA-enabled Windows ARM64 PyTorch is required. Pass -Wheelhouse with a compatible torch wheel."
    }
    $TorchWheel = Get-ChildItem -LiteralPath $Wheelhouse -Filter "torch-*.whl" |
        Sort-Object Name -Descending |
        Select-Object -First 1
    if (-not $TorchWheel) {
        throw "No torch wheel was found in $Wheelhouse."
    }
    & $PythonExe -m pip install --pre $TorchWheel.FullName
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install $($TorchWheel.Name)."
    }
}

& $PythonExe -m pip install "build>=1.2" "scikit-build-core>=0.10" "cmake>=3.31,<4" ninja numpy wheel
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install build dependencies."
}

$TorchCmakePrefix = & $PythonExe -c "import torch; print(torch.utils.cmake_prefix_path)"
if ($LASTEXITCODE -ne 0 -or -not $TorchCmakePrefix) {
    throw "Could not determine the PyTorch CMake prefix."
}

$CudaPath = $env:CUDA_PATH
if (-not $CudaPath) {
    $Nvcc = Get-Command nvcc -ErrorAction Stop
    $CudaPath = Split-Path -Parent (Split-Path -Parent $Nvcc.Source)
}
if (-not (Test-Path (Join-Path $CudaPath "bin\nvcc.exe"))) {
    throw "CUDA Toolkit was not found under $CudaPath."
}

$Vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $Vswhere)) {
    throw "vswhere.exe was not found. Install Visual Studio Build Tools with ARM64 C++ support."
}
$VsPath = & $Vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.ARM64 -property installationPath
if (-not $VsPath) {
    throw "Visual Studio ARM64 C++ build tools were not found."
}
$Vcvars = Join-Path $VsPath "VC\Auxiliary\Build\vcvarsall.bat"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$CmakeArgs = @(
    "-DCMAKE_CUDA_ARCHITECTURES=OFF",
    "-DSGL_KERNEL_CUDA_ARCHITECTURES=$CudaArchitectures",
    "-DSGL_KERNEL_COMPILE_THREADS=$NvccThreads",
    "-DSGL_KERNEL_BUILD_SM90=OFF",
    "-DSGL_KERNEL_ENABLE_FA3=OFF",
    "-DSGL_KERNEL_BUILD_INFLLM=OFF",
    "-DSGL_KERNEL_BUILD_SPATIAL=OFF",
    "-DSGL_KERNEL_BUILD_FLASHMLA=OFF",
    "-DSGL_KERNEL_INSTALL_TRITON_KERNELS=OFF",
    "-DSGL_KERNEL_ENABLE_SPARSE_FLASH_ATTN=OFF",
    "-DSGL_KERNEL_ENABLE_CUTLASS_MLA=OFF",
    "-DSGL_KERNEL_ENABLE_EXTENDED_CUDA_OPS=OFF"
) -join " "

$BuildCommand = @(
    "set `"PATH=$(Split-Path -Parent $Vswhere);%PATH%`"",
    "call `"$Vcvars`" arm64",
    "set `"CUDA_PATH=$CudaPath`"",
    "set `"CMAKE_PREFIX_PATH=$TorchCmakePrefix`"",
    "set `"CMAKE_GENERATOR=Ninja`"",
    "set `"CMAKE_BUILD_PARALLEL_LEVEL=$BuildJobs`"",
    "set `"TORCH_CUDA_ARCH_LIST=12.1`"",
    "set `"CMAKE_ARGS=$CmakeArgs`"",
    "cd /d `"$AotRoot`"",
    "`"$PythonExe`" -m build --wheel --no-isolation -Cbuild-dir=`"$BuildDir`" --outdir `"$OutDir`""
) -join " && "

& $env:ComSpec /d /s /c $BuildCommand
if ($LASTEXITCODE -ne 0) {
    throw "Windows ARM64 sglang-kernel build failed with exit code $LASTEXITCODE."
}

$Wheels = @(Get-ChildItem -LiteralPath $OutDir -Filter "*win_arm64.whl")
if ($Wheels.Count -ne 1) {
    throw "Expected one Windows ARM64 wheel in $OutDir, found $($Wheels.Count)."
}
Write-Output "Built $($Wheels[0].FullName)"
