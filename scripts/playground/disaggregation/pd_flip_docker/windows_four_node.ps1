param(
    [ValidateSet(
        "preflight",
        "write-env",
        "sync-env",
        "sync-code",
        "clone",
        "pull",
        "start-workers",
        "start-router",
        "start-monitor",
        "start-all",
        "status",
        "logs",
        "stop"
    )]
    [string]$Action = "preflight",
    [string]$RemoteRepo = "/root/sglang",
    [string]$RepoUrl = "https://github.com/TianciJ/sglang.git",
    [string]$EnvFile = "",
    [string]$Image = "sglang-pd-switch:tianciJ",
    [string]$ModelPath = "/models/deepseek_v3.1_terminus",
    [string]$ModelId = "deepseek_v3.1_terminus",
    [string]$TokenizerPath = "",
    [string]$Node0Url = "http://192.168.0.42:30000",
    [string]$Node1Url = "http://192.168.0.40:30000",
    [string]$Node2Url = "http://192.168.0.39:30000",
    [string]$Node3Url = "http://192.168.0.41:30000",
    [int]$TpSize = 8,
    [int]$DpSize = 1,
    [int]$Port = 30000,
    [int]$BootstrapPort = 8998,
    [double]$MemFractionStatic = 0.88,
    [string]$TransferBackend = "mooncake",
    [string]$IbDevice = "mlx5_0",
    [string]$MooncakeMaster = "10.0.0.10:50051",
    [string]$MooncakeMetadataServer = "http://10.0.0.10:8080/metadata",
    [string]$AdminApiKey = "",
    [string]$RouterHost = "127.0.0.1",
    [int]$RouterPort = 8000,
    [double]$TtftSloSeconds = 0.2,
    [double]$TpotSloSeconds = 0.02,
    [int]$PDFlipWindowSeconds = 30,
    [double]$PDFlipEnterThreshold = 0.9,
    [double]$PDFlipExitThreshold = 0.95,
    [double]$PDFlipCommitThreshold = 0.9,
    [string]$ExtraSGLangArgs = "--trust-remote-code --enable-metrics",
    [string]$ExtraDockerArgs = "",
    [string]$ExtraRouterArgs = "",
    [string]$RouterCratesMirrorUrl = "sparse+https://rsproxy.cn/index/",
    [string]$RouterDynamoTarballFallback = "1",
    [int]$MonitorIterations = 120,
    [int]$MonitorPollInterval = 1
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($AdminApiKey) -or $AdminApiKey -match "^(replace-with-|changeme|CHANGE_ME)") {
    throw "-AdminApiKey must be a non-placeholder secret; refusing unsafe orchestration"
}

$Nodes = @(
    @{ Name = "node0"; Host = "cloud-099"; Role = "prefill"; Session = "pd-node0" },
    @{ Name = "node1"; Host = "cloud-100"; Role = "decode"; Session = "pd-node1" },
    @{ Name = "node2"; Host = "cloud-101"; Role = "decode"; Session = "pd-node2" },
    @{ Name = "node3"; Host = "cloud-102"; Role = "decode"; Session = "pd-node3" }
)

$ControllerHost = "cloud-099"
$DockerRel = "scripts/playground/disaggregation/pd_flip_docker"
$RemoteDockerDir = "$RemoteRepo/$DockerRel"
$RemoteEnvFile = "$RemoteDockerDir/env.local"

if ([string]::IsNullOrWhiteSpace($EnvFile)) {
    $EnvFile = Join-Path $PSScriptRoot "env.local"
}
if ([string]::IsNullOrWhiteSpace($TokenizerPath)) {
    $TokenizerPath = $ModelPath
}

function Invoke-Remote {
    param(
        [string]$HostName,
        [string]$Command
    )

    Write-Host "[$HostName] $Command"
    ssh $HostName $Command
    if ($LASTEXITCODE -ne 0) {
        throw "ssh command failed on $HostName with exit code $LASTEXITCODE"
    }
}

function Quote-ShValue {
    param([string]$Value)
    if ([string]::IsNullOrEmpty($Value)) {
        return ""
    }
    if ($Value -match "^[A-Za-z0-9_./:@%+=,-]+$") {
        return $Value
    }
    return "'" + $Value.Replace("'", "'\''") + "'"
}

function Write-EnvFile {
    $lines = @(
        "SGLANG_REPO=$RemoteRepo",
        "IMAGE=$Image",
        "",
        "MODEL_PATH=$ModelPath",
        "MODEL_ID=$ModelId",
        "TOKENIZER_PATH=$TokenizerPath",
        "",
        "TP_SIZE=$TpSize",
        "DP_SIZE=$DpSize",
        "PORT=$Port",
        "BOOTSTRAP_PORT=$BootstrapPort",
        "MEM_FRACTION_STATIC=$MemFractionStatic",
        "",
        "TRANSFER_BACKEND=$TransferBackend",
        "IB_DEVICE=$IbDevice",
        "MOONCAKE_MASTER=$MooncakeMaster",
        "MOONCAKE_TE_META_DATA_SERVER=$MooncakeMetadataServer",
        "MOONCAKE_GLOBAL_SEGMENT_SIZE=0",
        "MOONCAKE_PROTOCOL=rdma",
        "MOONCAKE_DEVICE=$IbDevice",
        "MOONCAKE_STORE_PORT=8081",
        "HICACHE_STORAGE_BACKEND=mooncake",
        "HICACHE_WRITE_POLICY=write_through",
        "",
        "NODE0_HOST=cloud-099",
        "NODE1_HOST=cloud-100",
        "NODE2_HOST=cloud-101",
        "NODE3_HOST=cloud-102",
        "NODE0_ROLE=prefill",
        "NODE1_ROLE=decode",
        "NODE2_ROLE=decode",
        "NODE3_ROLE=decode",
        "",
        "ROUTER_HOST=$RouterHost",
        "ROUTER_PORT=$RouterPort",
        "NODE0=$Node0Url",
        "NODE1=$Node1Url",
        "NODE2=$Node2Url",
        "NODE3=$Node3Url",
        "",
        "TTFT_SLO_SECONDS=$TtftSloSeconds",
        "TPOT_SLO_SECONDS=$TpotSloSeconds",
        "PD_FLIP_WINDOW_SECONDS=$PDFlipWindowSeconds",
        "PD_FLIP_ENTER_THRESHOLD=$PDFlipEnterThreshold",
        "PD_FLIP_EXIT_THRESHOLD=$PDFlipExitThreshold",
        "PD_FLIP_COMMIT_THRESHOLD=$PDFlipCommitThreshold",
        "PD_FLIP_MONITOR_ITERATIONS=$MonitorIterations",
        "PD_FLIP_MONITOR_POLL_INTERVAL=$MonitorPollInterval",
        "PD_FLIP_FIRST_MIGRATION_RATIO=0.5",
        "PD_FLIP_OBSERVATION_SECONDS=10",
        "PD_FLIP_SLO_THRESHOLD=0.9",
        "PD_FLIP_MIN_PREFILL_SLO_SAMPLES=20",
        "PD_FLIP_MIN_DECODE_SLO_SAMPLES=20",
        "ADMIN_API_KEY=$(Quote-ShValue $AdminApiKey)",
        "PD_FLIP_ROUTER_ADMIN_API_KEY=$(Quote-ShValue $AdminApiKey)",
        "PD_FLIP_ARTIFACT_DIR=/sgl-workspace/sglang/pd-flip-artifacts/four-node-progressive",
        "",
        "EXTRA_SGLANG_ARGS=$(Quote-ShValue $ExtraSGLangArgs)",
        "EXTRA_DOCKER_ARGS=$(Quote-ShValue $ExtraDockerArgs)",
        "EXTRA_ROUTER_ARGS=$(Quote-ShValue $ExtraRouterArgs)",
        "",
        "ROUTER_CRATES_MIRROR_URL=$(Quote-ShValue $RouterCratesMirrorUrl)",
        "ROUTER_DYNAMO_TARBALL_FALLBACK=$(Quote-ShValue $RouterDynamoTarballFallback)"
    )
    Write-Host "[local] writing $EnvFile"
    [System.IO.File]::WriteAllText(
        $EnvFile,
        (($lines -join "`n") + "`n"),
        [System.Text.Encoding]::ASCII
    )
}

function Copy-EnvFile {
    if (!(Test-Path $EnvFile)) {
        throw "EnvFile not found: $EnvFile"
    }

    foreach ($node in $Nodes) {
        Invoke-Remote $node.Host "mkdir -p $RemoteDockerDir"
        $target = "$($node.Host):$RemoteEnvFile"
        Write-Host "[$($node.Host)] scp $EnvFile -> $RemoteEnvFile"
        scp $EnvFile $target
        if ($LASTEXITCODE -ne 0) {
            throw "scp env.local failed for $($node.Host)"
        }
        Invoke-Remote $node.Host "sed -i 's/\r$//' $RemoteEnvFile"
    }
}

function Sync-CodeArchive {
    $localRepo = (Resolve-Path (Join-Path $PSScriptRoot "../../../..")).Path
    $archive = Join-Path $env:TEMP "sglang-pd-flip-sync.tgz"
    if (Test-Path $archive) {
        Remove-Item $archive -Force
    }

    Write-Host "[local] packing $localRepo -> $archive"
    Push-Location $localRepo
    try {
        tar -czf $archive `
            --exclude .git `
            --exclude .pytest_cache `
            --exclude __pycache__ `
            --exclude "*.pyc" `
            .
        if ($LASTEXITCODE -ne 0) {
            throw "local tar failed with exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }

    foreach ($node in $Nodes) {
        $remoteArchive = "/tmp/sglang-pd-flip-sync.tgz"
        Write-Host "[$($node.Host)] scp $archive -> $remoteArchive"
        scp $archive "$($node.Host):$remoteArchive"
        if ($LASTEXITCODE -ne 0) {
            throw "scp code archive failed for $($node.Host)"
        }

        $cmd = "rm -rf $RemoteRepo && mkdir -p $RemoteRepo && tar -xzf $remoteArchive -C $RemoteRepo && find $RemoteRepo/scripts/playground/disaggregation/pd_flip_docker -type f \( -name '*.sh' -o -name '*.ps1' \) -exec sed -i 's/\r$//' {} + && chmod +x $RemoteDockerDir/*.sh"
        Invoke-Remote $node.Host $cmd
    }
}

function Invoke-Preflight {
    foreach ($node in $Nodes) {
        $cmd = "hostname; echo gpu_count=`$(nvidia-smi -L | wc -l); docker version --format '{{.Client.Version}}/{{.Server.Version}}' || docker -v; if test -d $RemoteRepo/.git && command -v git >/dev/null 2>&1; then git -C $RemoteRepo rev-parse --short HEAD; elif test -f $RemoteDockerDir/run_worker.sh; then echo repo_present_no_git=$RemoteRepo; else echo repo_missing=$RemoteRepo; fi; test -f $RemoteEnvFile && echo env_local=ok || echo env_local=missing"
        Invoke-Remote $node.Host $cmd
    }
}

function Invoke-Clone {
    foreach ($node in $Nodes) {
        $cmd = "if [ -d $RemoteRepo/.git ]; then git -C $RemoteRepo fetch --all --prune && git -C $RemoteRepo pull --ff-only; else mkdir -p `$(dirname $RemoteRepo) && git clone $RepoUrl $RemoteRepo; fi"
        Invoke-Remote $node.Host $cmd
    }
}

function Invoke-Pull {
    foreach ($node in $Nodes) {
        Invoke-Remote $node.Host "git -C $RemoteRepo fetch --all --prune && git -C $RemoteRepo pull --ff-only"
    }
}

function Start-Workers {
    foreach ($node in $Nodes) {
        $run = "ENV_FILE=$RemoteEnvFile ./run_worker.sh $($node.Role) 0.0.0.0"
        $cmd = "cd $RemoteDockerDir && if command -v tmux >/dev/null 2>&1; then tmux kill-session -t $($node.Session) 2>/dev/null || true; ENV_FILE=$RemoteEnvFile tmux new -d -s $($node.Session) './run_worker.sh $($node.Role) 0.0.0.0 |& tee worker.log'; else if [ -f $($node.Session).pid ]; then kill `$(cat $($node.Session).pid) 2>/dev/null || true; rm -f $($node.Session).pid; fi; nohup bash -lc '$run' > worker.log 2>&1 & echo `$! > $($node.Session).pid; fi"
        Invoke-Remote $node.Host $cmd
    }
}

function Start-Router {
    $run = "ENV_FILE=$RemoteEnvFile ./run_router.sh"
    $cmd = "cd $RemoteDockerDir && if command -v tmux >/dev/null 2>&1; then tmux kill-session -t pd-router 2>/dev/null || true; ENV_FILE=$RemoteEnvFile tmux new -d -s pd-router './run_router.sh |& tee router.log'; else if [ -f pd-router.pid ]; then kill `$(cat pd-router.pid) 2>/dev/null || true; rm -f pd-router.pid; fi; nohup bash -lc '$run' > router.log 2>&1 & echo `$! > pd-router.pid; fi"
    Invoke-Remote $ControllerHost $cmd
}

function Start-Monitor {
    $run = "ENV_FILE=$RemoteEnvFile PD_FLIP_MONITOR_ITERATIONS=$MonitorIterations PD_FLIP_MONITOR_POLL_INTERVAL=$MonitorPollInterval ./run_controller.sh monitor"
    $cmd = "cd $RemoteDockerDir && if command -v tmux >/dev/null 2>&1; then tmux kill-session -t pd-monitor 2>/dev/null || true; ENV_FILE=$RemoteEnvFile tmux new -d -s pd-monitor 'PD_FLIP_MONITOR_ITERATIONS=$MonitorIterations PD_FLIP_MONITOR_POLL_INTERVAL=$MonitorPollInterval ./run_controller.sh monitor |& tee monitor.log'; else if [ -f pd-monitor.pid ]; then kill `$(cat pd-monitor.pid) 2>/dev/null || true; rm -f pd-monitor.pid; fi; nohup bash -lc '$run' > monitor.log 2>&1 & echo `$! > pd-monitor.pid; fi"
    Invoke-Remote $ControllerHost $cmd
}

function Invoke-Status {
    foreach ($node in $Nodes) {
        Invoke-Remote $node.Host "source $RemoteEnvFile && if command -v tmux >/dev/null 2>&1; then tmux ls || true; else cd $RemoteDockerDir && for f in *.pid; do [ -f `$f ] && echo `$f=`$(cat `$f); done; fi; curl -fsS -H `"Authorization: Bearer `${ADMIN_API_KEY}`" http://127.0.0.1:30000/pd_flip/runtime_role/status"
    }
    Invoke-Remote $ControllerHost "source $RemoteEnvFile && curl -fsS -H `"Authorization: Bearer `${PD_FLIP_ROUTER_ADMIN_API_KEY}`" http://127.0.0.1:`${ROUTER_PORT}/pd_flip/router/workers"
}

function Invoke-Logs {
    foreach ($node in $Nodes) {
        Invoke-Remote $node.Host "cd $RemoteDockerDir && echo === worker.log === && tail -n 120 worker.log || true"
    }
    Invoke-Remote $ControllerHost "cd $RemoteDockerDir && echo === router.log === && tail -n 120 router.log || true; echo === monitor.log === && tail -n 120 monitor.log || true"
}

function Stop-Cluster {
    foreach ($node in $Nodes) {
        Invoke-Remote $node.Host "if command -v tmux >/dev/null 2>&1; then tmux kill-session -t $($node.Session) 2>/dev/null || true; else cd $RemoteDockerDir && if [ -f $($node.Session).pid ]; then kill `$(cat $($node.Session).pid) 2>/dev/null || true; rm -f $($node.Session).pid; fi; fi"
    }
    Invoke-Remote $ControllerHost "if command -v tmux >/dev/null 2>&1; then tmux kill-session -t pd-monitor 2>/dev/null || true; tmux kill-session -t pd-router 2>/dev/null || true; else cd $RemoteDockerDir && for f in pd-monitor.pid pd-router.pid; do if [ -f `$f ]; then kill `$(cat `$f) 2>/dev/null || true; rm -f `$f; fi; done; fi"
}

switch ($Action) {
    "preflight" { Invoke-Preflight }
    "write-env" { Write-EnvFile }
    "sync-env" { Copy-EnvFile }
    "sync-code" { Sync-CodeArchive }
    "clone" { Invoke-Clone }
    "pull" { Invoke-Pull }
    "start-workers" { Start-Workers }
    "start-router" { Start-Router }
    "start-monitor" { Start-Monitor }
    "start-all" {
        Start-Workers
        Start-Sleep -Seconds 20
        Start-Router
        Start-Sleep -Seconds 5
        Start-Monitor
    }
    "status" { Invoke-Status }
    "logs" { Invoke-Logs }
    "stop" { Stop-Cluster }
}
