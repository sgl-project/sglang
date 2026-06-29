param(
    [ValidateSet(
        "preflight",
        "sync-env",
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
    [string]$RemoteRepo = "/home/tiancij/sglang",
    [string]$EnvFile = "",
    [int]$MonitorIterations = 120,
    [int]$MonitorPollInterval = 1
)

$ErrorActionPreference = "Stop"

$Nodes = @(
    @{ Name = "node0"; Host = "cloud-099"; Role = "prefill"; Session = "pd-node0" },
    @{ Name = "node1"; Host = "cloud-100"; Role = "prefill"; Session = "pd-node1" },
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
    }
}

function Invoke-Preflight {
    foreach ($node in $Nodes) {
        Invoke-Remote $node.Host @"
hostname
echo gpu_count=`$(nvidia-smi -L | wc -l)
docker --version
test -d $RemoteRepo && git -C $RemoteRepo rev-parse --short HEAD || echo repo_missing=$RemoteRepo
test -f $RemoteEnvFile && echo env_local=ok || echo env_local=missing
"@
    }
}

function Invoke-Pull {
    foreach ($node in $Nodes) {
        Invoke-Remote $node.Host "git -C $RemoteRepo fetch --all --prune && git -C $RemoteRepo pull --ff-only"
    }
}

function Start-Workers {
    foreach ($node in $Nodes) {
        $cmd = "cd $RemoteDockerDir && tmux kill-session -t $($node.Session) 2>/dev/null || true; cd $RemoteDockerDir && ENV_FILE=$RemoteEnvFile tmux new -d -s $($node.Session) './run_worker.sh $($node.Role) 0.0.0.0 |& tee worker.log'"
        Invoke-Remote $node.Host $cmd
    }
}

function Start-Router {
    $cmd = "cd $RemoteDockerDir && tmux kill-session -t pd-router 2>/dev/null || true; cd $RemoteDockerDir && ENV_FILE=$RemoteEnvFile tmux new -d -s pd-router './run_router.sh |& tee router.log'"
    Invoke-Remote $ControllerHost $cmd
}

function Start-Monitor {
    $cmd = "cd $RemoteDockerDir && tmux kill-session -t pd-monitor 2>/dev/null || true; cd $RemoteDockerDir && ENV_FILE=$RemoteEnvFile tmux new -d -s pd-monitor 'PD_FLIP_MONITOR_ITERATIONS=$MonitorIterations PD_FLIP_MONITOR_POLL_INTERVAL=$MonitorPollInterval ./run_controller.sh monitor |& tee monitor.log'"
    Invoke-Remote $ControllerHost $cmd
}

function Invoke-Status {
    foreach ($node in $Nodes) {
        Invoke-Remote $node.Host "tmux ls || true; curl -fsS http://127.0.0.1:30000/pd_flip/runtime_role/status || true"
    }
    Invoke-Remote $ControllerHost "source $RemoteEnvFile && curl -fsS http://127.0.0.1:`${ROUTER_PORT}/pd_flip/router/workers || true"
}

function Invoke-Logs {
    foreach ($node in $Nodes) {
        Invoke-Remote $node.Host "cd $RemoteDockerDir && echo === worker.log === && tail -n 120 worker.log || true"
    }
    Invoke-Remote $ControllerHost "cd $RemoteDockerDir && echo === router.log === && tail -n 120 router.log || true; echo === monitor.log === && tail -n 120 monitor.log || true"
}

function Stop-Cluster {
    foreach ($node in $Nodes) {
        Invoke-Remote $node.Host "tmux kill-session -t $($node.Session) 2>/dev/null || true"
    }
    Invoke-Remote $ControllerHost "tmux kill-session -t pd-monitor 2>/dev/null || true; tmux kill-session -t pd-router 2>/dev/null || true"
}

switch ($Action) {
    "preflight" { Invoke-Preflight }
    "sync-env" { Copy-EnvFile }
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
