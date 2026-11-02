# Windows pipeline runner with a memory guard (no `make` needed).
# Each step runs as its own process; if its process tree exceeds -MaxStepGB or
# free RAM drops below -MinFreeGB, the step is killed before the machine thrashes.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_pipeline.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\run_pipeline.ps1 -Steps reconcile,evaluate
param(
    [string[]]$Steps = @('ingest', 'dbt-run', 'features', 'baselines', 'train', 'reconcile', 'backtest', 'evaluate', 'test'),
    [double]$MaxStepGB = 6,
    [double]$MinFreeGB = 2.5,
    # RAM + page file headroom; when this runs out Windows freezes or crashes.
    [double]$MinCommitGB = 1.5,
    [int]$HistoryDays = 365
)

$ErrorActionPreference = 'Stop'
# `powershell -File ... -Steps a,b` passes "a,b" as a single string.
$Steps = $Steps -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ }
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$commands = [ordered]@{
    'ingest'    = @('python', 'src/ingest.py')
    'dbt-run'   = @('dbt', 'build --project-dir dbt_m5 --profiles-dir dbt_m5')
    'features'  = @('python', 'src/features.py')
    'baselines' = @('python', 'src/baselines.py')
    'train'     = @('python', "src/train_lgbm.py --history-days $HistoryDays")
    'reconcile' = @('python', 'src/reconcile.py')
    'backtest'  = @('python', "src/backtest.py --window 2 --history-days $HistoryDays", 'python', "src/backtest.py --window 1 --history-days $HistoryDays")
    'evaluate'  = @('python', 'src/metrics.py', 'python', 'src/inventory_sim.py', 'python', 'src/backtest.py --summarize')
    'test'      = @('python', '-m pytest tests/ -v')
}

# Keep temp files off the (often nearly full) system drive.
$tmp = Join-Path $Root 'data\tmp'
New-Item -ItemType Directory -Force $tmp, (Join-Path $Root 'logs') | Out-Null
$env:TEMP = $tmp
$env:TMP = $tmp
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'

function Get-TreeIds([int]$rootId) {
    $all = Get-CimInstance Win32_Process -Property ProcessId, ParentProcessId
    $ids = [System.Collections.Generic.List[int]]::new()
    $ids.Add($rootId)
    for ($i = 0; $i -lt $ids.Count; $i++) {
        foreach ($p in $all) { if ($p.ParentProcessId -eq $ids[$i]) { $ids.Add([int]$p.ProcessId) } }
    }
    return $ids
}

function Invoke-Guarded([string]$step, [string]$exe, [string]$argLine) {
    $log = Join-Path $Root "logs\$step.log"
    $err = Join-Path $Root "logs\$step.err.log"
    Write-Host "`n=== $step : $exe $argLine" -ForegroundColor Cyan
    $proc = Start-Process -FilePath $exe -ArgumentList $argLine -NoNewWindow -PassThru `
        -RedirectStandardOutput $log -RedirectStandardError $err
    $null = $proc.Handle  # without an open handle, ExitCode stays empty
    $peak = 0.0
    while (-not $proc.HasExited) {
        Start-Sleep -Seconds 2
        $ids = Get-TreeIds $proc.Id
        $usedGB = (Get-Process -Id $ids -ErrorAction SilentlyContinue |
            Measure-Object WorkingSet64 -Sum).Sum / 1GB
        $os = Get-CimInstance Win32_OperatingSystem
        $freeGB = $os.FreePhysicalMemory / 1MB
        $commitGB = $os.FreeVirtualMemory / 1MB
        $peak = [math]::Max($peak, $usedGB)
        if ($usedGB -gt $MaxStepGB -or $freeGB -lt $MinFreeGB -or $commitGB -lt $MinCommitGB) {
            Write-Host ("ABORT {0}: step using {1:N1} GB, {2:N1} GB RAM free, {3:N1} GB commit free (limits {4} / {5} / {6} GB)" -f `
                $step, $usedGB, $freeGB, $commitGB, $MaxStepGB, $MinFreeGB, $MinCommitGB) -ForegroundColor Red
            foreach ($id in ($ids | Sort-Object -Descending)) { Stop-Process -Id $id -Force -ErrorAction SilentlyContinue }
            return $false
        }
    }
    $proc.WaitForExit()
    Get-Content $log -Tail 15 | Out-Host
    $ok = $proc.ExitCode -eq 0
    $color = if ($ok) { 'Green' } else { 'Red' }
    Write-Host ("=== {0} exit={1} peak={2:N1} GB (log: logs\{0}.log)" -f $step, $proc.ExitCode, $peak) -ForegroundColor $color
    if (-not $ok) { Get-Content $err -Tail 30 | Out-Host }
    return $ok
}

foreach ($step in $Steps) {
    if (-not $commands.Contains($step)) { throw "Unknown step '$step'. Valid: $($commands.Keys -join ', ')" }
    $cmd = $commands[$step]
    for ($i = 0; $i -lt $cmd.Count; $i += 2) {
        $name = if ($cmd.Count -gt 2) { "$step-$($i / 2 + 1)" } else { $step }
        if (-not (Invoke-Guarded $name $cmd[$i] $cmd[$i + 1])) {
            Write-Host "Pipeline stopped at '$name'." -ForegroundColor Red
            exit 1
        }
    }
}
Write-Host "`nPipeline complete." -ForegroundColor Green
