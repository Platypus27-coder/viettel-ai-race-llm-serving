# Viettel AI Race 2026 - local smoke-test runner.
# docker-compose.yml is the v6 incumbent and the only default artifact.

param(
    [Parameter(Mandatory = $false)]
    [string]$ComposeFile = "docker-compose.yml",

    [Parameter(Mandatory = $false)]
    [switch]$Detached,

    [Parameter(Mandatory = $false)]
    [switch]$Stop,

    [Parameter(Mandatory = $false)]
    [switch]$Benchmark,

    [Parameter(Mandatory = $false)]
    [switch]$TestAccuracy,

    [Parameter(Mandatory = $false)]
    [ValidateSet("quick", "gpqa")]
    [string]$AccuracyMode = "quick",

    [Parameter(Mandatory = $false)]
    [string]$BenchmarkOutput,

    [Parameter(Mandatory = $false)]
    [int]$HealthTimeoutSeconds = 120
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $PSScriptRoot

if ([IO.Path]::IsPathRooted($ComposeFile)) {
    $ComposeCandidate = $ComposeFile
} else {
    $ComposeCandidate = Join-Path $ProjectDir $ComposeFile
}
if (-not (Test-Path -LiteralPath $ComposeCandidate -PathType Leaf)) {
    throw "Compose file does not exist: $ComposeCandidate"
}
$ComposePath = (Resolve-Path -LiteralPath $ComposeCandidate).Path
$BaseUrl = "http://localhost:8000"
$TracePath = Join-Path $ProjectDir "019e649f-4e27-74db-82da-920f57b13786\grading-workload-spec.json"

function Wait-ForVllmHealth {
    param([int]$TimeoutSeconds)

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        try {
            $response = Invoke-WebRequest -Uri "$BaseUrl/health" -TimeoutSec 3 -UseBasicParsing
            if ($response.StatusCode -eq 200) {
                Write-Host "vLLM health check passed." -ForegroundColor Green
                return
            }
        } catch {
            # vLLM is still starting; retry until the deadline.
        }
        Start-Sleep -Seconds 2
    } while ((Get-Date) -lt $deadline)

    throw "vLLM did not become healthy within $TimeoutSeconds seconds. See: docker compose -f `"$ComposePath`" logs"
}

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host " Viettel AI Race 2026 - Local Runner" -ForegroundColor Cyan
Write-Host " Compose: $ComposePath" -ForegroundColor Yellow
Write-Host "=========================================" -ForegroundColor Cyan

if ($Stop) {
    docker compose -f $ComposePath down
    exit $LASTEXITCODE
}

if ($Benchmark) {
    if (-not $BenchmarkOutput) {
        $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
        $BenchmarkOutput = Join-Path $ProjectDir "benchmark\results\local-$stamp.json"
    }
    conda run -n viettel python "$ProjectDir\benchmark\benchmark_ers.py" `
        --trace $TracePath `
        --request-rate inf `
        --seed 42 `
        --runs 1 `
        --output $BenchmarkOutput
    exit $LASTEXITCODE
}

if ($TestAccuracy) {
    conda run -n viettel python "$ProjectDir\benchmark\test_accuracy.py" `
        --base-url $BaseUrl `
        --mode $AccuracyMode
    exit $LASTEXITCODE
}

docker compose -f $ComposePath config --quiet
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if ($Detached) {
    docker compose -f $ComposePath up -d
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    Wait-ForVllmHealth -TimeoutSeconds $HealthTimeoutSeconds
    Write-Host "Server started in the background." -ForegroundColor Green
    Write-Host "  Benchmark: .\scripts\run_local.ps1 -Benchmark" -ForegroundColor Yellow
    Write-Host "  Accuracy:  .\scripts\run_local.ps1 -TestAccuracy -AccuracyMode gpqa" -ForegroundColor Yellow
    Write-Host "  Stop:      .\scripts\run_local.ps1 -Stop" -ForegroundColor Yellow
    exit 0
}

Write-Host "Press Ctrl+C to stop." -ForegroundColor Yellow
docker compose -f $ComposePath up
