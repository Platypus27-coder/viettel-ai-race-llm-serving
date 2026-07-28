# ============================================================
# Viettel AI Race 2026 — Run locally for testing
# ============================================================
# PowerShell script

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("recovery", "tbt", "baseline")]
    [string]$Profile = "recovery",

    [Parameter(Mandatory=$false)]
    [switch]$Detached,

    [Parameter(Mandatory=$false)]
    [switch]$Stop,

    [Parameter(Mandatory=$false)]
    [switch]$Benchmark,

    [Parameter(Mandatory=$false)]
    [switch]$TestAccuracy
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $PSScriptRoot

# Map profile to compose file
$composeFiles = @{
    "recovery" = "docker-compose.yml"
    "tbt"      = "configs\docker-compose.slot-04-bf16-batch2048-seqs64.yml"
    "baseline" = "configs\docker-compose.baseline.yml"
}
$composeFile = Join-Path $ProjectDir $composeFiles[$Profile]

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  Viettel AI Race 2026 — Local Runner"
Write-Host "  Profile: $Profile"
Write-Host "=========================================" -ForegroundColor Cyan

if ($Stop) {
    Write-Host "Stopping containers..." -ForegroundColor Yellow
    docker compose -f $composeFile down
    exit 0
}

if ($Benchmark) {
    Write-Host "Running ERS benchmark..." -ForegroundColor Green
    conda run -n viettel python "$ProjectDir\benchmark\benchmark_ers.py" `
        --trace "$ProjectDir\019e649f-4e27-74db-82da-920f57b13786\grading-workload-spec.json" `
        --request-rate inf
    exit 0
}

if ($TestAccuracy) {
    Write-Host "Running accuracy test..." -ForegroundColor Green
    conda run -n viettel python "$ProjectDir\benchmark\test_accuracy.py" --mode quick
    exit 0
}

# Start server
Write-Host ""
Write-Host "Starting vLLM server with $Profile profile..." -ForegroundColor Green
Write-Host "Compose file: $composeFile" -ForegroundColor Yellow
Write-Host ""

if ($Detached) {
    docker compose -f $composeFile up -d
    Write-Host ""
    Write-Host "Server started in background." -ForegroundColor Green
    Write-Host "  Health check: curl http://localhost:8000/health" -ForegroundColor Yellow
    Write-Host "  View logs:    docker compose -f $composeFile logs -f" -ForegroundColor Yellow
    Write-Host "  Stop:         .\scripts\run_local.ps1 -Profile $Profile -Stop" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  Run benchmark: .\scripts\run_local.ps1 -Benchmark" -ForegroundColor Yellow
    Write-Host "  Test accuracy: .\scripts\run_local.ps1 -TestAccuracy" -ForegroundColor Yellow
} else {
    Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
    Write-Host ""
    docker compose -f $composeFile up
}
