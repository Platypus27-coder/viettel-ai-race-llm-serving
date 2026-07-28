# ============================================================
# Viettel AI Race 2026 — Build & Push Scripts
# ============================================================
# PowerShell script for Windows

param(
    [Parameter(Mandatory=$false)]
    [string]$DockerHubUsername = "",

    [Parameter(Mandatory=$false)]
    [string]$ImageTag = "v1",

    [Parameter(Mandatory=$false)]
    [switch]$BuildOnly,

    [Parameter(Mandatory=$false)]
    [switch]$PushOnly
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $PSScriptRoot
$ImageName = "viettel-ai-vllm"

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  Viettel AI Race 2026 — Build & Push"
Write-Host "=========================================" -ForegroundColor Cyan

# Validate Docker Hub username
if (-not $DockerHubUsername) {
    $DockerHubUsername = Read-Host "Enter your Docker Hub username"
}

$FullImageName = "${DockerHubUsername}/${ImageName}:${ImageTag}"
Write-Host "  Image: $FullImageName" -ForegroundColor Yellow
Write-Host ""

# ---- Build ----
if (-not $PushOnly) {
    Write-Host "[1/3] Building Docker image..." -ForegroundColor Green

    docker build `
        -t $FullImageName `
        -f "$ProjectDir\Dockerfile" `
        "$ProjectDir"

    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ❌ Build failed!" -ForegroundColor Red
        exit 1
    }
    Write-Host "  ✅ Build successful!" -ForegroundColor Green

    # Show image size
    $imageSize = docker image inspect $FullImageName --format '{{.Size}}' | ForEach-Object { [math]::Round([int64]$_ / 1GB, 2) }
    Write-Host "  Image size: ${imageSize} GB" -ForegroundColor Yellow
}

# ---- Push ----
if (-not $BuildOnly) {
    Write-Host ""
    Write-Host "[2/3] Pushing to Docker Hub..." -ForegroundColor Green
    Write-Host "  Make sure you are logged in: docker login" -ForegroundColor Yellow

    docker push $FullImageName

    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ❌ Push failed! Try: docker login" -ForegroundColor Red
        exit 1
    }
    Write-Host "  ✅ Push successful!" -ForegroundColor Green

    # Get image digest
    $digest = docker inspect --format='{{index .RepoDigests 0}}' $FullImageName 2>$null
    Write-Host "  Digest: $digest" -ForegroundColor Yellow
}

# ---- Update docker-compose ----
Write-Host ""
Write-Host "[3/3] Summary" -ForegroundColor Green
Write-Host "  ─────────────────────────────────────"
Write-Host "  Image:    $FullImageName"
Write-Host "  ─────────────────────────────────────"
Write-Host ""
Write-Host "  To use this image, update docker-compose.yml:" -ForegroundColor Cyan
Write-Host "    image: $FullImageName" -ForegroundColor White
Write-Host ""
Write-Host "  To submit:" -ForegroundColor Cyan
Write-Host "    1. Update docker-compose.yml with the image above"
Write-Host "    2. Upload docker-compose.yml to the BTC portal"
Write-Host "    3. Pin the image digest for reproducibility"
Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
