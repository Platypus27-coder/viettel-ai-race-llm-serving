# Build one digest-pinned custom vLLM candidate without mutating the root
# portal artifact. The two modes are isolated: ShortConv-FP8 or draft-only
# speculative decoding. Push only after local build and preflight pass.

param(
    [Parameter(Mandatory = $false)]
    [string]$DockerHubUsername = "",

    [Parameter(Mandatory = $false)]
    [string]$ImageRepository = "viettel-ai-vllm",

    [Parameter(Mandatory = $false)]
    [ValidateSet("shortconv-fp8", "speculative-draft")]
    [string]$Variant = "shortconv-fp8",

    [Parameter(Mandatory = $false)]
    [string]$ImageTag = "",

    [Parameter(Mandatory = $false)]
    [switch]$BuildOnly,

    [Parameter(Mandatory = $false)]
    [switch]$PushOnly,

    [Parameter(Mandatory = $false)]
    [switch]$NoCache
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $PSScriptRoot

if ($BuildOnly -and $PushOnly) {
    throw "Choose at most one of -BuildOnly and -PushOnly."
}
if (-not $DockerHubUsername) {
    $DockerHubUsername = Read-Host "Docker Hub username"
}
if (-not $DockerHubUsername) {
    throw "A Docker Hub username is required."
}
if (-not $ImageTag) {
    $ImageTag = "v0.22.1-$Variant"
}

$Image = "$DockerHubUsername/$ImageRepository`:$ImageTag"
$BakeDraft = if ($Variant -eq "speculative-draft") { "1" } else { "0" }
$EnableShortConvFp8 = if ($Variant -eq "shortconv-fp8") { "1" } else { "0" }

Write-Host "Viettel custom image build" -ForegroundColor Cyan
Write-Host "  Variant: $Variant"
Write-Host "  Image:   $Image"
Write-Host "  Draft baked: $BakeDraft"
Write-Host "  ShortConv FP8 patch: $EnableShortConvFp8"

if (-not $PushOnly) {
    $BuildArguments = @(
        "build",
        "--platform", "linux/amd64",
        "--build-arg", "ENABLE_SHORTCONV_FP8=$EnableShortConvFp8",
        "--build-arg", "BAKE_DRAFT_MODEL=$BakeDraft",
        "--tag", $Image,
        "--file", (Join-Path $ProjectDir "Dockerfile")
    )
    if ($NoCache) {
        $BuildArguments += "--no-cache"
    }
    $BuildArguments += $ProjectDir

    & docker @BuildArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Docker build failed."
    }

    $ImageSize = & docker image inspect $Image --format '{{.Size}}' 2>$null
    if ($LASTEXITCODE -eq 0 -and $ImageSize) {
        $ImageSizeGb = [Math]::Round(([Int64]$ImageSize / 1GB), 2)
        Write-Host "  Local image size: $ImageSizeGb GB"
    }
}

if (-not $BuildOnly) {
    Write-Host "Pushing $Image ..." -ForegroundColor Yellow
    & docker push $Image
    if ($LASTEXITCODE -ne 0) {
        throw "Docker push failed. Run 'docker login' and retry."
    }

    # Docker records the registry digest locally after a successful push.
    $RepoDigest = & docker image inspect $Image --format '{{index .RepoDigests 0}}' 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $RepoDigest) {
        Write-Warning "Push succeeded but the local digest was unavailable. Obtain it from the registry before rendering Compose."
        exit 0
    }

    Write-Host ""
    Write-Host "Immutable image reference:" -ForegroundColor Green
    Write-Host "  $RepoDigest"
    Write-Host ""
    Write-Host "Render (do not overwrite root yet):" -ForegroundColor Cyan
    if ($Variant -eq "speculative-draft") {
        Write-Host "  conda run -n viettel python scripts/select_submission.py --candidate speculative-draft --custom-image '$RepoDigest' --output artifacts/speculative-draft.yml"
    } else {
        Write-Host "  conda run -n viettel python scripts/select_submission.py --candidate shortconv-fp8 --custom-image '$RepoDigest' --output artifacts/shortconv-fp8.yml"
    }
}
