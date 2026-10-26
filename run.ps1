# Start SnapAssist on the Snapdragon HP PC (NPU + Genie).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path .venv)) {
    python -m venv .venv
    .\.venv\Scripts\pip install -q -r requirements.txt
}

$env:SNAP_ASR_BACKEND = "npu"
$env:SNAP_LLM_BACKEND = "genie"
.\.venv\Scripts\python -m app.server
