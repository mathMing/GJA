param([string]$Message = "Sync experiment code")
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $repoRoot
try {
    git add -- src run_all.py run_research.py config.json requirements.txt ARS_RUN_STATUS.md RESEARCH_RUNBOOK.md RUN_EXPERIMENTS.md .gitignore sync_github.ps1
    $pending = git diff --cached --name-only
    if ($pending) { git commit -m $Message }
    git push origin main
} finally { Pop-Location }
