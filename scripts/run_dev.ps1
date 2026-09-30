<#
.SYNOPSIS
    本地开发启动脚本（Windows PowerShell）。

.DESCRIPTION
    优先使用项目内 .venv 解释器启动 uvicorn，避免命中 Microsoft Store 的
    python 应用执行别名——那个占位程序被调用时「不报错、不输出、直接退出」，
    看起来就像服务启动失败。
    启动前依次检查：解释器是否装了依赖 → 端口是否被占用 → .env 是否存在。

.PARAMETER Port
    监听端口，默认 8000。

.PARAMETER BindAddress
    监听地址，默认 127.0.0.1。

.PARAMETER NoReload
    关闭热重载（等价于不传 --reload）。

.PARAMETER Diagnose
    只做环境体检（解释器 / 依赖 / 端口 / .env），不启动服务。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\run_dev.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\run_dev.ps1 -Diagnose

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\run_dev.ps1 -Port 8080 -NoReload
#>
param(
    [int]$Port = 8000,
    [string]$BindAddress = "127.0.0.1",
    [switch]$NoReload,
    [switch]$Diagnose
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

function Test-StoreAlias {
    param([string]$Path)
    if ($Path -like "*\WindowsApps\python*.exe") { return $true }
    return $false
}

function Get-PortOwner {
    param([int]$LocalPort)
    if (-not (Get-Command Get-NetTCPConnection -ErrorAction SilentlyContinue)) { return @() }
    return @(
        Get-NetTCPConnection -LocalPort $LocalPort -State Listen -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty OwningProcess -Unique
    )
}

# 调用某个 Python 解释器并返回「输出 + 退出码」。
# PowerShell 5.1 在 $ErrorActionPreference='Stop' 时，会把原生命令写到 stderr 的内容
# 升级为终止性错误（NativeCommandError），因此这里临时放宽再恢复。
function Invoke-Interpreter {
    param([string]$Exe, [string[]]$Pre, [string[]]$PyArgs)
    $saved = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $captured = (& $Exe @Pre @PyArgs 2>&1 | Out-String)
        $exitCode = $LASTEXITCODE
    } catch {
        $captured = $_.Exception.Message
        $exitCode = 1
    } finally {
        $ErrorActionPreference = $saved
    }
    if ($null -eq $exitCode) { $exitCode = 1 }
    return [pscustomobject]@{ Output = $captured.Trim(); ExitCode = $exitCode }
}

# ---------------- 1. 选择解释器：.venv -> py -3.10 -> python ----------------
$notes = New-Object System.Collections.Generic.List[string]
$chosen = $null
$chosenPre = @()
$chosenLabel = ""

$candidates = New-Object System.Collections.Generic.List[object]
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    $candidates.Add([pscustomobject]@{ Exe = $venvPython; Pre = @(); Label = ".venv 虚拟环境" })
}
$candidates.Add([pscustomobject]@{ Exe = "py"; Pre = @("-3.10"); Label = "py -3.10（系统 Python 3.10）" })
$candidates.Add([pscustomobject]@{ Exe = "python"; Pre = @(); Label = "python（PATH 中的 python）" })

foreach ($candidate in $candidates) {
    $exe = $candidate.Exe
    if (-not (Test-Path $exe)) {
        $cmd = Get-Command $candidate.Exe -ErrorAction SilentlyContinue
        if ($null -eq $cmd) {
            $notes.Add("跳过 $($candidate.Label)：命令不存在")
            continue
        }
        $exe = $cmd.Source
    }
    if (Test-StoreAlias -Path $exe) {
        $notes.Add("跳过 $($candidate.Label)：$exe 是 Microsoft Store 的应用执行别名，调用它不会产生任何输出。")
        continue
    }
    $pre = $candidate.Pre
    $probe = Invoke-Interpreter -Exe $exe -Pre $pre -PyArgs @((Join-Path $PSScriptRoot "check_env.py"))
    if ($probe.ExitCode -eq 0) {
        $chosen = $exe
        $chosenPre = $pre
        $chosenLabel = $candidate.Label
        break
    }
    $reason = "无法运行（退出码 $($probe.ExitCode)）"
    if ($probe.Output -like "DEPS_MISSING:*") {
        $reason = "缺少依赖：" + ($probe.Output -replace "^DEPS_MISSING:\s*", "")
    } elseif ($probe.Output -like "DEPS_BROKEN:*") {
        $reason = "依赖损坏：" + ($probe.Output -replace "^DEPS_BROKEN:\s*", "")
    } elseif ($probe.Output) {
        $reason = ($probe.Output -split "`r?`n")[0]
    }
    $notes.Add("跳过 $($candidate.Label)：$reason")
}

if ($null -eq $chosen) {
    Write-Host "[错误] 没有找到已安装本项目依赖的 Python 解释器。" -ForegroundColor Red
    foreach ($note in $notes) { Write-Host "  - $note" }
    Write-Host ""
    Write-Host "请安装依赖后重试："
    Write-Host "  py -3.10 -m venv .venv"
    Write-Host "  .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
    exit 1
}

Write-Host "[解释器] $chosenLabel -> $chosen" -ForegroundColor Cyan

# ---------------- 2. 端口占用预检 ----------------
$owners = @(Get-PortOwner -LocalPort $Port)
if ($owners.Count -gt 0) {
    Write-Host "[错误] 端口 $Port 已被占用，uvicorn 绑定会失败。" -ForegroundColor Red
    foreach ($ownerPid in $owners) {
        $procName = "?"
        $proc = Get-Process -Id $ownerPid -ErrorAction SilentlyContinue
        if ($proc) { $procName = $proc.ProcessName }
        Write-Host "  占用者：PID $ownerPid（$procName）"
    }
    Write-Host "  解决一：结束占用进程 —— taskkill /T /F /PID <上面的 PID>"
    Write-Host "  解决二：换端口 —— powershell -ExecutionPolicy Bypass -File scripts\run_dev.ps1 -Port 8080"
    exit 1
}

# ---------------- 3. .env ----------------
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "[提示] 未找到 .env，已从 .env.example 复制。请填写 OPENAI_API_KEY（未填写时 /api/v1/chat 返回 503）。" -ForegroundColor Yellow
}

# ---------------- 4. 体检模式 ----------------
if ($Diagnose) {
    Write-Host "[诊断] 项目目录 : $projectRoot"
    $report = Invoke-Interpreter -Exe $chosen -Pre $chosenPre `
        -PyArgs @((Join-Path $PSScriptRoot "check_env.py"), "--verbose", "--imports")
    foreach ($line in ($report.Output -split "`r?`n")) { Write-Host "  $line" }

    $configProbe = "from app.core.config import get_settings as g; s = g(); k = (s.openai_api_key or '').strip(); print('OPENAI_API_KEY:', ('已配置(len=%d)' % len(k)) if k else '未配置'); print('OPENAI_BASE_URL:', s.openai_base_url or '<官方默认>'); print('OPENAI_MODEL:', s.openai_model); print('WORKSPACE_DIR:', s.workspace_dir); print('CHECKPOINT_BACKEND:', s.checkpoint_backend); print('MAX_AGENT_STEPS:', s.max_agent_steps)"
    $configInfo = Invoke-Interpreter -Exe $chosen -Pre $chosenPre -PyArgs @("-c", $configProbe)
    if ($configInfo.ExitCode -eq 0) {
        Write-Host "[诊断] 配置加载 : ok（真实读取 app.core.config）"
        foreach ($line in ($configInfo.Output -split "`r?`n")) { Write-Host "  $line" }
    } else {
        $lastLine = ($configInfo.Output -split "`r?`n")[-1]
        Write-Host "[诊断] 配置加载 : 失败 —— $lastLine" -ForegroundColor Red
    }

    $envState = "已存在"
    if (-not (Test-Path ".env")) { $envState = "缺失（启动前会自动从 .env.example 复制）" }
    Write-Host "[诊断] .env      : $envState"
    Write-Host "[诊断] 端口 $Port : 空闲"
    Write-Host "[诊断] 结论      : 环境可启动（去掉 -Diagnose 即可运行服务）"
    exit 0
}

# ---------------- 5. 启动 ----------------
$uvicornArgs = @("-m", "uvicorn", "app.main:app", "--host", $BindAddress, "--port", $Port)
if (-not $NoReload) { $uvicornArgs += "--reload" }

Write-Host "[启动] $chosen $($uvicornArgs -join ' ')" -ForegroundColor Cyan
Write-Host "[地址] http://$($BindAddress):$Port/  （接口文档 /docs，健康检查 /api/v1/health）"
Write-Host ""

& $chosen @chosenPre @uvicornArgs
exit $LASTEXITCODE
