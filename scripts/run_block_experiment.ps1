# =============================================================================
# run_block_experiment.ps1
# Equivalente en PowerShell de run_block_experiment.sh
# Compila train.cu y lo corre con distintos tamaños de bloque (8, 16, 32),
# acumulando resultados en ../resultados/gpu_results.csv
#
# Uso:
#   .\run_block_experiment.ps1            -> usa epocas por defecto (1000)
#   .\run_block_experiment.ps1 50         -> usa 50 epocas (benchmark rapido)
# =============================================================================

param(
    [int]$Epochs = 0,   # 0 = usar el valor por defecto dentro de train.cu (1000)
    [int]$Seed = 42     # usa la MISMA semilla que tu corrida oficial para una comparación justa
)

New-Item -ItemType Directory -Force -Path "..\resultados" | Out-Null

Write-Host "-> Compilando train.cu ..."
nvcc -O3 -o ..\src\train.exe ..\src\train.cu
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] La compilacion con nvcc fallo." -ForegroundColor Red
    exit 1
}

$blockSizes = 8, 16, 32

foreach ($bs in $blockSizes) {
    Write-Host "=================================================="
    Write-Host "-> Ejecutando con block_size = ${bs}x${bs} (seed=$Seed)"
    Write-Host "=================================================="

    if ($Epochs -gt 0) {
        ..\src\train.exe $bs $Epochs $Seed
    } else {
        ..\src\train.exe $bs 1000 $Seed
    }
}

Write-Host ""
Write-Host "-> Experimento completo. Resultados acumulados en ..\resultados\gpu_results.csv"
Get-Content "..\resultados\gpu_results.csv"
