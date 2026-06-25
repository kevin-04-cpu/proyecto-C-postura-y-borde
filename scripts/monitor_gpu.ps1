# =============================================================================
# monitor_gpu.ps1
# Equivalente en PowerShell de monitor_gpu.sh
# Corre el entrenamiento mientras registra en segundo plano la utilizacion y
# memoria de la GPU con nvidia-smi.
#
# Uso:
#   .\monitor_gpu.ps1              -> block_size=32, epocas por defecto
#   .\monitor_gpu.ps1 16           -> block_size=16
#   .\monitor_gpu.ps1 32 50        -> block_size=32, 50 epocas
# =============================================================================

param(
    [int]$BlockSize = 32,
    [int]$Epochs = 0,
    [int]$Seed = 42
)

New-Item -ItemType Directory -Force -Path "..\resultados" | Out-Null
$logFile = "..\resultados\nvidia_smi_log.csv"

# Escribimos el encabezado nosotros mismos para que coincida con lo que
# espera plot_metrics.py, y luego nvidia-smi solo agrega filas (sin encabezado)
"timestamp,utilization_gpu_%,memory_used_MiB,memory_total_MiB" | Out-File -FilePath $logFile -Encoding utf8

Write-Host "-> Iniciando monitoreo de GPU en segundo plano..."
$smiArgs = "--query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits -l 1"
$smiProcess = Start-Process -FilePath "nvidia-smi" -ArgumentList $smiArgs `
    -RedirectStandardOutput "..\resultados\nvidia_smi_raw.csv" -NoNewWindow -PassThru

Write-Host "-> Lanzando entrenamiento (block_size=$BlockSize, seed=$Seed)..."
if ($Epochs -gt 0) {
    ..\src\train.exe $BlockSize $Epochs $Seed
} else {
    ..\src\train.exe $BlockSize 1000 $Seed
}

Write-Host "-> Entrenamiento terminado. Deteniendo monitoreo de GPU..."
Start-Sleep -Seconds 1
Stop-Process -Id $smiProcess.Id -Force -ErrorAction SilentlyContinue

# Unimos el encabezado + las filas capturadas por nvidia-smi
if (Test-Path "..\resultados\nvidia_smi_raw.csv") {
    Get-Content "..\resultados\nvidia_smi_raw.csv" | Add-Content -Path $logFile
    Remove-Item "..\resultados\nvidia_smi_raw.csv"
}

Write-Host "-> Log guardado en $logFile"

# Resumen rapido con Python (promedio y maximo)
python -c @"
import csv
utils, mems = [], []
with open(r'$logFile') as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            utils.append(float(row['utilization_gpu_%']))
            mems.append(float(row['memory_used_MiB']))
        except (ValueError, KeyError):
            continue
if utils:
    print(f'Utilizacion GPU -> promedio: {sum(utils)/len(utils):.1f}%  maximo: {max(utils):.1f}%')
    print(f'Memoria usada    -> promedio: {sum(mems)/len(mems):.1f} MiB  maximo: {max(mems):.1f} MiB')
else:
    print('No se capturaron muestras.')
"@
