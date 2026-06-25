# =============================================================================
# seed_search.ps1
# Prueba varias semillas de inicialización de pesos con pocas épocas, y
# reporta cuál da mejor accuracy/loss de validación, para elegir la semilla
# "oficial" que usarás en la corrida final, en el experimento de bloques y
# en la comparación CPU vs GPU.
#
# Uso:
#   powershell -ExecutionPolicy Bypass -File .\seed_search.ps1
#   powershell -ExecutionPolicy Bypass -File .\seed_search.ps1 -Epochs 100 -Seeds 1,7,42,123,2024
# =============================================================================

param(
    [int]$Epochs = 100,
    [int[]]$Seeds = @(1, 7, 42, 123, 2024),
    [int]$BlockSize = 32
)

New-Item -ItemType Directory -Force -Path "..\resultados" | Out-Null
$summaryFile = "..\resultados\seed_search_results.csv"
"seed,val_loss,val_acc,train_loss,train_acc" | Out-File -FilePath $summaryFile -Encoding utf8

Write-Host "-> Compilando train.cu ..."
nvcc -O3 -o ..\src\train.exe ..\src\train.cu
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] La compilacion con nvcc fallo." -ForegroundColor Red
    exit 1
}

foreach ($seed in $Seeds) {
    Write-Host "=================================================="
    Write-Host "-> Probando seed = $seed  (block=$BlockSize, epochs=$Epochs)"
    Write-Host "=================================================="

    ..\src\train.exe $BlockSize $Epochs $seed

    # Tras cada corrida, metrics_history.csv contiene la curva de ESTA corrida.
    # Tomamos la última fila (última época registrada) como resultado final.
    $historyPath = "..\resultados\metrics_history.csv"
    if (Test-Path $historyPath) {
        $lastLine = Get-Content $historyPath | Select-Object -Last 1
        $fields = $lastLine -split ","
        # columnas: epoch,train_loss,val_loss,train_acc,val_acc
        $trainLoss = $fields[1]
        $valLoss   = $fields[2]
        $trainAcc  = $fields[3]
        $valAcc    = $fields[4]
        "$seed,$valLoss,$valAcc,$trainLoss,$trainAcc" | Add-Content -Path $summaryFile
    } else {
        Write-Host "[ADVERTENCIA] No se encontró metrics_history.csv para seed=$seed"
    }
}

Write-Host ""
Write-Host "=================================================="
Write-Host "-> Resumen de todas las semillas probadas:"
Write-Host "=================================================="
$results = Import-Csv $summaryFile
$results | Sort-Object { [double]$_.val_acc } -Descending | Format-Table -AutoSize

$best = $results | Sort-Object { [double]$_.val_acc } -Descending | Select-Object -First 1
Write-Host ""
Write-Host "-> Mejor semilla: $($best.seed)  (val_acc = $($best.val_acc), val_loss = $($best.val_loss))" -ForegroundColor Green
Write-Host "-> Para usarla de forma fija en train.cu, cambia 'int seed = 42;' por 'int seed = $($best.seed);'"
