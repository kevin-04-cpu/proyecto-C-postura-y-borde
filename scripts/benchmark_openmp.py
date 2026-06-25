import os
import subprocess
import re
import matplotlib.pyplot as plt
import pandas as pd

# Configuración de rutas ajustadas para ejecutarse desde 'scripts'
EXE_PATH = "../src/preprocess.exe"  # Apunta al ejecutable en la carpeta 'src'
TARGET_SET = "train"           # El conjunto más pesado para medir rendimiento
THREADS_TO_TEST = [1, 2, 4, 8, 12]
OUTPUT_CSV = "../resultados/openmp_benchmark.csv"
OUTPUT_IMG = "../images_report/openmp_speedup.png"

def run_benchmark():
    results = []
    
    # Asegurar que existan las carpetas de salida
    os.makedirs("../resultados", exist_ok=True)
    os.makedirs("../images_report", exist_ok=True)

    print(f"=== Iniciando Benchmark OpenMP sobre el conjunto: {TARGET_SET} ===")
    
    for threads in THREADS_TO_TEST:
        print(f"Ejecutando con {threads} hilo(s)...", end="", flush=True)
        
        # Modificamos el entorno para que OpenMP detecte el número de hilos
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(threads)
        
        # Ejecutar el preprocesamiento capturando la salida de consola
        process = subprocess.run(
            [EXE_PATH, TARGET_SET], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True, 
            encoding="utf-8",
            env=env
        )
        
        # Buscar el tiempo en la salida usando expresiones regulares
        # Espera encontrar algo como: Tiempo de ejecución OpenMP (train): 2.543210 segundos
        match = re.search(r"Tiempo de ejecución OpenMP.*:\s+([\d.]+)\s+segundos", process.stdout)
        
        if match:
            execution_time = float(match.group(1))
            print(f" Completado en {execution_time:.4f} segundos.")
            results.append({"Hilos": threads, "Tiempo": execution_time})
        else:
            print(" ¡Error al capturar el tiempo! Verifica la salida del ejecutable.")
            # Descomenta la línea de abajo si necesitas depurar la salida de tu .exe
            # print(process.stdout) 

    if not results:
        print("No se recolectaron datos. Abortando.")
        return

    # Crear DataFrame y calcular Métricas (Speedup y Eficiencia)
    df = pd.DataFrame(results)
    t_serial = df.iloc[0]["Tiempo"]  # Tiempo con 1 hilo (Línea base)
    
    df["Speedup"] = t_serial / df["Tiempo"]
    df["Eficiencia"] = df["Speedup"] / df["Hilos"]
    
    # Guardar resultados en CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResultados guardados en: {OUTPUT_CSV}")
    print(df.to_string(index=False))
    
    # Generar las Gráficas requeridas
    generate_plots(df)

def generate_plots(df):
    plt.figure(figsize=(12, 5))
    
    # Gráfica 1: Tiempo de Ejecución vs Hilos
    plt.subplot(1, 2, 1)
    plt.plot(df["Hilos"], df["Tiempo"], marker='o', color='crimson', linewidth=2)
    plt.title("Tiempo de Ejecución vs Hilos")
    plt.xlabel("Número de Hilos")
    plt.ylabel("Tiempo (segundos)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(THREADS_TO_TEST)
    
    # Gráfica 2: Speedup Real vs Speedup Ideal
    plt.subplot(1, 2, 2)
    plt.plot(df["Hilos"], df["Speedup"], marker='s', color='blue', linewidth=2, label="Speedup Real")
    plt.plot(df["Hilos"], df["Hilos"], linestyle='--', color='gray', label="Speedup Ideal (Lineal)")
    plt.title("Speedup vs Número de Hilos")
    plt.xlabel("Número de Hilos")
    plt.ylabel("Speedup ($T_{1} / T_{P}$)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(THREADS_TO_TEST)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=300)
    print(f"Gráfica guardada en: {OUTPUT_IMG}")
    plt.show()

if __name__ == "__main__":
    run_benchmark()
