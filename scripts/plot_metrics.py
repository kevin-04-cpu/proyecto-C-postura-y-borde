"""
plot_metrics.py
Genera todas las gráficas que pide el reporte para la parte de CUDA:

  1. Curvas de pérdida y exactitud por época (train vs val)      -> curva_perdida.png, curva_exactitud.png
  2. Matriz de confusión del conjunto de prueba                  -> matriz_confusion.png
  3. Tiempo y speedup CPU vs GPU                                  -> cpu_vs_gpu.png
  4. Efecto del tamaño de bloque sobre el tiempo de entrenamiento -> tiempo_vs_bloque.png
  5. Utilización y memoria de GPU durante el entrenamiento        -> nvidia_smi.png  (si el log existe)

Uso (desde la carpeta 'scripts'):
    python plot_metrics.py

Requiere: pandas, matplotlib  (pip install pandas matplotlib)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "../resultados"
IMAGES_DIR = "../images_report"


def safe_read_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    print(f"[omitido] No se encontró: {path}")
    return None


def plot_loss_and_accuracy():
    df = safe_read_csv(os.path.join(RESULTS_DIR, "metrics_history.csv"))
    if df is None:
        return

    # Asegurar que la carpeta de imágenes existe
    os.makedirs(IMAGES_DIR, exist_ok=True)

    # --- Pérdida ---
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss", marker="o")
    plt.plot(df["epoch"], df["val_loss"], label="Val Loss", marker="o")
    plt.xlabel("Época")
    plt.ylabel("Pérdida (BCE)")
    plt.title("Curva de pérdida por época")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "curva_perdida.png"), dpi=150)
    plt.close()

    # --- Exactitud ---
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_acc"] * 100, label="Train Acc", marker="o")
    plt.plot(df["epoch"], df["val_acc"] * 100, label="Val Acc", marker="o")
    plt.xlabel("Época")
    plt.ylabel("Exactitud (%)")
    plt.title("Curva de exactitud por época")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "curva_exactitud.png"), dpi=150)
    plt.close()

    print("-> Guardadas: curva_perdida.png, curva_exactitud.png en", IMAGES_DIR)


def plot_confusion_matrix():
    df = safe_read_csv(os.path.join(RESULTS_DIR, "gpu_results.csv"))
    if df is None:
        return
    
    os.makedirs(IMAGES_DIR, exist_ok=True)
    
    # Usamos la corrida "oficial" = la de mayor número de épocas (asumimos que
    # los experimentos cortos, como el de tamaño de bloque, usan menos épocas
    # que la corrida final completa).
    row = df.loc[df["epochs"].idxmax()]
    tn, fp, fn, tp = row["tn"], row["fp"], row["fn"], row["tp"]
    matrix = [[tn, fp], [fn, tp]]

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred. 0", "Pred. 1"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Real 0", "Real 1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(matrix[i][j]), ha="center", va="center",
                     color="white" if matrix[i][j] > (tp + tn + fp + fn) / 4 else "black",
                     fontsize=16, fontweight="bold")
    ax.set_title(f"Matriz de Confusión (Test)\nAccuracy={row['accuracy']*100:.1f}%  "
                 f"Precision={row['precision']:.2f}  Recall={row['recall']:.2f}  F1={row['f1']:.2f}")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "matriz_confusion.png"), dpi=150)
    plt.close()
    print("-> Guardada: matriz_confusion.png en", IMAGES_DIR)


def plot_cpu_vs_gpu():
    gpu_df = safe_read_csv(os.path.join(RESULTS_DIR, "gpu_results.csv"))
    cpu_df = safe_read_csv(os.path.join(RESULTS_DIR, "cpu_results.csv"))
    if gpu_df is None or cpu_df is None:
        return
    
    os.makedirs(IMAGES_DIR, exist_ok=True)

    # Para que la comparación sea justa, buscamos un número de épocas que
    # exista tanto en gpu_results.csv (block_size=32) como en cpu_results.csv.
    gpu32 = gpu_df[gpu_df["block_size"] == 32]
    common_epochs = set(gpu32["epochs"]).intersection(set(cpu_df["epochs"]))
    if not common_epochs:
        print("[omitido] No hay un número de épocas en común entre cpu_results.csv y "
              "gpu_results.csv (block=32). Corre ambos con el mismo número de épocas, "
              "ej: .\\train_cpu.exe 50  y  .\\train.exe 32 50")
        return

    target_epochs = max(common_epochs)  # la corrida más reciente con épocas en común
    gpu_row = gpu32[gpu32["epochs"] == target_epochs].iloc[-1]
    cpu_row = cpu_df[cpu_df["epochs"] == target_epochs].iloc[-1]

    gpu_time = gpu_row["gpu_time_seconds"]
    cpu_time = cpu_row["cpu_time_seconds"]
    speedup = cpu_time / gpu_time if gpu_time > 0 else 0

    plt.figure(figsize=(6, 5))
    bars = plt.bar(["CPU (serial)", "GPU (CUDA)"], [cpu_time, gpu_time], color=["#d9534f", "#5cb85c"])
    plt.ylabel("Tiempo de entrenamiento (segundos)")
    plt.title(f"Tiempo de entrenamiento CPU vs GPU ({target_epochs} épocas)\nSpeedup = {speedup:.1f}x")
    for b, t in zip(bars, [cpu_time, gpu_time]):
        plt.text(b.get_x() + b.get_width() / 2, t, f"{t:.2f}s", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "cpu_vs_gpu.png"), dpi=150)
    plt.close()
    print(f"-> Guardada: cpu_vs_gpu.png en {IMAGES_DIR} (épocas={target_epochs}, speedup calculado: {speedup:.2f}x)")


def plot_block_size_effect():
    df = safe_read_csv(os.path.join(RESULTS_DIR, "gpu_results.csv"))
    if df is None:
        return
    
    os.makedirs(IMAGES_DIR, exist_ok=True)

    # Buscamos el grupo de épocas que tenga al menos 2 tamaños de bloque
    # distintos (ese es el experimento real de "efecto del tamaño de bloque").
    # Así evitamos mezclar la corrida oficial larga con el benchmark corto.
    candidate_groups = df.groupby("epochs")["block_size"].nunique()
    valid_epochs = candidate_groups[candidate_groups >= 2].index.tolist()
    if not valid_epochs:
        print("[omitido] Solo hay una corrida de GPU; corre run_block_experiment.ps1 "
              "para comparar tamaños de bloque.")
        return

    target_epochs = max(valid_epochs)
    subset = df[df["epochs"] == target_epochs]
    grouped = subset.groupby("block_size")["gpu_time_seconds"].mean().reset_index()

    plt.figure(figsize=(6, 5))
    plt.bar(grouped["block_size"].astype(str) + "x" + grouped["block_size"].astype(str),
            grouped["gpu_time_seconds"], color="#5bc0de")
    plt.xlabel("Tamaño de bloque")
    plt.ylabel("Tiempo de entrenamiento (segundos)")
    plt.title(f"Efecto del tamaño de bloque sobre el tiempo de entrenamiento\n({target_epochs} épocas)")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "tiempo_vs_bloque.png"), dpi=150)
    plt.close()
    print(f"-> Guardada: tiempo_vs_bloque.png en {IMAGES_DIR} (épocas={target_epochs})")


def plot_nvidia_smi():
    path = os.path.join(RESULTS_DIR, "nvidia_smi_log.csv")
    df = safe_read_csv(path)
    if df is None:
        return
    
    os.makedirs(IMAGES_DIR, exist_ok=True)
    df.columns = [c.strip() for c in df.columns]
    util_col = [c for c in df.columns if "utilization" in c][0]
    mem_col = [c for c in df.columns if "memory_used" in c][0]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(df[util_col].astype(float).values, color="#5cb85c", label="Utilización GPU (%)")
    ax1.set_ylabel("Utilización GPU (%)", color="#5cb85c")
    ax1.set_xlabel("Muestra (cada ~1s)")

    ax2 = ax1.twinx()
    ax2.plot(df[mem_col].astype(float).values, color="#d9534f", label="Memoria usada (MiB)")
    ax2.set_ylabel("Memoria usada (MiB)", color="#d9534f")

    plt.title("Utilización y memoria de GPU durante el entrenamiento")
    fig.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "nvidia_smi.png"), dpi=150)
    plt.close()
    print("-> Guardada: nvidia_smi.png en", IMAGES_DIR)


if __name__ == "__main__":
    plot_loss_and_accuracy()
    plot_confusion_matrix()
    plot_cpu_vs_gpu()
    plot_block_size_effect()
    plot_nvidia_smi()
    print("\nListo. Revisa las carpetas:", RESULTS_DIR, "e", IMAGES_DIR)
