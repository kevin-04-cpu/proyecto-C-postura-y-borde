[← Back to Main README](../README.md) 

# Performance Report and Benchmarks 

<div align="center">
  <img src="https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExcnc1MDhrNW00ZTh3M2V0Y3FlYnhtcW13N2lkc2tnc2I0aW5pYzhteiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/l0HlJIp1dIZzimEBq/giphy.gif" alt="Benchmarks" />
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#environment-specifications">Environment Specifications</a>
      <ul>
        <li><a href="#cpu-hardware">CPU Hardware</a></li>
        <li><a href="#gpu-hardware">GPU Hardware</a></li>
      </ul>
    </li>
    <li>
      <a href="#dataset-analytics">Dataset Analytics</a>
      <ul>
        <li><a href="#data-collection-and-variety">Data Collection and Variety</a></li>
        <li><a href="#data-splitting">Data Splitting</a></li>
      </ul>
    </li>
    <li>
      <a href="#neural-network-architecture">Neural Network Architecture</a>
      <ul>
        <li><a href="#mlp-layers-and-parameters">MLP Layers and Parameters</a></li>
      </ul>
    </li>
    <li>
      <a href="#stage-1-performance-openmp">Stage 1 Performance: OpenMP</a>
      <ul>
        <li><a href="#openmp-execution-times">Execution Times Table</a></li>
        <li><a href="#openmp-speedup-analysis">Speedup Chart & Analysis</a></li>
        <li><a href="#openmp-reflection-questions">Stage 1 Reflection Questions</a></li>
      </ul>
    </li>
    <li>
      <a href="#stage-2-performance-cuda">Stage 2 Performance: CUDA</a>
      <ul>
        <li><a href="#cpu-vs-gpu-execution-times">CPU vs GPU Execution Times</a></li>
        <li><a href="#block-size-impact-analysis">Block Size Impact Analysis</a></li>
        <li><a href="#cuda-reflection-questions">Stage 2 Reflection Questions</a></li>
      </ul>
    </li>
    <li>
      <a href="#stage-3-model-evaluation-metrics">Stage 3 Model Evaluation Metrics</a>
      <ul>
        <li><a href="#training-loss-and-accuracy-curves">Training Loss and Accuracy Curves</a></li>
        <li><a href="#confusion-matrix">Confusion Matrix</a></li>
        <li><a href="#final-classification-metrics">Precision, Recall, and F1-Score</a></li>
      </ul>
    </li>
    <li><a href="#conclusions">Conclusions</a></li>
    <li><a href="#acknowledgments-and-references">Acknowledgments and References</a></li>
  </ol>
</details>

## Environment Specifications

Detalle del Procesador (CPU) utilizado para OpenMP (Modelo, número de núcleos físicos e hilos lógicos).
Detalle de la Tarjeta Gráfica (GPU) utilizada para CUDA (Modelo, núcleos CUDA y memoria VRAM disponible).

### CPU Hardware
### GPU Hardware


## Dataset Analytics
Tabla resumen con la cantidad de imágenes recolectadas por clase, balance del conjunto, variedad de sujetos y condiciones de luz aplicadas.

Declaración numérica de la división de datos: Entrenamiento (70%), Validación (15%) y Prueba (15%).

224  test.
1042 entrenamineto.
223 validaciones.

### Data Collection and Variety

### Data Splitting

## Neural Network Architecture

Detailed description of the Multilayer Perceptron (MLP) designed from scratch for binary classification.The network processes flattened grayscale image vectors to output a single probability.

### MLP Layers and Parameters

The network consists of an input layer, a single hidden dense layer, and a dense output layer. Below is the structural specification and the exact dimension of each layer:

Input Layer ($L_0$):** 4,096 nodes, representing the flattened $64 \times 64$ pixels of the preprocessed image.

* **Hidden Layer ($L_1$):** Dense layer with [REPLACE: Number of hidden neurons, e.g., 64 or 128] neurons, utilizing the Rectified Linear Unit (ReLU) activation function.
* **Output Layer ($L_2$):** A single neuron with a Sigmoid activation function to map the binary probability ($0$ or $1$).

#### Layer Properties and Learnable Parameters

| Layer | Type | Input Size | Output Size | Activation Function | Weights Matrix | Bias Vector | Total Parameters |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hidden ($L_1$)** | Dense | 4,096 | $W_1$ | ReLU | $4096 \times W_1$ | $W_1 \times 1$ | $(4096 \times W_1) + W_1$ |
| **Output ($L_2$)** | Dense | $W_1$ | 1 | Sigmoid | $W_1 \times 1$ | $1 \times 1$ | $W_1 + 1$ |
| **Total** | | | | | | | **[REPLACE: Sum of parameters]** |

> *Note: Replace $W_1$ with the hidden size you experimented with (e.g., 64 or 128).*

#### Mathematical Parameter Calculations (Formula)
To verify the intellectual and mathematical rigor of the network, parameters are calculated using the formula: $\text{Parameters} = (\text{inputs} \times \text{outputs}) + \text{biases}$.

* **Hidden Layer ($L_1$):** $(4,096 \times W_1) + W_1$
* **Output Layer ($L_2$):** $(W_1 \times 1) + 1$

## Stage 1 Performance: OpenMP

Tabla de tiempos medidos: Tiempo serial de un hilo frente a tiempos con 2, 4, 8... hilos en OpenMP.

Gráfica de aceleración (Speedup vs. Número de hilos).

Respuestas técnicas a las preguntas de reflexión de OpenMP (Limitaciones por Ley de Amdahl y cuello de botella de E/S de disco).

### Execution Times Table
### Speedup Chart & Analysis
### Stage 1 Reflection Question



## Stage 2 Performance: CUDA

Tabla comparativa de tiempos de entrenamiento: CPU vs. GPU.

Tabla de rendimiento según variaciones en el tamaño de bloque de hilos CUDA (16×16, 32×32).

Capturas de pantalla adjuntas de la terminal ejecutando nvidia-smi durante el procesamiento intensivo.

Respuestas técnicas a las preguntas de reflexión de CUDA (Ventajas del Matmul paralelo, abstracción de PyTorch e impacto del tamaño de los datos en el Speedup).

### CPU vs GPU Execution Times
### Block Size Impact Analysis
### Stage 2 Reflection Questions



## Stage 3 Model Evaluation Metrics

Gráficas de las curvas de entrenamiento: Pérdida (Loss) y Exactitud (Accuracy) decrecientes/crecientes por época.

Matriz de Confusión final calculada estrictamente sobre el conjunto de prueba aislado.

Cuadro comparativo de métricas finales: Precisión, Recall, F1-Score y Accuracy global.

### Training Loss and Accuracy Curves
### Confusion Matrix
### Precision, Recall, and F1-Score

## Conclusions

## References
Enlaces a la documentación oficial consultada (NVIDIA CUDA C Programming Guide, especificaciones de OpenMP y librerías utilizadas).
