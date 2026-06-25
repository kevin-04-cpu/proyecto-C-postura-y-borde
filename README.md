<div align="center">
  <img src="images_report/logo.png" alt="Logo" width="80" height="80">

  <h3 align="center">Visión por computadora para la detección de postura</h3>

  <p align="center">
    Arquitectura híbrida CPU/GPU utilizando OpenMP y CUDA para clasificación desde cero
  </p>
</div>

<details>
  <summary>Tabla de contenidos</summary>
  <ol>
    <li>
      <a href="#sobre-el-proyecto">Sobre el proyecto</a>
      <ul>
        <li><a href="#contexto-del-proyecto">Contexto del proyecto</a></li>
        <li><a href="#arquitectura-del-pipeline">Arquitectura del pipeline</a></li>
        <li><a href="#construido-con">Construido con</a></li>
      </ul>
    </li>
    <li>
      <a href="#primeros-pasos">Primeros pasos</a>
      <ul>
        <li><a href="#requisitos-previos">Requisitos previos</a></li>
        <li><a href="#instalación-y-configuración">Instalación y configuración</a></li>
      </ul>
    </li>
    <li>
      <a href="#uso-y-ejecución">Uso y ejecución</a>
      <ul>
        <li><a href="#etapa-1-preprocesamiento-con-openmp">Etapa 1: Preprocesamiento con OpenMP</a></li>
        <li><a href="#etapa-2-entrenamiento-con-cuda">Etapa 2: Entrenamiento con CUDA</a></li>
        <li><a href="#etapa-3-aplicación-de-streamlit">Etapa 3: Aplicación de Streamlit</a></li>
      </ul>
    </li>
    <li><a href="#informe-de-rendimiento-y-benchmarks">Informe de rendimiento y benchmarks</a></li>
  </ol>
</details>

<br>
 
## Sobre el proyecto

### Contexto del proyecto
Este proyecto implementa un pipeline de clasificación de imágenes de extremo a extremo (end-to-end) construido desde cero para detectar la postura al sentarse (encorvado vs. erguido). En lugar de utilizar frameworks de aprendizaje profundo preentrenados, el sistema gestiona el ciclo de vida completo: creación del conjunto de datos local, preprocesamiento en CPU multinúcleo, entrenamiento de una red neuronal personalizada en GPU y despliegue en tiempo real.  

El objetivo de optimización se centra en realizar un benchmark de los patrones de computación paralela, medir el aceleramiento (speedup) y analizar los límites de utilización del hardware a través de ejecuciones seriales y paralelas.

### Arquitectura del pipeline
El sistema se divide en tres fases de ejecución distintas:

![Pipeline del proyecto](images_report/pipeline.png)

### Construido con

<table>
  <tr>
    <td align="center" width="150">
      <img src="https://img.shields.io/badge/C-A8B9CC?style=for-the-badge&logo=c&logoColor=white" alt="C"><br>
      <img src="https://img.shields.io/badge/OpenMP-5CA5E6?style=for-the-badge&logo=openmp&logoColor=white" alt="OpenMP">
    </td>
    <td>
      Maneja bucles de ejecución paralela de datos de bajo nivel y multinúcleo sobre el directorio de imágenes.
    </td>
  </tr>
  <tr>
    <td align="center" width="150">
      <img src="https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA">
    </td>
    <td>
      Gestiona la planificación de hilos acelerada por hardware, la asignación de memoria global/compartida y los kernels de ejecución de matrices en la GPU.
    </td>
  </tr>
  <tr>
    <td align="center" width="150">
      <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"><br>
      <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
    </td>
    <td>
      Ejecuta el diseño de prototipado rápido para la interacción del usuario y la predicción de inferencia en tiempo real.
    </td>
  </tr>
</table>
<br>

## Primeros pasos

### Requisitos previos
Antes de configurar y ejecutar este proyecto, asegúrate de que tu sistema cumpla con los siguientes requisitos de hardware y software:

* **Sistema operativo**: Linux (se recomienda Ubuntu 20.04 LTS o posterior) o Windows configurado con WSL2.
* **Compilador y flags**: `gcc` con soporte para OpenMP (por ejemplo, `-fopenmp`).
* **CUDA Toolkit**: NVIDIA CUDA Toolkit (v11.0 o superior) que coincida con la capacidad del controlador de tu GPU.
* **Entorno de Python**: Python 3.8 o superior.

### Instalación y configuración

1. **Clonar el repositorio**
   ```bash
   git clone https://github.com/Daxdzzzy/proyecto-C-postura-y-borde
   cd proyecto-C-postura-y-borde
   ```

2. **Descargar el conjunto de datos**
   Los archivos de imágenes están excluidos de este repositorio mediante `.gitignore`. Descárgalos manualmente desde el siguiente enlace:
   https://drive.google.com/drive/folders/1osc-LFSSpwpy6_MkisDl1tH1SWocND23?usp=drive_link

   Coloca las carpetas descargadas dentro del directorio `images/` en la raíz del proyecto, de modo que la estructura quede así:
   ```
   images/
   ├── postura_recta/
   └── postura_encorvada/
   ```

3. **Instalar las dependencias de Python**
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   pip install streamlit numpy pillow scikit-learn
   ```

4. **Dividir el conjunto de datos**
   Ejecuta el script de organización para dividir las imágenes en conjuntos de entrenamiento, validación y prueba (70/15/15) con estratificación por clase. Esto genera la estructura que consume el preprocesador en C:
   ```bash
   python scripts/organizar_dataset.py
   ```

   Resultado esperado en `dataset/processed/`:
   ```
   dataset/processed/
   ├── train.txt
   ├── val.txt
   ├── test.txt
   ├── train/
   │   ├── postura_recta/
   │   └── postura_encorvada/
   ├── val/
   │   ├── postura_recta/
   │   └── postura_encorvada/
   └── test/
       ├── postura_recta/
       └── postura_encorvada/
   ```

## Uso y ejecución

Este proyecto se ejecuta en tres etapas secuenciales. Debes seguir este orden para procesar correctamente los datos, entrenar el modelo y ejecutar la interfaz.

### *Etapa 1: Preprocesamiento (OpenMP)*

Procesa archivos de imagen independientes en un bucle paralelizado para convertir las entradas a escala de grises, aplicar filtros de detección de bordes de silueta, redimensionar las estructuras a $64 \times 64$ píxeles y exportar una matriz aplanada al disco.

1. Compilar el script de preprocesamiento: Compila el código fuente en C utilizando `gcc` con la flag de OpenMP habilitada:
    ```
    gcc -fopenmp preprocess.c -o preprocess -lm
    ```

2. Ejecutar benchmarks de ejecución: Prueba el pipeline cambiando el número de hilos activos para evaluar la escalabilidad y medir el aceleramiento (speedup) frente a la ejecución serial base:
    ```
    # Ejecutar la ejecución serial (1 hilo)
    OMP_NUM_THREADS=1 ./preprocess

    # Ejecutar la ejecución paralela (por ejemplo, 2, 4, 8 hilos)
    OMP_NUM_THREADS=4 ./preprocess
    ```


Esto exportará la matriz de imágenes aplanada (`dataset.bin` o `dataset.csv`) y las etiquetas directamente al disco.

### *Etapa 2: Entrenamiento del modelo (CUDA)*

Carga el conjunto de datos aplanado para entrenar un Perceptrón Multicapa (MLP) utilizando kernels explícitos de GPU para la multiplicación de matrices, adición de sesgos (bias), funciones de activación (ReLU/Sigmoid), cálculo de la pérdida de Entropía Cruzada Binaria y retropropagación (backpropagation).

1. Compilar la aplicación de entrenamiento CUDA: Compila los kernels del dispositivo y el código de host utilizando el compilador de NVIDIA CUDA (`nvcc`):
    ```
    nvcc train.cu -o train_mlp
    ```


2. Ejecutar el entrenamiento del modelo: Ejecuta el ejecutable de entrenamiento para realizar la pasada hacia adelante (forward pass) de la red, calcular la pérdida de Entropía Cruzada Binaria, computar los gradientes de retropropagación y optimizar los pesos mediante SGD:
    ```
    ./train_mlp
    ```


Al finalizar, este proceso exporta automáticamente las matrices de pesos y sesgos entrenados a un archivo externo (por ejemplo, `weights.bin`).

### *Etapa 3: Aplicación (Streamlit)*

Una aplicación de despliegue ligera que carga los binarios crudos para ejecutar inferencias sobre nuevas capturas de la cámara web.

1. Iniciar la interfaz web: Asegúrate de que tu entorno virtual de Python esté activo e inicia el servidor de Streamlit:

    ```
    streamlit run app.py
    ```

2. Realizar inferencia: Abre la URL local proporcionada por Streamlit en tu navegador (normalmente `http://localhost:8501`).

* Sube una imagen o proporciona un archivo estático para activar el motor de inferencia de Python.
* El backend aplica exactamente las mismas transformaciones de preprocesamiento (escala de grises, filtros de borde Sobel y redimensionamiento a $64 \times 64$) antes de ejecutar la pasada hacia adelante con tus pesos guardados para generar la clasificación final de la postura.

## Informe de rendimiento y benchmarks

Para un análisis detallado de los tiempos de ejecución, especificaciones de hardware, gráficos de aceleramiento (speedup) y preguntas de reflexión académica, consulta el [Informe de rendimiento y benchmarks](https://www.google.com/search?q=./docs/BENCHMARK_REPORT.md).
