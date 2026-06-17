#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define INPUT_SIZE 4096
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 1

#define CHECK_CUDA(call)                                                       \
  {                                                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      printf("[Error CUDA] %s:%d: %s\n", _FILE, __LINE_,                       \
             cudaGetErrorString(err));                                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

// -------------------------------------------------------------------------
// KERNELS DE CUDA (DEVICE)
// -------------------------------------------------------------------------

// Kernel Capa 1: Multiplicación de Matrices + Bias + Activación ReLU
_global_ void forward_layer1(float *X, float *W1, float *b1, float *A1,
                             int num_images) {
  int row = blockIdx.y * blockDim.y +
            threadIdx.y; // Índice de la imagen (Ej. 0 a 1041)
  int col = blockIdx.x * blockDim.x +
            threadIdx.x; // Índice de la neurona oculta (0 a 127)

  if (row < num_images && col < HIDDEN_SIZE) {
    float sum = 0.0f;
    // Producto punto: Fila de X por la Fila de W1 (Traspuesta implícita)
    for (int i = 0; i < INPUT_SIZE; i++) {
      sum += X[row * INPUT_SIZE + i] * W1[col * INPUT_SIZE + i];
    }
    sum += b1[col]; // Suma del sesgo

    // Función de Activación ReLU
    A1[row * HIDDEN_SIZE + col] = fmaxf(0.0f, sum);
  }
}

// Kernel Capa 2: Multiplicación de Matrices + Bias + Activación Sigmoide
_global_ void forward_layer2(float *A1, float *W2, float *b2, float *A2,
                             int num_images) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x +
            threadIdx.x; // Índice de la neurona de salida (Solo 0)

  if (row < num_images && col < OUTPUT_SIZE) {
    float sum = 0.0f;
    for (int i = 0; i < HIDDEN_SIZE; i++) {
      sum += A1[row * HIDDEN_SIZE + i] * W2[col * HIDDEN_SIZE + i];
    }
    sum += b2[col];

    // Función de Activación Sigmoide
    A2[row * OUTPUT_SIZE + col] = 1.0f / (1.0f + expf(-sum));
  }
}

// -------------------------------------------------------------------------
// FUNCIONES DE HOST (CPU)
// -------------------------------------------------------------------------

int load_bin_float(const char *filepath, float **data) {
  FILE *f = fopen(filepath, "rb");
  if (!f)
    return -1;
  fseek(f, 0, SEEK_END);
  long filesize = ftell(f);
  rewind(f);
  int num_elements = filesize / sizeof(float);
  data = (float)malloc(filesize);
  size_t read_elements = fread(*data, sizeof(float), num_elements, f);
  fclose(f);
  return num_elements;
}

int load_bin_int(const char *filepath, int **data) {
  FILE *f = fopen(filepath, "rb");
  if (!f)
    return -1;
  fseek(f, 0, SEEK_END);
  long filesize = ftell(f);
  rewind(f);
  int num_elements = filesize / sizeof(int);
  data = (int)malloc(filesize);
  size_t read_elements = fread(*data, sizeof(int), num_elements, f);
  fclose(f);
  return num_elements;
}

void init_weights(float *array, int size, int input_connections) {
  float limit = sqrt(2.0f / input_connections);
  for (int i = 0; i < size; i++) {
    float rand_val = (float)rand() / (float)RAND_MAX;
    array[i] = (rand_val * 2.0f - 1.0f) * limit;
  }
}

int main() {
  srand(time(NULL));

  printf("=== INICIANDO MOTOR DE ENTRENAMIENTO CUDA ===\n");

  // 1. CARGA DE DATOS EN EL HOST
  float *h_X_train;
  int *h_y_train;
  int total_features =
      load_bin_float("dataset/processed/X_train.bin", &h_X_train);
  int total_labels = load_bin_int("dataset/processed/y_train.bin", &h_y_train);

  if (total_features <= 0 || total_labels <= 0)
    return -1;
  int num_images = total_features / INPUT_SIZE;
  printf("-> [OK] Dataset list. Imágenes: %d\n", num_images);

  // 2. INICIALIZACIÓN DE PARÁMETROS
  float h_W1 = (float)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
  float h_b1 = (float)calloc(HIDDEN_SIZE, sizeof(float));
  float h_W2 = (float)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
  float h_b2 = (float)calloc(OUTPUT_SIZE, sizeof(float));

  init_weights(h_W1, HIDDEN_SIZE * INPUT_SIZE, INPUT_SIZE);
  init_weights(h_W2, OUTPUT_SIZE * HIDDEN_SIZE, HIDDEN_SIZE);

  // 3. RESERVA EN MEMORIA GLOBAL (VRAM)
  float *d_X_train, *d_W1, *d_b1, *d_W2, *d_b2;
  float *d_A1, *d_A2; // Punteros para las matrices de Activación (Resultados)
  int *d_y_train;

  size_t size_X = num_images * INPUT_SIZE * sizeof(float);
  size_t size_y = num_images * sizeof(int);
  size_t size_W1 = HIDDEN_SIZE * INPUT_SIZE * sizeof(float);
  size_t size_b1 = HIDDEN_SIZE * sizeof(float);
  size_t size_W2 = OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float);
  size_t size_b2 = OUTPUT_SIZE * sizeof(float);

  // Tamaños dinámicos basados en la cantidad de imágenes
  size_t size_A1 = num_images * HIDDEN_SIZE * sizeof(float);
  size_t size_A2 = num_images * OUTPUT_SIZE * sizeof(float);

  CHECK_CUDA(cudaMalloc((void **)&d_X_train, size_X));
  CHECK_CUDA(cudaMalloc((void **)&d_y_train, size_y));
  CHECK_CUDA(cudaMalloc((void **)&d_W1, size_W1));
  CHECK_CUDA(cudaMalloc((void **)&d_b1, size_b1));
  CHECK_CUDA(cudaMalloc((void **)&d_W2, size_W2));
  CHECK_CUDA(cudaMalloc((void **)&d_b2, size_b2));

  CHECK_CUDA(
      cudaMalloc((void **)&d_A1, size_A1)); // Memoria para salidas Capa 1
  CHECK_CUDA(
      cudaMalloc((void **)&d_A2, size_A2)); // Memoria para predicciones Capa 2

  // 4. TRANSFERENCIA AL DEVICE
  CHECK_CUDA(cudaMemcpy(d_X_train, h_X_train, size_X, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_y_train, h_y_train, size_y, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_W1, h_W1, size_W1, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b1, h_b1, size_b1, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_W2, h_W2, size_W2, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b2, h_b2, size_b2, cudaMemcpyHostToDevice));

  // 5. EJECUCIÓN DE KERNELS (FORWARD PASS)
  printf("5. Ejecutando Forward Pass en CUDA...\n");

  // Configuración del Grid y Bloques (32x32 hilos por bloque es altamente
  // eficiente)
  dim3 threadsPerBlock(32, 32);

  // Calcular la grilla necesaria para cubrir todas las imágenes y neuronas
  dim3 numBlocks1((HIDDEN_SIZE + threadsPerBlock.x - 1) / threadsPerBlock.x,
                  (num_images + threadsPerBlock.y - 1) / threadsPerBlock.y);

  dim3 numBlocks2((OUTPUT_SIZE + threadsPerBlock.x - 1) / threadsPerBlock.x,
                  (num_images + threadsPerBlock.y - 1) / threadsPerBlock.y);

  // Lanzamiento asíncrono
  forward_layer1<<<numBlocks1, threadsPerBlock>>>(d_X_train, d_W1, d_b1, d_A1,
                                                  num_images);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize()); // Sincronizar para asegurar que A1
                                       // terminó de calcularse

  forward_layer2<<<numBlocks2, threadsPerBlock>>>(d_A1, d_W2, d_b2, d_A2,
                                                  num_images);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(
      cudaDeviceSynchronize()); // Sincronizar para asegurar que A2 terminó

  printf("-> [ÉXITO] Forward Pass completado. Las predicciones (A2) están en "
         "la VRAM.\n");

  // Limpieza
  free(h_X_train);
  free(h_y_train);
  free(h_W1);
  free(h_b1);
  free(h_W2);
  free(h_b2);
  cudaFree(d_X_train);
  cudaFree(d_y_train);
  cudaFree(d_W1);
  cudaFree(d_b1);
  cudaFree(d_W2);
  cudaFree(d_b2);
  cudaFree(d_A1);
  cudaFree(d_A2);

  return 0;
}