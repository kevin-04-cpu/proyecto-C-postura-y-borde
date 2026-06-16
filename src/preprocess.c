#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>

#define INPUT_SIZE 4096
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 1

// 1. MACRO PARA ATRAPAR ERRORES DE CUDA AL VUELO
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("[Error CUDA] %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// (Tus funciones load_bin_float y load_bin_int van aquí, no las borres)
int load_bin_float(const char* filepath, float** data) {
    FILE* f = fopen(filepath, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long filesize = ftell(f);
    rewind(f);
    int num_elements = filesize / sizeof(float);
    *data = (float*)malloc(filesize);
    size_t read_elements = fread(*data, sizeof(float), num_elements, f);
    fclose(f);
    return num_elements;
}

int load_bin_int(const char* filepath, int** data) {
    FILE* f = fopen(filepath, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long filesize = ftell(f);
    rewind(f);
    int num_elements = filesize / sizeof(int);
    *data = (int*)malloc(filesize);
    size_t read_elements = fread(*data, sizeof(int), num_elements, f);
    fclose(f);
    return num_elements;
}

// 2. Función para inicializar pesos aleatorios (He Initialization simplificada)
void init_weights(float* array, int size, int input_connections) {
    float limit = sqrt(2.0f / input_connections);
    for (int i = 0; i < size; i++) {
        float rand_val = (float)rand() / (float)RAND_MAX; // [0, 1]
        array[i] = (rand_val * 2.0f - 1.0f) * limit;      // [-limit, limit]
    }
}

int main() {
    srand(time(NULL)); // Semilla para la aleatoriedad de los pesos

    printf("=== INICIANDO MOTOR DE ENTRENAMIENTO CUDA ===\n");

    // ==========================================
    // FASE 1: HOST (CPU)
    // ==========================================
    float* h_X_train;
    int* h_y_train;

    printf("1. Cargando tensores desde almacenamiento...\n");
    int total_features = load_bin_float("dataset/processed/X_train.bin", &h_X_train);
    int total_labels = load_bin_int("dataset/processed/y_train.bin", &h_y_train);

    if (total_features <= 0 || total_labels <= 0) return -1;
    int num_images = total_features / INPUT_SIZE;

    printf("-> [OK] Dataset listo en RAM. Imagenes: %d\n", num_images);

    // Inicializando pesos y sesgos en el Host
    printf("2. Inicializando arquitectura MLP (Pesos y Sesgos)...\n");
    float *h_W1 = (float*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    float *h_b1 = (float*)calloc(HIDDEN_SIZE, sizeof(float)); // Sesgos inician en 0
    float *h_W2 = (float*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    float *h_b2 = (float*)calloc(OUTPUT_SIZE, sizeof(float));

    init_weights(h_W1, HIDDEN_SIZE * INPUT_SIZE, INPUT_SIZE);
    init_weights(h_W2, OUTPUT_SIZE * HIDDEN_SIZE, HIDDEN_SIZE);

    // ==========================================
    // FASE 2: DEVICE (GPU)
    // ==========================================
    printf("3. Reservando memoria en la GPU (cudaMalloc)...\n");
    float *d_X_train, *d_W1, *d_b1, *d_W2, *d_b2;
    int *d_y_train;

    // Tamaños en bytes
    size_t size_X = num_images * INPUT_SIZE * sizeof(float);
    size_t size_y = num_images * sizeof(int);
    size_t size_W1 = HIDDEN_SIZE * INPUT_SIZE * sizeof(float);
    size_t size_b1 = HIDDEN_SIZE * sizeof(float);
    size_t size_W2 = OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float);
    size_t size_b2 = OUTPUT_SIZE * sizeof(float);

    // Asignación en VRAM
    CHECK_CUDA(cudaMalloc((void**)&d_X_train, size_X));
    CHECK_CUDA(cudaMalloc((void**)&d_y_train, size_y));
    CHECK_CUDA(cudaMalloc((void**)&d_W1, size_W1));
    CHECK_CUDA(cudaMalloc((void**)&d_b1, size_b1));
    CHECK_CUDA(cudaMalloc((void**)&d_W2, size_W2));
    CHECK_CUDA(cudaMalloc((void**)&d_b2, size_b2));

    printf("4. Transfiriendo datos Host -> Device (cudaMemcpy)...\n");
    CHECK_CUDA(cudaMemcpy(d_X_train, h_X_train, size_X, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y_train, h_y_train, size_y, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W1, h_W1, size_W1, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b1, h_b1, size_b1, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W2, h_W2, size_W2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b2, h_b2, size_b2, cudaMemcpyHostToDevice));

    printf("-> [ÉXITO] Todo el modelo y dataset estan alojados en la VRAM de la GTX 1650.\n");

    // Limpieza de memoria temporal en Host
    free(h_X_train); free(h_y_train);
    free(h_W1); free(h_b1); free(h_W2); free(h_b2);

    // Limpieza de memoria en Device
    cudaFree(d_X_train); cudaFree(d_y_train);
    cudaFree(d_W1); cudaFree(d_b1); cudaFree(d_W2); cudaFree(d_b2);

    return 0;
}