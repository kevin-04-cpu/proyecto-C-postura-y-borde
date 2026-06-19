#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>

#define INPUT_SIZE 4096
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 1

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("[Error CUDA] %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// -------------------------------------------------------------------------
// KERNELS: FORWARD PASS
// -------------------------------------------------------------------------

__global__ void forward_layer1(float* X, float* W1, float* b1, float* A1, int num_images) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    if (row < num_images && col < HIDDEN_SIZE) {
        float sum = 0.0f;
        for (int i = 0; i < INPUT_SIZE; i++) {
            sum += X[row * INPUT_SIZE + i] * W1[col * INPUT_SIZE + i];
        }
        sum += b1[col]; 
        A1[row * HIDDEN_SIZE + col] = fmaxf(0.0f, sum); // ReLU
    }
}

__global__ void forward_layer2(float* A1, float* W2, float* b2, float* A2, int num_images) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    if (row < num_images && col < OUTPUT_SIZE) {
        float sum = 0.0f;
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            sum += A1[row * HIDDEN_SIZE + i] * W2[col * HIDDEN_SIZE + i];
        }
        sum += b2[col];
        A2[row * OUTPUT_SIZE + col] = 1.0f / (1.0f + expf(-sum)); // Sigmoide
    }
}

// -------------------------------------------------------------------------
// KERNELS: BACKWARD PASS & SGD
// -------------------------------------------------------------------------

__global__ void compute_dz2(float* A2, int* Y, float* dZ2, int num_images) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_images) {
        dZ2[i] = A2[i] - (float)Y[i]; // Derivada BCE + Sigmoide
    }
}

__global__ void compute_dz1(float* dZ2, float* W2, float* A1, float* dZ1, int num_images) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    if (row < num_images && col < HIDDEN_SIZE) {
        float relu_deriv = (A1[row * HIDDEN_SIZE + col] > 0.0f) ? 1.0f : 0.0f;
        dZ1[row * HIDDEN_SIZE + col] = dZ2[row] * W2[col] * relu_deriv;
    }
}

__global__ void update_layer2(float* dZ2, float* A1, float* W2, float* b2, float learning_rate, int num_images) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    if (col < HIDDEN_SIZE) {
        float dw = 0.0f;
        for (int i = 0; i < num_images; i++) {
            dw += dZ2[i] * A1[i * HIDDEN_SIZE + col];
        }
        W2[col] -= learning_rate * (dw / num_images);

        // El hilo 0 se encarga del bias para evitar condiciones de carrera
        if (col == 0) {
            float db = 0.0f;
            for (int i = 0; i < num_images; i++) db += dZ2[i];
            b2[0] -= learning_rate * (db / num_images);
        }
    }
}

__global__ void update_layer1(float* dZ1, float* X, float* W1, float* b1, float learning_rate, int num_images) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Neurona oculta (0 a 127)
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Píxel de entrada (0 a 4095)

    if (row < HIDDEN_SIZE && col < INPUT_SIZE) {
        float dw = 0.0f;
        for (int i = 0; i < num_images; i++) {
            dw += dZ1[i * HIDDEN_SIZE + row] * X[i * INPUT_SIZE + col];
        }
        W1[row * INPUT_SIZE + col] -= learning_rate * (dw / num_images);

        // La columna 0 de cada fila (neurona) actualiza su respectivo bias
        if (col == 0) {
            float db = 0.0f;
            for (int i = 0; i < num_images; i++) db += dZ1[i * HIDDEN_SIZE + row];
            b1[row] -= learning_rate * (db / num_images);
        }
    }
}

// -------------------------------------------------------------------------
// FUNCIONES DE HOST (CPU)
// -------------------------------------------------------------------------

float calculate_bce_loss(float* h_A2, int* h_y, int num_images) {
    float loss = 0.0f;
    for (int i = 0; i < num_images; i++) {
        // Clamping para evitar log(0) que genera NaN
        float a = fmaxf(1e-7f, fminf(1.0f - 1e-7f, h_A2[i])); 
        loss += - (h_y[i] * logf(a) + (1 - h_y[i]) * logf(1 - a));
    }
    return loss / num_images;
}

int load_bin_float(const char* filepath, float** data) {
    FILE* f = fopen(filepath, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END); long filesize = ftell(f); rewind(f);
    int num_elements = filesize / sizeof(float);
    *data = (float*)malloc(filesize);
    fread(*data, sizeof(float), num_elements, f);
    fclose(f); return num_elements;
}

int load_bin_int(const char* filepath, int** data) {
    FILE* f = fopen(filepath, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END); long filesize = ftell(f); rewind(f);
    int num_elements = filesize / sizeof(int);
    *data = (int*)malloc(filesize);
    fread(*data, sizeof(int), num_elements, f);
    fclose(f); return num_elements;
}

void init_weights(float* array, int size, int input_connections) {
    float limit = sqrt(2.0f / input_connections);
    for (int i = 0; i < size; i++) {
        float rand_val = (float)rand() / (float)RAND_MAX;
        array[i] = (rand_val * 2.0f - 1.0f) * limit; 
    }
}

// -------------------------------------------------------------------------
// PROGRAMA PRINCIPAL
// -------------------------------------------------------------------------

int main() {
    srand(time(NULL));

    printf("=== INICIANDO MOTOR DE ENTRENAMIENTO CUDA ===\n");

    // =========================================================================
    // 1. CARGA DE TODOS LOS DATASETS EN EL HOST (CPU)
    // =========================================================================
    float *h_X_train, *h_X_val, *h_X_test;
    int *h_y_train, *h_y_val, *h_y_test;

    printf("1. Cargando tensores (Train, Val, Test) desde almacenamiento...\n");
    
    // Carga Train
    int total_features_train = load_bin_float("../dataset/processed/X_train.bin", &h_X_train);
    int num_images_train = total_features_train / INPUT_SIZE;
    load_bin_int("../dataset/processed/y_train.bin", &h_y_train);

    // Carga Validación
    int total_features_val = load_bin_float("../dataset/processed/X_val.bin", &h_X_val);
    int num_images_val = total_features_val / INPUT_SIZE;
    load_bin_int("../dataset/processed/y_val.bin", &h_y_val);

    // Carga Test
    int total_features_test = load_bin_float("../dataset/processed/X_test.bin", &h_X_test);
    int num_images_test = total_features_test / INPUT_SIZE;
    load_bin_int("../dataset/processed/y_test.bin", &h_y_test);

    printf("-> [OK] Datasets cargados:\n   - Train: %d imágenes\n   - Val:   %d imágenes\n   - Test:  %d imágenes\n", 
           num_images_train, num_images_val, num_images_test);

    // Arrancar contenedores para evaluar pérdidas en CPU
    float* h_A2_train = (float*)malloc(num_images_train * sizeof(float));
    float* h_A2_val = (float*)malloc(num_images_val * sizeof(float));

    // =========================================================================
    // 2. INICIALIZACIÓN DE PARÁMETROS
    // =========================================================================
    float *h_W1 = (float*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    float *h_b1 = (float*)calloc(HIDDEN_SIZE, sizeof(float));
    float *h_W2 = (float*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    float *h_b2 = (float*)calloc(OUTPUT_SIZE, sizeof(float));
    init_weights(h_W1, HIDDEN_SIZE * INPUT_SIZE, INPUT_SIZE);
    init_weights(h_W2, OUTPUT_SIZE * HIDDEN_SIZE, HIDDEN_SIZE);

    // =========================================================================
    // 3. RESERVA DE MEMORIA GLOBAL EN DEVICE (VRAM)
    // =========================================================================
    float *d_X_train, *d_X_val, *d_X_test;
    float *d_W1, *d_b1, *d_W2, *d_b2;
    float *d_A1_train, *d_A2_train, *d_A1_val, *d_A2_val, *d_A1_test, *d_A2_test;
    float *d_Z1, *d_Z2;
    int *d_y_train, *d_y_val;

    // Alloc datasets
    CHECK_CUDA(cudaMalloc((void**)&d_X_train, num_images_train * INPUT_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_y_train, num_images_train * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_X_val, num_images_val * INPUT_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_y_val, num_images_val * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_X_test, num_images_test * INPUT_SIZE * sizeof(float)));

    // Alloc pesos
    CHECK_CUDA(cudaMalloc((void**)&d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_b1, HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_b2, OUTPUT_SIZE * sizeof(float)));

    // Alloc Activaciones intermedio/salida
    CHECK_CUDA(cudaMalloc((void**)&d_A1_train, num_images_train * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_A2_train, num_images_train * OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_A1_val, num_images_val * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_A2_val, num_images_val * OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_A1_test, num_images_test * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_A2_test, num_images_test * OUTPUT_SIZE * sizeof(float)));

    // Alloc matrices de error (exclusivas del tamaño de entrenamiento)
    CHECK_CUDA(cudaMalloc((void**)&d_Z1, num_images_train * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_Z2, num_images_train * sizeof(float)));

    // =========================================================================
    // 4. TRANSFERENCIA INICIAL HOST -> DEVICE
    // =========================================================================
    CHECK_CUDA(cudaMemcpy(d_X_train, h_X_train, num_images_train * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y_train, h_y_train, num_images_train * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_X_val, h_X_val, num_images_val * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y_val, h_y_val, num_images_val * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_X_test, h_X_test, num_images_test * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    
    CHECK_CUDA(cudaMemcpy(d_W1, h_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b1, h_b1, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W2, h_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b2, h_b2, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    // =========================================================================
    // 5. CONFIGURACIÓN DE GRIDS DINÁMICAS
    // =========================================================================
    dim3 threadsPerBlock(32, 32);
    
    // Grids para Train
    dim3 grid_FW1_train((HIDDEN_SIZE + 31) / 32, (num_images_train + 31) / 32);
    dim3 grid_FW2_train((OUTPUT_SIZE + 31) / 32, (num_images_train + 31) / 32);
    
    // Grids para Validación
    dim3 grid_FW1_val((HIDDEN_SIZE + 31) / 32, (num_images_val + 31) / 32);
    dim3 grid_FW2_val((OUTPUT_SIZE + 31) / 32, (num_images_val + 31) / 32);

    int threads1D = 256;
    int blocks1D_train = (num_images_train + threads1D - 1) / threads1D;
    
    dim3 grid_BW2((HIDDEN_SIZE + 31) / 32, 1);
    dim3 grid_BW1((INPUT_SIZE + 31) / 32, (HIDDEN_SIZE + 31) / 32);

    // =========================================================================
    // 6. BUCLE DE ENTRENAMIENTO CON MONITOREO DE VALIDACIÓN
    // =========================================================================
    int epochs = 1000;
    float learning_rate = 0.05f;

    printf("\n-> Hiperparámetros: Épocas = %d | LR = %.3f\n", epochs, learning_rate);
    printf("==================================================\n");

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start)); CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, 0));

    for (int epoch = 1; epoch <= epochs; epoch++) {
        // --- 6.1 FORWARD & BACKWARD (TRAIN DATA) ---
        forward_layer1<<<grid_FW1_train, threadsPerBlock>>>(d_X_train, d_W1, d_b1, d_A1_train, num_images_train);
        forward_layer2<<<grid_FW2_train, threadsPerBlock>>>(d_A1_train, d_W2, d_b2, d_A2_train, num_images_train);

        compute_dz2<<<blocks1D_train, threads1D>>>(d_A2_train, d_y_train, d_Z2, num_images_train);
        compute_dz1<<<grid_FW1_train, threadsPerBlock>>>(d_Z2, d_W2, d_A1_train, d_Z1, num_images_train);

        update_layer2<<<grid_BW2, threadsPerBlock>>>(d_Z2, d_A1_train, d_W2, d_b2, learning_rate, num_images_train);
        update_layer1<<<grid_BW1, threadsPerBlock>>>(d_Z1, d_X_train, d_W1, d_b1, learning_rate, num_images_train);
        
        // --- 6.2 EVALUAR VALIDACIÓN (SOLO FORWARD, NO ACTUALIZA PESOS) ---
        if (epoch % 100 == 0 || epoch == 1) {
            forward_layer1<<<grid_FW1_val, threadsPerBlock>>>(d_X_val, d_W1, d_b1, d_A1_val, num_images_val);
            forward_layer2<<<grid_FW2_val, threadsPerBlock>>>(d_A1_val, d_W2, d_b2, d_A2_val, num_images_val);
            
            CHECK_CUDA(cudaDeviceSynchronize());

            // Descargar predicciones de Train y Val a la CPU
            CHECK_CUDA(cudaMemcpy(h_A2_train, d_A2_train, num_images_train * sizeof(float), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(h_A2_val, d_A2_val, num_images_val * sizeof(float), cudaMemcpyDeviceToHost));

            float train_loss = calculate_bce_loss(h_A2_train, h_y_train, num_images_train);
            float val_loss = calculate_bce_loss(h_A2_val, h_y_val, num_images_val);

            printf("Época %4d/%d -> Train Loss: %f | Val Loss: %f\n", epoch, epochs, train_loss, val_loss);
        }
    }

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float gpu_time = 0; CHECK_CUDA(cudaEventElapsedTime(&gpu_time, start, stop));
    CHECK_CUDA(cudaEventDestroy(start)); CHECK_CUDA(cudaEventDestroy(stop));

    printf("==================================================\n");
    printf("-> [ÉXITO] Entrenamiento CUDA finalizado.\n");
    printf("Tiempo total de entrenamiento en GPU: %f segundos\n", gpu_time / 1000.0f);
    printf("==================================================\n");

    // =========================================================================
    // 7. EVALUACIÓN FINAL DEFINITIVA CON EL CONJUNTO DE PRUEBA (TEST SET)
    // =========================================================================
    printf("7. Iniciando Inferencia sobre el Dataset de Prueba (Caja Negra)...\n");
    
    dim3 grid_test1((HIDDEN_SIZE + 31) / 32, (num_images_test + 31) / 32);
    dim3 grid_test2((OUTPUT_SIZE + 31) / 32, (num_images_test + 31) / 32);

    forward_layer1<<<grid_test1, threadsPerBlock>>>(d_X_test, d_W1, d_b1, d_A1_test, num_images_test);
    forward_layer2<<<grid_test2, threadsPerBlock>>>(d_A1_test, d_W2, d_b2, d_A2_test, num_images_test);
    CHECK_CUDA(cudaDeviceSynchronize());

    float* h_A2_test = (float*)malloc(num_images_test * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_A2_test, d_A2_test, num_images_test * sizeof(float), cudaMemcpyDeviceToHost));

    int tp = 0, tn = 0, fp = 0, fn = 0;
    for (int i = 0; i < num_images_test; i++) {
        int pred = (h_A2_test[i] >= 0.5f) ? 1 : 0;
        int real = h_X_test[i]; // Nota: la carga lee etiquetas y características en orden
        
        if (h_y_test[i] == 1 && pred == 1) tp++;
        else if (h_y_test[i] == 0 && pred == 0) tn++;
        else if (h_y_test[i] == 0 && pred == 1) fp++;
        else if (h_y_test[i] == 1 && pred == 0) fn++;
    }

    printf("\n=== METRICAS DE EVALUACION FINAL (TEST DATA) ===\n");
    printf("Exactitud (Accuracy) Final: %.2f%%\n", ((float)(tp + tn) / num_images_test) * 100.0f);
    printf("--------------------------------------------------\n");
    printf("Matriz de Confusión:\n");
    printf("               Pred. Recta (0)   Pred. Encorvada (1)\n");
    printf("Real Recta (0):      %d                %d\n", tn, fp);
    printf("Real Encorv. (1):    %d                %d\n", fn, tp);
    printf("==================================================\n");

    // =========================================================================
    // 8. EXPORTAR PESOS PARA STREAMLIT
    // =========================================================================
    printf("8. Volcando parámetros optimizados a disco...\n");
    float* final_W1 = (float*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    float* final_b1 = (float*)malloc(HIDDEN_SIZE * sizeof(float));
    float* final_W2 = (float*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    float* final_b2 = (float*)malloc(OUTPUT_SIZE * sizeof(float));

    CHECK_CUDA(cudaMemcpy(final_W1, d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(final_b1, d_b1, HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(final_W2, d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(final_b2, d_b2, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    FILE* fw1 = fopen("../dataset/processed/W1.bin", "wb"); FILE* fb1 = fopen("../dataset/processed/b1.bin", "wb");
    FILE* fw2 = fopen("../dataset/processed/W2.bin", "wb"); FILE* fb2 = fopen("../dataset/processed/b2.bin", "wb");

    if (fw1 && fb1 && fw2 && fb2) {
        fwrite(final_W1, sizeof(float), HIDDEN_SIZE * INPUT_SIZE, fw1);
        fwrite(final_b1, sizeof(float), HIDDEN_SIZE, fb1);
        fwrite(final_W2, sizeof(float), OUTPUT_SIZE * HIDDEN_SIZE, fw2);
        fwrite(final_b2, sizeof(float), OUTPUT_SIZE, fb2);
        fclose(fw1); fclose(fb1); fclose(fw2); fclose(fb2);
        printf("-> [OK] Modelos exportados correctamente.\n");
    }
    printf("==================================================\n");

    // Libres
    free(h_X_train); free(h_y_train); free(h_X_val); free(h_y_val); free(h_X_test); free(h_y_test);
    free(h_A2_train); free(h_A2_val); free(h_A2_test); free(final_W1); free(final_b1); free(final_W2); free(final_b2);
    cudaFree(d_X_train); cudaFree(d_y_train); cudaFree(d_X_val); cudaFree(d_y_val); cudaFree(d_X_test);
    cudaFree(d_W1); cudaFree(d_b1); cudaFree(d_W2); cudaFree(d_b2);
    cudaFree(d_A1_train); cudaFree(d_A2_train); cudaFree(d_A1_val); cudaFree(d_A2_val); cudaFree(d_A1_test); cudaFree(d_A2_test);
    cudaFree(d_Z1); cudaFree(d_Z2);

    return 0;
}
