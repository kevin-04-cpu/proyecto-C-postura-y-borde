#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>

#define INPUT_SIZE 4096
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 1
#define BLOCK_SIZE 8

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

__global__ void update_layer2(float* dZ2, float* A1, float* W2, float* b2, float learning_rate, float l2_lambda, int num_images) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    if (col < HIDDEN_SIZE) {
        float dw = 0.0f;
        for (int i = 0; i < num_images; i++) {
            dw += dZ2[i] * A1[i * HIDDEN_SIZE + col];
        }
        // Regularización L2 (weight decay): penaliza pesos grandes para reducir
        // el sobreajuste. No se aplica al bias (práctica estándar).
        W2[col] -= learning_rate * (dw / num_images + l2_lambda * W2[col]);

        // El hilo 0 se encarga del bias para evitar condiciones de carrera
        if (col == 0) {
            float db = 0.0f;
            for (int i = 0; i < num_images; i++) db += dZ2[i];
            b2[0] -= learning_rate * (db / num_images);
        }
    }
}

__global__ void update_layer1(float* dZ1, float* X, float* W1, float* b1, float learning_rate, float l2_lambda, int num_images) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Neurona oculta (0 a 127)
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Píxel de entrada (0 a 4095)

    if (row < HIDDEN_SIZE && col < INPUT_SIZE) {
        float dw = 0.0f;
        for (int i = 0; i < num_images; i++) {
            dw += dZ1[i * HIDDEN_SIZE + row] * X[i * INPUT_SIZE + col];
        }
        int idx = row * INPUT_SIZE + col;
        W1[idx] -= learning_rate * (dw / num_images + l2_lambda * W1[idx]);

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

// Exactitud (accuracy) a partir de las predicciones (umbral 0.5)
float calculate_accuracy(float* h_A2, int* h_y, int num_images) {
    int correct = 0;
    for (int i = 0; i < num_images; i++) {
        int pred = (h_A2[i] >= 0.5f) ? 1 : 0;
        if (pred == h_y[i]) correct++;
    }
    return (float)correct / num_images;
}

// Calcula TP/TN/FP/FN, y a partir de ellos precisión, recall y F1
typedef struct {
    int tp, tn, fp, fn;
    float accuracy, precision, recall, f1;
} ConfusionMetrics;

ConfusionMetrics calculate_confusion_metrics(float* h_A2, int* h_y, int num_images) {
    ConfusionMetrics m = {0, 0, 0, 0, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = 0; i < num_images; i++) {
        int pred = (h_A2[i] >= 0.5f) ? 1 : 0;
        int real = h_y[i];
        if (real == 1 && pred == 1) m.tp++;
        else if (real == 0 && pred == 0) m.tn++;
        else if (real == 0 && pred == 1) m.fp++;
        else if (real == 1 && pred == 0) m.fn++;
    }
    m.accuracy = (float)(m.tp + m.tn) / num_images;
    m.precision = (m.tp + m.fp > 0) ? (float)m.tp / (m.tp + m.fp) : 0.0f;
    m.recall    = (m.tp + m.fn > 0) ? (float)m.tp / (m.tp + m.fn) : 0.0f;
    m.f1 = (m.precision + m.recall > 0.0f)
               ? 2.0f * (m.precision * m.recall) / (m.precision + m.recall)
               : 0.0f;
    return m;
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
    // Pesos iniciados aleatoriamente en cada corrida (producción).
    // Si necesitas reproducibilidad para depurar, cambia esto por srand(42).
    srand(time(NULL));

    printf("=== INICIANDO MOTOR DE ENTRENAMIENTO CUDA (block=%dx%d) ===\n", BLOCK_SIZE, BLOCK_SIZE);

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
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    
    // Grids para Train
    dim3 grid_FW1_train((HIDDEN_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, (num_images_train + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 grid_FW2_train((OUTPUT_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, (num_images_train + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Grids para Validación
    dim3 grid_FW1_val((HIDDEN_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, (num_images_val + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 grid_FW2_val((OUTPUT_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, (num_images_val + BLOCK_SIZE - 1) / BLOCK_SIZE);

    int threads1D = 256;
    int blocks1D_train = (num_images_train + threads1D - 1) / threads1D;
    
    dim3 grid_BW2((HIDDEN_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
    dim3 grid_BW1((INPUT_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, (HIDDEN_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // =========================================================================
    // 6. BUCLE DE ENTRENAMIENTO CON MONITOREO DE VALIDACIÓN
    // =========================================================================
    int epochs = 1000;
    float learning_rate = 0.05f;

    // Regularización L2 (weight decay): reduce el sobreajuste penalizando
    // pesos grandes. Early stopping: si la pérdida de validación no mejora
    // durante `patience` evaluaciones, se detiene y se usan los pesos de la
    // mejor época (no los de la última, que suelen estar sobreajustados).
    float l2_lambda = 0.0005f;
    int patience = 4;

    printf("\n-> Hiperparámetros: Épocas = %d | LR = %.3f | L2 = %.5f | Paciencia = %d\n",
           epochs, learning_rate, l2_lambda, patience);
    printf("==================================================\n");

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start)); CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, 0));

    int log_every = 100; // imprime progreso cada 100 épocas

    // --- Buffers para guardar el MEJOR modelo visto durante el entrenamiento
    // (el de menor val_loss), en vez de quedarnos solo con los pesos de la
    // última época, que suelen estar sobreajustados. ---
    float* best_W1 = (float*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    float* best_b1 = (float*)malloc(HIDDEN_SIZE * sizeof(float));
    float* best_W2 = (float*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    float* best_b2 = (float*)malloc(OUTPUT_SIZE * sizeof(float));
    float best_val_loss = 1e30f;
    int best_epoch = 0;
    int evals_without_improvement = 0;

    for (int epoch = 1; epoch <= epochs; epoch++) {
        // --- 6.1 FORWARD & BACKWARD (TRAIN DATA) ---
        forward_layer1<<<grid_FW1_train, threadsPerBlock>>>(d_X_train, d_W1, d_b1, d_A1_train, num_images_train);
        forward_layer2<<<grid_FW2_train, threadsPerBlock>>>(d_A1_train, d_W2, d_b2, d_A2_train, num_images_train);

        compute_dz2<<<blocks1D_train, threads1D>>>(d_A2_train, d_y_train, d_Z2, num_images_train);
        compute_dz1<<<grid_FW1_train, threadsPerBlock>>>(d_Z2, d_W2, d_A1_train, d_Z1, num_images_train);

        update_layer2<<<grid_BW2, threadsPerBlock>>>(d_Z2, d_A1_train, d_W2, d_b2, learning_rate, l2_lambda, num_images_train);
        update_layer1<<<grid_BW1, threadsPerBlock>>>(d_Z1, d_X_train, d_W1, d_b1, learning_rate, l2_lambda, num_images_train);
        
        // --- 6.2 EVALUAR VALIDACIÓN (SOLO FORWARD, NO ACTUALIZA PESOS) ---
        if (epoch % log_every == 0 || epoch == 1 || epoch == epochs) {
            forward_layer1<<<grid_FW1_val, threadsPerBlock>>>(d_X_val, d_W1, d_b1, d_A1_val, num_images_val);
            forward_layer2<<<grid_FW2_val, threadsPerBlock>>>(d_A1_val, d_W2, d_b2, d_A2_val, num_images_val);
            
            CHECK_CUDA(cudaDeviceSynchronize());

            // Descargar predicciones de Train y Val a la CPU
            CHECK_CUDA(cudaMemcpy(h_A2_train, d_A2_train, num_images_train * sizeof(float), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(h_A2_val, d_A2_val, num_images_val * sizeof(float), cudaMemcpyDeviceToHost));

            float train_loss = calculate_bce_loss(h_A2_train, h_y_train, num_images_train);
            float val_loss = calculate_bce_loss(h_A2_val, h_y_val, num_images_val);
            float train_acc = calculate_accuracy(h_A2_train, h_y_train, num_images_train);
            float val_acc = calculate_accuracy(h_A2_val, h_y_val, num_images_val);

            printf("Época %4d/%d -> Train Loss: %f | Val Loss: %f | Train Acc: %.2f%% | Val Acc: %.2f%%\n",
                   epoch, epochs, train_loss, val_loss, train_acc * 100.0f, val_acc * 100.0f);

            // --- Guardar el mejor modelo visto hasta ahora (menor val_loss) ---
            if (val_loss < best_val_loss) {
                best_val_loss = val_loss;
                best_epoch = epoch;
                evals_without_improvement = 0;
                CHECK_CUDA(cudaMemcpy(best_W1, d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaMemcpy(best_b1, d_b1, HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaMemcpy(best_W2, d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaMemcpy(best_b2, d_b2, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
            } else {
                evals_without_improvement++;
            }

            // --- Early stopping: si val_loss no mejora en `patience` evaluaciones
            // seguidas, detenemos el entrenamiento para no seguir sobreajustando ---
            if (evals_without_improvement >= patience) {
                printf("-> [EARLY STOPPING] Sin mejora en val_loss durante %d evaluaciones. "
                       "Mejor época: %d (val_loss=%f). Deteniendo entrenamiento.\n",
                       patience, best_epoch, best_val_loss);
                break;
            }
        }
    }

    // --- Restaurar en la GPU los pesos del MEJOR modelo (menor val_loss),
    // no los de la última época, antes de evaluar en test y exportar. ---
    if (best_epoch > 0) {
        printf("-> Restaurando pesos de la mejor época (%d, val_loss=%f) para la evaluación final.\n",
               best_epoch, best_val_loss);
        CHECK_CUDA(cudaMemcpy(d_W1, best_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_b1, best_b1, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_W2, best_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_b2, best_b2, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    }
    free(best_W1); free(best_b1); free(best_W2); free(best_b2);

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
    
    dim3 grid_test1((HIDDEN_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, (num_images_test + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 grid_test2((OUTPUT_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, (num_images_test + BLOCK_SIZE - 1) / BLOCK_SIZE);

    forward_layer1<<<grid_test1, threadsPerBlock>>>(d_X_test, d_W1, d_b1, d_A1_test, num_images_test);
    forward_layer2<<<grid_test2, threadsPerBlock>>>(d_A1_test, d_W2, d_b2, d_A2_test, num_images_test);
    CHECK_CUDA(cudaDeviceSynchronize());

    float* h_A2_test = (float*)malloc(num_images_test * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_A2_test, d_A2_test, num_images_test * sizeof(float), cudaMemcpyDeviceToHost));

    ConfusionMetrics m = calculate_confusion_metrics(h_A2_test, h_y_test, num_images_test);

    printf("\n=== METRICAS DE EVALUACION FINAL (TEST DATA) ===\n");
    printf("Exactitud (Accuracy):  %.2f%%\n", m.accuracy * 100.0f);
    printf("Precision:             %.4f\n", m.precision);
    printf("Recall:                %.4f\n", m.recall);
    printf("F1-Score:              %.4f\n", m.f1);
    printf("--------------------------------------------------\n");
    printf("Matriz de Confusión:\n");
    printf("                 Pred. 0          Pred. 1\n");
    printf("Real 0:          %4d             %4d\n", m.tn, m.fp);
    printf("Real 1:          %4d             %4d\n", m.fn, m.tp);
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

        FILE* fw1 = fopen("../weights/W1.bin", "wb"); FILE* fb1 = fopen("../weights/b1.bin", "wb");
    FILE* fw2 = fopen("../weights/W2.bin", "wb"); FILE* fb2 = fopen("../weights/b2.bin", "wb");

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
