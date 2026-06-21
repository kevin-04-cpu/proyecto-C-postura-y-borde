/* =========================================================================
 * train_cpu.c
 * Versión EN SERIE (1 solo hilo de CPU) de la misma red que train.cu, usada
 * únicamente para medir el tiempo de entrenamiento en CPU y compararlo
 * contra el tiempo en GPU (speedup = tiempo_cpu / tiempo_gpu).
 *
 * Misma arquitectura, mismos datos, mismo número de épocas y misma tasa de
 * aprendizaje que train.cu, para que la comparación sea justa.
 *
 * Compilar:   gcc -O2 -o train_cpu train_cpu.c -lm
 * Ejecutar:   ./train_cpu [epochs]
 *             ./train_cpu          -> usa 1000 épocas (igual que train.cu)
 *             ./train_cpu 50       -> benchmark corto de 50 épocas
 * ========================================================================= */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define INPUT_SIZE 4096
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 1

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

float calculate_bce_loss(float* A2, int* y, int n) {
    float loss = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = fmaxf(1e-7f, fminf(1.0f - 1e-7f, A2[i]));
        loss += -(y[i] * logf(a) + (1 - y[i]) * logf(1 - a));
    }
    return loss / n;
}

float calculate_accuracy(float* A2, int* y, int n) {
    int correct = 0;
    for (int i = 0; i < n; i++) {
        int pred = (A2[i] >= 0.5f) ? 1 : 0;
        if (pred == y[i]) correct++;
    }
    return (float)correct / n;
}

int main(int argc, char** argv) {
    int seed = 42;
    int epochs = 1000;
    if (argc > 1) {
        int e = atoi(argv[1]);
        if (e > 0) epochs = e;
    }
    if (argc > 2) {
        seed = atoi(argv[2]);
    }
    srand(seed); // misma semilla que train.cu para que la comparación sea justa
    float learning_rate = 0.05f;

    printf("=== ENTRENAMIENTO CPU (SERIAL, 1 HILO, seed=%d) ===\n", seed);

    // --- 1. Carga de datos (mismos archivos que usa train.cu) ---
    float *X_train, *X_val;
    int *y_train, *y_val;

    int total_features_train = load_bin_float("../dataset/processed/X_train.bin", &X_train);
    if (total_features_train < 0) { printf("[ERROR] No se pudo abrir X_train.bin\n"); return 1; }
    int N = total_features_train / INPUT_SIZE;
    load_bin_int("../dataset/processed/y_train.bin", &y_train);

    int total_features_val = load_bin_float("../dataset/processed/X_val.bin", &X_val);
    int N_val = total_features_val / INPUT_SIZE;
    load_bin_int("../dataset/processed/y_val.bin", &y_val);

    printf("-> Imágenes de entrenamiento: %d | validación: %d | épocas: %d\n", N, N_val, epochs);

    // --- 2. Inicialización de pesos (mismo esquema que la GPU) ---
    float *W1 = (float*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    float *b1 = (float*)calloc(HIDDEN_SIZE, sizeof(float));
    float *W2 = (float*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    float *b2 = (float*)calloc(OUTPUT_SIZE, sizeof(float));
    init_weights(W1, HIDDEN_SIZE * INPUT_SIZE, INPUT_SIZE);
    init_weights(W2, OUTPUT_SIZE * HIDDEN_SIZE, HIDDEN_SIZE);

    float *A1 = (float*)malloc(N * HIDDEN_SIZE * sizeof(float));
    float *A2 = (float*)malloc(N * OUTPUT_SIZE * sizeof(float));
    float *A1_val = (float*)malloc(N_val * HIDDEN_SIZE * sizeof(float));
    float *A2_val = (float*)malloc(N_val * OUTPUT_SIZE * sizeof(float));
    float *dZ1 = (float*)malloc(N * HIDDEN_SIZE * sizeof(float));
    float *dZ2 = (float*)malloc(N * sizeof(float));

    FILE* f_history = fopen("../resultados/metrics_history_cpu.csv", "w");
    if (f_history) fprintf(f_history, "epoch,train_loss,val_loss,train_acc,val_acc\n");

    printf("==================================================\n");

    clock_t t_start = clock();

    for (int epoch = 1; epoch <= epochs; epoch++) {
        // --- FORWARD layer 1 ---
        for (int row = 0; row < N; row++) {
            for (int col = 0; col < HIDDEN_SIZE; col++) {
                float sum = 0.0f;
                for (int i = 0; i < INPUT_SIZE; i++) {
                    sum += X_train[row * INPUT_SIZE + i] * W1[col * INPUT_SIZE + i];
                }
                sum += b1[col];
                A1[row * HIDDEN_SIZE + col] = fmaxf(0.0f, sum); // ReLU
            }
        }

        // --- FORWARD layer 2 ---
        for (int row = 0; row < N; row++) {
            float sum = 0.0f;
            for (int i = 0; i < HIDDEN_SIZE; i++) {
                sum += A1[row * HIDDEN_SIZE + i] * W2[i];
            }
            sum += b2[0];
            A2[row] = 1.0f / (1.0f + expf(-sum)); // Sigmoide
        }

        // --- BACKWARD: dZ2 ---
        for (int i = 0; i < N; i++) {
            dZ2[i] = A2[i] - (float)y_train[i];
        }

        // --- BACKWARD: dZ1 ---
        for (int row = 0; row < N; row++) {
            for (int col = 0; col < HIDDEN_SIZE; col++) {
                float relu_deriv = (A1[row * HIDDEN_SIZE + col] > 0.0f) ? 1.0f : 0.0f;
                dZ1[row * HIDDEN_SIZE + col] = dZ2[row] * W2[col] * relu_deriv;
            }
        }

        // --- UPDATE layer 2 (W2, b2) ---
        for (int col = 0; col < HIDDEN_SIZE; col++) {
            float dw = 0.0f;
            for (int i = 0; i < N; i++) dw += dZ2[i] * A1[i * HIDDEN_SIZE + col];
            W2[col] -= learning_rate * (dw / N);
        }
        {
            float db = 0.0f;
            for (int i = 0; i < N; i++) db += dZ2[i];
            b2[0] -= learning_rate * (db / N);
        }

        // --- UPDATE layer 1 (W1, b1) ---
        for (int row = 0; row < HIDDEN_SIZE; row++) {
            for (int col = 0; col < INPUT_SIZE; col++) {
                float dw = 0.0f;
                for (int i = 0; i < N; i++) dw += dZ1[i * HIDDEN_SIZE + row] * X_train[i * INPUT_SIZE + col];
                W1[row * INPUT_SIZE + col] -= learning_rate * (dw / N);
            }
            float db = 0.0f;
            for (int i = 0; i < N; i++) db += dZ1[i * HIDDEN_SIZE + row];
            b1[row] -= learning_rate * (db / N);
        }

        // --- Evaluación de validación cada 100 épocas (igual que train.cu) ---
        if (epoch % 100 == 0 || epoch == 1) {
            for (int row = 0; row < N_val; row++) {
                for (int col = 0; col < HIDDEN_SIZE; col++) {
                    float sum = 0.0f;
                    for (int i = 0; i < INPUT_SIZE; i++) sum += X_val[row * INPUT_SIZE + i] * W1[col * INPUT_SIZE + i];
                    sum += b1[col];
                    A1_val[row * HIDDEN_SIZE + col] = fmaxf(0.0f, sum);
                }
                float sum2 = 0.0f;
                for (int i = 0; i < HIDDEN_SIZE; i++) sum2 += A1_val[row * HIDDEN_SIZE + i] * W2[i];
                sum2 += b2[0];
                A2_val[row] = 1.0f / (1.0f + expf(-sum2));
            }

            float train_loss = calculate_bce_loss(A2, y_train, N);
            float val_loss = calculate_bce_loss(A2_val, y_val, N_val);
            float train_acc = calculate_accuracy(A2, y_train, N);
            float val_acc = calculate_accuracy(A2_val, y_val, N_val);

            printf("Época %4d/%d -> Train Loss: %f | Val Loss: %f | Train Acc: %.2f%% | Val Acc: %.2f%%\n",
                   epoch, epochs, train_loss, val_loss, train_acc * 100.0f, val_acc * 100.0f);

            if (f_history) fprintf(f_history, "%d,%f,%f,%f,%f\n", epoch, train_loss, val_loss, train_acc, val_acc);
        }
    }

    clock_t t_end = clock();
    double cpu_time = (double)(t_end - t_start) / CLOCKS_PER_SEC;

    if (f_history) fclose(f_history);

    printf("==================================================\n");
    printf("-> [ÉXITO] Entrenamiento CPU finalizado.\n");
    printf("Tiempo total de entrenamiento en CPU: %f segundos\n", cpu_time);
    printf("==================================================\n");

    // Guardar el tiempo en un CSV para construir la tabla CPU vs GPU del reporte
    FILE* f_results = fopen("../resultados/cpu_results.csv", "a");
    if (f_results) {
        fseek(f_results, 0, SEEK_END);
        if (ftell(f_results) == 0) fprintf(f_results, "epochs,cpu_time_seconds,num_images_train\n");
        fprintf(f_results, "%d,%f,%d\n", epochs, cpu_time, N);
        fclose(f_results);
        printf("-> [OK] Resultado añadido a ../resultados/cpu_results.csv\n");
    }

    free(X_train); free(y_train); free(X_val); free(y_val);
    free(W1); free(b1); free(W2); free(b2);
    free(A1); free(A2); free(A1_val); free(A2_val); free(dZ1); free(dZ2);

    return 0;
}
