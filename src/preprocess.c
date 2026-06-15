#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

// Definiciones para STB (Deben ir antes de incluir los headers, y SÓLO en un archivo .c)
#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "../include/stb_image_resize2.h"

// Definiciones de la arquitectura del proyecto
#define IMG_SIZE 64
#define FLATTENED_SIZE (IMG_SIZE * IMG_SIZE)
#define MAX_PATH_LENGTH 256

typedef struct {
    char path[MAX_PATH_LENGTH];
    int label;
} ImageRecord;

// Función para leer el txt (Tu implementación original)
void load_dataset_list(const char* txt_path, ImageRecord** dataset, int* count) {
    FILE* file = fopen(txt_path, "r");
    if (!file) {
        printf("Error: No se pudo abrir %s\n", txt_path);
        exit(1);
    }

    // 1. Contar líneas primero
    *count = 0;
    char buffer[512];
    while (fgets(buffer, sizeof(buffer), file)) {
        // Ignorar líneas vacías si las hay
        if (strlen(buffer) > 1) {
            (*count)++;
        }
    }
    rewind(file);

    // 2. Reservar memoria
    *dataset = (ImageRecord*)malloc((*count) * sizeof(ImageRecord));

    // 3. Leer líneas de forma segura con fgets
    int i = 0;
    while (fgets(buffer, sizeof(buffer), file)) {
        // Eliminar saltos de línea (\n y \r) al final de la cadena
        buffer[strcspn(buffer, "\r\n")] = 0;

        if (strlen(buffer) == 0) continue;

        // Encontrar el último espacio que separa la ruta de la etiqueta
        char* last_space = strrchr(buffer, ' ');
        if (last_space != NULL) {
            *last_space = '\0'; // Dividimos la cadena en dos
            
            // Copiar la ruta de forma segura
            strncpy((*dataset)[i].path, buffer, MAX_PATH_LENGTH - 1);
            (*dataset)[i].path[MAX_PATH_LENGTH - 1] = '\0';
            
            // Convertir la etiqueta a entero
            (*dataset)[i].label = atoi(last_space + 1);
            i++;
        }
    }
    fclose(file);
}

// Implementación del Filtro Sobel (Paso Alto)
void apply_sobel(unsigned char* input, unsigned char* output, int width, int height) {
    int gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    // Bucle para recorrer la imagen ignorando el borde de 1 pixel
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int sum_x = 0;
            int sum_y = 0;

            // Aplicar la convolución 3x3
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    int pixel_val = input[(y + i) * width + (x + j)];
                    sum_x += pixel_val * gx[i + 1][j + 1];
                    sum_y += pixel_val * gy[i + 1][j + 1];
                }
            }

            // Calcular magnitud
            int magnitude = (int)sqrt((double)(sum_x * sum_x + sum_y * sum_y));
            
            // Limitar a 255 (saturación)
            if (magnitude > 255) magnitude = 255;
            if (magnitude < 0) magnitude = 0;

            output[y * width + x] = (unsigned char)magnitude;
        }
    }
}

int main(int argc, char* argv[]) {
    // Validar que el usuario pase el argumento (train, val o test)
    if (argc < 2) {
        printf("Error: Debes especificar el conjunto a procesar.\n");
        printf("Uso sugerido: %s [train | val | test]\n", argv[0]);
        return -1;
    }

    char* target = argv[1];
    char txt_path[256];
    char out_bin_X[256];
    char out_bin_y[256];

    // Construir dinámicamente las rutas de entrada y salida
    sprintf(txt_path, "../dataset/processed/%s.txt", target);
    sprintf(out_bin_X, "../dataset/processed/X_%s.bin", target);
    sprintf(out_bin_y, "../dataset/processed/y_%s.bin", target);

    ImageRecord* dataset_list;
    int num_images;

    printf("==================================================\n");
    printf("Iniciando Preprocesamiento para el conjunto: %s\n", target);
    printf("Cargando lista desde: %s\n", txt_path);
    
    load_dataset_list(txt_path, &dataset_list, &num_images);
    printf("Imágenes totales encontradas: %d\n", num_images);
    printf("==================================================\n");

    float* X_data = (float*)malloc(num_images * FLATTENED_SIZE * sizeof(float));
    int* y_data = (int*)malloc(num_images * sizeof(int));

    if (!X_data || !y_data) {
        printf("Error de memoria. No se pudo alojar la matriz.\n");
        return -1;
    }

    omp_set_num_threads(12);
    double start_time = omp_get_wtime();

    #pragma omp parallel for schedule(dynamic) num_threads(12)
    for (int i = 0; i < num_images; i++) {
        y_data[i] = dataset_list[i].label;

        char full_path[512];
        sprintf(full_path, "../dataset/processed/%s", dataset_list[i].path);
        float* mi_fila_matriz = &X_data[i * FLATTENED_SIZE];

        int width, height, channels;
        unsigned char* img_data = stbi_load(full_path, &width, &height, &channels, 1);
        
        if (!img_data) {
            // Imprime un aviso simple si salta alguna imagen corrupta
            printf("[Aviso] Saltada imagen corrupta en hilo %d: %s\n", omp_get_thread_num(), dataset_list[i].path);
            // Inicializar fila en 0 para evitar basura en memoria si falla
            memset(mi_fila_matriz, 0, FLATTENED_SIZE * sizeof(float));
            continue; 
        }

        unsigned char* resized_img = (unsigned char*)malloc(IMG_SIZE * IMG_SIZE);
        stbir_resize_uint8_linear(img_data, width, height, 0, 
                                  resized_img, IMG_SIZE, IMG_SIZE, 0, 1);
        
        unsigned char* sobel_img = (unsigned char*)calloc(IMG_SIZE * IMG_SIZE, sizeof(unsigned char));
        apply_sobel(resized_img, sobel_img, IMG_SIZE, IMG_SIZE);

        for (int p = 0; p < FLATTENED_SIZE; p++) {
            mi_fila_matriz[p] = (float)sobel_img[p] / 255.0f;
        }

        stbi_image_free(img_data);
        free(resized_img);
        free(sobel_img);
    }

    double end_time = omp_get_wtime();
    printf("\nTiempo de ejecución OpenMP (%s): %f segundos\n", target, end_time - start_time);

    printf("Guardando archivos binarios...\n");
    FILE* fx = fopen(out_bin_X, "wb");
    FILE* fy = fopen(out_bin_y, "wb");
    
    if(fx && fy) {
        fwrite(X_data, sizeof(float), num_images * FLATTENED_SIZE, fx);
        fwrite(y_data, sizeof(int), num_images, fy);
        fclose(fx);
        fclose(fy);
        printf("Archivos binarios guardados exitosamente:\n -> %s\n -> %s\n", out_bin_X, out_bin_y);
    } else {
        printf("Error crítico al intentar guardar los archivos binarios.\n");
    }

    free(X_data);
    free(y_data);
    free(dataset_list);
    printf("==================================================\n\n");

    return 0;
}