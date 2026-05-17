#include "gputk.h"
#include "fa_cpu.hpp"
#include <vector>
#include <cstdlib>
#include <ctime>

static char *base_dir;

//data for Q, K, and V
static void generate_attention_data(int N, int d, float *Q, float *K, float *V) {
    for (int i = 0; i < N * d; i++) {
        Q[i] = ((float)(rand() % 20) - 5) / 5.0f;
        K[i] = ((float)(rand() % 20) - 5) / 5.0f;
        V[i] = ((float)(rand() % 20) - 5) / 5.0f;
    }
}

// Write to file
static void write_data(char *file_name, float *data, int height, int width) {
    int ii, jj;
    FILE *handle = fopen(file_name, "w");
    fprintf(handle, "%d %d\n", height, width);
    for (ii = 0; ii < height; ii++) {
        for (jj = 0; jj < width; jj++) {
            fprintf(handle, "%.6f", *data++);
            if (jj != width - 1) {
                fprintf(handle, " ");
            }
        }
        if (ii != height - 1) {
            fprintf(handle, "\n");
        }
    }
    fflush(handle);
    fclose(handle);
}

// dataset for attention testing
static void create_attention_dataset(int datasetNum, int N, int d) {
    const char *dir_name =
        gpuTKDirectory_create(gpuTKPath_join(base_dir, datasetNum));

    char *Q_file_name = gpuTKPath_join(dir_name, "Q.raw");
    char *K_file_name = gpuTKPath_join(dir_name, "K.raw");
    char *V_file_name = gpuTKPath_join(dir_name, "V.raw");
    char *output_file_name = gpuTKPath_join(dir_name, "output.raw");

    // Allocate and generate data
    float *Q_data = (float *)malloc(sizeof(float) * N * d);
    float *K_data = (float *)malloc(sizeof(float) * N * d);
    float *V_data = (float *)malloc(sizeof(float) * N * d);
    generate_attention_data(N, d, Q_data, K_data, V_data);

    // Compute attention output
    std::vector<float> O = standard_attention(
        std::vector<float>(Q_data, Q_data + N * d),
        std::vector<float>(K_data, K_data + N * d),
        std::vector<float>(V_data, V_data + N * d),
        N, d
    );

    // Write input and output data to files
    write_data(Q_file_name, Q_data, N, d);
    write_data(K_file_name, K_data, N, d);
    write_data(V_file_name, V_data, N, d);
    write_data(output_file_name, O.data(), N, d);

    // Free memory
    free(Q_data);
    free(K_data);
    free(V_data);
}

int main() {
    base_dir = gpuTKPath_join(gpuTKDirectory_current(), "data/fa");
    srand(time(0));

    // Create datasets with different dimensions
    create_attention_dataset(0, 16, 16);
    create_attention_dataset(1, 32, 32);
    create_attention_dataset(2, 64, 64);
    create_attention_dataset(3, 128, 64);
    create_attention_dataset(4, 64, 128);
    create_attention_dataset(5, 256, 128);
    create_attention_dataset(6, 128, 256);
    create_attention_dataset(7, 512, 64);
    create_attention_dataset(8, 64, 512);

    return 0;
}
