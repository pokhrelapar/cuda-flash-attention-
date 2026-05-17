#include <stdio.h>
#include <stdlib.h>

#include "fa_gpu.cuh"

static float *generate_data(int height, int width) {
    float *data = (float *)malloc(sizeof(float) * width * height);
    int i;
    for (i = 0; i < width * height; i++) {
        data[i] = ((float)(rand() % 20) - 5) / 5.0f;
    }
    return data;
}

int main(int argc, char **argv) {
    float *hostQ; // The Q matrix
    float *hostK; // The K matrix
    float *hostV; // The V matrix
    float *hostO; // The output O matrix
    float *attention_scores; // The attention scores matrix
    float *deviceQ;
    float *deviceK;
    float *deviceV;
    float *deviceO;
    int N; // sequence length
    int d; // head dimension

    if (argc != 3) {
        printf("Usage: %s sequence_length head_dimension\n", argv[0]);
        return 1;
    }

    N = atoi(argv[1]);
    d = atoi(argv[2]);

    hostQ = generate_data(N, d);
    hostK = generate_data(N, d);
    hostV = generate_data(N, d);

    hostO = (float *)malloc(N * d * sizeof(float));
    attention_scores = (float *)malloc(N * N * sizeof(float));

    computeNaiveAttention(hostQ, hostK, hostV,attention_scores, hostO, N, d);

    free(hostQ);
    free(hostK);
    free(hostV);
    free(hostO);
    free(attention_scores);

    return 0;
}