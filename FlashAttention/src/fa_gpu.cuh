#pragma once

// Error checking macro
#define gpuTKCheck(stmt)                                                       \
    do {                                                                       \
        cudaError_t err = stmt;                                                \
        if (err != cudaSuccess) {                                              \
            gpuTKLog(ERROR, "Failed to run stmt ", #stmt);                     \
            return -1;                                                         \
        }                                                                      \
    } while (0)

void computeNaiveAttention(float* queryMatrix, float* keyMatrix, float* valueMatrix, float* attentionMatrix, float* outputMatrix, int numSamples, int featureDimension);

void transposeMatrix(const float* keyMatrix, float* transposedKeyMatrix, int N, int d);
