#include "gputk.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BLOCK_SIZE 32
#define MEM_WIDTH 32
#define TILE_WIDTH 32

#define NUM_SAMPLES 1024
#define FEATURE_DIMENSION 1024




// CUDA kernel for scaled dot-product QK^T
__global__ void computeScoresKernel(float* queryMatrix, float* keyMatrix, float* scoreMatrix) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Row index in the matrix
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Column index in the matrix

    if (row < NUM_SAMPLES && col < NUM_SAMPLES) {
        float score = 0.0f;
        for (int d = 0; d < FEATURE_DIMENSION; ++d) {
            score += queryMatrix[row * FEATURE_DIMENSION + d] * keyMatrix[col * FEATURE_DIMENSION + d];
        }
        scoreMatrix[row * NUM_SAMPLES + col] = score / sqrtf(static_cast<float>(FEATURE_DIMENSION)); // Scale
    }
}

__global__ void applySoftmaxKernel(float* scoreMatrix, float* softmaxMatrix) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Row index

    if (row < NUM_SAMPLES) {
        float maxScore = -1e30f;

        // Find max score for numerical stability
        for (int col = 0; col < NUM_SAMPLES; ++col) {
            maxScore = fmaxf(maxScore, scoreMatrix[row * NUM_SAMPLES + col]);
        }

        float sumExp = 0.0f;

        // Compute exponentials
        for (int col = 0; col < NUM_SAMPLES; ++col) {
            softmaxMatrix[row * NUM_SAMPLES + col] = expf(scoreMatrix[row * NUM_SAMPLES + col] - maxScore);
            sumExp += softmaxMatrix[row * NUM_SAMPLES + col];
        }

        // Normalize
        for (int col = 0; col < NUM_SAMPLES; ++col) {
            softmaxMatrix[row * NUM_SAMPLES + col] /= sumExp;
        }
    }
}


// CUDA kernel for computing final output matrix = softmax_scores * V
__global__ void computeOutputKernel(float* softmaxMatrix, float* valueMatrix, float* outputMatrix) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Row index in the output
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Column index in the output

    if (row < NUM_SAMPLES && col < FEATURE_DIMENSION) {
        float result = 0.0f;
        for (int k = 0; k < NUM_SAMPLES; ++k) {
            result += softmaxMatrix[row * NUM_SAMPLES + k] * valueMatrix[k * FEATURE_DIMENSION + col];
        }
        outputMatrix[row * FEATURE_DIMENSION + col] = result;
    }
}


// GPU-based implementation of naive Attention :Global Memory
void computeNaiveAttention(float* queryMatrix, float* keyMatrix, float* valueMatrix, float* attentionMatrix, float* outputMatrix) {
    // Device pointers
    float* d_queryMatrix, * d_keyMatrix, * d_valueMatrix;
    float* d_scoreMatrix, * d_softmaxMatrix, * d_outputMatrix;

    // Allocate device memory
    cudaMalloc(&d_queryMatrix, NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float));
    cudaMalloc(&d_keyMatrix, NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float));
    cudaMalloc(&d_valueMatrix, NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float));
    cudaMalloc(&d_scoreMatrix, NUM_SAMPLES * NUM_SAMPLES * sizeof(float));
    cudaMalloc(&d_softmaxMatrix, NUM_SAMPLES * NUM_SAMPLES * sizeof(float));
    cudaMalloc(&d_outputMatrix, NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_queryMatrix, queryMatrix, NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_keyMatrix, keyMatrix, NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_valueMatrix, valueMatrix, NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float), cudaMemcpyHostToDevice);

    // Configure thread block and grid dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE); // 16x16 threads per block
    dim3 gridDim((NUM_SAMPLES + blockDim.x - 1) / blockDim.x, (NUM_SAMPLES + blockDim.y - 1) / blockDim.y);

    // Compute QK^T
    computeScoresKernel << <gridDim, blockDim >> > (d_queryMatrix, d_keyMatrix, d_scoreMatrix);
    cudaDeviceSynchronize();

    // Apply softmax to scores
    dim3 softmaxBlockDim(1, 256);
    dim3 softmaxGridDim(1, (NUM_SAMPLES + softmaxBlockDim.y - 1) / softmaxBlockDim.y);
    applySoftmaxKernel << <softmaxGridDim, softmaxBlockDim >> > (d_scoreMatrix, d_softmaxMatrix);
    cudaDeviceSynchronize();

    // Compute final output
    dim3 outputBlock(BLOCK_SIZE, BLOCK_SIZE); // 16x16 threads for output matrix
    dim3 outputGrid((FEATURE_DIMENSION + outputBlock.x - 1) / outputBlock.x, (NUM_SAMPLES + outputBlock.y - 1) / outputBlock.y);
    computeOutputKernel << <outputGrid, outputBlock >> > (d_softmaxMatrix, d_valueMatrix, d_outputMatrix);
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(attentionMatrix, d_softmaxMatrix, NUM_SAMPLES * NUM_SAMPLES * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(outputMatrix, d_outputMatrix, NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_queryMatrix);
    cudaFree(d_keyMatrix);
    cudaFree(d_valueMatrix);
    cudaFree(d_scoreMatrix);
    cudaFree(d_softmaxMatrix);
    cudaFree(d_outputMatrix);
}


