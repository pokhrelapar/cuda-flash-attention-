#include "gputk.h"

#include "fa_gpu.cuh"

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

// CUDA kernel for scaled dot-product QK^T
__global__ void computeScoresKernel(float* queryMatrix, float* keyMatrix, float* scoreMatrix, int numSamples, int featureDimension) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Row index in the matrix
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Column index in the matrix

    if (row < numSamples && col < numSamples) {
        float score = 0.0f;
        for (int d = 0; d < featureDimension; ++d) {
            score += queryMatrix[row * featureDimension + d] * keyMatrix[col * featureDimension + d];
        }
        scoreMatrix[row * numSamples + col] = score / sqrtf(static_cast<float>(featureDimension)); // Scale
    }
}

__global__ void applySoftmaxKernel(float* scoreMatrix, float* softmaxMatrix, int numSamples, int featureDimension) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Row index

    if (row < numSamples) {
        float maxScore = -1e30f;

        // Find max score for numerical stability
        for (int col = 0; col < numSamples; ++col) {
            maxScore = fmaxf(maxScore, scoreMatrix[row * numSamples + col]);
        }

        float sumExp = 0.0f;

        // Compute exponentials
        for (int col = 0; col < numSamples; ++col) {
            softmaxMatrix[row * numSamples + col] = expf(scoreMatrix[row * numSamples + col] - maxScore);
            sumExp += softmaxMatrix[row * numSamples + col];
        }

        // Normalize
        for (int col = 0; col < numSamples; ++col) {
            softmaxMatrix[row * numSamples + col] /= sumExp;
        }
    }
}

// CUDA kernel for computing final output matrix = softmax_scores * V
__global__ void computeOutputKernel(float* softmaxMatrix, float* valueMatrix, float* outputMatrix, int numSamples, int featureDimension) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Row index in the output
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Column index in the output

    if (row < numSamples && col < featureDimension) {
        float result = 0.0f;
        for (int k = 0; k < numSamples; ++k) {
            result += softmaxMatrix[row * numSamples + k] * valueMatrix[k * featureDimension + col];
        }
        outputMatrix[row * featureDimension + col] = result;
    }
}

// Transpose Matrix
void transposeMatrix(const float* inputMatrix, float* transposedMatrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            transposedMatrix[j * rows + i] = inputMatrix[i * cols + j];
        }
    }
}

__global__ void shared_compute_scores(float* queryMatrix, float* keyTransposeMatrix, float* attentionScores, int numSamples, int featureDimension) {

    int threadX = threadIdx.x;
    int threadY = threadIdx.y;
    int blockX = blockIdx.x;
    int blockY = blockIdx.y;

    int scoreColumnIndex = blockX * TILE_WIDTH + threadX;
    int scoreRowIndex = blockY * TILE_WIDTH + threadY;
    float scoreValue = 0.0f;

    // Determine the number of phases
    int numPhases = (featureDimension + TILE_WIDTH - 1) / TILE_WIDTH;

    // Initialize shared memory
    __shared__ float sharedQuery[MEM_WIDTH][MEM_WIDTH];
    __shared__ float sharedKeyTranspose[MEM_WIDTH][MEM_WIDTH];

    // Iterate through phases
    for (int phase = 0; phase < numPhases; phase++) {
        
        if (phase * TILE_WIDTH + threadX < featureDimension && blockY * TILE_WIDTH + threadY < numSamples) {
            sharedQuery[threadY][threadX] = queryMatrix[(blockY * TILE_WIDTH + threadY) * featureDimension + phase * TILE_WIDTH + threadX];
        }
        else {
            sharedQuery[threadY][threadX] = 0.0f;
        }

        if (phase * TILE_WIDTH + threadY < featureDimension && blockX * TILE_WIDTH + threadX < numSamples) {
            sharedKeyTranspose[threadY][threadX] = keyTransposeMatrix[(phase * TILE_WIDTH + threadY) * numSamples + blockX * TILE_WIDTH + threadX];
        }
        else {
            sharedKeyTranspose[threadY][threadX] = 0.0f;
        }
        __syncthreads();

     
        if (scoreColumnIndex < numSamples && scoreRowIndex < numSamples) {
            // Cumulatively add the scores_value based on elements in the tile
            for (int i = 0; i < TILE_WIDTH; i++) {
                scoreValue += sharedQuery[threadY][i] * sharedKeyTranspose[i][threadX];
            }
        }

        __syncthreads();
    }
    
    if (scoreColumnIndex < numSamples && scoreRowIndex < numSamples) {
        attentionScores[scoreRowIndex * numSamples + scoreColumnIndex] = scoreValue / sqrtf(static_cast<float>(featureDimension));
    }
}

__global__ void shared_softmax(float* attentionScores, float* softmaxScores, int numSamples) {
    int rowIndex = blockIdx.y * blockDim.y + threadIdx.y;

    if (rowIndex < numSamples) {
        float maxScore = -1e30f;

        // Find max score for numerical stability
        for (int colIndex = 0; colIndex < numSamples; ++colIndex) {
            maxScore = fmaxf(maxScore, attentionScores[rowIndex * numSamples + colIndex]);
        }

        float sumExp = 0.0f;

        for (int colIndex = 0; colIndex < numSamples; ++colIndex) {
            softmaxScores[rowIndex * numSamples + colIndex] = expf(attentionScores[rowIndex * numSamples + colIndex] - maxScore);
            sumExp += softmaxScores[rowIndex * numSamples + colIndex];
        }

        for (int colIndex = 0; colIndex < numSamples; ++colIndex) {
            softmaxScores[rowIndex * numSamples + colIndex] /= sumExp;
        }
    }
}

__global__ void shared_compute_output(float* softmaxScores, float* valueMatrix, float* outputMatrix, int numSamples, int featureDimension) {

    int threadX = threadIdx.x;
    int threadY = threadIdx.y;
    int blockX = blockIdx.x;
    int blockY = blockIdx.y;

    int outputColumnIndex = blockX * TILE_WIDTH + threadX;
    int outputRowIndex = blockY * TILE_WIDTH + threadY;
    float outputValue = 0.0f;

    int numPhases = (numSamples + TILE_WIDTH - 1) / TILE_WIDTH;

    // Initialize shared memory
    __shared__ float sharedSoftmaxScores[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sharedValueMatrix[TILE_WIDTH][TILE_WIDTH];

    for (int phase = 0; phase < numPhases; phase++) {
        // Load valid elements into shared memory
        if (phase * TILE_WIDTH + threadX < numSamples && blockY * TILE_WIDTH + threadY < numSamples) {
            sharedSoftmaxScores[threadY][threadX] = softmaxScores[(blockY * TILE_WIDTH + threadY) * numSamples + phase * TILE_WIDTH + threadX];
        }
        else {
            sharedSoftmaxScores[threadY][threadX] = 0.0f;
        }

        if (phase * TILE_WIDTH + threadY < numSamples && blockX * TILE_WIDTH + threadX < featureDimension) {
            sharedValueMatrix[threadY][threadX] = valueMatrix[(phase * TILE_WIDTH + threadY) * featureDimension + blockX * TILE_WIDTH + threadX];
        }
        else {
            sharedValueMatrix[threadY][threadX] = 0.0f;
        }

        __syncthreads();

        if (outputColumnIndex < featureDimension && outputRowIndex < numSamples) {
            for (int i = 0; i < TILE_WIDTH; i++) {
                outputValue += sharedSoftmaxScores[threadY][i] * sharedValueMatrix[i][threadX];
            }
        }

        __syncthreads();
    }

    if (outputColumnIndex < featureDimension && outputRowIndex < numSamples) {
        outputMatrix[outputRowIndex * featureDimension + outputColumnIndex] = outputValue;
    }
}

// GPU-based implementation of naive Attention :Global Memory
void computeNaiveAttention(float* queryMatrix, float* keyMatrix, float* valueMatrix, float* attentionMatrix, float* outputMatrix, int numSamples, int featureDimension) {
    // Device pointers
    float* d_queryMatrix, * d_keyMatrix, * d_valueMatrix;
    float* d_scoreMatrix, * d_softmaxMatrix, * d_outputMatrix;

    // Allocate device memory
    cudaMalloc(&d_queryMatrix, numSamples * featureDimension * sizeof(float));
    cudaMalloc(&d_keyMatrix, numSamples * featureDimension * sizeof(float));
    cudaMalloc(&d_valueMatrix, numSamples * featureDimension * sizeof(float));
    cudaMalloc(&d_scoreMatrix, numSamples * numSamples * sizeof(float));
    cudaMalloc(&d_softmaxMatrix, numSamples * numSamples * sizeof(float));
    cudaMalloc(&d_outputMatrix, numSamples * featureDimension * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_queryMatrix, queryMatrix, numSamples * featureDimension * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_keyMatrix, keyMatrix, numSamples * featureDimension * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_valueMatrix, valueMatrix, numSamples * featureDimension * sizeof(float), cudaMemcpyHostToDevice);

    // Configure thread block and grid dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE); // 16x16 threads per block
    dim3 gridDim((numSamples + blockDim.x - 1) / blockDim.x, (numSamples + blockDim.y - 1) / blockDim.y);
    // Compute QK^T
    computeScoresKernel << <gridDim, blockDim >> > (d_queryMatrix, d_keyMatrix, d_scoreMatrix, numSamples, featureDimension);
    cudaDeviceSynchronize();

    // Apply softmax to scores
    dim3 softmaxBlockDim(1, 256);
    dim3 softmaxGridDim(1, (numSamples + softmaxBlockDim.y - 1) / softmaxBlockDim.y);
    applySoftmaxKernel << <softmaxGridDim, softmaxBlockDim >> > (d_scoreMatrix, d_softmaxMatrix, numSamples, featureDimension);
    cudaDeviceSynchronize();

    // Compute final output
    dim3 outputBlock(BLOCK_SIZE, BLOCK_SIZE); // 16x16 threads for output matrix
    dim3 outputGrid((featureDimension + outputBlock.x - 1) / outputBlock.x, (numSamples + outputBlock.y - 1) / outputBlock.y);
    computeOutputKernel << <outputGrid, outputBlock >> > (d_softmaxMatrix, d_valueMatrix, d_outputMatrix, numSamples, featureDimension);
    cudaDeviceSynchronize();

    dim3 blockDimension(TILE_WIDTH, TILE_WIDTH); // 32*32 threads per block
    dim3 gridDimension((numSamples + blockDimension.x - 1) / blockDimension.x, (numSamples + blockDimension.y - 1) / blockDimension.y);

    // Launch kernels for flash attention
    shared_compute_scores << <gridDimension, blockDimension >> > (d_queryMatrix, d_keyMatrix, d_scoreMatrix, numSamples, featureDimension);
    cudaDeviceSynchronize();

    dim3 softmaxBlockDimension(1, BLOCK_SIZE);
    dim3 softmaxGridDimension(1, (numSamples + softmaxBlockDimension.y - 1) / softmaxBlockDimension.y);

    shared_softmax << <softmaxGridDimension, softmaxBlockDimension >> > (d_scoreMatrix, d_softmaxMatrix, numSamples);
    cudaDeviceSynchronize();

    dim3 outputBlockFlashAttention(TILE_WIDTH, TILE_WIDTH); // 32*32 threads for output matrix
    dim3 outputGridFlashAttention((featureDimension + outputBlockFlashAttention.x - 1) / outputBlockFlashAttention.x, (numSamples + outputBlockFlashAttention.y - 1) / outputBlockFlashAttention.y);
    shared_compute_output << <outputGridFlashAttention, outputBlockFlashAttention >> > (d_softmaxMatrix, d_valueMatrix, d_outputMatrix, numSamples, featureDimension);
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(attentionMatrix, d_softmaxMatrix, numSamples * numSamples * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(outputMatrix, d_outputMatrix, numSamples * featureDimension * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_queryMatrix);
    cudaFree(d_keyMatrix);
    cudaFree(d_valueMatrix);
    cudaFree(d_scoreMatrix);
    cudaFree(d_softmaxMatrix);
    cudaFree(d_outputMatrix);
}
