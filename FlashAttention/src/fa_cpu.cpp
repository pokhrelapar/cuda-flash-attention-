#include "fa_cpu.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <vector>

// Transpose
std::vector<float> transposeMatrix(
    const std::vector<float>& input,
    int rows, int cols) {

    std::vector<float> output(cols * rows);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            output[j * rows + i] = input[i * cols + j];
        }
    }

    return output;
}



// new cpu code

// N = sequence length, d = head dimension
// Matrices are represented as flat 1D vectors in row-major order
std::vector<float> standard_attention(
    const std::vector<float>& Q, 
    const std::vector<float>& K, 
    const std::vector<float>& V, 
    int N, int d) {

    // Transpose K → shape becomes (d x N)
    std::vector<float> K_T = transposeMatrix(K, N, d);

    float scale = 1.0f / std::sqrt((float)d);

    // S = Q * K^T
    std::vector<float> S(N * N, 0.0f);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;

            for (int k = 0; k < d; ++k) {
                // Now accessing contiguous memory in K_T
                sum += Q[i * d + k] * K_T[k * N + j];
            }

            S[i * N + j] = sum * scale;
        }
    }

    // Softmax
    std::vector<float> P(N * N, 0.0f);
    for (int i = 0; i < N; ++i) {
        float max_val = S[i * N];
        for (int j = 1; j < N; ++j) {
            max_val = std::max(max_val, S[i * N + j]);
        }

        float sum_exp = 0.0f;
        for (int j = 0; j < N; ++j) {
            P[i * N + j] = std::exp(S[i * N + j] - max_val);
            sum_exp += P[i * N + j];
        }

        for (int j = 0; j < N; ++j) {
            P[i * N + j] /= sum_exp;
        }
    }

    // O = P * V
    std::vector<float> O(N * d, 0.0f);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < d; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += P[i * N + k] * V[k * d + j];
            }
            O[i * d + j] = sum;
        }
    }

    return O;
}