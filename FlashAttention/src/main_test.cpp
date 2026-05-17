#include "gputk.h"
#include "fa_gpu.cuh"
#include "fa_cpu.hpp"

int main(int argc, char **argv) {
    gpuTKArg_t args;
    float *Q, *K, *V, *transposeK, *hostO, *attention_scores;
    int N, D;

    args = gpuTKArg_read(argc, argv);

    gpuTKTime_start(Generic, "Importing data and creating memory on host");
    int qRows, qCols;
    Q = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 0), &qRows, &qCols);

    int kRows, kCols;
    K = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 1), &kRows, &kCols);

    int vRows, vCols;
    V = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 2), &vRows, &vCols);

    // Validate 
    if (qRows != kRows || qCols != kCols || qRows != vRows || qCols != vCols) {
        gpuTKLog(FATAL, "Q, K, and V matrices must have the same dimensions");
    }

    N = qRows;
    D = qCols;

    // Calculate output matrix
    hostO = (float *)malloc(N * D * sizeof(float));
    attention_scores = (float *)malloc(N * N * sizeof(float));
    transposeK = (float *)malloc(N * D * sizeof(float));
    gpuTKTime_stop(Generic, "Importing data and creating memory on host");

    gpuTKLog(TRACE, "Dimensions: N=", N, ", D=", D);
    transposeMatrix(K, transposeK, N, D);

    computeNaiveAttention(Q, transposeK, V, attention_scores, hostO, N, D);

    gpuTKSolution(args, hostO, N, D);

    free(Q);
    free(K);
    free(V);
    free(transposeK);
    free(hostO);
    free(attention_scores);

    return 0;
}

