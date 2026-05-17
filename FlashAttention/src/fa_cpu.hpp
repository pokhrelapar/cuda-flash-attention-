#pragma once
#include <vector>
#include <cmath>
#include <algorithm>

std::vector<float> standard_attention(
    const std::vector<float>& Q,
    const std::vector<float>& K,
    const std::vector<float>& V,
    int N,
    int d
);

std::vector<float> transposeMatrix(
    const std::vector<float>& input,
    int rows, int cols
);
