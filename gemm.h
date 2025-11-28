#ifndef GEMM_H
#define GEMM_H

#include <stddef.h>

#define TILE_SIZE 64

void gemm_naive(const float* left, const size_t lrows, const size_t lcols, const float* right,
                const size_t rrows, const size_t rcols, const char right_is_transposed,
                float* result);

#endif
