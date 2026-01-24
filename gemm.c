#include "gemm.h"
#include <stdio.h>
#include <stddef.h>
#include <omp.h>

void gemm_naive(const float* left, const size_t lrows, const size_t lcols, const float* right,
                const size_t rrows, const size_t rcols, const char right_is_transposed,
                float* result) {
    if (right_is_transposed) {
        if (lcols != rcols) {
            fprintf(stderr, "incorrect matrix shape\n");
            return;
        }
#pragma omp parallel for
        for (size_t lrow = 0; lrow < lrows; lrow++) {
            for (size_t rrow = 0; rrow < rrows; rrow++) {
                for (size_t lcol = 0; lcol < lcols; lcol++) {
                    result[rrows * lrow + rrow] +=
                        left[lcols * lrow + lcol] * right[rcols * rrow + lcol];
                }
            }
        }
    } else {
        if (lcols != rrows) {
            fprintf(stderr, "incorrect matrix shape\n");
            return;
        }
#pragma omp parallel for
        for (size_t lrow = 0; lrow < lrows; lrow++) {
            for (size_t rcol = 0; rcol < rcols; rcol++) {
                for (size_t lcol = 0; lcol < lcols; lcol++) {
                    result[rcols * lrow + rcol] +=
                        left[lcols * lrow + lcol] * right[rcols * lcol + rcol];
                }
            }
        }
    }
}
