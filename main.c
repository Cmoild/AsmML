#include "gemm.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

extern int print(const char* s);

int main() {
    size_t size = 1000;
    float* a = (float*)aligned_alloc(32, size * size * sizeof(float));
    float* b = (float*)aligned_alloc(32, size * size * sizeof(float));
    float* res = (float*)aligned_alloc(32, size * size * sizeof(float));
    // float a[size * size] = {23};
    // float b[size * size] = {345};
    // float res[size * size];

    gemm_naive(a, size, size, b, size, size, 0, res);
    // for (size_t i = 0; i < 5; i++) {
    //     for (size_t j = 0; j < 5; j++) {
    //         printf("%f ", res[i * 5 + j]);
    //     }
    //     printf("\n");
    // }

    free(a);
    free(b);
    free(res);
    return 0;
}
