#include "gemm.h"
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

extern int print(const char* s);
extern void matmul(const float* left, const size_t lrows, const size_t lcols, const float* right,
                   const size_t rrows, const size_t rcols, const char right_is_transposed,
                   float* result);

// m - rows A
// k - cols A, rows B
// n - cols B
extern void matmul_naive(const float* A, const float* B, const float* C, size_t m, size_t k,
                         size_t n);

// m - rows A
// n - rows B
// k - cols A, cols B
extern void matmul_t_naive(const float* A, const float* B, const float* C, size_t m, size_t n,
                           size_t k);

extern void relu(const float* M, size_t n);

extern void exp8fv(const float* V);

#define ARRAY_SIZE (8)
int main() {
    // assert(ARR_SIZE % 8 == 0 && ARR_SIZE == 8);
    float a[ARRAY_SIZE] = {1, 2, -3, 4, 5, 6, 7, 8};
    float b[ARRAY_SIZE] = {9, -8, 7, 6, -5, 4, 3, 2};
    float res[2 * 2] = {0};

    // matmul(a, size, size, b, size, size, 0, res);
    // matmul_t_naive(a, b, res, 2, 2, 2);
    // for (size_t i = 0; i < 2; i++) {
    //     for (size_t j = 0; j < 2; j++) {
    //         printf("%f ", res[i * 2 + j]);
    //     }
    //     printf("\n");
    // }
    //
    // relu(res, 4);
    // for (size_t i = 0; i < 2; i++) {
    //     for (size_t j = 0; j < 2; j++) {
    //         printf("%f ", res[i * 2 + j]);
    //     }
    //     printf("\n");
    // }

    float arr[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    exp8fv(arr);
    for (size_t i = 0; i < 8; i++)
        printf("%f ", arr[i]);
    printf("\n");
    // free(a);
    // free(b);
    // free(res);
    return 0;
}
