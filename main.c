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

extern void relu(float* M, size_t n);

extern void exp8fv(float* V);

extern void expNfv(float* V, size_t n);

extern void softmax(float* M, size_t n, size_t m);

extern float maxNfv(float* V, size_t n);

#define ARRAY_SIZE (8)
int main() {
    // assert(ARR_SIZE % 8 == 0 && ARR_SIZE == 8);
    float a[20] = {-0.8937, -0.7988, -1.9367, 1.4457,  0.4375,  -1.3356, 0.6597,
                   -0.1773, -0.6089, -0.0871, -0.4192, -0.0208, -1.4332, 1.2833,
                   1.3104,  0.6812,  -1.9798, 0.1717,  0.6637,  -0.4301};
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
    softmax(a, 2, 10);
    // float arr[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    // exp8fv(arr);
    for (size_t i = 0; i < 20; i++)
        printf("%f ", a[i]);
    printf("\n");
    // printf("%f\n", maxNfv(a, 8));
    // printf("%f\n", maxNfv(b, 8));
    // free(a);
    // free(b);
    // free(res);
    return 0;
}
