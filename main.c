#include "gemm.h"
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

extern int print(const char* s);

// m - rows A
// k - cols A, rows B
// n - cols B
extern void matmul_naive(const float* A, const float* B, float* C, size_t m, size_t k, size_t n);

// m - rows A
// n - rows B
// k - cols A, cols B
extern void matmul_t_naive(const float* A, const float* B, float* C, size_t m, size_t n, size_t k);

// m - rows A
// k - cols A
// n - cols B
extern void matmul_left_t_naive(const float* A, const float* B, float* C, size_t m, size_t k,
                                size_t n);

extern void relu(float* M, size_t n);

extern void exp8fv(float* V);

extern void expNfv(float* V, size_t n);

extern void softmax(float* M, size_t n, size_t m);

extern float maxNfv(float* V, size_t n);

struct Linear {
    float* weight;
    size_t weight_m;
    size_t weight_n;
    float* bias;
    size_t bias_n;
    float* input;
    size_t input_m;
    size_t input_n;
    float* output;
    size_t output_m;
    size_t output_n;
    float* grad_input;
    float* grad_output;
    float* grad_weight;
    float* grad_bias;
};

_Static_assert(offsetof(struct Linear, weight) == 0, "");
_Static_assert(offsetof(struct Linear, weight_m) == 8, "");
_Static_assert(offsetof(struct Linear, weight_n) == 16, "");
_Static_assert(offsetof(struct Linear, bias) == 24, "");
_Static_assert(offsetof(struct Linear, bias_n) == 32, "");
_Static_assert(offsetof(struct Linear, input) == 40, "");
_Static_assert(offsetof(struct Linear, input_m) == 48, "");
_Static_assert(offsetof(struct Linear, input_n) == 56, "");
_Static_assert(offsetof(struct Linear, output) == 64, "");
_Static_assert(offsetof(struct Linear, output_m) == 72, "");
_Static_assert(offsetof(struct Linear, output_n) == 80, "");
_Static_assert(offsetof(struct Linear, grad_input) == 88, "");
_Static_assert(offsetof(struct Linear, grad_output) == 96, "");
_Static_assert(offsetof(struct Linear, grad_weight) == 104, "");
_Static_assert(offsetof(struct Linear, grad_bias) == 112, "");

extern void linear_forward(struct Linear* module);

extern void memcopy(void* dst, void* src, size_t n);

extern void batch_sum(const float* M, float* v, size_t n, size_t m);

extern void linear_update_grad(struct Linear* module);

void test_linear_forward() {
    float bias[3] = {1, 2, 3};
    float weight[6 * 3] = {0.1, 1.1, 2.1, 0.2, 1.2, 2.2, 0.3, 1.3, 2.3,
                           0.4, 1.4, 2.4, 0.5, 1.5, 2.5, 0.6, 1.6, 2.6};
    float weight_t[3 * 6] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.1, 1.2, 1.3,
                             1.4, 1.5, 1.6, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6};
    float input[2 * 6] = {1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16};
    float output[2 * 3] = {0};
    struct Linear module = {
        .weight = weight_t,
        .weight_m = 3,
        .weight_n = 6,
        .bias = bias,
        .bias_n = 3,
        .input = input,
        .input_m = 2,
        .input_n = 6,
        .output = output,
        .output_m = 2,
        .output_n = 3,
    };

    linear_forward(&module);

    for (int i = 0; i < 2 * 3; ++i)
        printf("%f, ", output[i]);
    puts("");
}

void test_linear_update_grad() {
    float bias[3] = {1, 2, 3};
    float weight_t[3 * 6] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.1, 1.2, 1.3,
                             1.4, 1.5, 1.6, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6};
    float input[2 * 6] = {1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16};
    float grad_input[2 * 6] = {0};
    float grad_output[2 * 3] = {2.1, 2.2, 2.3, 2.4, 2.5, 2.6};
    float grad_bias[3] = {0};
    float grad_weight[3 * 6] = {0};
    struct Linear module = {.weight = weight_t,
                            .weight_m = 3,
                            .weight_n = 6,
                            .bias = bias,
                            .bias_n = 3,
                            .input = input,
                            .input_m = 2,
                            .input_n = 6,
                            .output_m = 2,
                            .output_n = 3,
                            .grad_input = grad_input,
                            .grad_output = grad_output,
                            .grad_weight = grad_weight,
                            .grad_bias = grad_bias};

    linear_update_grad(&module);

    puts("Grad output");
    for (int i = 0; i < 2 * 6; ++i)
        printf("%f, ", grad_input[i]);
    puts("\nGrad weight");
    for (int i = 0; i < 3 * 6; ++i)
        printf("%f, ", grad_weight[i]);
    puts("\nGrad output");
    for (int i = 0; i < 3; ++i)
        printf("%f, ", grad_bias[i]);
    puts("");
}

int main() {
    test_linear_update_grad();
    return 0;
}
