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

extern void relu(float* M, size_t n, float* M_out);

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

struct ReLU {
    float* input;
    size_t input_m;
    size_t input_n;
    float* output;
    float* grad_input;
    float* grad_output;
};

_Static_assert(offsetof(struct ReLU, input) == 0, "");
_Static_assert(offsetof(struct ReLU, input_m) == 8, "");
_Static_assert(offsetof(struct ReLU, input_n) == 16, "");
_Static_assert(offsetof(struct ReLU, output) == 24, "");
_Static_assert(offsetof(struct ReLU, grad_input) == 32, "");
_Static_assert(offsetof(struct ReLU, grad_output) == 40, "");

extern void relu_forward(struct ReLU* module);

extern void relu_update_grad(struct ReLU* module);

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
    float weight_t[3 * 6] = {0.12440023, -0.42457172, 0.3063704,  0.38398394,  -0.79650676,
                             -0.5316126, 0.05219063,  -0.1291055, -0.00685904, -0.34825373,
                             0.35901272, 0.31753224,  0.02695692, 0.4601943,   0.19085988,
                             -0.3508047, 0.15054187,  -0.39146218};
    float input[2 * 6] = {0.3047171, -1.0399841, 0.7504512,   0.9405647,  -1.9510351, -1.3021795,
                          0.1278404, -0.3162426, -0.01680116, -0.8530439, 0.879398,   0.7777919};
    float grad_input[2 * 6] = {0};
    float grad_output[2 * 3] = {0.0660307,   1.1272413,  0.46750933,
                                -0.85929245, 0.36875078, -0.95888263};
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

    puts("Grad input");
    for (int i = 0; i < 2 * 6; ++i)
        printf("%f, ", grad_input[i]);
    puts("\nGrad weight");
    for (int i = 0; i < 3 * 6; ++i)
        printf("%f, ", grad_weight[i]);
    puts("\nGrad bias");
    for (int i = 0; i < 3; ++i)
        printf("%f, ", grad_bias[i]);
    puts("");
}

void test_relu_forward() {
    float test[25] = {0.1072, 0.1223, -1.2947, -0.2245, 0.2226,  0.6319,  -1.1761, -2.0376, -1.4152,
                      0.0078, 0.4336, 0.3302,  -1.1719, -0.5094, -0.6923, 0.0433,  1.5905,  -0.0333,
                      0.1904, 0.9133, 0.5219,  -0.8280, -0.5921, 1.1436,  -1.2585};
    float res[25] = {0};
    struct ReLU module = {.input = test, .input_m = 5, .input_n = 5, .output = res};
    relu_forward(&module);
    for (int i = 0; i < 25; i++)
        printf("%f, ", res[i]);
    puts("");
}

void test_relu_update_grad() {
    float test[25] = {0.9604,  -0.3194, -1.0436, -0.2665, -0.3246, 1.5302,  -0.3800, 1.3150, 1.3504,
                      -2.7083, -1.4846, -0.0906, 2.0149,  0.8364,  -2.3374, -0.0776, 0.9906, 0.3975,
                      -1.7088, 0.1813,  1.3356,  -0.7327, 0.8508,  0.7693,  -0.9871};
    float grad_input[25] = {0};
    float grad_output[25] = {1.5683,  -1.0189, 1.0044,  -1.2993, -1.1357, 0.9425,  -0.5050,
                             2.3649,  -0.9042, 0.2715,  -0.2236, 0.6351,  -1.0446, 0.1262,
                             1.1680,  -0.8709, 0.0865,  2.8323,  0.5354,  -0.8447, 2.3348,
                             -1.6300, -0.0472, -1.3245, 0.5434};
    struct ReLU module = {.input = test,
                          .input_m = 5,
                          .input_n = 5,
                          .grad_input = grad_input,
                          .grad_output = grad_output};
    relu_update_grad(&module);
    for (int i = 0; i < 25; i++)
        printf("%f, ", grad_input[i]);
    puts("");
}

int main() {
    test_linear_update_grad();
    return 0;
}
