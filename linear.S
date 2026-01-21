# struct Linear {
#     float* weight;
#     size_t weight_m;
#     size_t weight_n;
#     float* bias;
#     size_t bias_m;
#     size_t bias_n;
#     float* input;
#     size_t input_m;
#     size_t input_n;
#     float* output;
#     size_t output_m;
#     size_t output_n;
#     float* grad_input;
#     float* grad_output;
# };

    .section .text
    .globl linear_forward
    .type linear_forward, @function
# void linear_forward
# args: struct Linear* module
linear_forward:
    
    .size linear_forward, . - linear_forward
