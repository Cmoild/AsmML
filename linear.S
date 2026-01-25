# struct Linear {
#     float* weight
#     size_t weight_m
#     size_t weight_n
#     float* bias
#     size_t bias_n
#     float* input
#     size_t input_m
#     size_t input_n
#     float* output
#     size_t output_m
#     size_t output_n
#     float* grad_input
#     float* grad_output
#     float* grad_weight
#     float* grad_bias
# }

    .section .text
    .globl linear_forward
    .type linear_forward, @function
# void linear_forward
# args: struct Linear* module
linear_forward:
    push %rbp
    mov %rsp, %rbp

    # Algorithm:
    # - copy bias output_m (batch size) times into output
    # - output = input @ weight.T + output, or
    # matmul_t(input, weight, output, input_m, weight_m, input_n)

    mov %rdi, %r8
    mov 0x48(%r8), %rax

    .local .Lbias_copy_loop
.Lbias_copy_loop:
    test %rax, %rax
    jz .Lbias_copy_loop_end
    dec %rax

    push %rax

    mov 0x20(%r8), %rdx
    imul %rdx, %rax
    mov 0x40(%r8), %rdi
    mov 0x18(%r8), %rsi
    lea (%rdi, %rax, 4), %rdi
    shl $2, %rdx

    push %r8
    call memcopy
    pop %r8

    pop %rax
    jmp .Lbias_copy_loop
    .local .Lbias_copy_loop_end
.Lbias_copy_loop_end:

    mov 0x28(%r8), %rdi
    mov 0x0(%r8), %rsi
    mov 0x40(%r8), %rdx
    mov 0x30(%r8), %rcx
    mov 0x10(%r8), %r9
    mov 0x8(%r8), %r8
    call matmul_t_naive

    mov %rbp, %rsp
    pop %rbp
    ret
    .size linear_forward, . - linear_forward

    .globl batch_sum
    .type batch_sum, @function
# void batch_sum
# args: float* M, float* v,
# size_t n (num of blocks), size_t m (block size)
# len(M) == n * m, len(v) == m
batch_sum:
    mov %rcx, %r8
    xor %rax, %rax
    vxorps %ymm0, %ymm0, %ymm0

    and $~7, %r8
    jz .Lclear_output_end
.Lclear_output:
    vmovups %ymm0, (%rsi, %rax, 4)

    add $8, %rax
    cmp %r8, %rax
    jb .Lclear_output
.Lclear_output_end:
    mov %rcx, %r9
    and $7, %r9
    jz .Lclear_end

    lea (%rsi, %rax, 4), %r8
    xor %rax, %rax
.Lclear_tail:
    vmovss %xmm0, (%r8, %rax, 4)

    add $8, %rax
    cmp %r9, %rax
    jb .Lclear_tail
.Lclear_end:

    mov %rdx, %r8
.Lblocks_loop:
    test %r8, %r8
    jz .Lblocks_loop_end
    dec %r8

    mov %rcx, %r9
    xor %rax, %rax
    and $~7, %r9
    jz .Lsimd_loop_end
.Lsimd_loop:
    vmovups (
    // TODO: доделать эту штуку

    jmp .Lblocks_loop
.Lblocks_loop_end:
    vzeroupper
    ret
    .size batch_sum, . - batch_sum

    .globl linear_update_grad
    .type linear_update_grad, @function
# void linear_update_grad
# args: struct Linear* module
linear_update_grad:
    push %rbp
    mov %rsp, %rbp

    # Algotithm:
    # - compute grad_input
    # - compute grad_weight
    # - compute grad_bias

    

    mov %rbp, %rsp
    pop %rbp
    ret
    .size linear_update_grad, . - linear_update_grad
