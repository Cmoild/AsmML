    .section .text
    .globl relu
    .type relu, @function
# void relu
# args: float* M, size_t n, float* M_out
relu:
    vxorps %xmm0, %xmm0, %xmm0
    xor %rax, %rax

    mov %rsi, %rcx
    and $~7, %rcx
    jz .Lsimd_loop_end

    .local .Lsimd_loop
.Lsimd_loop:
    vmovups (%rdi, %rax, 4), %ymm1
    vmaxps %ymm1, %ymm0, %ymm2
    vmovups %ymm2, (%rdx, %rax, 4)

    add $8, %rax
    cmp %rcx, %rax
    jb .Lsimd_loop

    .local .Lsimd_loop_end
.Lsimd_loop_end:
    mov %rsi, %r8
    and $7, %r8
    jz .Lend

    lea (%rdi, %rcx, 4), %rdi
    lea (%rdx, %rcx, 4), %rdx
    xor %rax, %rax

    .local .Ltail_loop
.Ltail_loop:
    vmovss (%rdi, %rax, 4), %xmm1
    vmaxss %xmm1, %xmm0, %xmm2
    vmovss %xmm2, (%rdx, %rax, 4)

    inc %rax
    cmp %r8, %rax
    jb .Ltail_loop

    .local .Lend
.Lend:
    vzeroupper
    ret
    .size relu, . - relu

# struct ReLU {
#     float* input
#     size_t input_m
#     size_t input_n
#     float* output
#     float* grad_input
#     float* grad_output
# }

    .globl relu_forward
    .type relu_forward, @function
# void relu_forward
# args: struct ReLU* module
relu_forward:
    push %rbp
    mov %rsp, %rbp

    mov 0x8(%rdi), %rsi
    imul 0x10(%rdi), %rsi
    mov 0x18(%rdi), %rdx
    mov (%rdi), %rdi

    call relu

    mov %rbp, %rsp
    pop %rbp
    ret
    .size relu_forward, . - relu_forward

    .globl relu_update_grad
    .type relu_update_grad, @function
# void relu_update_grad
# args: struct ReLU* module
relu_update_grad:
    mov 0x8(%rdi), %rsi
    imul 0x10(%rdi), %rsi

    vxorps %ymm0, %ymm0, %ymm0
    xor %rax, %rax

    mov %rsi, %rcx
    and $~7, %rcx
    jz .Lupdate_grad_simd_loop_end

    mov 0x28(%rdi), %rdx # grad_output
    mov 0x20(%rdi), %r9 # grad_input
    mov (%rdi), %rdi # input

.Lupdate_grad_simd_loop:
    vmovups (%rdi, %rax, 4), %ymm1
    vcmpps $0x1e, %ymm0, %ymm1, %ymm1 # GT_OQ, %ymm1 > %ymm0
    vmovups (%rdx, %rax, 4), %ymm2
    vandps %ymm1, %ymm2, %ymm1
    vmovups %ymm1, (%r9, %rax, 4)

    add $8, %rax
    cmp %rcx, %rax
    jb .Lupdate_grad_simd_loop

.Lupdate_grad_simd_loop_end:
    mov %rsi, %r8
    and $7, %r8
    jz .Lupdate_grad_end

    lea (%rdi, %rcx, 4), %rdi
    lea (%rdx, %rcx, 4), %rdx
    lea (%r9, %rcx, 4), %r9
    xor %rax, %rax

.Lupdate_grad_tail_loop:
    vmovss (%rdi, %rax, 4), %xmm1
    vcmpss $0x1e, %xmm0, %xmm1, %xmm1 # GT_OQ, %ymm1 > %ymm0
    vmovss (%rdx, %rax, 4), %xmm2
    vandps %xmm1, %xmm2, %xmm1
    vmovss %xmm1, (%r9, %rax, 4)

    inc %rax
    cmp %r8, %rax
    jb .Lupdate_grad_tail_loop

.Lupdate_grad_end:
    vzeroupper
    ret
    .size relu_update_grad, . - relu_update_grad
