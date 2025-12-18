    .section .text
    .globl relu
    .type relu, @function
# void relu
# args: float* M, size_t n
relu:
    push %rbx
    vxorps %xmm0, %xmm0, %xmm0
    xor %rax, %rax

    mov %rsi, %rcx
    and $~7, %rcx
    jz .Lsimd_loop_end

    .local .Lsimd_loop
.Lsimd_loop:
    vmovups (%rdi, %rax, 4), %ymm1
    vmaxps %ymm1, %ymm0, %ymm2
    vmovups %ymm2, (%rdi, %rax, 4)

    add $8, %rax
    cmp %rcx, %rax
    jb .Lsimd_loop

    .local .Lsimd_loop_end
.Lsimd_loop_end:
    mov %rsi, %rdx
    and $7, %rdx
    jz .Lend

    lea (%rdi, %rcx, 4), %rbx
    xor %rax, %rax

    .local .Ltail_loop
.Ltail_loop:
    vmovss (%rbx, %rax, 4), %xmm1
    vmaxss %xmm1, %xmm0, %xmm2
    vmovss %xmm2, (%rbx, %rax, 4)

    inc %rax
    cmp %rdx, %rax
    jb .Ltail_loop

    .local .Lend
.Lend:
    vzeroupper
    pop %rbx
    ret
    .size relu, . - relu

