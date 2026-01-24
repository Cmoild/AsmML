    .section .text
    .globl matmul_t_naive
    .type matmul_t_naive, @function
# void matmul_t
# args: float* A, float* B, float* C, size_t m, size_t n, size_t k
matmul_t_naive:
    push %rbp
    mov %rsp, %rbp
    sub $0x18, %rsp

    mov %rcx, (%rsp)
    mov %r8, 0x8(%rsp)
    mov %r9, 0x10(%rsp)

    .local .Llrow_loop
.Llrow_loop:
    test %rcx, %rcx
    jz .Llrow_loop_end
    dec %rcx
    mov 0x8(%rsp), %r8
    .local .Lrrow_loop
.Lrrow_loop:
    test %r8, %r8
    jz .Lrrow_loop_end
    dec %r8
    mov 0x10(%rsp), %r9
    .local .Llcol_loop
.Llcol_loop:
    test %r9, %r9
    jz .Llcol_loop_end
    dec %r9

    mov 0x8(%rsp), %rax
    imul %rcx, %rax
    add %r8, %rax
    mov %rax, %r10
    movss (%rdx, %rax, 4), %xmm0

    mov 0x10(%rsp), %rax
    imul %rcx, %rax
    add %r9, %rax
    movss (%rdi, %rax, 4), %xmm1

    mov 0x10(%rsp), %rax
    imul %r8, %rax
    add %r9, %rax
    movss (%rsi, %rax, 4), %xmm2

    vfmadd231ss %xmm1, %xmm2, %xmm0

    movss %xmm0, (%rdx, %r10, 4)

    jmp .Llcol_loop
    .local .Llcol_loop_end
.Llcol_loop_end:
    jmp .Lrrow_loop
    .local .Lrrow_loop_end
.Lrrow_loop_end:
    jmp .Llrow_loop
    .local .Llrow_loop_end
.Llrow_loop_end:

    add $0x18, %rsp
    pop %rbp
    ret
    .size matmul_t_naive, . - matmul_t_naive

