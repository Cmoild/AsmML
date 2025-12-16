    .section .text
    .globl exp8fv
    .type exp8fv, @function
# void exp8fv
# args: float* V
exp8fv:
#   vmovups (%rdi), %ymm0
    mov $8, %rax
    sub $8, %rsp

    .local .Lsimple_loop
.Lsimple_loop:
    test %rax, %rax
    jz .Lsimple_loop_end
    dec %rax
    movss (%rdi, %rax, 4), %xmm0
    push %rax
    call expf@plt
    pop %rax
    movss %xmm0, (%rdi, %rax, 4)
    jmp .Lsimple_loop
    .local .Lsimple_loop_end
.Lsimple_loop_end:
    add $8, %rsp
    ret
    .size exp8fv, . - exp8fv

    .globl softmax
    .type softmax, @function
# void softmax
# args: float* M, size_t n (num of elements), size_t m (block size)
softmax:
    push %rbp
    mov %rsp, %rbp

    

    pop %rbp
    ret
    .size softmax, . - softmax
