    .section .text
    .globl set_seed
    .type set_seed, @function
# void set_seed
# args: uint32_t seed
set_seed:
    lea .Lrandom_seed(%rip), %rax
    mov %edi, (%rax)
    ret
    .size set_seed, . - set_seed

    .globl randu32
    .type randu32, @function
# uint32_t randu32
randu32:
    mov .Lrandom_seed(%rip), %eax
    mov %eax, %edi
    shl $13, %edi
    xor %edi, %eax
    mov %eax, %edi
    shr $17, %edi
    xor %edi, %eax
    mov %eax, %edi
    shl $5, %edi
    xor %edi, %eax
    lea .Lrandom_seed(%rip), %rdi
    mov %eax, (%rdi)
    ret
    .size randu32, . - randu32

    .globl rand_uniform_float
    .type rand_uniform_float, @function
# float rand_uniform_float
# args: float scale, float bias
rand_uniform_float:
    sub $8, %rsp
    call randu32
    add $8, %rsp
    mov %eax, %eax
    vcvtsi2ss %rax, %xmm3, %xmm3
    vmovss .LCu32_max(%rip), %xmm2
    vdivss %xmm2, %xmm3, %xmm3
    vmulss %xmm3, %xmm0, %xmm0
    vaddss %xmm1, %xmm0, %xmm0
    ret
    .size rand_uniform_float, . - rand_uniform_float

    .section .data
.Lrandom_seed:
.int 1

    .section .rodata
.LCu32_max:
.float 4294967296.0
