    .section .text
    .globl exp8fv
    .type exp8fv, @function
# void exp8fv
# args: float* V
exp8fv:
    vmovups (%rdi), %ymm0
    vbroadcastss .LC_log2e, %ymm1
    vmulps %ymm0, %ymm1, %ymm0 // y = x * math.log2(math.e)
    vroundps $1, %ymm0, %ymm1 // n = math.floor(y)
    vsubps %ymm1, %ymm0, %ymm0 // f = y - n

    vbroadcastss .LC_a4, %ymm2
    vbroadcastss .LC_a5, %ymm3
    vfmadd231ps %ymm3, %ymm0, %ymm2 // a4 + f * a5
    vbroadcastss .LC_a3, %ymm3
    vfmadd231ps %ymm2, %ymm0, %ymm3 // a3 + f * (...)
    vbroadcastss .LC_a2, %ymm2
    vfmadd231ps %ymm3, %ymm0, %ymm2 // a2 + f * (...)
    vbroadcastss .LC_a1, %ymm3
    vfmadd231ps %ymm2, %ymm0, %ymm3 // a1 + f * (...)
    vbroadcastss .LC_a0, %ymm2
    vfmadd231ps %ymm3, %ymm0, %ymm2 // a0 + f * (...)

    vcvtps2dq %ymm1, %ymm3 // int(n)
    vpbroadcastd .LC_IEEE754_bias, %ymm1
    vpaddd %ymm1, %ymm3, %ymm3 // n + 127
    vpslld $23, %ymm3, %ymm3 // (n + 127) << 23
    vmulps %ymm2, %ymm3, %ymm0 // 2**n * two_f

    vmovups %ymm0, (%rdi)
    ret
    .size exp8fv, . - exp8fv

    .globl exp1f
    .type exp1f, @function
# void exp1f
# args: float* V
exp1f:
    
    .size exp1f, . - exp1f

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

    .section .rodata
.LC_log2e:
.float 1.4426950408889634
.LC_a0:
.float 1.0
.LC_a1:
.float 0.6931471805599453
.LC_a2:
.float 0.2402265069591007
.LC_a3:
.float 0.05550410866482158
.LC_a4:
.float 0.009618129107628477
.LC_a5:
.float 0.001333355814642844
.LC_IEEE754_bias:
.int 127
