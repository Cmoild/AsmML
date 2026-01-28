    .section .text
    .globl exp8fv
    .type exp8fv, @function
# void exp8fv
# args: float* V
exp8fv:
    vmovups (%rdi), %ymm0
    vbroadcastss .LC_log2e(%rip), %ymm1
    vmulps %ymm0, %ymm1, %ymm0 // y = x * math.log2(math.e)
    vroundps $1, %ymm0, %ymm1 // n = math.floor(y)
    vsubps %ymm1, %ymm0, %ymm0 // f = y - n

    vbroadcastss .LC_a4(%rip), %ymm2
    vbroadcastss .LC_a5(%rip), %ymm3
    vfmadd231ps %ymm3, %ymm0, %ymm2 // a4 + f * a5
    vbroadcastss .LC_a3(%rip), %ymm3
    vfmadd231ps %ymm2, %ymm0, %ymm3 // a3 + f * (...)
    vbroadcastss .LC_a2(%rip), %ymm2
    vfmadd231ps %ymm3, %ymm0, %ymm2 // a2 + f * (...)
    vbroadcastss .LC_a1(%rip), %ymm3
    vfmadd231ps %ymm2, %ymm0, %ymm3 // a1 + f * (...)
    vbroadcastss .LC_a0(%rip), %ymm2
    vfmadd231ps %ymm3, %ymm0, %ymm2 // a0 + f * (...)

    vcvtps2dq %ymm1, %ymm3 // int(n)
    vpbroadcastd .LC_IEEE754_bias(%rip), %ymm1
    vpaddd %ymm1, %ymm3, %ymm3 // n + 127
    vpslld $23, %ymm3, %ymm3 // (n + 127) << 23
    vmulps %ymm2, %ymm3, %ymm0 // 2**n * two_f

    vmovups %ymm0, (%rdi)
    vzeroupper
    ret
    .size exp8fv, . - exp8fv

    .globl expNfv
    .type expNfv, @function
# void expNfv
# args: float* V, size_t n
expNfv:
    .local .LexpNfv_loop
.LexpNfv_loop:
    test %rsi, %rsi
    jz .LexpNfv_loop_end
    dec %rsi

    vmovss (%rdi, %rsi, 4), %xmm0
    vmovss .LC_log2e(%rip), %xmm1
    vmulss %xmm0, %xmm1, %xmm0
    roundss $1, %xmm0, %xmm1
    vsubss %xmm1, %xmm0, %xmm0

    vmovss .LC_a4(%rip), %xmm2
    vmovss .LC_a5(%rip), %xmm3
    vfmadd231ss %xmm3, %xmm0, %xmm2
    vmovss .LC_a3(%rip), %xmm3
    vfmadd231ss %xmm2, %xmm0, %xmm3
    vmovss .LC_a2(%rip), %xmm2
    vfmadd231ss %xmm3, %xmm0, %xmm2
    vmovss .LC_a1(%rip), %xmm3
    vfmadd231ss %xmm2, %xmm0, %xmm3
    vmovss .LC_a0(%rip), %xmm2
    vfmadd231ss %xmm3, %xmm0, %xmm2

    vcvtss2si %xmm1, %eax
    add .LC_IEEE754_bias(%rip), %eax
    shl $23, %eax
    vmovd %eax, %xmm3
    vmulss %xmm2, %xmm3, %xmm0

    vmovss %xmm0, (%rdi, %rsi, 4)

    jmp .LexpNfv_loop
    .local .LexpNfv_loop_end
.LexpNfv_loop_end:
    vzeroupper
    ret
    .size expNfv, . - expNfv

    .globl maxNfv
    .type maxNfv, @function
# float maxNfv
# args: float* V, size_t n
maxNfv:
    vmovss (%rdi), %xmm0
    .local .Lmax_loop
.Lmax_loop:
    test %rsi, %rsi
    jz .Lmax_loop_end
    dec %rsi

    vmovss (%rdi, %rsi, 4), %xmm1
    vmaxss %xmm1, %xmm0, %xmm0

    jmp .Lmax_loop
    .local .Lmax_loop_end
.Lmax_loop_end:
    vzeroupper
    ret
    .size maxNfv, . - maxNfv

    .globl softmax
    .type softmax, @function
# void softmax
# args: float* M, size_t n (num of blocks), size_t m (block size)
# len(M) == n * m
softmax:
    push %rbp
    mov %rsp, %rbp

    mov %rsi, %rcx

    push %rdi
    push %rsi

    .local .Lblocks_loop
.Lblocks_loop:
    test %rcx, %rcx
    jz .Lblocks_loop_end
    dec %rcx

    mov %rdx, %rax
    imul %rcx, %rax
    shl $2, %rax
    add 0x8(%rsp), %rax
    push %rax

    mov %rax, %rdi
    mov %rdx, %rsi
    sub $8, %rsp
    call maxNfv

    add $4, %rsp
    vmovss %xmm0, (%rsp)

    xor %rax, %rax
    mov 0x4(%rsp), %r9

    mov %rdx, %r8
    and $~7, %r8
    jz .Lin_block_sub_max_and_exp_loop_end
    .local .Lin_block_sub_max_and_exp_loop
.Lin_block_sub_max_and_exp_loop:
    vbroadcastss (%rsp), %ymm0
    vmovups (%r9, %rax, 4), %ymm1
    vsubps %ymm0, %ymm1, %ymm1
    vmovups %ymm1, (%r9, %rax, 4)

    sub $0xc, %rsp
    lea (%r9, %rax, 4), %rdi
    call exp8fv
    add $0xc, %rsp

    add $8, %rax
    cmp %r8, %rax
    jb .Lin_block_sub_max_and_exp_loop
    .local .Lin_block_sub_max_and_exp_loop_end
.Lin_block_sub_max_and_exp_loop_end:
    lea (%r9, %rax, 4), %r9
    xor %rax, %rax
    mov %rdx, %r8
    and $7, %r8
    jz .Lin_block_sub_max_tail_end

    .local .Lin_block_sub_max_tail
.Lin_block_sub_max_tail:
    vmovss (%rsp), %xmm0
    vmovss (%r9, %rax, 4), %xmm1
    vsubss %xmm0, %xmm1, %xmm1
    vmovss %xmm1, (%r9, %rax, 4)

    inc %rax
    cmp %r8, %rax
    jb .Lin_block_sub_max_tail
    .local .Lin_block_sub_max_tail_end
.Lin_block_sub_max_tail_end:

    add $4, %rsp
    mov %r9, %rdi
    mov %rax, %rsi
    call expNfv

    mov %rdx, %rax
    mov (%rsp), %rdi
    vxorps %xmm0, %xmm0, %xmm0
    .local .Lin_block_sum_loop
.Lin_block_sum_loop:
    test %rax, %rax
    jz .Lin_block_sum_loop_end
    dec %rax

    vaddss (%rdi, %rax, 4), %xmm0, %xmm0

    jmp .Lin_block_sum_loop
    .local .Lin_block_sum_loop_end
.Lin_block_sum_loop_end:

    mov %rdx, %rax
    mov (%rsp), %rdi
    .local .Lin_block_div_loop
.Lin_block_div_loop:
    test %rax, %rax
    jz .Lin_block_div_loop_end
    dec %rax

    vmovss (%rdi, %rax, 4), %xmm1
    vdivss %xmm0, %xmm1, %xmm1
    vmovss %xmm1, (%rdi, %rax, 4)

    jmp .Lin_block_div_loop
    .local .Lin_block_div_loop_end
.Lin_block_div_loop_end:

    pop %rax

    jmp .Lblocks_loop
    .local .Lblocks_loop_end
.Lblocks_loop_end:
    vzeroupper
    mov %rbp, %rsp
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
