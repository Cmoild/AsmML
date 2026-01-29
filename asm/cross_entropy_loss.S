    .section .text
    .globl ce_loss_grad
    .type ce_loss_grad, @function
# void ce_loss_grad
# args: float* true, float* pred, size_t n, size_t m, float* grad_loss
# true.shape == (n, m)
ce_loss_grad:
    imul %rdx, %rcx
    mov %rcx, %r9

    vcvtsi2ss %rdx, %xmm0, %xmm0
    vbroadcastss %xmm0, %ymm0

    xor %rax, %rax
    and $~7, %r9
    jz .Lsimd_sub_div_loop_end
.Lsimd_sub_div_loop:
    vmovups (%rdi, %rax, 4), %ymm1
    vmovups (%rsi, %rax, 4), %ymm2
    vsubps %ymm1, %ymm2, %ymm1
    vdivps %ymm0, %ymm1, %ymm1
    vmovups %ymm1, (%r8, %rax, 4)

    add $8, %rax
    cmp %r9, %rax
    jb .Lsimd_sub_div_loop
.Lsimd_sub_div_loop_end:
    mov %rcx, %r9
    and $7, %r9
    jz .Lce_loss_grad_end

    lea (%rdi, %rax, 4), %rdi
    lea (%rsi, %rax, 4), %rsi
    lea (%r8, %rax, 4), %r8
    xor %rax, %rax
.Lsub_div_loop_tail:
    vmovss (%rdi, %rax, 4), %xmm1
    vmovss (%rsi, %rax, 4), %xmm2
    vsubss %xmm1, %xmm2, %xmm1
    vdivss %xmm0, %xmm1, %xmm1
    vmovss %xmm1, (%r8, %rax, 4)

    inc %rax
    cmp %r9, %rax
    jb .Lsub_div_loop_tail
.Lce_loss_grad_end:
    vzeroupper
    ret
    .size ce_loss_grad, . - ce_loss_grad

    .globl logNfv
    .type logNfv, @function
# void logNfv
# args: float* v, size_t n
logNfv:
    mov %rsi, %rdx
    vbroadcastss .LC_a0(%rip), %ymm5
    vbroadcastss .LC_a1(%rip), %ymm6
    vbroadcastss .LC_a2(%rip), %ymm7
    vbroadcastss .LC_two(%rip), %ymm8
    vbroadcastss .LC_ln2(%rip), %ymm9
    vpbroadcastd .Lmantissa_mask(%rip), %ymm10
    vpbroadcastd .LC_IEEE754_bias(%rip), %ymm11
    vpslld $23, %ymm11, %ymm12
    vbroadcastss .LC_one(%rip), %ymm13

    xor %rax, %rax
    and $~7, %rdx
    jz .Lsimd_log_loop_end
.Lsimd_log_loop:
    vmovups (%rdi, %rax, 4), %ymm3
    vandps %ymm10, %ymm3, %ymm0
    vpaddd %ymm12, %ymm0, %ymm0 // mantissa
    vpsrld $23, %ymm3, %ymm2
    vpsubd %ymm11, %ymm2, %ymm1
    vcvtdq2ps %ymm1, %ymm1 // exponent
    vsubps %ymm13, %ymm0, %ymm2 // m - 1.
    vaddps %ymm13, %ymm0, %ymm3 // m + 1.
    vdivps %ymm3, %ymm2, %ymm2 // f
    vmulps %ymm2, %ymm2, %ymm3 // f ** 2
    // p = f + f * f**2 * (a0 + f**2 * (a1 + f**2 * a2))
    vmovaps %ymm6, %ymm4
    vfmadd231ps %ymm3, %ymm7, %ymm4

    vmovaps %ymm5, %ymm0
    vfmadd231ps %ymm3, %ymm4, %ymm0
    vmulps %ymm0, %ymm3, %ymm0

    vfmadd231ps %ymm0, %ymm2, %ymm2 // p

    vmulps %ymm2, %ymm8, %ymm0
    vfmadd231ps %ymm1, %ymm9, %ymm0

    vmovups %ymm0, (%rdi, %rax, 4)

    add $8, %rax
    cmp %rdx, %rax
    jb .Lsimd_log_loop
.Lsimd_log_loop_end:
    mov %rsi, %rdx
    and $7, %rdx
    jz .Llog_end

    lea (%rdi, %rax, 4), %rdi
    xor %rax, %rax
.Llog_loop_tail:
    vmovss (%rdi, %rax, 4), %xmm3
    vandps %xmm10, %xmm3, %xmm0
    vpaddd %xmm12, %xmm0, %xmm0 // mantissa
    vpsrld $23, %xmm3, %xmm2
    vpsubd %xmm11, %xmm2, %xmm1
    vcvtdq2ps %xmm1, %xmm1 // exponent
    vsubss %xmm13, %xmm0, %xmm2 // m - 1.
    vaddss %xmm13, %xmm0, %xmm3 // m + 1.
    vdivss %xmm3, %xmm2, %xmm2 // f
    vmulss %xmm2, %xmm2, %xmm3 // f ** 2
    // p = f + f * f**2 * (a0 + f**2 * (a1 + f**2 * a2))
    vmovaps %xmm6, %xmm4
    vfmadd231ss %xmm3, %xmm7, %xmm4

    vmovaps %xmm5, %xmm0
    vfmadd231ss %xmm3, %xmm4, %xmm0
    vmulss %xmm0, %xmm3, %xmm0

    vfmadd231ss %xmm0, %xmm2, %xmm2 // p

    vmulss %xmm2, %xmm8, %xmm0
    vfmadd231ss %xmm1, %xmm9, %xmm0

    vmovss %xmm0, (%rdi, %rax, 4)

    inc %rax
    cmp %rdx, %rax
    jb .Llog_loop_tail
.Llog_end:
    vzeroupper
    ret
    .size logNfv, . - logNfv

    .globl ce_loss_val
    .type ce_loss_val, @function
# float ce_loss_val
# args: float* true, float* pred, size_t n, size_t m
# true.shape == (n, m)
ce_loss_val:
    push %rbp
    mov %rsp, %rbp

    mov %rdx, %r9
    imul %rcx, %r9

    push %rdi
    push %rsi
    push %rdx
    push %rcx
    push %r8
    push %r9

    mov %rsi, %rdi
    mov %r9, %rsi
    call logNfv

    pop %r9
    pop %r8
    pop %rcx
    pop %rdx
    pop %rsi
    pop %rdi

    vxorps %xmm0, %xmm0, %xmm0

    vbroadcastss .LC_negative_zero(%rip), %ymm1

    xor %rax, %rax
    mov %r9, %r10
    and $~7, %r10
    jz .Lsimd_mul_neg_loop_end
.Lsimd_mul_neg_loop:
    vmovups (%rdi, %rax, 4), %ymm2
    vmovups (%rsi, %rax, 4), %ymm3
    vmulps %ymm2, %ymm3, %ymm2
    vxorps %ymm2, %ymm1, %ymm2
    vmovups %ymm2, (%rdi, %rax, 4)

    add $8, %rax
    cmp %r10, %rax
    jb .Lsimd_mul_neg_loop
.Lsimd_mul_neg_loop_end:
    mov %r9, %r10
    and $7, %r10
    jz .Lmul_neg_loop_tail_end

    push %rdi
    push %rsi
    lea (%rdi, %rax, 4), %rdi
    lea (%rsi, %rax, 4), %rsi
    xor %rax, %rax
.Lmul_neg_loop_tail:
    vmovss (%rdi, %rax, 4), %xmm2
    vmovss (%rsi, %rax, 4), %xmm3
    vmulss %xmm2, %xmm3, %xmm2
    vxorps %xmm2, %xmm1, %xmm2
    vmovss %xmm2, (%rdi, %rax, 4)

    inc %rax
    cmp %r10, %rax
    jb .Lmul_neg_loop_tail
    pop %rsi
    pop %rdi
.Lmul_neg_loop_tail_end:

    vcvtsi2ss %rdx, %xmm3, %xmm3
.Lblocks_loop:
    test %rdx, %rdx
    jz .Lblocks_loop_end
    dec %rdx

    vxorps %xmm1, %xmm1, %xmm1
    mov %rcx, %rax
.Lvector_loop:
    test %rax, %rax
    jz .Lvector_loop_end
    dec %rax

    mov %rcx, %r9
    imul %rdx, %r9
    add %rax, %r9
    vmovss (%rdi, %r9, 4), %xmm2
    vaddss %xmm1, %xmm2, %xmm1

    jmp .Lvector_loop
.Lvector_loop_end:
    vaddss %xmm0, %xmm1, %xmm0
    jmp .Lblocks_loop
.Lblocks_loop_end:
    vdivss %xmm3, %xmm0, %xmm0

    mov %rbp, %rsp
    pop %rbp
    vzeroupper
    ret
    .size ce_loss_val, . - ce_loss_val

    .section .rodata
.Lmantissa_mask:
.int 0x7FFFFF
.LC_IEEE754_bias:
.int 127
.LC_one:
.float 1.
.LC_two:
.float 2.
.LC_a2:
.float 0.14285714285714285
.LC_a1:
.float 0.2
.LC_a0:
.float 0.3333333333333333
.LC_ln2:
.float 0.6931471805599453
.LC_negative_zero:
.int 0x80000000
