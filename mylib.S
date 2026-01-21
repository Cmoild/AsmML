    .section .text
    .globl memcopy
    .type memcopy, @function
# void memcopy
# args: void* destinatioin, void* source, size_t n_bytes
memcopy:
    xor %rax, %rax

    mov %rdx, %rcx
    and $~31, %rcx
    jz .Lsimd256_loop_end

    .local .Lsimd256_loop
.Lsimd256_loop:
    vmovdqu (%rsi, %rax, 1), %ymm1
    vmovdqu %ymm1, (%rdi, %rax, 1)

    add $32, %rax
    cmp %rcx, %rax
    jb .Lsimd256_loop

    .local .Lsimd256_loop_end
.Lsimd256_loop_end:
    mov %rdx, %r8
    and $31, %r8
    jz .Lend

    // 31 bytes or less
    mov %rdx, %rax
    sub %rcx, %rax
    cmp $16, %rax
    jb .Lsimd128_end

    .local .Lsimd128
.Lsimd128:
    vmovdqu (%rsi, %rcx, 1), %xmm1
    vmovdqu %xmm1, (%rdi, %rcx, 1)

    add $16, %rcx
    sub $16, %rax
    .local .Lsimd128_end
.Lsimd128_end:
    // 15 bytes or less
    cmp $8, %rax
    jb .Lcopy8_end

    .local .Lcopy8
.Lcopy8:
    movq (%rsi, %rcx, 1), %r9
    movq %r9, (%rdi, %rcx, 1)

    add $8, %rcx
    sub $8, %rax
    .local .Lcopy8_end
.Lcopy8_end:
    // 7 bytes or less
    cmp $4, %rax
    jb .Lcopy4_end

    .local .Lcopy4
.Lcopy4:
    movl (%rsi, %rcx, 1), %r9d
    movl %r9d, (%rdi, %rcx, 1)

    add $4, %rcx
    sub $4, %rax
    .local .Lcopy4_end
.Lcopy4_end:
    // 3 bytes or less
    cmp $2, %rax
    jb .Lcopy2_end

    .local .Lcopy2
.Lcopy2:
    movw (%rsi, %rcx, 1), %r9w
    movw %r9w, (%rdi, %rcx, 1)

    add $2, %rcx
    sub $2, %rax
    .local .Lcopy2_end
.Lcopy2_end:
    // 1 byte or less
    cmp $1, %rax
    jb .Lend

    .local .Lcopy1
.Lcopy1:
    movb (%rsi, %rcx, 1), %r9b
    movb %r9b, (%rdi, %rcx, 1)

    .local .Lend
.Lend:
    vzeroupper
    ret
    .size memcopy, . - memcopy

