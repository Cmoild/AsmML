    .section .text
    .globl matmul
    .type matmul, @function
# void matmul
# args: float* left, size_t lrows, size_t lcols, float* right,
#       size_t rrows, size_t rcols, char right_is_transposed, float* result
matmul:
    mov 0x8(%rsp), %rax
    test %rax, %rax
    jz .Lif_right_is_not_transposed
# not implemented
    ret
.Lif_right_is_not_transposed:
    cmp %rdx, %r8
    jne .Lincorrect_shape // lcols != rrows

# counter (lrow)
    mov %rsi, %r10
# counter (rcol)
    mov %r9, %r11
# calling aligned_alloc
    push %rdi
    push %rsi
    sub $8, %rsp

    mov $32, %rdi
    mov $32, %rsi // 8x1 kernel
    call aligned_alloc@plt

    add $8, %rsp
    pop %rsi
    pop %rdi
# null pointer test
    test %rax, %rax
    jz .Lkernel_is_null

# lcol counter
    push %r12
    mov %rdx, %r12

    .p2align 5
.Lleft_matrix_rows_loop:
    test %r10, %r10
    jz .Lleft_matrix_rows_loop_end
    sub $8, %r10
#   vmovaps (%rdi, %r10, 0x4), %ymm0
.Lright_matrix_cols_loop:
    test %r11, %r11
    jz .Lright_matrix_cols_loop_end
    sub $8, %r11
.Lleft_matrix_cols_loop:
    test %r12, %r12
    jz .Lleft_matrix_cols_loop_end

.Lleft_matrix_cols_loop_end:
    jmp .Lleft_matrix_cols_loop
.Lright_matrix_cols_loop_end:
    jmp .Lleft_matrix_rows_loop
.Lleft_matrix_rows_loop_end:
    pop %r12
    ret

.Lkernel_is_null:
    ret

.Lincorrect_shape:
    lea incorrect_shape_string(%rip), %rsi
    mov $incorrect_shape_string_len, %rdx
    mov $2, %rdi
    mov $1, %rax
    syscall
    ret
    .size matmul, . - matmul

    .section .rodata
incorrect_shape_string:
    .asciz "incorrect matrix shape\n"
incorrect_shape_string_len = . - incorrect_shape_string
