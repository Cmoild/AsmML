    .section .text
    .globl print
    .type print, @function
# int print(const char* s)
# returns 1 if s == NULL, 0 otherwise
print:
    mov $1, %rax # output value is 1 by default
    test %rdi, %rdi
    jz .Lfuncend

    # loop while character is not 0
    xor %rdx, %rdx # rdx - count arg in syscall
.Lloop:
    movb (%rdi, %rdx, 1), %al
    test %al, %al
    jz .Lend
    inc %rdx
    jmp .Lloop
.Lend:
    mov %rdi, %rsi # buf = s
    mov $1, %rdi # stdout
    mov $1, %rax # SYS_WRITE
    syscall
.Lfuncend:
    ret
    .size print, . - print

