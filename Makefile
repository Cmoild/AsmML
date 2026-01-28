.PHONY: debug objdump clean

CC := gcc
DEBUG_FLAGS := -g -O0 -fno-omit-frame-pointer -no-pie

debug:
	$(CC) $(DEBUG_FLAGS) asm/*.S c/*.c -o main -lm

objdump:
	objdump -D main > main.dump.S

clean:
	rm -f main main.dump.S *.o
