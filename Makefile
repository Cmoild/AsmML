.PHONY: debug objdump clean

CC := gcc
DEBUG_FLAGS := -g -O0 -fno-omit-frame-pointer -no-pie
BUILD := build
SO := tests/lib.so

ASM_OBJS := $(patsubst asm/%.S,$(BUILD)/%.o,$(wildcard asm/*.S))

debug:
	$(CC) $(DEBUG_FLAGS) asm/*.S c/*.c -o main -lm

objdump:
	objdump -D main > main.dump.S

clean:
	rm -rf $(BUILD) main main.dump.S *.o

$(BUILD)/%.o: asm/%.S
	mkdir -p $(BUILD)
	gcc -fPIC -c $< -o $@

test: $(ASM_OBJS) $(C_OBJS)
	$(CC) -shared -o $(SO) $^
