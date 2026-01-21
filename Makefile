all:
	gcc -c print.S
	gcc -c matmul_naive.S
	gcc -c main.c -O2
	gcc main.o print.o matmul_naive.o -o main

debug:
	gcc -g -O0 -fno-omit-frame-pointer -no-pie print.S gemm.c matmul_naive.S matmul_t_naive.S relu.S softmax.S mylib.S main.c -o main -lm

objdump:
	objdump -D main > main.dump.S

clean:
	rm -f main main.dump.S main.o print.o

gemm:
	gcc main.c gemm.c -o main -O3 -fopenmp -mavx2 -march=native
