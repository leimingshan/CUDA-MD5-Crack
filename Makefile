INCLUDE_PATH = /usr/local/cuda/include
LIB_PATH = /usr/local/cuda/lib
CUDA_CC = /usr/local/cuda/bin/nvcc
CUDA_CFLAGS = -I$(INCLUDE_PATH) -keep -DGPU_BENCHMARK -DBENCHMARK #-DDEBUG
CC = gcc
CFLAGS = -I$(INCLUDE_PATH) -L$(LIB_PATH) -lm -lcudart -DBENCHMARK #-DDEBUG

main: md5.o
	$(CC) md5.o -o main $(CFLAGS)

md5.o: md5.cu
	$(CUDA_CC) md5.cu -c -o md5.o $(CUDA_CFLAGS)

clean:
	${RM} -r md5.cpp* md5.cuda* md5.o md5.fatbin* md5.ptx
	${RM} -r md5.sm_10* md5.module_id md5.hash md5.cu.cpp.ii main

