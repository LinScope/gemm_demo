#CUDA_PATH = /usr/local/cuda
#CUDA_INC_DIC = ${CUDA_PATH}/include
#CUDA_LIB_DIC = ${CUDA_PATH}/lib64
#NVCC = ${CUDA_PATH}/bin/nvcc

GXX=/usr/bin/g++

#test: test_${TEST_ID}.cu
#	${NVCC} -std=c++11 -o test --ptxas-options=-v --generate-code arch=compute_75,code=sm_75 -L${CUDA_LIB_DIC} -I${CUDA_INC_DIC} test_${TEST_ID}.cu

test: src/main.cpp
	${GXX} -std=c++14 -S -O2 src/gemm.cpp src/main.cpp
	${GXX} -std=c++14 -o build/test -O2 src/gemm.cpp src/main.cpp -march=armv8-a

run: test
	build/test

clean:
	rm -f build/test
