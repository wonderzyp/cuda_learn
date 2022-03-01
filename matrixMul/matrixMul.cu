#include <stdio.h>
#include <assert.h>

#include <cuda_runtime.h>


void ConstantInit(float *data, int size, float val){
    for (int i=0; i<size; ++i){
        data[i] = val;
    }
}

// a simple test of matrix multiplication using cuda
int MatrixMultiply(int argc, char **argv, 
                    int block_size, const dim3 &dimsA, const dim3 &dimsB){
    
}
