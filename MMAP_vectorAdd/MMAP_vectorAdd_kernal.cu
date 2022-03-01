extern "C" __global__ void vecAdd_kernal(const float *A, const float *B,
                                        float *C, int N) {
    int i=blockDim.x*blockIdx.x + threadIdx.x;

    if (i>=N) return;

    C[i] = A[i] + B[i];
}