//简单的向量加法：A+B=C


#include <cuda_runtime.h>
#include <stdio.h>


__global__ void vectorAdd(const float *A, const float *B, float *C,
                          int numElements) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i<numElements) {
    C[i] = A[i] + B[i] + 0.0f;
  }
}


int main()
{
  int numElements = 50000;

  size_t size = numElements * sizeof(float);

  float *h_A = (float *)malloc(size);
  float *h_B = (float *)malloc(size);
  float *h_C = (float *)malloc(size);

//分配内存后，需验证操作是否成功
  if (h_A == nullptr || h_B == nullptr || h_C==nullptr){
    fprintf(stderr, "Failed to allocate host vectors\n");
    exit(EXIT_FAILURE);
  }

  // Initialize the host input vectors
  for (int i = 0; i < numElements; ++i) {
    h_A[i] = rand() / (float)RAND_MAX;
    h_B[i] = rand() / (float)RAND_MAX;
  }

  float *d_A = nullptr;
  float *d_B = nullptr;
  float *d_C = nullptr;
  cudaMalloc((void **)&d_A, size);
  cudaMalloc((void **)&d_B, size);
  cudaMalloc((void **)&d_C, size);

  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocksPerGrid = (numElements+threadsPerBlock-1) / threadsPerBlock;
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
        threadsPerBlock);

  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);


  //验证GPU计算是否准确
  for (int i = 0; i < numElements; ++i) {
    if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
      fprintf(stderr, "Result verification failed at element %d!\n", i);
      exit(EXIT_FAILURE);
    }
  }
  printf("Test PASSED\n");


  //释放资源
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  free(h_A);
  free(h_B);
  free(h_C);

  printf("Done\n");
  return 0;

}