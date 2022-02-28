// CPU与GPU同时执行


#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>


int main()
{
  //show the device information
  int devID=0;
  cudaDeviceProp deviceProps;
  checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
  printf("CUDA device [%s]\n", deviceProps.name);

}

