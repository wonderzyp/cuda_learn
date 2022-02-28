// CPU与GPU同时执行


#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_timer.h>

__global__ void d_fun(int *g_data, int inc_value){
  int idx = threadIdx.x+blockDim.x*blockIdx.x;
  g_data[idx] = g_data[idx]+inc_value;
}

bool correct_output(int *data, const int n, const int x) {
  for (int i = 0; i < n; i++)
    if (data[i] != x) {
      printf("Error! data[%d] = %d, ref = %d\n", i, data[i], x);
      return false;
    }

  return true;
}


int main()
{
  //show the device information
  int devID=0;
  cudaDeviceProp deviceProps;
  checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
  printf("CUDA device [%s]\n", deviceProps.name);

  int n = 16*1024*1024;
  int nbytes = n*sizeof(int);
  int value =26;
  
  // 分配内存
  int *a = 0;
  checkCudaErrors(cudaMallocHost((void**)&a, nbytes));
  memset(a, 0, nbytes);

  int *d_a=0;
  checkCudaErrors(cudaMalloc((void**)&d_a, nbytes));
  checkCudaErrors(cudaMemset(d_a, 255, nbytes));

  dim3 threads = dim3(512, 1);
  dim3 blocks = dim3(n/threads.x ,1);

  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  StopWatchInterface *timer = nullptr;
  sdkCreateTimer(&timer);
  sdkResetTimer(&timer);

  checkCudaErrors(cudaDeviceSynchronize());
  float gpu_time =0.0f;

  sdkStartTimer(&timer);
  cudaEventRecord(start,0);
  cudaMemcpyAsync(d_a, a, nbytes, cudaMemcpyHostToDevice,0);
  d_fun<<<blocks, threads,0,0>>> (d_a, value);
  cudaMemcpyAsync(a, d_a, nbytes, cudaMemcpyDeviceToHost,0);
  cudaEventRecord(stop, 0);
  sdkStopTimer(&timer);

// CPU可在GPU计算的同时进行工作
  unsigned long int counter =0;
  while (cudaEventQuery(stop)==cudaErrorNotReady){
    ++counter;
  }

  checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));

  printf("time spent executing by the GPU: %.2fms\n", gpu_time);
  printf("time spent by CPU in CUDA calls: %.2fms\n", sdkGetTimerValue(&timer));
  printf("CPU executed %lu iterations while waiting for GPU to finish\n",
         counter);

  bool checkResult = correct_output(a,n,value);

  // release resources
  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));
  checkCudaErrors(cudaFreeHost(a));
  checkCudaErrors(cudaFree(d_a));

  exit(checkResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

