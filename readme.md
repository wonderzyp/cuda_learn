写在前面：
---
本仓库借鉴cuda官方案例学习，地址：[https://github.com/nvidia/cuda-samples](https://github.com/nvidia/cuda-samples)


asyncAPI.cu
---
令CPU和GPU同时运行

MMAP_vectorAdd
---
相同长度的向量相加：$A+B=C$



helper_timer.h
---
- QueryPerformanceCounter：查询性能计时器
- QueryPerformanceFrequency：查询计时器的频率

此函数在windows与linux下各有一套实现方式，但**调用时没有区别**，用户感受不到差异。
关键在于定义的接口类，及`reinterpret_cast`

bilateralFilter
---
双边滤波器：一种保持边缘的非线性平滑滤波器
主要有三个参数：
- 高斯增量(gaussian delta)
- 欧几里得增量(euclidean delta)
- 迭代次数