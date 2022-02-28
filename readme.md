写在前面：
---
本仓库借鉴cuda官方案例学习，地址：[https://github.com/nvidia/cuda-samples](https://github.com/nvidia/cuda-samples)


asyncAPI.cu
---
令CPU和GPU同时运行


helper_timer.h
---
- QueryPerformanceCounter：查询性能计时器
- QueryPerformanceFrequency：查询计时器的频率

此函数在windows与linux下各有一套实现方式，但**调用时没有区别**，用户感受不到差异。
关键在于定义的接口类，及`reinterpret_cast`

