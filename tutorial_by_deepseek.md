# CUDA Programming 核心函数详解（C++ 开发者指南）

作为有 C++ 基础的开发者，理解 CUDA 的关键在于掌握 **主机端（Host）API** 和 **设备端（Device）函数** 的配合使用。CUDA 编程模型本质上是异构计算——CPU 负责控制逻辑，GPU 负责数据并行计算。以下按功能模块系统梳理核心函数的用法。

## 1. 设备管理（Device Management）

在任何 CUDA 操作之前，首先需要识别并初始化可用的 GPU 设备。

```cpp
#include <cuda_runtime.h>
#include <iostream>

void initDevice() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);  // 获取可用 GPU 数量
    
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);  // 获取设备属性
        std::cout << "Device " << i << ": " << prop.name << "\n";
        std::cout << "  SM count: " << prop.multiProcessorCount << "\n";
        std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << "\n";
    }
    
    cudaSetDevice(0);  // 选择设备 0 作为当前操作对象
}
```

**关键函数说明：**

| 函数 | 用途 |
|------|------|
| `cudaGetDeviceCount` | 获取可用设备数量 |
| `cudaGetDeviceProperties` | 获取设备详细属性（SM 数量、内存大小、计算能力等） |
| `cudaSetDevice` | 设置当前线程要使用的 GPU 设备 |
| `cudaDeviceSynchronize` | 阻塞主机，直到设备上所有任务完成（调试常用） |

**环境变量控制：**
- `CUDA_VISIBLE_DEVICES`：限制应用可见的 GPU 列表，常用于多用户环境隔离 

---

## 2. 内存管理（Memory Management）

### 2.1 传统同步内存 API

最基础的内存操作模式——分配、拷贝、释放。

```cpp
// 主机端（CPU）和设备端（GPU）内存分配与传输
float* h_data = new float[N];        // 可分页主机内存
float* d_data = nullptr;

// 分配设备内存
cudaMalloc(&d_data, N * sizeof(float));

// 数据从主机复制到设备
cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

// 启动内核...
kernel<<<grid, block>>>(d_data, N);

// 结果从设备复制回主机
cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);

// 释放资源
cudaFree(d_data);
delete[] h_data;
```

**传输方向参数：**
- `cudaMemcpyHostToDevice`：CPU → GPU
- `cudaMemcpyDeviceToHost`：GPU → CPU
- `cudaMemcpyDeviceToDevice`：GPU → GPU（同一设备内）

### 2.2 锁页内存（Pinned Memory）

普通 `new/malloc` 分配的是**可分页内存**，GPU 无法直接通过 DMA 访问，需要先拷贝到临时的锁页缓冲区。使用 `cudaMallocHost` 分配锁页内存可提高传输带宽。

```cpp
float* h_pinned;
cudaMallocHost(&h_pinned, N * sizeof(float));  // 分配锁页内存
// ... 使用后释放
cudaFreeHost(h_pinned);
```

**注意：** 锁页内存会占用物理内存，过量使用可能导致系统性能下降。

### 2.3 流序内存分配器（Stream-Ordered Memory Allocator）

CUDA 13+ 推荐使用异步内存分配，可与流中的其他操作并执行序化 。

```cpp
void* ptr;
cudaMallocAsync(&ptr, size, stream);   // 在指定流中异步分配
kernel<<<..., stream>>>(ptr, ...);      // 使用分配的内存
cudaFreeAsync(ptr, stream);             // 异步释放，无需同步主机
```

**优势：** 避免 `cudaMalloc/cudaFree` 导致的全局同步，支持内存池复用，显著降低分配开销 。

### 2.4 内存池（Memory Pool）

```cpp
cudaMemPool_t memPool;
cudaDeviceGetDefaultMemPool(&memPool, 0);  // 获取默认内存池

// 设置内存池属性
cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReleaseThreshold, &threshold);
```

---

## 3. 流管理（Stream Management）

CUDA 流（Stream）是一个**命令队列**，同一流内的操作按顺序执行，不同流之间可以并发执行。

```cpp
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// 在不同流中异步执行
cudaMemcpyAsync(d_data1, h_data1, size, cudaMemcpyHostToDevice, stream1);
kernel1<<<grid, block, 0, stream1>>>(d_data1);

cudaMemcpyAsync(d_data2, h_data2, size, cudaMemcpyHostToDevice, stream2);
kernel2<<<grid, block, 0, stream2>>>(d_data2);

// 同步方式
cudaStreamSynchronize(stream1);  // 等待特定流完成
cudaDeviceSynchronize();         // 等待所有流完成

cudaStreamDestroy(stream1);
cudaStreamDestroy(stream2);
```

**并发条件：**
- 使用 `cudaMemcpyAsync` 而非同步版 `cudaMemcpy`
- 主机内存必须是锁页内存（`cudaMallocHost`）
- 内核与数据传输可以重叠（需设备支持）

---

## 4. 内核启动与执行配置

### 4.1 内核定义与调用

```cpp
// 内核定义：__global__ 表示在设备端执行、可从主机调用
__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    // 计算全局线程索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// 主机端调用语法：kernel<<<gridDim, blockDim, sharedMem, stream>>>(args)
int threadsPerBlock = 256;
int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
```

**执行配置参数：**
- `gridDim`：Grid 中 Block 的数量（`dim3` 类型，最高三维）
- `blockDim`：每个 Block 中 Thread 的数量（`dim3` 类型）
- `sharedMem`：动态分配的共享内存字节数（可选，默认 0）
- `stream`：关联的流（可选，默认 0 即默认流）

### 4.2 内置变量

在内核中可直接使用以下变量获取线程索引：

| 变量 | 含义 |
|------|------|
| `threadIdx.x/y/z` | 线程在 Block 内的索引 |
| `blockIdx.x/y/z` | Block 在 Grid 中的索引 |
| `blockDim.x/y/z` | Block 在各维度的线程数 |
| `gridDim.x/y/z` | Grid 在各维度的 Block 数 |

---

## 5. 同步函数（Synchronization）

CUDA 提供多个层次的同步原语。

### 5.1 主机端同步

```cpp
cudaDeviceSynchronize();      // 等待设备上所有任务完成
cudaStreamSynchronize(stream); // 等待指定流中的任务完成
cudaEventSynchronize(event);   // 等待指定事件发生
```

### 5.2 设备端同步

```cpp
__global__ void kernel() {
    // Block 内所有线程同步（最常用）
    __syncthreads();
    
    // Warp 内线程同步（32 线程为一组）
    __syncwarp();
    
    // 线程栅栏（Memory Fence）
    __threadfence();        // 全局内存栅栏
    __threadfence_block();  // Block 级栅栏
}
```

**`__syncthreads()` 注意事项：**
- 必须在 Block 内**所有线程都会执行到**的路径上调用
- 不能放在条件分支内（除非该分支对所有线程统一）
- 常用于共享内存归约前后的同步点 

### 5.3 异步栅栏（Asynchronous Barrier）

CUDA 13+ 引入 `cuda::barrier` 实现更细粒度的异步同步，支持 split arrive/wait 模式 ：

```cpp
#include <cuda/barrier>
#include <cooperative_groups.h>

__global__ void async_example() {
    __shared__ cuda::barrier<cuda::thread_scope_block> bar;
    auto block = cooperative_groups::this_thread_block();
    
    // 单线程初始化
    if (block.thread_rank() == 0) {
        init(&bar, block.size());
    }
    block.sync();
    
    // Arrive（非阻塞）
    auto token = bar.arrive();
    // ... 在此期间可执行其他计算 ...
    // Wait（阻塞直到所有线程 arrive）
    bar.wait(std::move(token));
}
```

---

## 6. 共享内存（Shared Memory）

共享内存是 Block 内线程共用的**片上高速缓存**（L1 级别），延迟远低于全局内存。

### 6.1 静态分配

```cpp
__global__ void staticSharedKernel() {
    __shared__ float cache[256];  // 编译期确定大小
    
    int tid = threadIdx.x;
    cache[tid] = tid * 1.0f;
    __syncthreads();
    // 使用 cache...
}
```

### 6.2 动态分配

```cpp
__global__ void dynamicSharedKernel() {
    extern __shared__ float cache[];  // 大小由主机端启动时指定
    
    // 使用方式与静态相同
}

// 主机端启动时指定共享内存大小
size_t sharedMemSize = blockSize * sizeof(float);
kernel<<<grid, block, sharedMemSize>>>(args);
```

### 6.3 归约（Reduction）典型模式

```cpp
__global__ void reduce(float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 加载到共享内存
    sdata[tid] = (idx < n) ? input[idx] : 0;
    __syncthreads();
    
    // 并行归约（要求 blockDim 是 2 的幂）
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // 每个 Block 的结果写回全局内存
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}
```

---

## 7. 原子操作（Atomic Operations）

当多个线程需要同时修改同一内存位置时，必须使用原子操作避免数据竞争。

```cpp
__global__ void histogram(int* data, int* bins, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int bin = data[idx] % 10;
        atomicAdd(&bins[bin], 1);  // 原子加法
    }
}
```

**常用原子函数：**
- `atomicAdd` / `atomicSub` — 加减
- `atomicExch` — 交换
- `atomicMin` / `atomicMax` — 最小/最大值
- `atomicInc` / `atomicDec` — 环绕递增/递减
- `atomicCAS` — Compare-And-Swap（可实现自定义原子操作）

---

## 8. 错误处理

CUDA 函数通常返回 `cudaError_t` 枚举值，建议封装成检查宏。

```cpp
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// 使用示例
CUDA_CHECK(cudaMalloc(&d_ptr, size));
CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice));

// 内核启动后检查
kernel<<<grid, block>>>(args);
CUDA_CHECK(cudaGetLastError());  // 捕获启动配置错误
CUDA_CHECK(cudaDeviceSynchronize());  // 捕获内核执行错误
```

---

## 9. 事件（Event）与性能计时

CUDA Event 可用于精确计时和流间依赖管理。

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, stream);   // 记录开始事件
kernel<<<grid, block, 0, stream>>>(args);
cudaEventRecord(stop, stream);    // 记录结束事件

cudaEventSynchronize(stop);       // 等待事件完成
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
printf("Kernel time: %.3f ms\n", milliseconds);

cudaEventDestroy(start);
cudaEventDestroy(stop);
```

**流间依赖：**
```cpp
cudaStreamWaitEvent(stream2, event1);  // stream2 等待 event1 完成后再继续
```

---

## 10. 统一内存（Unified Memory）

统一内存简化了内存管理——CPU 和 GPU 使用同一指针，迁移由驱动自动处理。

```cpp
float* data;
cudaMallocManaged(&data, N * sizeof(float));

// 初始化（CPU 访问）
for (int i = 0; i < N; i++) data[i] = i * 0.1f;

// GPU 访问
kernel<<<grid, block>>>(data, N);
cudaDeviceSynchronize();

// CPU 再次访问结果
printf("Result: %f\n", data[0]);

cudaFree(data);
```

**性能优化建议：**
- 使用 `cudaMemPrefetchAsync` 预迁移数据到指定设备
- 使用 `cudaMemAdvise` 给驱动提供访问模式提示

---

## 11. 协作组（Cooperative Groups）

协作组是 CUDA 9+ 引入的抽象层，提供更灵活、安全的线程协作方式。

```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void coop_kernel() {
    cg::thread_block block = cg::this_thread_block();  // 当前 Block
    cg::thread_group warp = cg::tiled_partition<32>(block);  // Warp 分组
    
    // Block 级同步（更安全的 __syncthreads 替代）
    block.sync();
    
    // Warp 级 shuffle 操作
    int val = warp.shfl_down(data, 1);
}
```

**优势：**
- 更清晰的语义和类型安全
- 支持显式的 warp 级归约（`cg::tiled_partition`）
- 模块化设计，未来架构兼容性更好

---

## 12. 全局内存访问优化要点

这是影响性能最关键的环节。GPU 以 **Warp（32 线程）** 为单位执行，以 **32 字节内存事务** 为单位访问全局内存 。

### 12.1 合并访问（Coalesced Access）

**高效模式——连续访问：**
```cpp
__global__ void coalesced(float* in, float* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        out[tid] = in[tid] * 2.0f;  // 相邻线程访问相邻地址
    }
}
```
一个 Warp 的 32 个线程访问连续的 128 字节，仅需 4 次 32 字节内存事务 。

**低效模式——跨步访问：**
```cpp
__global__ void strided(float* in, float* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        out[tid] = in[(tid * 32) % n] * 2.0f;  // 跨步访问
    }
}
```
每个线程请求的 4 字节都来自不同的 32 字节扇区，导致大量数据被读取但未使用 。

### 12.2 分析工具

使用 Nsight Compute 分析内存合并效率：
```bash
# 全局内存分析
ncu --metrics group:memory__dram_table ./app
# L1 缓存分析
ncu --metrics group:memory__first_level_cache_table ./app
```

---

## 13. 环境变量速查表

| 变量 | 作用 | 典型值 |
|------|------|--------|
| `CUDA_VISIBLE_DEVICES` | 限制可见 GPU | `0,1` 或 GPU UUID  |
| `CUDA_LAUNCH_BLOCKING` | 同步执行内核（便于调试） | `1`  |
| `CUDA_CACHE_DISABLE` | 禁用 JIT 缓存 | `1`  |
| `CUDA_DEVICE_MAX_CONNECTIONS` | 并发工作队列数 | `8`（默认）  |

---

## 小结：典型程序框架

```cpp
int main() {
    // 1. 设备选择
    CUDA_CHECK(cudaSetDevice(0));
    
    // 2. 内存分配
    float *h_data, *d_data;
    CUDA_CHECK(cudaMallocHost(&h_data, N * sizeof(float)));  // 锁页
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
    
    // 3. 数据初始化
    init_data(h_data, N);
    
    // 4. 数据传输 + 流创建
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMemcpyAsync(d_data, h_data, N * sizeof(float), 
                               cudaMemcpyHostToDevice, stream));
    
    // 5. 内核执行
    int block = 256;
    int grid = (N + block - 1) / block;
    kernel<<<grid, block, 0, stream>>>(d_data, N);
    CUDA_CHECK(cudaGetLastError());
    
    // 6. 结果回传
    CUDA_CHECK(cudaMemcpyAsync(h_data, d_data, N * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    
    // 7. 同步与清理
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFreeHost(h_data));
    
    return 0;
}
```