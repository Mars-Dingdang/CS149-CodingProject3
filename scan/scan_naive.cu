#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"

#define THREADS_PER_BLOCK 256


// helper function to round an integer up to the next power of 2
static inline int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

// exclusive_scan --
//
// Implementation of an exclusive scan on global memory array `input`,
// with results placed in global memory `result`.
//
// N is the logical size of the input and output arrays, however
// students can assume that both the start and result arrays we
// allocated with next power-of-two sizes as described by the comments
// in cudaScan().  This is helpful, since your parallel scan
// will likely write to memory locations beyond N, but of course not
// greater than N rounded up to the next power of 2.
//
// Also, as per the comments in cudaScan(), you can implement an
// "in-place" scan, since the timing harness makes a copy of input and
// places it in result
__global__ void upsweep_kernel(int *result, int d, int nxt_d, int rounded_N);
__global__ void downsweep_kernel(int *result, int d, int nxt_d, int rounded_N);
void exclusive_scan(int* input, int N, int* result)
{

    // CS149 TODO:
    //
    // Implement your exclusive scan implementation here.  Keep in
    // mind that although the arguments to this function are device
    // allocated arrays, this is a function that is running in a thread
    // on the CPU.  Your implementation will need to make multiple calls
    // to CUDA kernel functions (that you must write) to implement the
    // scan.
    
    int rounded_N = nextPow2(N);
    if(input != result) cudaMemcpy(result, input, N * sizeof(int), cudaMemcpyDeviceToDevice);
    if(rounded_N > N) cudaMemset(result + N, 0, (rounded_N - N) * sizeof(int));
    for(int d = 1; d < rounded_N; d <<= 1) {
        int nxt_d = d << 1;
        int num_work_items = rounded_N / nxt_d;
        int num_blocks = (num_work_items + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        if (num_blocks > 65535) num_blocks = 65535;
        upsweep_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(result, d, nxt_d, rounded_N);
    }
    cudaMemset(result + rounded_N - 1, 0, sizeof(int));
    for(int d = rounded_N >> 1; d > 0; d >>= 1) {
        int nxt_d = d << 1;
        int num_work_items = rounded_N / nxt_d;
        int num_blocks = (num_work_items + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        if (num_blocks > 65535) num_blocks = 65535;
        downsweep_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(result, d, nxt_d, rounded_N);
    }
}

__global__ void upsweep_kernel(int *result, int d, int nxt_d, int rounded_N) {
    // blockIdx是块索引，blockDim是每个块的线程数，threadIdx是线程索引
    // gridDim.x 是块的数量，blockDim.x是每个块的线程数，因此gridDim.x * blockDim.x是总线程数
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int num_work_items = rounded_N / nxt_d;

    for (; idx < num_work_items; idx += stride) { // 注意num_blocks数量有限，因此需要循环来保证所有工作项都被处理到
        int offset = idx * nxt_d;
        result[offset + nxt_d - 1] += result[offset + d - 1];
    }
}

__global__ void downsweep_kernel(int *result, int d, int nxt_d, int rounded_N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int num_work_items = rounded_N / nxt_d;

    for (; idx < num_work_items; idx += stride) {
        int offset = idx * nxt_d;
        int tmp = result[offset + d - 1];
        result[offset + d - 1] = result[offset + nxt_d - 1];
        result[offset + nxt_d - 1] += tmp;
    }
}


//
// cudaScan --
//
// This function is a timing wrapper around the student's
// implementation of scan - it copies the input to the GPU
// and times the invocation of the exclusive_scan() function
// above. Students should not modify it.
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    int* device_input;
    int N = end - inarray;  

    // This code rounds the arrays provided to exclusive_scan up
    // to a power of 2, but elements after the end of the original
    // input are left uninitialized and not checked for correctness.
    //
    // Student implementations of exclusive_scan may assume an array's
    // allocated length is a power of 2 for simplicity. This will
    // result in extra work on non-power-of-2 inputs, but it's worth
    // the simplicity of a power of two only solution.

    int rounded_length = nextPow2(end - inarray);
    
    cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);

    // For convenience, both the input and output vectors on the
    // device are initialized to the input values. This means that
    // students are free to implement an in-place scan on the result
    // vector if desired.  If you do this, you will need to keep this
    // in mind when calling exclusive_scan from find_repeats.
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_input, N, device_result);

    // Wait for completion
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
       
    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// cudaScanThrust --
//
// Wrapper around the Thrust library's exclusive scan function
// As above in cudaScan(), this function copies the input to the GPU
// and times only the execution of the scan itself.
//
// Students are not expected to produce implementations that achieve
// performance that is competition to the Thrust version, but it is fun to try.
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
    
    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
   
    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);

    thrust::device_free(d_input);
    thrust::device_free(d_output);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// find_repeats --
//
// Given an array of integers `device_input`, returns an array of all
// indices `i` for which `device_input[i] == device_input[i+1]`.
//
// Returns the total number of pairs found
__global__ void find_repeats_kernel(int *device_input, int length, int *exclusive_sum);
__global__ void write_device_output_kernel(int *exclusive_sum, int *device_output, int length);
int find_repeats(int* device_input, int length, int* device_output) {

    // CS149 TODO:
    //
    // Implement this function. You will probably want to
    // make use of one or more calls to exclusive_scan(), as well as
    // additional CUDA kernel launches.
    //    
    // Note: As in the scan code, the calling code ensures that
    // allocated arrays are a power of 2 in size, so you can use your
    // exclusive_scan function with them. However, your implementation
    // must ensure that the results of find_repeats are correct given
    // the actual array length.
    if(length < 1) return 0;
    int cnt = 0;
    // int *device_cnt = nullptr;
    int *exclusive_sum = nullptr;
    // cudaMalloc((void **)&device_cnt, sizeof(int));
    // cudaMemcpy(device_cnt, &cnt, sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&exclusive_sum, sizeof(int) * nextPow2(length));
    cudaMemset(exclusive_sum, 0, sizeof(int) * nextPow2(length)); // initialize exclusive_sum to 0 to avoid uninitialized padding
    int num_blocks = (length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    find_repeats_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(device_input, length, exclusive_sum);
    cudaDeviceSynchronize();
    // cudaMemcpy(&cnt, device_cnt, sizeof(int), cudaMemcpyDeviceToHost);
    exclusive_scan(exclusive_sum, length, exclusive_sum);
    write_device_output_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(exclusive_sum, device_output, length);
    cudaDeviceSynchronize();
    cudaMemcpy(&cnt, exclusive_sum + length - 1, sizeof(int), cudaMemcpyDeviceToHost);
    // cudaFree(device_cnt);
    cudaFree(exclusive_sum);
    return cnt; 
}

__global__ void find_repeats_kernel(int *device_input, int length, int *exclusive_sum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < length - 1) {
        exclusive_sum[idx] = (device_input[idx] == device_input[idx + 1]) ? 1 : 0;
        // atomicAdd(device_cnt, exclusive_sum[idx]);
        // device_output[idx] = (device_input[idx] == device_input[idx + 1]) ? idx : 0;
    }
}

__global__ void write_device_output_kernel(int *exclusive_sum, int *device_output, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < length - 1 && exclusive_sum[idx] != exclusive_sum[idx + 1]) {
        device_output[exclusive_sum[idx]] = idx;
    }
}


//
// cudaFindRepeats --
//
// Timing wrapper around find_repeats. You should not modify this function.
double cudaFindRepeats(int *input, int length, int *output, int *output_length) {

    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    double startTime = CycleTimer::currentSeconds();
    
    int result = find_repeats(device_input, length, device_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    // set output count and results array
    *output_length = result;
    cudaMemcpy(output, device_output, length * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    float duration = endTime - startTime; 
    return duration;
}



void printCudaInfo()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n"); 
}
