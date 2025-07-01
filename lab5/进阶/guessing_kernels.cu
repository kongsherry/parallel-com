#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <cstring>
#include <cstdio>
#include <memory>
#include <stdexcept>
using namespace std;

// 定义最大字符串长度
constexpr int MAX_LEN = 64;

// CUDA错误检查宏
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA错误 @ %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        throw std::runtime_error("CUDA运行时错误"); \
    } \
} while(0)

// GPU核函数：为每个输入字符串添加前缀
__global__ void generate_batch_kernel(
    const char* values, const char* prefixes, 
    const int* counts, char* output, int pt_count) {
    
    // 计算当前PT的索引
    int pt_idx = blockIdx.x;
    if(pt_idx >= pt_count) return;
    
    // 获取当前PT的前缀和值范围
    const char* prefix = prefixes + pt_idx * MAX_LEN;
    int start_idx = (pt_idx == 0) ? 0 : counts[pt_idx-1];
    int end_idx = counts[pt_idx];
    
    // 每个线程处理一个值
    int value_idx = threadIdx.x;
    if(value_idx >= (end_idx - start_idx)) return;
    
    // 计算输出位置
    char* out = output + (start_idx + value_idx) * MAX_LEN;
    
    // 1. 复制前缀
    int p = 0;
    while(prefix[p] != '\0' && p < MAX_LEN - 1) {
        out[p] = prefix[p];
        p++;
    }
    
    // 2. 追加值
    const char* val = values + (start_idx + value_idx) * MAX_LEN;
    int j = 0;
    while(val[j] != '\0' && p < MAX_LEN - 1) {
        out[p++] = val[j++];
    }
    out[p] = '\0';
}

extern "C" void gpu_generate_batch(
    const char* values, const char* prefixes, 
    const int* counts, int pt_count, 
    std::vector<std::string>& result) {
    
    // 计算总value数
    int total_values = 0;
    std::vector<int> host_counts(counts, counts + pt_count);
    for(int cnt : host_counts) total_values += cnt;
    
    // 分配设备内存
    char *d_values, *d_prefixes, *d_output;
    int *d_counts;
    
    size_t values_size = total_values * MAX_LEN;
    size_t prefixes_size = pt_count * MAX_LEN;
    size_t output_size = total_values * MAX_LEN;
    size_t counts_size = pt_count * sizeof(int);
    
    CUDA_CHECK(cudaMalloc(&d_values, values_size));
    CUDA_CHECK(cudaMalloc(&d_prefixes, prefixes_size));
    CUDA_CHECK(cudaMalloc(&d_output, output_size));
    CUDA_CHECK(cudaMalloc(&d_counts, counts_size));
    
    // 拷贝数据到设备
    CUDA_CHECK(cudaMemcpy(d_values, values, values_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_prefixes, prefixes, prefixes_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_counts, counts, counts_size, cudaMemcpyHostToDevice));
    
    // 计算并启动核函数
    dim3 grid(pt_count);
    dim3 block(256); // 每个PT最多256个线程
    
    generate_batch_kernel<<<grid, block>>>(d_values, d_prefixes, d_counts, d_output, pt_count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 拷贝结果回主机
    std::vector<char> host_output(output_size);
    CUDA_CHECK(cudaMemcpy(host_output.data(), d_output, output_size, cudaMemcpyDeviceToHost));
    
    // 转换为字符串
    result.clear();
    result.reserve(total_values);
    for(int i = 0; i < total_values; ++i) {
        result.emplace_back(host_output.data() + i * MAX_LEN);
    }
    
    // 释放设备内存
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_prefixes));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_counts));
}
