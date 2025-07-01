#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <cstring>
#include <cstdio>
#include <memory>
#include <stdexcept>

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
__global__ void generate_kernel(const char *input, char *output, const char *prefix, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x; // 计算当前线程处理的元素索引
    if (i >= count) return; // 超出范围则返回

    const char *in = input + i * MAX_LEN;  // 当前线程处理的输入字符串
    char *out = output + i * MAX_LEN;     // 当前线程处理的输出位置

    // 1. 复制前缀部分
    int p = 0;
    while (prefix[p] != '\0' && p < MAX_LEN - 1) {
        out[p] = prefix[p];
        p++;
    }

    // 2. 追加原始字符串内容
    int j = 0;
    while (in[j] != '\0' && p < MAX_LEN - 1) {
        out[p++] = in[j++];
    }
    out[p] = '\0';  // 确保字符串正确终止
}

// 主处理函数（外部C接口）
extern "C" void gpu_generate(const char *flat_input, int count, const std::string &prefix, std::vector<std::string> &result) {
    // 处理空输入情况
    if (count <= 0) {
        result.clear();
        return;
    }

    const size_t total_size = count * MAX_LEN;
    
    // 1. 分配设备内存
    char *d_input = nullptr;
    char *d_output = nullptr;
    char *d_prefix = nullptr;
    
    try {
        CUDA_CHECK(cudaMalloc(&d_input, total_size));
        CUDA_CHECK(cudaMalloc(&d_output, total_size));
        CUDA_CHECK(cudaMalloc(&d_prefix, MAX_LEN));

        // 2. 拷贝数据到设备
        CUDA_CHECK(cudaMemcpy(d_input, flat_input, total_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_prefix, prefix.c_str(), prefix.size() + 1, cudaMemcpyHostToDevice));

        // 3. 计算并启动核函数
        const int block_size = 256;  // 每个块的线程数
        const int grid_size = (count + block_size - 1) / block_size;  // 计算需要的块数
        
        generate_kernel<<<grid_size, block_size>>>(d_input, d_output, d_prefix, count);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // 4. 将结果拷贝回主机
        std::unique_ptr<char[]> flat_output(new char[total_size]);
        CUDA_CHECK(cudaMemcpy(flat_output.get(), d_output, total_size, cudaMemcpyDeviceToHost));

        // 5. 转换为字符串向量
        result.clear();
        result.reserve(count);  // 预分配空间提高效率
        for (int i = 0; i < count; ++i) {
            result.emplace_back(flat_output.get() + i * MAX_LEN);
        }

    } catch (...) {
        // 确保异常时也能释放内存
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
        if (d_prefix) cudaFree(d_prefix);
        throw;
    }

    // 6. 释放设备内存
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_prefix));
}
