#include "gpu.hpp"
#include <array>
#include <cstdio>
#include <future>
#include <chrono>
#include <cmath>

using namespace gpu;

static const char *kGelu = R"(
const GELU_SCALING_FACTOR: f32 = 0.7978845608028654; // sqrt(2.0 / PI)
@group(0) @binding(0) var<storage, read_write> inp: array<{{precision}}>;
@group(0) @binding(1) var<storage, read_write> out: array<{{precision}}>;
@compute @workgroup_size({{workgroupSize}})
fn main(
    @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let i: u32 = GlobalInvocationID.x;
    if (i < arrayLength(&inp)) {
        let x: f32 = inp[i];
        out[i] = select(0.5 * x * (1.0 + tanh(GELU_SCALING_FACTOR 
                 * (x + .044715 * x * x * x))), x, x > 10.0);
    }
}
)";

// CPU version of GELU
inline float cpu_gelu(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    return (x > 10.0f)
        ? x
        : 0.5f * x * (1.0f + std::tanh(sqrt_2_over_pi * (x + 0.044715f * x * x * x)));
}

int main(int argc, char **argv) {
    printf("\033[2J\033[1;1H");
    printf("\nHello gpu.cpp!\n");
    printf("--------------\n\n");

    constexpr size_t N = 10000000; // 10 million for clear timing
    std::array<float, 12> inputShow;
    std::vector<float> inputArr(N), outputArr(N), cpuArr(N);

    for (size_t i = 0; i < N; ++i) {
        inputArr[i] = static_cast<float>(i) / 10.0f; // dummy input data
        if (i < 12) inputShow[i] = inputArr[i];
    }

    // ======== CPU GELU ========
    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; ++i) {
        cpuArr[i] = cpu_gelu(inputArr[i]);
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double>(cpu_end - cpu_start).count();

    // ======== GPU GELU ========
    Context ctx = createContext();
    Tensor input = createTensor(ctx, Shape{N}, kf32, inputArr.data());
    Tensor output = createTensor(ctx, Shape{N}, kf32);

    std::promise<void> promise;
    std::future<void> future = promise.get_future();

    Kernel op = createKernel(ctx, {kGelu, 256, kf32},
                             Bindings{input, output},
                             {cdiv(N, 256), 1, 1});

    auto gpu_start = std::chrono::high_resolution_clock::now();
    dispatchKernel(ctx, op, promise);
    wait(ctx, future);
    toCPU(ctx, output, outputArr.data(), sizeof(float) * N);
    auto gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_time = std::chrono::duration<double>(gpu_end - gpu_start).count();

    // ======== Compare and print ========
    printf("=== GELU Benchmark (N = %zu) ===\n", N);
    printf("CPU time: %.4f s\n", cpu_time);
    printf("WebGPU time: %.4f s\n", gpu_time);

    // Compare first 12 results
    printf("\n  Input    |   CPU GELU   |   GPU GELU   | Abs Diff\n");
    printf("-----------|--------------|--------------|----------\n");
    for (int i = 0; i < 12; ++i) {
        printf("%9.2f | %12.6f | %12.6f | %8.6f\n",
            inputShow[i], cpuArr[i], outputArr[i], std::fabs(cpuArr[i] - outputArr[i]));
    }

    // Mean absolute difference
    double abs_diff = 0.0;
    for (size_t i = 0; i < N; ++i) abs_diff += std::fabs(cpuArr[i] - outputArr[i]);
    abs_diff /= N;
    printf("\nMean absolute difference (CPU vs GPU): %.8f\n", abs_diff);

    printf("\nDone!\n");
    return 0;
}
