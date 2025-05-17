// benchmark_dense.cpp
#include "gpu.hpp"
#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <cmath>
#include <future>
#include <algorithm>

using namespace gpu;

// Full dense (matrix-vector + bias) WGSL kernel,
// with all storage buffers marked read_write so arrayLength() works.
static const char* kDenseWGSL = R"(
@group(0) @binding(0) var<storage, read_write> input:   array<f32>;
@group(0) @binding(1) var<storage, read_write> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> bias:    array<f32>;
@group(0) @binding(3) var<storage, read_write> output:  array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let i: u32 = GlobalInvocationID.x;
    let outSize: u32 = arrayLength(&bias);
    let inSize:  u32 = arrayLength(&input);
    if (i < outSize) {
        var sum: f32 = bias[i];
        for (var j: u32 = 0u; j < inSize; j = j + 1u) {
            sum = sum + input[j] * weights[i * inSize + j];
        }
        output[i] = sum;
    }
}
)";

// CPU reference dense
void dense_cpu(const std::vector<float>& in,
               const std::vector<float>& W,
               const std::vector<float>& B,
               std::vector<float>& out,
               int inSize, int outSize) {
    for (int i = 0; i < outSize; ++i) {
        float s = B[i];
        for (int j = 0; j < inSize; ++j) {
            s += in[j] * W[i * inSize + j];
        }
        out[i] = s;
    }
}

int main() {
    const int inSize  = 1024;
    const int outSize = 1024;

    std::mt19937                        rng(42);
    std::uniform_real_distribution<float> dist(-1,1);

    std::vector<float> in(inSize), W(inSize*outSize), B(outSize);
    std::vector<float> cpu_out(outSize), gpu_out(outSize);

    for (auto &x: in)  x = dist(rng);
    for (auto &x: W)   x = dist(rng);
    for (auto &x: B)   x = dist(rng);

    // CPU
    auto t0 = std::chrono::high_resolution_clock::now();
    dense_cpu(in,W,B,cpu_out,inSize,outSize);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double,std::milli>(t1-t0).count();
    std::cout << "CPU time: " << cpu_ms << " ms\n";

    // GPU
    Context ctx = createContext();
    if (!ctx.device) {
        std::cerr<<"ERROR: no GPU device\n";
        return 1;
    }

    auto in_t   = createTensor(ctx, Shape{size_t(inSize)},         kf32, in.data());
    auto W_t    = createTensor(ctx, Shape{size_t(inSize*outSize)}, kf32, W.data());
    auto B_t    = createTensor(ctx, Shape{size_t(outSize)},        kf32, B.data());
    auto out_t  = createTensor(ctx, Shape{size_t(outSize)},        kf32);

    Bindings b{in_t, W_t, B_t, out_t};
    uint32_t groups = (outSize + 255)/256;
    auto kernel = createKernel(ctx, {kDenseWGSL, 256, kf32}, b, {groups,1,1});

    std::promise<void> p; auto f = p.get_future();
    auto t2 = std::chrono::high_resolution_clock::now();
    dispatchKernel(ctx, kernel, p);
    wait(ctx, f);
    auto t3 = std::chrono::high_resolution_clock::now();
    double gpu_ms = std::chrono::duration<double,std::milli>(t3-t2).count();
    std::cout << "GPU time: " << gpu_ms << " ms\n";

    toCPU(ctx, out_t, gpu_out.data(), sizeof(float)*gpu_out.size());

    double sumErr = 0, maxErr = 0;
    for (int i = 0; i < outSize; ++i) {
        double e = std::abs(cpu_out[i] - gpu_out[i]);
        sumErr += e;
        maxErr = std::max(maxErr, e);
    }
    std::cout << "Mean abs diff: " << (sumErr/outSize) << "\n"
              << "Max  abs diff: " << maxErr << "\n";
    return 0;
}
