// benchmark_forward_layer.cpp
#include "gpu.hpp"
#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <cmath>
#include <future>
#include <algorithm>

using namespace gpu;

// CPU version of forwardLayer with ReLU
void forwardLayer_cpu(
    const std::vector<float>& input,
    const std::vector<float>& weights,
    const std::vector<float>& bias,
    std::vector<float>& output,
    int inSize,
    int outSize
) {
    for (int i = 0; i < outSize; ++i) {
        float sum = bias[i];
        for (int j = 0; j < inSize; ++j) {
            sum += input[j] * weights[i * inSize + j];
        }
        // ReLU activation
        output[i] = sum < 0.0f ? 0.0f : sum;
    }
}

// WGSL kernel: compute dot-product + bias + ReLU
static const char* kForwardLayerWGSL = R"(
@group(0) @binding(0) var<storage, read_write> input:   array<f32>;
@group(0) @binding(1) var<storage, read_write> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> bias:    array<f32>;
@group(0) @binding(3) var<storage, read_write> output:  array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let outSize = arrayLength(&bias);
    let inSize  = arrayLength(&input);
    if (i < outSize) {
        var sum: f32 = bias[i];
        for (var j: u32 = 0u; j < inSize; j = j + 1u) {
            sum = sum + input[j] * weights[i * inSize + j];
        }
        // ReLU
        if (sum < 0.0) { sum = 0.0; }
        output[i] = sum;
    }
}
)";

int main() {
    // layer dimensions
    const int inSize  = 2048;
    const int outSize = 1024;

    // random initialization
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float>
        input(inSize),
        weights(inSize * outSize),
        bias(outSize),
        cpu_out(outSize),
        gpu_out(outSize);

    for (auto &v : input)   v = dist(rng);
    for (auto &w : weights) w = dist(rng);
    for (auto &b : bias)    b = dist(rng);

    // --- CPU run ---
    auto t0 = std::chrono::high_resolution_clock::now();
    forwardLayer_cpu(input, weights, bias, cpu_out, inSize, outSize);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double,std::milli>(t1 - t0).count();
    std::cout << "CPU forwardLayer: " << cpu_ms << " ms\n";

    // --- GPU setup ---
    Context ctx = createContext();
    if (!ctx.device) {
        std::cerr << "ERROR: no GPU device\n";
        return 1;
    }

    // create tensors
    Tensor in_t  = createTensor(ctx, Shape{size_t(inSize)},         kf32, input.data());
    Tensor w_t   = createTensor(ctx, Shape{size_t(inSize*outSize)}, kf32, weights.data());
    Tensor b_t   = createTensor(ctx, Shape{size_t(outSize)},        kf32, bias.data());
    Tensor out_t = createTensor(ctx, Shape{size_t(outSize)},        kf32);

    Bindings bindings{in_t, w_t, b_t, out_t};
    uint32_t groups = (outSize + 255) / 256;
    Kernel kernel = createKernel(
        ctx,
        { kForwardLayerWGSL, 256, kf32 },
        bindings,
        { groups, 1, 1 }
    );

    // --- GPU run ---
    std::promise<void> p;
    auto f = p.get_future();
    auto t2 = std::chrono::high_resolution_clock::now();
    dispatchKernel(ctx, kernel, p);
    wait(ctx, f);
    auto t3 = std::chrono::high_resolution_clock::now();
    double gpu_ms = std::chrono::duration<double,std::milli>(t3 - t2).count();
    std::cout << "GPU forwardLayer: " << gpu_ms << " ms\n";

    // copy back
    toCPU(ctx, out_t, gpu_out.data(), sizeof(float)*gpu_out.size());

    // --- compare ---
    double sumErr = 0.0, maxErr = 0.0;
    for (int i = 0; i < outSize; ++i) {
        double d = std::abs(cpu_out[i] - gpu_out[i]);
        sumErr += d;
        if (d > maxErr) maxErr = d;
    }
    std::cout << "Mean abs diff: " << (sumErr / outSize) << "\n";
    std::cout << "Max  abs diff: " << maxErr << "\n";
    return 0;
}
