// benchmark_forward_layer_big.cpp
#include "gpu.hpp"
#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <future>

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
    // Layer sizes to test
    const int inSizes[] = { 256, 1024, 4096, 8192 };
    const int outSizes[] = { 128, 512, 2048, 4096 };
    const int numTests = sizeof(inSizes) / sizeof(inSizes[0]);
    const int numRuns = 20; // Runs per config

    for (int t = 0; t < numTests; ++t) {
        int inSize = inSizes[t];
        int outSize = outSizes[t];
        std::cout << "\n--- Benchmarking inSize: " << inSize << ", outSize: " << outSize << " ---\n";

        std::vector<float> input(inSize), weights(inSize * outSize), bias(outSize), cpu_out(outSize), gpu_out(outSize);
        std::mt19937 rng(12345);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto &v : input)   v = dist(rng);
        for (auto &w : weights) w = dist(rng);
        for (auto &b : bias)    b = dist(rng);

        // --- CPU runs ---
        double cpu_total = 0.0, cpu_best = 1e9;
        for (int run = 0; run < numRuns; ++run) {
            auto t0 = std::chrono::high_resolution_clock::now();
            forwardLayer_cpu(input, weights, bias, cpu_out, inSize, outSize);
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double,std::milli>(t1 - t0).count();
            cpu_total += ms;
            if (ms < cpu_best) cpu_best = ms;
        }
        std::cout << "CPU forwardLayer avg: " << (cpu_total / numRuns) << " ms (best: " << cpu_best << " ms)\n";

        // --- GPU setup ---
        Context ctx = createContext();
        if (!ctx.device) {
            std::cerr << "ERROR: no GPU device\n";
            return 1;
        }
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

        // --- GPU runs ---
        double gpu_total = 0.0, gpu_best = 1e9;
        for (int run = 0; run < numRuns; ++run) {
            std::promise<void> p;
            auto f = p.get_future();
            auto t2 = std::chrono::high_resolution_clock::now();
            dispatchKernel(ctx, kernel, p);
            wait(ctx, f);
            auto t3 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double,std::milli>(t3 - t2).count();
            gpu_total += ms;
            if (ms < gpu_best) gpu_best = ms;
        }
        std::cout << "GPU forwardLayer avg: " << (gpu_total / numRuns) << " ms (best: " << gpu_best << " ms)\n";

        toCPU(ctx, out_t, gpu_out.data(), sizeof(float)*gpu_out.size());

        // Check errors on the final run
        double sumErr = 0.0, maxErr = 0.0;
        for (int i = 0; i < outSize; ++i) {
            double d = std::abs(cpu_out[i] - gpu_out[i]);
            sumErr += d;
            if (d > maxErr) maxErr = d;
        }
        std::cout << "Mean abs diff: " << (sumErr / outSize) << ", Max abs diff: " << maxErr << "\n";
    }
    return 0;
}
