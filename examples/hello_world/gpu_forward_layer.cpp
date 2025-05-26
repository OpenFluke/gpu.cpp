#include "gpu.hpp"
#include <nlohmann/json.hpp>
#include <future>
#include <vector>
#include <string>
#include <cstring>
#include <iostream>
#include <iterator>
#include <sstream>

using namespace gpu;
using json = nlohmann::json;

// Updated WGSL kernel with correct access qualifiers
static const char* kForwardLayer = R"(
const GELU_SCALING_FACTOR: f32 = 0.7978845608028654;

// Declare storage buffers with explicit access qualifiers
@group(0) @binding(0) var<storage, read> inputs      : array<f32>;  // Read-only
@group(0) @binding(1) var<storage, read> weights     : array<f32>;  // Read-only
@group(0) @binding(2) var<storage, read> offsets     : array<f32>;  // Read-only
@group(0) @binding(3) var<storage, read> counts      : array<f32>;  // Read-only
@group(0) @binding(4) var<storage, read> biases      : array<f32>;  // Read-only
@group(0) @binding(5) var<storage, read_write> output : array<f32>; // Read-write (allow writing)
@group(0) @binding(6) var<storage, read> activations : array<f32>;  // Read-only

fn activate(x: f32, kind: u32) -> f32 {
  if (kind == 1u)       { return max(0.0, x); }                    // ReLU
  else if (kind == 2u)  { return select(0.01 * x, x, x > 0.0); }   // Leaky ReLU
  else if (kind == 3u)  { return tanh(x); }                        // Tanh
  else if (kind == 4u)  { return 1.0 / (1.0 + exp(-x)); }          // Sigmoid
  else if (kind == 5u)  {                                          // GELU
    return 0.5 * x * (1.0 + tanh(GELU_SCALING_FACTOR * (x + 0.044715 * x * x * x)));
  }
  return x;  // Default: linear activation
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i        = gid.x;
  let numNeurs = arrayLength(&biases);
  if (i >= numNeurs) { return; }

  let start = bitcast<u32>(offsets[i]);
  let cnt   = bitcast<u32>(counts[i]);

  var sum: f32 = biases[i];
  for (var j: u32 = 0u; j < cnt; j = j + 1u) {
    let idx = start + j;
    sum += inputs[idx] * weights[idx];
  }

  let actId = bitcast<u32>(activations[i]);
  output[i] = activate(sum, actId);  // Now allowed since output is read_write
}
)";

uint32_t activationToId(const std::string &act) {
    if (act == "relu")        return 1;
    if (act == "leaky_relu")  return 2;
    if (act == "tanh")        return 3;
    if (act == "sigmoid")     return 4;
    if (act == "gelu")        return 5;
    return 0;  // Default: linear
}

template<typename T>
void bitcastVec(const std::vector<uint32_t>& src, std::vector<T>& dst) {
    static_assert(sizeof(T) == sizeof(uint32_t), "Sizes must match");
    dst.resize(src.size());
    std::memcpy(dst.data(), src.data(), src.size() * sizeof(uint32_t));
}

int main() {
    // 1) Create context ONCE
    Context ctx;
    try {
        ctx = createContext();
    } catch (const std::exception& ex) {
        std::cerr << "[FATAL] Failed to create GPU context: " << ex.what() << std::endl;
        return 1;
    }

    // 2) Loop: read a line, process, output, repeat
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) {
            std::cout << "[]\n" << std::flush;
            continue;
        }

        json inputData;
        try {
            inputData = json::parse(line);
        } catch (const json::exception &e) {
            std::cerr << "[JSON parse error] " << e.what() << std::endl;
            std::cout << "[]\n" << std::flush;
            continue;
        }

        size_t N = inputData.size();
        if (N == 0) {
            std::cerr << "[Input] No neurons in input." << std::endl;
            std::cout << "[]\n" << std::flush;
            continue;
        }

        std::vector<float> inFlat, wFlat, biases;
        std::vector<uint32_t> offs, cnts, actIds;
        bool layer_ok = true;
        for (size_t i = 0; i < N; ++i) {
            const auto &n = inputData[i];
            if (!n.contains("inputs") || !n.contains("weights") ||
                !n.contains("bias")   || !n.contains("activation")) {
                std::cerr << "[Input] Neuron " << i << " missing fields." << std::endl;
                layer_ok = false;
                break;
            }
            auto &ins = n["inputs"];
            auto &wts = n["weights"];
            if (ins.size() != wts.size()) {
                std::cerr << "[Input] Neuron " << i << " inputs/weights mismatch." << std::endl;
                layer_ok = false;
                break;
            }
            offs.push_back(static_cast<uint32_t>(inFlat.size()));
            cnts.push_back(static_cast<uint32_t>(ins.size()));
            for (size_t k = 0; k < ins.size(); ++k) {
                inFlat.push_back(ins[k].get<float>());
                wFlat.push_back(wts[k].get<float>());
            }
            biases.push_back(n["bias"].get<float>());
            actIds.push_back(activationToId(n["activation"].get<std::string>()));
        }

        if (!layer_ok) {
            std::cout << "[]\n" << std::flush;
            continue;
        }

        // Bit-cast u32 → f32 for WGSL
        std::vector<float> offsF, cntsF, actsF;
        bitcastVec(offs, offsF);
        bitcastVec(cnts, cntsF);
        bitcastVec(actIds, actsF);

        // --- Build tensors & dispatch compute ---
        std::vector<float> out(N);
        try {
            Tensor tIn    = createTensor(ctx, Shape{inFlat.size()},  kf32, inFlat.data());
            Tensor tW     = createTensor(ctx, Shape{wFlat.size()},   kf32, wFlat.data());
            Tensor tOfs   = createTensor(ctx, Shape{offsF.size()},   kf32, offsF.data());
            Tensor tCnt   = createTensor(ctx, Shape{cntsF.size()},   kf32, cntsF.data());
            Tensor tB     = createTensor(ctx, Shape{N},              kf32, biases.data());
            Tensor tAct   = createTensor(ctx, Shape{N},              kf32, actsF.data());
            // Output tensor must be WRITABLE (usage flag set in createTensor)
            Tensor tOut   = createTensor(ctx, Shape{N},              kf32);

            std::promise<void> p;
            auto fut = p.get_future();
            Kernel k = createKernel(ctx, {kForwardLayer, 64, kf32},
                                    Bindings{tIn,tW,tOfs,tCnt,tB,tOut,tAct},
                                    { cdiv(N,64), 1, 1 });
            dispatchKernel(ctx, k, p);
            wait(ctx, fut);

            toCPU(ctx, tOut, out.data(), sizeof(float)*N);

        } catch (const std::exception& ex) {
            std::cerr << "[Kernel/Dispatch ERROR] " << ex.what() << std::endl;
            std::fill(out.begin(), out.end(), 0.0f);
        }

        // --- Output result (and nothing else) to stdout ---
        std::cout << json(out).dump() << std::endl;
        std::cout.flush();
    }
    return 0;
}