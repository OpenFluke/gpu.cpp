// gpu_forward_layer.cpp
#include "gpu.hpp"               // Dawn/WebGPU helpers
#include <nlohmann/json.hpp>
#include <future>
#include <vector>
#include <string>
#include <cstring>               // std::memcpy
#include <iostream>
#include <iterator>              // std::istreambuf_iterator

using namespace gpu;
using json = nlohmann::json;

/* ───────────────────────── WGSL kernel ──────────────────────────────── */
static const char* kForwardLayer = R"(
const GELU_SCALING_FACTOR: f32 = 0.7978845608028654;

@group(0) @binding(0) var<storage, read>  inputs      : array<f32>;
@group(0) @binding(1) var<storage, read>  weights     : array<f32>;
@group(0) @binding(2) var<storage, read>  offsets     : array<f32>; // bit-cast u32
@group(0) @binding(3) var<storage, read>  counts      : array<f32>; // bit-cast u32
@group(0) @binding(4) var<storage, read>  biases      : array<f32>;
@group(0) @binding(5) var<storage, write> output      : array<f32>;
@group(0) @binding(6) var<storage, read>  activations : array<f32>; // bit-cast u32

fn activate(x: f32, kind: u32) -> f32 {
  if (kind == 1u)       { return max(0.0, x); }                        // ReLU
  else if (kind == 2u)  { return select(0.01 * x, x, x > 0.0); }       // LeakyReLU
  else if (kind == 3u)  { return tanh(x); }
  else if (kind == 4u)  { return 1.0 / (1.0 + exp(-x)); }              // Sigmoid
  else if (kind == 5u)  {                                               // GELU
    return 0.5 * x * (1.0 + tanh(GELU_SCALING_FACTOR *
                                 (x + 0.044715 * x * x * x)));
  }
  return x;                                                            // Linear
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
  output[i] = activate(sum, actId);
}
)";

/* ───────────────────── helper functions ─────────────────────────────── */
uint32_t activationToId(const std::string &act) {
  if (act == "relu")        return 1;
  if (act == "leaky_relu")  return 2;
  if (act == "tanh")        return 3;
  if (act == "sigmoid")     return 4;
  if (act == "gelu")        return 5;
  return 0;                         // Linear / unknown
}

template<typename T>
void bitcastVec(const std::vector<uint32_t>& src, std::vector<T>& dst) {
  static_assert(sizeof(T) == sizeof(uint32_t), "Sizes must match");
  dst.resize(src.size());
  std::memcpy(dst.data(), src.data(), src.size()*sizeof(uint32_t));
}

/* ─────────────────────────── main ───────────────────────────────────── */
int main() {
  // 1) Read all of stdin into a string
  std::string inJson((std::istreambuf_iterator<char>(std::cin)),
                     std::istreambuf_iterator<char>());

  // 2) Parse JSON array of neurons
  json inputData;
  try {
    inputData = json::parse(inJson);
  } catch (const json::exception &e) {
    std::cerr << "JSON parse error: " << e.what() << std::endl;
    return 1;
  }

  size_t N = inputData.size();
  if (N == 0) {
    std::cerr << "No neurons in input." << std::endl;
    return 1;
  }

  // 3) Flatten data into big buffers + record per-neuron offsets/counts
  std::vector<float>    inFlat, wFlat, biases;
  std::vector<uint32_t> offs, cnts, actIds;

  for (size_t i = 0; i < N; ++i) {
    const auto &n = inputData[i];
    if (!n.contains("inputs") || !n.contains("weights") ||
        !n.contains("bias")   || !n.contains("activation")) {
      std::cerr << "Neuron " << i << " missing fields." << std::endl;
      return 1;
    }
    auto &ins = n["inputs"];
    auto &wts = n["weights"];
    if (ins.size() != wts.size()) {
      std::cerr << "Neuron " << i << " inputs/weights mismatch." << std::endl;
      return 1;
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

  // 4) Bit-cast u32 → f32 so WGSL can interpret offsets/counts/acts
  std::vector<float> offsF, cntsF, actsF;
  bitcastVec(offs, offsF);
  bitcastVec(cnts, cntsF);
  bitcastVec(actIds, actsF);

  // 5) Build tensors & dispatch compute
  Context ctx   = createContext();
  Tensor tIn    = createTensor(ctx, Shape{inFlat.size()},  kf32, inFlat.data());
  Tensor tW     = createTensor(ctx, Shape{wFlat.size()},   kf32, wFlat.data());
  Tensor tOfs   = createTensor(ctx, Shape{offsF.size()},   kf32, offsF.data());
  Tensor tCnt   = createTensor(ctx, Shape{cntsF.size()},   kf32, cntsF.data());
  Tensor tB     = createTensor(ctx, Shape{N},             kf32, biases.data());
  Tensor tOut   = createTensor(ctx, Shape{N},             kf32);
  Tensor tAct   = createTensor(ctx, Shape{N},             kf32, actsF.data());

  std::promise<void> p;
  auto fut = p.get_future();
  Kernel k = createKernel(ctx, {kForwardLayer, 64, kf32},
                          Bindings{tIn,tW,tOfs,tCnt,tB,tOut,tAct},
                          { cdiv(N,64), 1, 1 });
  dispatchKernel(ctx, k, p);
  wait(ctx, fut);

  // 6) Read back result and dump **only** JSON to stdout
  std::vector<float> out(N);
  toCPU(ctx, tOut, out.data(), sizeof(float)*N);
  std::cout << json(out).dump() << std::endl;

  return 0;
}
