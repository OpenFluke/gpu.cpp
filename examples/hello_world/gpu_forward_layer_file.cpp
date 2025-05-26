// gpu_forward_layer.cpp
#include "gpu.hpp"               // Dawn/WebGPU helpers (your existing header)
#include <fstream>
#include <vector>
#include <string>
#include <future>
#include <nlohmann/json.hpp>
#include <cstring>               // std::memcpy
#include <cstdint>
#include <iostream>

using namespace gpu;
using json = nlohmann::json;

/* ───────────────────────── WGSL kernel ──────────────────────────────── */
static const char *kForwardLayer = R"(
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
  static_assert(sizeof(T) == sizeof(uint32_t));
  dst.resize(src.size());
  std::memcpy(dst.data(), src.data(), src.size() * sizeof(uint32_t));
}

/* ─────────────────────────── main ───────────────────────────────────── */
int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "Usage: gpu_forward_layer <input.json> <output.json>\n";
    return 1;
  }
  const std::string inputFile  = argv[1];
  const std::string outputFile = argv[2];

  /* ── read JSON ------------------------------------------------------- */
  json inputData;
  {
    std::ifstream in(inputFile);
    if (!in) { std::cerr << "Cannot open " << inputFile << '\n'; return 1; }
    try { in >> inputData; }
    catch (const json::exception &e) {
      std::cerr << "JSON parse error: " << e.what() << '\n'; return 1;
    }
  }

  const size_t numNeurons = inputData.size();
  if (numNeurons == 0) { std::cerr << "No neurons in input.\n"; return 1; }

  /* ── flatten per-neuron data ---------------------------------------- */
  std::vector<float>        inputsFlat, weightsFlat, biases;
  std::vector<uint32_t>     offsets, counts, actIds;

  for (size_t i = 0; i < numNeurons; ++i) {
    const auto &n = inputData[i];

    if (!n.contains("inputs") || !n.contains("weights") ||
        !n.contains("bias")   || !n.contains("activation")) {
      std::cerr << "Neuron " << i << " missing required fields.\n"; return 1;
    }
    const auto &ins = n["inputs"];
    const auto &wts = n["weights"];
    if (ins.size() != wts.size()) {
      std::cerr << "Neuron " << i << " inputs/weights length differ.\n";
      return 1;
    }

    offsets.push_back(static_cast<uint32_t>(inputsFlat.size()));
    counts .push_back(static_cast<uint32_t>(ins.size()));

    for (size_t k = 0; k < ins.size(); ++k) {
      inputsFlat .push_back(ins[k].get<float>());
      weightsFlat.push_back(wts[k].get<float>());
    }
    biases .push_back(n["bias"].get<float>());
    actIds .push_back(activationToId(n["activation"].get<std::string>()));
  }

  /* ── bit-cast u32 vectors to f32 so WGSL can read them -------------- */
  std::vector<float> offsetsF, countsF, actsF;
  bitcastVec(offsets, offsetsF);
  bitcastVec(counts , countsF);
  bitcastVec(actIds , actsF);

  /* ── create GPU tensors --------------------------------------------- */
  Context ctx = createContext();

  Tensor tInputs   = createTensor(ctx, Shape{inputsFlat.size()} , kf32, inputsFlat.data());
  Tensor tWeights  = createTensor(ctx, Shape{weightsFlat.size()}, kf32, weightsFlat.data());
  Tensor tOffsets  = createTensor(ctx, Shape{offsetsF.size()}  , kf32, offsetsF.data());
  Tensor tCounts   = createTensor(ctx, Shape{countsF.size()}   , kf32, countsF.data());
  Tensor tBiases   = createTensor(ctx, Shape{numNeurons}       , kf32, biases.data());
  Tensor tOutput   = createTensor(ctx, Shape{numNeurons}       , kf32);          // GPU writes here
  Tensor tActs     = createTensor(ctx, Shape{numNeurons}       , kf32, actsF.data());

  /* ── launch kernel --------------------------------------------------- */
  std::promise<void> promise;
  std::future<void>  fut = promise.get_future();

  Kernel kernel = createKernel(
      ctx, {kForwardLayer, 64, kf32},
      /* bindings order MUST match @binding indices in WGSL */
      Bindings{ tInputs, tWeights, tOffsets, tCounts, tBiases,
                tOutput, tActs },
      { cdiv(numNeurons, 64), 1, 1 });

  dispatchKernel(ctx, kernel, promise);
  wait(ctx, fut);              // block until the GPU work finishes

  /* ── read back & write JSON ----------------------------------------- */
  std::vector<float> outVals(numNeurons);
  toCPU(ctx, tOutput, outVals.data(), sizeof(float) * numNeurons);

  std::ofstream out(outputFile);
  if (!out) { std::cerr << "Cannot open " << outputFile << '\n'; return 1; }
  out << json(outVals).dump(2) << '\n';

  return 0;
}
