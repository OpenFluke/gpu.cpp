#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include "gpu.hpp" // Wire in your GPU kernels here, if needed.

using json = nlohmann::json;
using namespace std;

// ----- Paragon persistence format structures -----
struct Conn {
    int L, X, Y;
    float W;
};

struct Neuron {
    float Bias;
    string Act;
    vector<Conn> In;
    float Value = 0;
};

struct Layer {
    int W, H;
    vector<vector<Neuron>> Neurons; // [H][W]
};

// -- Robust parser for "layers" array --
vector<Layer> parse_layers(const json& js_layers) {
    vector<Layer> layers;
    for (size_t lidx = 0; lidx < js_layers.size(); ++lidx) {
        const auto& jl = js_layers.at(lidx);
        if (!jl.contains("w") || !jl.contains("h") || !jl.contains("n")) {
            cerr << "Layer " << lidx << " missing one of 'w', 'h', or 'n'!" << endl;
            exit(2);
        }
        Layer layer;
        layer.W = jl["w"];
        layer.H = jl["h"];
        const auto& nmat = jl["n"];
        if (nmat.size() != layer.H) {
            cerr << "Layer " << lidx << " 'n' array height mismatch (expected " << layer.H << ", got " << nmat.size() << ")!" << endl;
            exit(2);
        }
        layer.Neurons.resize(layer.H);
        for (int y = 0; y < layer.H; ++y) {
            const auto& row = nmat[y];
            if (row.size() != layer.W) {
                cerr << "Layer " << lidx << " row " << y << " width mismatch (expected " << layer.W << ", got " << row.size() << ")!" << endl;
                exit(2);
            }
            for (int x = 0; x < layer.W; ++x) {
                const auto& jn = row[x];
                if (!jn.contains("b") || !jn.contains("a") || !jn.contains("in")) {
                    cerr << "Neuron at (" << lidx << "," << y << "," << x << ") missing 'b', 'a', or 'in'!" << endl;
                    exit(2);
                }
                Neuron neuron;
                neuron.Bias = jn["b"];
                neuron.Act = jn["a"];
                const auto& in_arr = jn["in"];
                for (size_t cidx = 0; cidx < in_arr.size(); ++cidx) {
                    const auto& jc = in_arr[cidx];
                    if (!jc.contains("layer") || !jc.contains("x") || !jc.contains("y") || !jc.contains("w")) {
                        cerr << "Connection missing key at neuron (" << lidx << "," << y << "," << x << "), conn #" << cidx << endl;
                        exit(2);
                    }
                    Conn c;
                    c.L = jc["layer"];
                    c.X = jc["x"];
                    c.Y = jc["y"];
                    c.W = jc["w"];
                    neuron.In.push_back(c);
                }
                layer.Neurons[y].push_back(neuron);
            }
        }
        layers.push_back(layer);
    }
    return layers;
}

// -- Activation (CPU for now, but drop GPU code here) --
float gpu_activation(float x, const string& act) {
    if (act == "leaky_relu") return x > 0 ? x : 0.01f * x;
    if (act == "gelu") {
        float k = 0.7978845608028654f; // sqrt(2/PI)
        return 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
    }
    return x; // Identity fallback
}

// -- Forward pass --
vector<float> forward_pass(vector<Layer>& layers) {
    for (size_t l = 1; l < layers.size(); ++l) {
        Layer& curr = layers[l];
        for (int y = 0; y < curr.H; ++y) {
            for (int x = 0; x < curr.W; ++x) {
                Neuron& n = curr.Neurons[y][x];
                float sum = n.Bias;
                for (const Conn& c : n.In) {
                    if (c.L >= 0 && c.L < (int)layers.size()
                        && c.Y >= 0 && c.Y < (int)layers[c.L].H
                        && c.X >= 0 && c.X < (int)layers[c.L].W) {
                        sum += c.W * layers[c.L].Neurons[c.Y][c.X].Value;
                    }
                }
                n.Value = gpu_activation(sum, n.Act);
            }
        }
    }
    // Output = last layer's values, flattened.
    vector<float> result;
    const Layer& last = layers.back();
    for (int y = 0; y < last.H; ++y)
        for (int x = 0; x < last.W; ++x)
            result.push_back(last.Neurons[y][x].Value);
    return result;
}

int main(int argc, char** argv) {
    string json_data;
    if (argc > 1) {
        ifstream fin(argv[1]);
        if (!fin) {
            cerr << "Error opening file: " << argv[1] << endl;
            return 1;
        }
        json_data = string(istreambuf_iterator<char>(fin), {});
    } else {
        json_data = string(istreambuf_iterator<char>(cin), {});
    }

    json net_json;
    try {
        net_json = json::parse(json_data);
    } catch (const std::exception& e) {
        cerr << "JSON parse error: " << e.what() << endl;
        return 1;
    }

    // Print top-level keys
    cout << "Top-level keys: ";
    for (auto it = net_json.begin(); it != net_json.end(); ++it)
        cout << it.key() << " ";
    cout << endl;

    if (net_json.contains("layers")) {
        const auto& layers = net_json["layers"];
        cout << "layers size: " << layers.size() << endl;

        if (!layers.empty()) {
            const auto& first_layer = layers[0];
            cout << "First layer keys: ";
            for (auto it = first_layer.begin(); it != first_layer.end(); ++it)
                cout << it.key() << " ";
            cout << endl;

            if (first_layer.contains("n")) {
                const auto& neuron_matrix = first_layer["n"];
                cout << "neuron_matrix size: " << neuron_matrix.size() << endl;
                if (!neuron_matrix.empty()) {
                    const auto& first_row = neuron_matrix[0];
                    cout << "first_row size: " << first_row.size() << endl;
                    if (!first_row.empty()) {
                        const auto& first_neuron = first_row[0];
                        cout << "First neuron keys: ";
                        for (auto it = first_neuron.begin(); it != first_neuron.end(); ++it)
                            cout << it.key() << " ";
                        cout << endl;
                    } else {
                        cout << "First neuron row is empty." << endl;
                    }
                } else {
                    cout << "Neuron matrix is empty." << endl;
                }
            } else {
                cout << "First layer has no key 'n'!" << endl;
            }
        } else {
            cout << "Layers array is empty." << endl;
        }
    } else {
        cout << "No 'layers' key at top level!" << endl;
    }

    return 0;
}