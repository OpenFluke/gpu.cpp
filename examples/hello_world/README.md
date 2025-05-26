clang++ -std=c++17 \
 -I../.. \
 -I../../third_party/headers \
 -L../../third_party/lib \
 -lwebgpu_dawn \
 gpu_forward_layer.cpp -o gpu_forward_layer

(base) samuel@Steamy:~/git/PARAGON/NeuralArena/typeGpu1$ ./gpu_forward_layer layer1_input.json layer1_output.json
./gpu_forward_layer: error while loading shared libraries: libwebgpu_dawn.so: cannot open shared object file: No such file or directory
(base) samuel@Steamy:~/git/PARAGON/NeuralArena/typeGpu1$ find / -name libwebgpu_dawn.so 2>/dev/null
/home/samuel/git/gpu.cpp/third_party/lib/libwebgpu_dawn.so
(base) samuel@Steamy:~/git/PARAGON/NeuralArena/typeGpu1$ ./gpu_forward_layer
./gpu_forward_layer: error while loading shared libraries: libwebgpu_dawn.so: cannot open shared object file: No such file or directory
(base) samuel@Steamy:~/git/PARAGON/NeuralArena/typeGpu1$ export LD_LIBRARY_PATH=/home/samuel/git/gpu.cpp/third_party/lib:$LD_LIBRARY_PATH
./gpu_forward_layer
Usage: gpu_forward_layer <input.json> <output.json>
(base) samuel@Steamy:~/git/PARAGON/NeuralArena/typeGpu1$ ./gpu_forward_layer
Usage: gpu_forward_layer <input.json> <output.json>
