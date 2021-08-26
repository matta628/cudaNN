#include <random>
#include <array>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <assert.h>

constexpr size_t N_INPUTS = 128*1024;
constexpr size_t N_NODES = 8192;
static_assert(N_NODES%1024 == 0, "N_NODES must be multiple of 1024");

#define gpu_assert(rv) gpu_assert_h((rv), __FILE__, __LINE__)
void
gpu_assert_h(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

std::default_random_engine eng;

__global__ void
dotprod(const float *const input_layer, const float *const weights_data, float *const dotprods) {

    /*
    printf("input %f\n", *input_layer);
    printf("weight %f\n", *weights_data);
    */

    // Use the thread ID as the node ID.
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    // printf("ID %d, weights_data=0x%p\n", id, weights_data);

    // weights points to the beginning of the weights for this node (each thread handles a node).
    const float *const weights = weights_data + id*N_INPUTS;
    // printf("ID %d, weights_data=0x%p\n", id, weights_data);

    /*
    for (size_t i = 0; i < N_INPUTS; i++) {
        printf("[ID %d] weight[%d]=%f\n", id, int(i), weights[i]);
    }
    */

    // Compute dot product for this node.
    float dp = 0;
    for (size_t i = 0; i < N_INPUTS; i++) {
        dp += weights[i]*input_layer[i];
    }
    dotprods[id] = dp;
}

#if 0

__global__ void
dotprod_coalesced(const float *const input_layer, const float *const weights_data, float *const dotprods) {

    /*
    printf("input %f\n", *input_layer);
    printf("weight %f\n", *weights_data);
    */

    // Use the thread ID as the node index.
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    // printf("ID %d, weights_data=0x%p\n", id, weights_data);

    // weights points to the beginning of the weights for this node (each thread handles a node).
    const float *const weights = weights_data + id*N_INPUTS;
    // printf("ID %d, weights_data=0x%p\n", id, weights_data);

    /*
    for (size_t i = 0; i < N_INPUTS; i++) {
        printf("[ID %d] weight[%d]=%f\n", id, int(i), weights[i]);
    }
    */

    // Compute dot product for this node.
    float dp = 0;
    for (size_t i = 0; i < N_INPUTS; i++) {
        dp += weights[i]*input_layer[i];
    }
    dotprods[id] = dp;
}

#endif

int
main() {

    using dist_t = std::uniform_real_distribution<float>;
    // Use this to print output to prevent compiler optimizations.
    FILE *devnull = fopen("/dev/null", "w");
    assert(devnull != nullptr);

    // Fill input vector.
    std::vector<float> input_layer(N_INPUTS);
    std::generate(input_layer.begin(),  input_layer.end(),
        []() { return dist_t{-1, 1}(eng); }
    );

    /*
    for (size_t i = 0; i < N_INPUTS; i++) {
        printf("Input %zu: %f\n", i, input_layer.at(i));
    }
    */

    std::vector<std::array<float, N_INPUTS>> weights(N_NODES);

    // Create weight matrix.
    for (size_t i = 0; i < N_NODES; i++) {
        for (size_t j = 0; j < N_INPUTS; j++) {
            weights.at(i).at(j) = dist_t{-1, 1}(eng);
        }
    }

    /*
    for (size_t i = 0; i < N_NODES; i++) {
        printf("Node %zu: \n", i);
        for (size_t j = 0; j < N_INPUTS; j++) {
            printf("Element %zu: %f\n", j, weights.at(i).at(j));
        }
    }
    */

    // Now do the dot products.
    static std::array<float, N_NODES> dot_prods;
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N_NODES; i++) {
        float dp = 0;
        for (size_t j = 0; j < N_INPUTS; j++) {
            dp += weights.at(i).at(j)*input_layer.at(j);
        }
        dot_prods.at(i) = dp;
    }
    //std::chrono::duration<float> dt = std::chrono::high_resolution_clock::now() - start;
    std::chrono::duration<double, std::milli> dt = std::chrono::high_resolution_clock::now() - start;

    // Print out dot prods to /dev/null so that the compiler doesn't optimise it out.
    for (size_t i = 0; i < N_NODES; i++) {
        fprintf(devnull, "Node %zu DP: %f\n", i, dot_prods.at(i));
    }

    printf("Time required on CPU: %f\n", dt.count()/1'000.0);

    /*
     * Now do device.
     */

    cudaError_t rv_ce;

    float *d_input_layer;
    rv_ce = cudaMalloc(&d_input_layer, N_INPUTS*sizeof(float));
    gpu_assert(rv_ce);

    rv_ce = cudaMemcpy(d_input_layer, input_layer.data(), N_INPUTS*sizeof(float), cudaMemcpyHostToDevice);
    gpu_assert(rv_ce);

    // Allocate and copy weights.
    float *d_weights = nullptr;
    rv_ce = cudaMalloc(&d_weights, N_NODES*N_INPUTS*sizeof(float));
    gpu_assert(rv_ce);
    // printf("d_weights: 0x%p\n", d_weights);

    {
        float *d_p = d_weights;
        for (size_t i = 0; i < N_NODES; i++) {
            rv_ce = cudaMemcpy(d_p, weights.at(i).data(), N_INPUTS*sizeof(float), cudaMemcpyHostToDevice);
            gpu_assert(rv_ce);
            d_p += weights.at(i).size();
        }
    }

    // Allocate dotprod space on device.
    float *d_dotprods;
    rv_ce = cudaMalloc(&d_dotprods, N_NODES*sizeof(float));
    gpu_assert(rv_ce);

    /*
    {
        static float x = 3.14;
        rv_ce = cudaMemcpy(d_weights, &x, sizeof(float), cudaMemcpyHostToDevice);
        gpu_assert(rv_ce);
    }
    */

    // Execute kernel.
    {
        auto start = std::chrono::high_resolution_clock::now();
        //dotprod<<<1, N_NODES>>>(d_input_layer, d_weights, d_dotprods);
        dotprod<<<N_NODES/1024, 1024>>>(d_input_layer, d_weights, d_dotprods);
        gpu_assert(cudaDeviceSynchronize());
        gpu_assert(cudaPeekAtLastError());

        // Copy dotprod back to host.
        std::array<float, N_NODES> dotprods;
        rv_ce = cudaMemcpy(dotprods.data(), d_dotprods, N_NODES*sizeof(float), cudaMemcpyDeviceToHost);
        gpu_assert(rv_ce);

        std::chrono::duration<double, std::milli> dt
         = std::chrono::high_resolution_clock::now() - start;

        /*
        for (size_t i = 0; i < dotprods.size(); i++) {
            printf("Node %zu dotprod=%f\n", i, dotprods.at(i));
        }
        */

        printf("Time required on GPU: %f\n", dt.count()/1'000.0);
    }

    #if 0

    // Now rearange so as to get memory coalescing.
    {
        std::array<std::array<loat, N_NODES>, N_INPUTS>> weights_coalesced;
        for (std:size_t i_w = 0; i_w < N_INPUTS, i_w++) {
            for (std:size_t i_n = 0; i_n < N_NODES, i_n++) {
                weights_coalesced.at(i_w).at(i_n) = weights.at(i_n).at(i_w);
            }
        }

        
        float *d_input_layer;
        rv_ce = cudaMalloc(&d_input_layer, N_INPUTS*sizeof(float));
        gpu_assert(rv_ce);

        rv_ce = cudaMemcpy(d_input_layer, input_layer.data(), N_INPUTS*sizeof(float), cudaMemcpyHostToDevice);
        gpu_assert(rv_ce);

        // Allocate and copy weights.
        float *d_weights = nullptr;
        rv_ce = cudaMalloc(&d_weights, N_NODES*N_INPUTS*sizeof(float));
        gpu_assert(rv_ce);
        // printf("d_weights: 0x%p\n", d_weights);

        {
            float *d_p = d_weights;
            for (size_t i = 0; i < N_NODES; i++) {
                rv_ce = cudaMemcpy(d_p, weights.at(i).data(), N_INPUTS*sizeof(float), cudaMemcpyHostToDevice);
                gpu_assert(rv_ce);
                d_p += weights.at(i).size();
            }
        }
    }

    #endif
}

// vim: set ts=4 sw=4 ai et:
