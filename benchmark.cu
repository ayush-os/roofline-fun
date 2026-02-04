#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <iomanip>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

struct BenchResult {
    std::string name;
    double time_ms;
    double bandwidth_gb_s;
    double tflops;
};

// Simple Timer for Kernel Execution
class GPUTimer {
    cudaEvent_t start, stop;
public:
    GPUTimer() { cudaEventCreate(&start); cudaEventCreate(&stop); }
    ~GPUTimer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
    void Start() { cudaEventRecord(start); }
    void Stop() { cudaEventRecord(stop); cudaDeviceSynchronize(); }
    float Elapsed() {
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

// --- KERNELS ---

// 1. Vector Add (Memory-Bound)
__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// 2. Naive GEMM (Compute-Bound, CUDA Cores)
__global__ void naiveGEMM(const float* a, const float* b, float* c, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += a[row * N + k] * b[k * N + col];
        }
        c[row * N + col] = sum;
    }
}

// 3. Fused Multiply-Add Stress (Peak FLOPs test)
__global__ void fmaStress(float* out, int iterations) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (float)i;
    float multiplier = 1.000001f;
    #pragma unroll
    for (int j = 0; j < iterations; j++) {
        val = fma(val, multiplier, 0.5f); // Use FMA instruction specifically
    }
    out[i] = val;
}

#include <mma.h>
using namespace nvcuda;

__global__ void tensorCoreGEMM(const half* a, const half* b, float* c) {
    // Fragment storage for Tensor Core operations
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // Load, Multiply-Accumulate, and Store
    wmma::load_matrix_sync(a_frag, a, 16);
    wmma::load_matrix_sync(b_frag, b, 16);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_row_major);
}

// --- RUNNER LOGIC ---

void runBenchmarks() {
    const int N = 1 << 25; // ~33 million elements for VectorAdd
    const int MAT_SIZE = 2048; // for GEMM
    size_t bytes = N * sizeof(float);
    size_t mat_bytes = MAT_SIZE * MAT_SIZE * sizeof(float);

    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_c, bytes));

    GPUTimer timer;

    // --- Benchmark 1: Vector Add ---
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    timer.Start();
    vectorAdd<<<blocks, threads>>>(d_a, d_b, d_c, N);
    timer.Stop();
    
    float vAdd_ms = timer.Elapsed();
    // BW = (Reads: 2*bytes + Writes: 1*bytes) / Time
    double vAdd_bw = (3.0 * bytes) / (vAdd_ms / 1000.0) / 1e9;
    std::cout << "VectorAdd: " << vAdd_ms << " ms | BW: " << vAdd_bw << " GB/s" << std::endl;

    // --- Benchmark 2: GEMM ---
    float *d_matA, *d_matB, *d_matC;
    CHECK_CUDA(cudaMalloc(&d_matA, mat_bytes));
    CHECK_CUDA(cudaMalloc(&d_matB, mat_bytes));
    CHECK_CUDA(cudaMalloc(&d_matC, mat_bytes));

    dim3 block_dim(32, 32);
    dim3 grid_dim((MAT_SIZE + 31) / 32, (MAT_SIZE + 31) / 32);
    
    timer.Start();
    naiveGEMM<<<grid_dim, block_dim>>>(d_matA, d_matB, d_matC, MAT_SIZE);
    timer.Stop();

    float gemm_ms = timer.Elapsed();
    double flops = 2.0 * MAT_SIZE * MAT_SIZE * MAT_SIZE;
    double gemm_tflops = (flops / 1e12) / (gemm_ms / 1000.0);
    std::cout << "NaiveGEMM: " << gemm_ms << " ms | TFLOPS: " << gemm_tflops << std::endl;

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaFree(d_matA); cudaFree(d_matB); cudaFree(d_matC);
}

int main() {
    runBenchmarks();
    return 0;
}