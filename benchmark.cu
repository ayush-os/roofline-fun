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

// Tile size for Shared Memory GEMM
const int TILE_SIZE = 32;

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

// 2. Tiled Shared Memory GEMM (New)
__global__ void sharedMemoryGEMM(const float* A, const float* B, float* C, int N) {
    // Allocate shared memory for tiles
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float tmp = 0.0f;

    // Loop over tiles required to compute the result
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Collaborative Load: Each thread loads one element of the tile into shared memory
        if (row < N && (t * TILE_SIZE + tx) < N)
            s_A[ty][tx] = A[row * N + t * TILE_SIZE + tx];
        else
            s_A[ty][tx] = 0.0f;

        if (col < N && (t * TILE_SIZE + ty) < N)
            s_B[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        else
            s_B[ty][tx] = 0.0f;

        // Ensure all threads have finished loading tiles
        __syncthreads();

        // Compute partial product from this tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            tmp += s_A[ty][k] * s_B[k][tx];
        }

        // Wait for all threads before moving to the next tile
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = tmp;
    }
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

    GPUTimer timer;
    double flops = 2.0 * MAT_SIZE * MAT_SIZE * MAT_SIZE;

    // --- Naive Benchmark ---
    dim3 block_dim(32, 32);
    dim3 grid_dim((MAT_SIZE + 31) / 32, (MAT_SIZE + 31) / 32);
    
    timer.Start();
    naiveGEMM<<<grid_dim, block_dim>>>(d_matA, d_matB, d_matC, MAT_SIZE);
    timer.Stop();
    std::cout << std::left << std::setw(20) << "NaiveGEMM:" << timer.Elapsed() << " ms | " 
              << (flops / 1e12) / (timer.Elapsed() / 1000.0) << " TFLOPS" << std::endl;

    // --- Shared Memory (Tiled) Benchmark ---
    timer.Start();
    sharedMemoryGEMM<<<grid_dim, block_dim>>>(d_matA, d_matB, d_matC, MAT_SIZE);
    timer.Stop();
    std::cout << std::left << std::setw(20) << "SharedMemGEMM:" << timer.Elapsed() << " ms | " 
              << (flops / 1e12) / (timer.Elapsed() / 1000.0) << " TFLOPS" << std::endl;

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaFree(d_matA); cudaFree(d_matB); cudaFree(d_matC);
}

int main() {
    runBenchmarks();
    return 0;
}