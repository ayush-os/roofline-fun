#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <iomanip>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

// Configuration
const int TILE_SIZE = 32;
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

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

// 1. Naive GEMM (FP32)
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

// 2. Tiled Shared Memory GEMM (FP32)
__global__ void sharedMemoryGEMM(const float* A, const float* B, float* C, int N) {
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float tmp = 0.0f;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < N && (t * TILE_SIZE + tx) < N)
            s_A[ty][tx] = A[row * N + t * TILE_SIZE + tx];
        else
            s_A[ty][tx] = 0.0f;

        if (col < N && (t * TILE_SIZE + ty) < N)
            s_B[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        else
            s_B[ty][tx] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            tmp += s_A[ty][k] * s_B[k][tx];
        }
        __syncthreads();
    }

    if (row < N && col < N) C[row * N + col] = tmp;
}

// 3. Tensor Core GEMM (FP16 Input, FP32 Accumulate)
__global__ void tensorCoreGEMM(const half* A, const half* B, float* C, int N) {
    // Each warp computes a 16x16 tile
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    int warpM = (blockIdx.y * blockDim.y + threadIdx.y) / 32;
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x);

    for (int i = 0; i < N; i += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = i;
        int bRow = i;
        int bCol = warpN * WMMA_N;

        if (aRow < N && aCol < N && bRow < N && bCol < N) {
            wmma::load_matrix_sync(a_frag, A + aRow * N + aCol, N);
            wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    if (cRow < N && cCol < N) {
        wmma::store_matrix_sync(C + cRow * N + cCol, acc_frag, N, wmma::mem_row_major);
    }
}

// --- RUNNER LOGIC ---

int main() {
    const int MAT_SIZE = 2048; 
    const size_t f_size = MAT_SIZE * MAT_SIZE * sizeof(float);
    const size_t h_size = MAT_SIZE * MAT_SIZE * sizeof(half);

    float *d_A, *d_B, *d_C;
    half *d_A_half, *d_B_half;

    CHECK_CUDA(cudaMalloc(&d_A, f_size));
    CHECK_CUDA(cudaMalloc(&d_B, f_size));
    CHECK_CUDA(cudaMalloc(&d_C, f_size));
    CHECK_CUDA(cudaMalloc(&d_A_half, h_size));
    CHECK_CUDA(cudaMalloc(&d_B_half, h_size));

    // Initialize with dummy data (A100 loves non-zeroes)
    // Note: In a real test, you'd use a kernel to convert FP32 to FP16 on GPU
    
    GPUTimer timer;
    double flops = 2.0 * MAT_SIZE * MAT_SIZE * MAT_SIZE;

    std::cout << "Matrix Size: " << MAT_SIZE << " x " << MAT_SIZE << "\n";
    std::cout << std::string(50, '-') << "\n";

    // 1. Naive
    dim3 block_naive(32, 32);
    dim3 grid_naive((MAT_SIZE + 31) / 32, (MAT_SIZE + 31) / 32);
    timer.Start();
    naiveGEMM<<<grid_naive, block_naive>>>(d_A, d_B, d_C, MAT_SIZE);
    timer.Stop();
    std::cout << std::left << std::setw(20) << "Naive GEMM:" << timer.Elapsed() << " ms | " 
              << (flops / 1e12) / (timer.Elapsed() / 1000.0) << " TFLOPS" << std::endl;

    // 2. Shared Memory
    timer.Start();
    sharedMemoryGEMM<<<grid_naive, block_naive>>>(d_A, d_B, d_C, MAT_SIZE);
    timer.Stop();
    std::cout << std::left << std::setw(20) << "Shared Mem GEMM:" << timer.Elapsed() << " ms | " 
              << (flops / 1e12) / (timer.Elapsed() / 1000.0) << " TFLOPS" << std::endl;

    // 3. Tensor Core
    // For WMMA, we need 32 threads per warp. 
    // This grid setup ensures one warp per 16x16 output tile.
    dim3 block_tc(128, 1); // 4 warps per block
    dim3 grid_tc((MAT_SIZE + (WMMA_N * 4) - 1) / (WMMA_N * 4), (MAT_SIZE + WMMA_M - 1) / WMMA_M);
    
    timer.Start();
    tensorCoreGEMM<<<grid_tc, block_tc>>>(d_A_half, d_B_half, d_C, MAT_SIZE);
    timer.Stop();
    std::cout << std::left << std::setw(20) << "Tensor Core GEMM:" << timer.Elapsed() << " ms | " 
              << (flops / 1e12) / (timer.Elapsed() / 1000.0) << " TFLOPS" << std::endl;

    CHECK_CUDA(cudaFree(d_A)); CHECK_CUDA(cudaFree(d_B)); CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_A_half)); CHECK_CUDA(cudaFree(d_B_half));

    return 0;
}