#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <iomanip>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

// Configuration
const int KERNEL_RADIUS = 2;
const int KERNEL_SIZE = 2 * KERNEL_RADIUS + 1;
const int BLOCK_SIZE = 32;
// The shared memory needs to be larger than the block to account for the "halo"
const int SHMEM_STRIDE = BLOCK_SIZE + 2 * KERNEL_RADIUS;

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

// 1. Naive Convolution (Global Memory Heavy)
__global__ void naiveConv2d(const float* input, const float* mask, float* output, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height && col < width) {
        float sum = 0.0f;
        for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; ++i) {
            for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; ++j) {
                int r = row + i;
                int c = col + j;
                if (r >= 0 && r < height && c >= 0 && c < width) {
                    sum += input[r * width + c] * mask[(i + KERNEL_RADIUS) * KERNEL_SIZE + (j + KERNEL_RADIUS)];
                }
            }
        }
        output[row * width + col] = sum;
    }
}

// 2. Optimized Shared Memory Convolution
__global__ void sharedConv2d(const float* input, const float* mask, float* output, int width, int height) {
    // __shared__ float s_mask[KERNEL_SIZE * KERNEL_SIZE]; // Could also put mask in shared
    __shared__ float s_data[SHMEM_STRIDE][SHMEM_STRIDE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * blockDim.x + tx;
    int row = blockIdx.y * blockDim.y + ty;

    // Collaborative load into shared memory
    // Each thread loads its primary pixel, plus some threads load the halo
    for (int i = ty; i < SHMEM_STRIDE; i += BLOCK_SIZE) {
        for (int j = tx; j < SHMEM_STRIDE; j += BLOCK_SIZE) {
            int r = blockIdx.y * BLOCK_SIZE + i - KERNEL_RADIUS;
            int c = blockIdx.x * BLOCK_SIZE + j - KERNEL_RADIUS;

            if (r >= 0 && r < height && c >= 0 && c < width)
                s_data[i][j] = input[r * width + c];
            else
                s_data[i][j] = 0.0f;
        }
    }
    __syncthreads();

    if (row < height && col < width) {
        float sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < KERNEL_SIZE; ++i) {
            #pragma unroll
            for (int j = 0; j < KERNEL_SIZE; ++j) {
                sum += s_data[ty + i][tx + j] * mask[i * KERNEL_SIZE + j];
            }
        }
        output[row * width + col] = sum;
    }
}

int main() {
    const int W = 4096;
    const int H = 4096;
    size_t img_bytes = W * H * sizeof(float);
    size_t mask_bytes = KERNEL_SIZE * KERNEL_SIZE * sizeof(float);

    float *h_mask = new float[KERNEL_SIZE * KERNEL_SIZE];
    for(int i=0; i < KERNEL_SIZE*KERNEL_SIZE; ++i) h_mask[i] = 1.0f / (KERNEL_SIZE*KERNEL_SIZE);

    float *d_in, *d_out, *d_mask;
    CHECK_CUDA(cudaMalloc(&d_in, img_bytes));
    CHECK_CUDA(cudaMalloc(&d_out, img_bytes));
    CHECK_CUDA(cudaMalloc(&d_mask, mask_bytes));
    CHECK_CUDA(cudaMemcpy(d_mask, h_mask, mask_bytes, cudaMemcpyHostToDevice));

    GPUTimer timer;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((W + BLOCK_SIZE - 1) / BLOCK_SIZE, (H + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Naive Run
    timer.Start();
    naiveConv2d<<<grid, block>>>(d_in, d_mask, d_out, W, H);
    timer.Stop();
    std::cout << "Naive Conv2d:  " << timer.Elapsed() << " ms" << std::endl;

    // Shared Run
    timer.Start();
    sharedConv2d<<<grid, block>>>(d_in, d_mask, d_out, W, H);
    timer.Stop();
    std::cout << "Shared Conv2d: " << timer.Elapsed() << " ms" << std::endl;

    // Cleanup
    delete[] h_mask;
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_mask);
    return 0;
}