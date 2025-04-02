#include "cuda_fp16.h"
#include <stdint.h>

#define WARP_SIZE 32

#ifndef USE_ROCM
  #define VLLM_SHFL_XOR_SYNC(var, lane_mask) __shfl_xor_sync(uint32_t(-1), var, lane_mask)
#else
  #define VLLM_SHFL_XOR_SYNC(var, lane_mask) __shfl_xor(var, lane_mask)
#endif

template<typename T>
__inline__ __device__ T warpReduceSum(T val) {
#pragma unroll
  for (int mask = WARP_SIZE/2; mask > 0; mask >>= 1)
    val += VLLM_SHFL_XOR_SYNC(val, mask);
  return val;
}

__inline__ __device__ constexpr int _calculateLaneMask(int warp_size) {
  return warp_size - 1;
}

__inline__ __device__ constexpr int _calculateWidShift(int warp_size) {
  return 5 + (warp_size >> 6);
}

/* Calculate the sum of all elements in a block */
template<typename T>
__inline__ __device__ T blockReduceSum(T val) {
  static __shared__ T shared[WARP_SIZE];
  constexpr auto LANE_MASK = _calculateLaneMask(WARP_SIZE);
  constexpr auto WID_SHIFT = _calculateWidShift(WARP_SIZE);
  int lane = threadIdx.x & LANE_MASK;
  int wid = threadIdx.x >> WID_SHIFT;

  val = warpReduceSum<T>(val);

  if (lane == 0)
    shared[wid] = val;

  __syncthreads();

  /* Use blockDim.x / 32 since blockDim.x may not be a multiple of 32 */
  val = (threadIdx.x < (blockDim.x / 32)) ? shared[lane] : (T)(0.0f);
  val = warpReduceSum<T>(val);
  return val;
}

// Generic macro for f32 and f16
#define FUSED_NORM_SCALE_SHIFT_OP(FN_NAME, TYPENAME) \
extern "C" __global__ void FN_NAME(\
  TYPENAME* __restrict__ out, \
  const TYPENAME* __restrict__ input, \
  const TYPENAME* __restrict__ norm_weight, \
  const TYPENAME* __restrict__ mod_scale, \
  const TYPENAME* __restrict__ mod_shift, \
  const float epsilon, \
  const int num_tokens, \
  const int hidden_size) {\
  __shared__ float s_norm_factor;\
  float variance = 0.0f;\
  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {\
    const float x = (float) input[blockIdx.x * hidden_size + idx];\
    variance += x * x;\
  }\
  variance = blockReduceSum<float>(variance);\
  if (threadIdx.x == 0) {\
    s_norm_factor = rsqrtf(variance / hidden_size + epsilon);\
  }\
  __syncthreads();\
  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {\
    const int index = blockIdx.x * hidden_size + idx; \
    float x = (float) input[index];\
    float norm_w = (float) norm_weight[idx];\
    float normalized = x * s_norm_factor * norm_w;\
    float scale = (float) mod_scale[index];\
    float shift = (float) mod_shift[index];\
    out[index] = (TYPENAME)(normalized * (scale + 1.0f) + shift);\
  }\
}

// Specialized macro for BF16 (using __nv_bfloat16) – accumulate in float for precision
#if __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#define FUSED_NORM_SCALE_SHIFT_OP_BF16(FN_NAME) \
extern "C" __global__ void FN_NAME(\
  __nv_bfloat16* __restrict__ out, \
  const __nv_bfloat16* __restrict__ input, \
  const __nv_bfloat16* __restrict__ norm_weight, \
  const __nv_bfloat16* __restrict__ mod_scale, \
  const __nv_bfloat16* __restrict__ mod_shift, \
  const float epsilon, \
  const int num_tokens, \
  const int hidden_size) {\
  __shared__ float s_norm_factor;\
  float variance = 0.0f;\
  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {\
    const float x = __bfloat162float(input[blockIdx.x * hidden_size + idx]);\
    variance += x * x;\
  }\
  variance = blockReduceSum<float>(variance);\
  if (threadIdx.x == 0) {\
    s_norm_factor = rsqrtf(variance / hidden_size + epsilon);\
  }\
  __syncthreads();\
  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {\
    const int index = blockIdx.x * hidden_size + idx; \
    float x = __bfloat162float(input[index]);\
    float norm_w = __bfloat162float(norm_weight[idx]);\
    float normalized = x * s_norm_factor * norm_w;\
    float scale = __bfloat162float(mod_scale[index]);\
    float shift = __bfloat162float(mod_shift[index]);\
    out[index] = __float2bfloat16(normalized * (scale + 1.0f) + shift);\
  }\
}
#endif

#if __CUDA_ARCH__ >= 900  // Hopper-specific optimizations
extern "C" __global__ void fused_norm_scale_shift_bf16_hopper(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ norm_weight,
    const __nv_bfloat16* __restrict__ mod_scale,
    const __nv_bfloat16* __restrict__ mod_shift,
    const float epsilon,
    const int num_tokens,
    const int hidden_size) {

    __shared__ float s_norm_factor;
    // Store weights in shared memory - they're reused for all tokens in the block
    extern __shared__ __nv_bfloat16 s_norm_weights[];

    // Load weights into shared memory
    for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
        s_norm_weights[idx] = norm_weight[idx];
    }

    // Ensure input pointers are aligned to 4-byte boundaries for vector loads
    if (reinterpret_cast<uintptr_t>(input) % 4 == 0) {
        // Use vectorized implementation

        // Vectorized variance calculation
        const int vector_hidden_size = hidden_size / 2;
        float variance = 0.0f;

        // Process two elements at once using vector loads
        const __nv_bfloat162* input2 = reinterpret_cast<const __nv_bfloat162*>(input);

        for (int idx = threadIdx.x; idx < vector_hidden_size; idx += blockDim.x) {
            __nv_bfloat162 vec = input2[blockIdx.x * vector_hidden_size + idx];
            float2 values = __bfloat1622float2(vec);
            variance += values.x * values.x + values.y * values.y;

            // Example of prefetching technique
            __nv_bfloat162 next_input_vec;
            if (idx + blockDim.x < vector_hidden_size) {
                next_input_vec = input2[blockIdx.x * vector_hidden_size + idx + blockDim.x];
            }
        }

        // Handle odd-sized hidden dimensions
        if (hidden_size % 2 == 1 && threadIdx.x == 0) {
            float x = __bfloat162float(input[blockIdx.x * hidden_size + (hidden_size - 1)]);
            variance += x * x;
        }

        variance = blockReduceSum<float>(variance);
        if (threadIdx.x == 0) {
            s_norm_factor = rsqrtf(variance / hidden_size + epsilon);
        }
        __syncthreads();

        // Vectorized output calculation
        __nv_bfloat162* out2 = reinterpret_cast<__nv_bfloat162*>(out);
        const __nv_bfloat162* mod_scale2 = reinterpret_cast<const __nv_bfloat162*>(mod_scale);
        const __nv_bfloat162* mod_shift2 = reinterpret_cast<const __nv_bfloat162*>(mod_shift);

        for (int idx = threadIdx.x; idx < vector_hidden_size; idx += blockDim.x) {
            int vector_idx = blockIdx.x * vector_hidden_size + idx;

            // Load pairs of values
            __nv_bfloat162 input_vec = input2[vector_idx];
            __nv_bfloat162 scale_vec = mod_scale2[vector_idx];
            __nv_bfloat162 shift_vec = mod_shift2[vector_idx];

            // Convert to float2 for processing
            float2 input_f2 = __bfloat1622float2(input_vec);
            float2 scale_f2 = __bfloat1622float2(scale_vec);
            float2 shift_f2 = __bfloat1622float2(shift_vec);

            // First element in pair
            float norm_w1 = __bfloat162float(s_norm_weights[idx*2]);
            float normalized1 = input_f2.x * s_norm_factor * norm_w1;
            float result1 = normalized1 * (scale_f2.x + 1.0f) + shift_f2.x;

            // Second element in pair
            float norm_w2 = __bfloat162float(s_norm_weights[idx*2 + 1]);
            float normalized2 = input_f2.y * s_norm_factor * norm_w2;
            float result2 = normalized2 * (scale_f2.y + 1.0f) + shift_f2.y;

            // Convert back to vector bf16 and store
            float2 result_f2 = {result1, result2};
            out2[vector_idx] = __float22bfloat162_rn(result_f2);
        }

        // Handle odd-sized hidden dimensions
        if (hidden_size % 2 == 1 && threadIdx.x == 0) {
            int last_idx = blockIdx.x * hidden_size + (hidden_size - 1);
            float x = __bfloat162float(input[last_idx]);
            float norm_w = __bfloat162float(s_norm_weights[hidden_size - 1]);
            float normalized = x * s_norm_factor * norm_w;
            float scale = __bfloat162float(mod_scale[last_idx]);
            float shift = __bfloat162float(mod_shift[last_idx]);
            out[last_idx] = __float2bfloat16(normalized * (scale + 1.0f) + shift);
        }
    } else {
        // Fall back to non-vectorized version

        float variance = 0.0f;
        for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
            const float x = __bfloat162float(input[blockIdx.x * hidden_size + idx]);
            variance += x * x;
        }
        variance = blockReduceSum<float>(variance);
        if (threadIdx.x == 0) {
            s_norm_factor = rsqrtf(variance / hidden_size + epsilon);
        }
        __syncthreads();

        for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
            const int index = blockIdx.x * hidden_size + idx;
            float x = __bfloat162float(input[index]);
            float norm_w = __bfloat162float(norm_weight[idx]);
            float normalized = x * s_norm_factor * norm_w;
            float scale = __bfloat162float(mod_scale[index]);
            float shift = __bfloat162float(mod_shift[index]);
            out[index] = __float2bfloat16(normalized * (scale + 1.0f) + shift);
        }
    }
}

// Calculate appropriate shared memory size
size_t shared_mem_size = hidden_size * sizeof(__nv_bfloat16);

// Adjust block size based on hidden dimension
int block_size = (hidden_size <= 1024) ? 256 : 512;

// Choose optimal block size for Hopper (test these!)
dim3 blockDim(block_size);
dim3 gridDim(num_tokens);

// Launch with shared memory allocation
fused_norm_scale_shift_bf16_hopper<<<gridDim, blockDim, shared_mem_size>>>(/*...*/);
#endif

// Instantiate for f32 and f16 using the generic macro.
FUSED_NORM_SCALE_SHIFT_OP(fused_norm_scale_shift_f32, float)
FUSED_NORM_SCALE_SHIFT_OP(fused_norm_scale_shift_f16, __half)

#if __CUDA_ARCH__ >= 800
FUSED_NORM_SCALE_SHIFT_OP_BF16(fused_norm_scale_shift_bf16)
#endif
