#include "cuda_fp16.h"

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

// Macro to fuse normalization (RMS norm) with modulation (scale and shift)
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
}\

FUSED_NORM_SCALE_SHIFT_OP(fused_norm_scale_shift_f32, float) \
FUSED_NORM_SCALE_SHIFT_OP(fused_norm_scale_shift_f16, __half)

#if __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
FUSED_NORM_SCALE_SHIFT_OP(fused_norm_scale_shift_bf16, __nv_bfloat16)
#endif
