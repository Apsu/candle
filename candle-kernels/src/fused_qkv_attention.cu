#include "cuda_fp16.h"
#include <stdint.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 256  // Tunable parameter

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

template<typename T>
__inline__ __device__ T warpReduceMax(T val) {
  #pragma unroll
  for (int mask = WARP_SIZE/2; mask > 0; mask >>= 1)
    val = max(val, VLLM_SHFL_XOR_SYNC(val, mask));
  return val;
}

#if __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>

extern "C" __global__ void fused_qkv_attention_bf16(
    __nv_bfloat16* __restrict__ output,             // Output tensor
    const __nv_bfloat16* __restrict__ input,        // Input tensor (xs)
    const __nv_bfloat16* __restrict__ qkv_weights,  // QKV projection weights
    const __nv_bfloat16* __restrict__ qkv_bias,     // Optional QKV bias
    const __nv_bfloat16* __restrict__ query_norm,   // Q normalization weights
    const __nv_bfloat16* __restrict__ key_norm,     // K normalization weights
    const __nv_bfloat16* __restrict__ proj_weights, // Output projection weights
    const __nv_bfloat16* __restrict__ proj_bias,    // Output projection bias
    const __nv_bfloat16* __restrict__ pe,           // Positional encoding
    const int batch_size,
    const int seq_length,
    const int hidden_size,
    const int num_heads,
    const bool has_qkv_bias,
    const bool has_proj_bias) {

    // Calculate dimensions
    const int head_dim = hidden_size / num_heads;
    const int qkv_size = hidden_size * 3;

    // Allocate shared memory regions
    extern __shared__ __nv_bfloat16 shared_mem[];
    __nv_bfloat16* s_qkv = shared_mem;                        // For QKV intermediate results
    __nv_bfloat16* s_query_norm = &s_qkv[BLOCK_SIZE * 3];     // For query norm weights
    __nv_bfloat16* s_key_norm = &s_query_norm[head_dim];      // For key norm weights
    __nv_bfloat16* s_attn = &s_key_norm[head_dim];            // For attention scores

    // Load norm weights into shared memory
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        s_query_norm[i] = query_norm[i];
        s_key_norm[i] = key_norm[i];
    }
    __syncthreads();

    // Identify which token this block is processing
    const int token_idx = blockIdx.x;
    const int batch_idx = token_idx / seq_length;
    const int seq_idx = token_idx % seq_length;

    // Step 1: QKV Projection (matrix multiplication)
    // Each thread handles multiple elements of the input vector
    for (int i = threadIdx.x; i < qkv_size; i += blockDim.x) {
        float sum = 0.0f;

        // Vectorized dot product
        const int vec_hidden_size = hidden_size / 2;
        const __nv_bfloat162* input2 = reinterpret_cast<const __nv_bfloat162*>(
            &input[batch_idx * seq_length * hidden_size + seq_idx * hidden_size]);

        for (int j = 0; j < vec_hidden_size; j++) {
            __nv_bfloat162 in_vec = input2[j];
            __nv_bfloat162 weight_vec = reinterpret_cast<const __nv_bfloat162*>(
                &qkv_weights[i * hidden_size])[j];

            float2 in_f2 = __bfloat1622float2(in_vec);
            float2 weight_f2 = __bfloat1622float2(weight_vec);

            sum += in_f2.x * weight_f2.x + in_f2.y * weight_f2.y;
        }

        // Handle odd-sized hidden dimension
        if (hidden_size % 2 == 1) {
            int last_idx = hidden_size - 1;
            sum += __bfloat162float(input[batch_idx * seq_length * hidden_size +
                                        seq_idx * hidden_size + last_idx]) *
                  __bfloat162float(qkv_weights[i * hidden_size + last_idx]);
        }

        // Add bias if present
        if (has_qkv_bias) {
            sum += __bfloat162float(qkv_bias[i]);
        }

        // Store result in shared memory (Q, K, V interleaved)
        s_qkv[i] = __float2bfloat16(sum);
    }
    __syncthreads();

    // Step 2 & 3: Split QKV and apply normalization
    // Calculate logical indices for Q, K, V in the shared memory
    const int local_seq_idx = seq_idx % seq_length;

    // Each thread handles one head dimension across Q, K, or V
    for (int head_idx = 0; head_idx < num_heads; head_idx++) {
        for (int dim_idx = threadIdx.x; dim_idx < head_dim; dim_idx += blockDim.x) {
            const int q_idx = head_idx * head_dim + dim_idx;
            const int k_idx = hidden_size + head_idx * head_dim + dim_idx;
            const int v_idx = 2 * hidden_size + head_idx * head_dim + dim_idx;

            float q_val = __bfloat162float(s_qkv[q_idx]);
            float k_val = __bfloat162float(s_qkv[k_idx]);

            // Compute normalization factor for Q
            float q_norm_factor = 0.0f;
            for (int j = 0; j < head_dim; j++) {
                float val = __bfloat162float(s_qkv[head_idx * head_dim + j]);
                q_norm_factor += val * val;
            }
            q_norm_factor = rsqrtf(q_norm_factor / head_dim + 1e-6f);

            // Compute normalization factor for K
            float k_norm_factor = 0.0f;
            for (int j = 0; j < head_dim; j++) {
                float val = __bfloat162float(s_qkv[hidden_size + head_idx * head_dim + j]);
                k_norm_factor += val * val;
            }
            k_norm_factor = rsqrtf(k_norm_factor / head_dim + 1e-6f);

            // Apply normalization
            q_val = q_val * q_norm_factor * __bfloat162float(s_query_norm[dim_idx]);
            k_val = k_val * k_norm_factor * __bfloat162float(s_key_norm[dim_idx]);

            // Step 4: Apply RoPE (Rotary Positional Embedding)
            float q_rot = 0.0f, k_rot = 0.0f;
            if (dim_idx % 2 == 0 && dim_idx + 1 < head_dim) {
                // Safe access for paired indices
                float cos_val = __bfloat162float(pe[local_seq_idx * head_dim/2 + dim_idx/2]);
                float sin_val = __bfloat162float(pe[local_seq_idx * head_dim/2 + dim_idx/2 + head_dim/4]);

                q_rot = q_val * cos_val - __bfloat162float(s_qkv[head_idx * head_dim + dim_idx+1]) * sin_val;
                k_rot = k_val * cos_val - __bfloat162float(s_qkv[hidden_size + head_idx * head_dim + dim_idx+1]) * sin_val;
            } else if (dim_idx % 2 == 0) {
                // Handle boundary case
                float cos_val = __bfloat162float(pe[local_seq_idx * head_dim/2 + dim_idx/2]);
                q_rot = q_val * cos_val;
                k_rot = k_val * cos_val;
            } else {
                // Odd indices - sine
                float cos_val = __bfloat162float(pe[local_seq_idx * head_dim/2 + (dim_idx-1)/2]);
                float sin_val = __bfloat162float(pe[local_seq_idx * head_dim/2 + (dim_idx-1)/2 + head_dim/4]);

                q_rot = q_val * sin_val + __bfloat162float(s_qkv[head_idx * head_dim + dim_idx-1]) * cos_val;
                k_rot = k_val * sin_val + __bfloat162float(s_qkv[hidden_size + head_idx * head_dim + dim_idx-1]) * cos_val;
            }

            // Update Q, K with rotated values
            s_qkv[q_idx] = __float2bfloat16(q_rot);
            s_qkv[k_idx] = __float2bfloat16(k_rot);
        }
    }
    __syncthreads();

    // Step 5 & 6: Compute attention scores and apply softmax
    // Each thread computes one attention score
    for (int head_idx = 0; head_idx < num_heads; head_idx++) {
        const int attn_offset = head_idx * seq_length * seq_length;

        for (int j = threadIdx.x; j < seq_length; j += blockDim.x) {
            float attn_score = 0.0f;

            // Dot product of Q[head_idx, seq_idx] and K[head_idx, j]
            for (int dim_idx = 0; dim_idx < head_dim; dim_idx++) {
                float q_val = __bfloat162float(s_qkv[head_idx * head_dim + dim_idx]);
                float k_val = __bfloat162float(s_qkv[hidden_size + head_idx * head_dim + dim_idx]);
                attn_score += q_val * k_val;
            }

            // Scale by sqrt(head_dim)
            attn_score /= sqrtf(head_dim);

            // Store for softmax
            s_attn[head_idx * seq_length + j] = __float2bfloat16(attn_score);
        }
        __syncthreads();

        // Apply softmax
        if (threadIdx.x < seq_length) {
            int j = threadIdx.x;

            // Find max for numerical stability
            float max_val = -INFINITY;
            for (int k = 0; k < seq_length; k++) {
                max_val = fmaxf(max_val, __bfloat162float(s_attn[head_idx * seq_length + k]));
            }

            // Compute exp(score - max) and sum
            float sum = 0.0f;
            for (int k = 0; k < seq_length; k++) {
                float val = expf(__bfloat162float(s_attn[head_idx * seq_length + k]) - max_val);
                s_attn[head_idx * seq_length + k] = __float2bfloat16(val);
                sum += val;
            }

            // Normalize
            for (int k = 0; k < seq_length; k++) {
                s_attn[head_idx * seq_length + k] = __float2bfloat16(
                    __bfloat162float(s_attn[head_idx * seq_length + k]) / sum);
            }
        }
        __syncthreads();

        // Step 7: Apply attention scores to V
        for (int dim_idx = threadIdx.x; dim_idx < head_dim; dim_idx += blockDim.x) {
            float weighted_sum = 0.0f;

            for (int j = 0; j < seq_length; j++) {
                float attn_val = __bfloat162float(s_attn[head_idx * seq_length + j]);
                float v_val = __bfloat162float(s_qkv[2 * hidden_size + head_idx * head_dim + dim_idx]);
                weighted_sum += attn_val * v_val;
            }

            // Store result back
            const int out_idx = batch_idx * seq_length * hidden_size +
                               seq_idx * hidden_size +
                               head_idx * head_dim + dim_idx;

            output[out_idx] = __float2bfloat16(weighted_sum);
        }
    }
    __syncthreads();

    // Step 8: Apply output projection
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float sum = 0.0f;
        const int out_offset = batch_idx * seq_length * hidden_size + seq_idx * hidden_size;

        for (int j = 0; j < hidden_size; j++) {
            sum += __bfloat162float(output[out_offset + j]) *
                   __bfloat162float(proj_weights[i * hidden_size + j]);
        }

        if (has_proj_bias) {
            sum += __bfloat162float(proj_bias[i]);
        }

        output[out_offset + i] = __float2bfloat16(sum);
    }
}
#endif

// Similar implementation for F16
#if __CUDA_ARCH__ >= 530
// F16 implementation similar to above
#endif

// F32 implementation
extern "C" __global__ void fused_qkv_attention_f32(
    // Same parameters as BF16 but with float type
) {
    // Similar implementation for F32
}
