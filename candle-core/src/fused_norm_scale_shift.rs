// File: /home/ubuntu/dev/candle/candle-core/src/fused_norm_scale_shift.rs
//! Fused Norm-Scale-Shift Operation
//!
//! This module implements a fused operator that applies normalization followed by
//! modulation (scale and shift) in a single operation, improving performance.

use crate::{Error, Result, Tensor};

/// Applies a fused normalization and modulation operation.
///
/// # Arguments
///
/// * `input` - The input tensor of shape `(batch_size, hidden_size)`
/// * `norm_weight` - Normalization weights
/// * `mod_scale` - Modulation scale tensor
/// * `mod_shift` - Modulation shift tensor
/// * `epsilon` - Small value for numerical stability
///
/// # Returns
///
/// A new tensor with the fused operation applied
pub fn fused_norm_scale_shift(
    input: &Tensor,
    norm_weight: &Tensor,
    mod_scale: &Tensor,
    mod_shift: &Tensor,
    epsilon: f32,
) -> Result<Tensor> {
    let dims = input.dims();
    if dims.len() < 2 {
        return Err(Error::Msg(format!("Expected tensor with at least 2 dimensions, got: {:?}", dims)));
    }

    // For tensors with more than 2 dimensions, we treat them as 2D by flattening all but the last dimension
    let original_shape = dims.to_vec();
    let hidden_dim = dims[dims.len() - 1];
    let flat_size: usize = dims.iter().take(dims.len() - 1).product();

    // Reshape to [flat_size, hidden_dim] for processing
    let x = if dims.len() > 2 {
        input.reshape((flat_size, hidden_dim))?
    } else {
        input.clone()
    };

    // Compute normalization (RMS-norm style)
    let x2 = x.sqr()?;
    let variance = x2.sum_keepdim(1)? / hidden_dim as f64;
    let norm_factor = (variance? + epsilon as f64)?.sqrt()?.recip()?;

    // Apply normalization with the weight
    let normalized = x.broadcast_mul(&norm_factor)?.broadcast_mul(norm_weight)?;

    // Apply scale_shift operation
    let scale_plus_one = (mod_scale + 1.0)?;
    let result = normalized.broadcast_mul(&scale_plus_one)?.broadcast_add(mod_shift)?;

    // Reshape back to original dimensions if needed
    if dims.len() > 2 {
        result.reshape(original_shape)
    } else {
        Ok(result)
    }
}
