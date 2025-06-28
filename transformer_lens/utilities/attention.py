"""Attention.
Utilities for attention components.
"""
import einops
import torch
import torch.nn.functional as F
from jaxtyping import Float


def simple_attn_linear(
    input: Float[torch.Tensor, "batch pos d_model"],
    w: Float[torch.Tensor, "head_index d_model d_head"],
    b: Float[torch.Tensor, "head_index d_head"] = None,
) -> Float[torch.Tensor, "batch pos head_index d_head"]:
    """Linear layer for attention calculation."""
    if input.device != w.device:
        w = w.to(input.device)
    # Keep track of original shape for reshaping
    n_heads = w.shape[0]
    d_head = w.shape[2]
    
    # Rearrange weight matrix
    w = einops.rearrange(w, "head_index d_model d_head -> (head_index d_head) d_model")
    
    # Use torch.matmul and reshape to [batch, pos, n_heads, d_head]
    result = torch.matmul(input, w.T).reshape(input.shape[0], input.shape[1], n_heads, d_head)
    
    # Add bias if provided
    if b is not None:
        if b.device != result.device:
            b = b.to(result.device)
        result = result + b
    
    return result


def complex_attn_linear(
    input: Float[torch.Tensor, "batch pos head_index d_model"],
    w: Float[torch.Tensor, "head_index d_model d_head"],
    b: Float[torch.Tensor, "head_index d_head"] = None,
) -> Float[torch.Tensor, "batch pos head_index d_head"]:
    """Linear layer for attention calculation.
    This is almost the same as simple_attn_linear, but the input tensor has an extra head_index dimension, used when calculating the input of each attention head separately.
    """
    # Handle case where input has a different head dimension
    if input.shape[2] != w.shape[0]:
        # Average over head dimension and expand to match weight heads
        input = input.mean(dim=2, keepdim=True).expand(-1, -1, w.shape[0], -1)
    
    # Use einops.einsum for efficient computation
    result = einops.einsum(
        input,
        w,
        "batch pos head_index d_model, head_index d_model d_head -> batch pos head_index d_head",
    )
    
    # Add bias if provided
    if b is not None:
        if b.device != result.device:
            b = b.to(result.device)
        result = result + b
    
    return result