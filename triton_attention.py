import math
import torch
import triton
import triton.language as tl

from custom_modules import scaled_dot_product_attention

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    key_tile_index = tl.program_id(2)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    raise NotImplementedError

class PyTorchFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, is_causal: bool=False):
        # assert query.is_cuda, "Expected query is cuda"
        # assert key.is_cuda, "Expected key is cuda"
        # assert value.is_cuda, "Expected value is cuda"
        # if attn_mask is not None:
        #     assert attn_mask.is_cuda, "Expected attn_mask is cuda"
        
        ctx.Q_TILE_SIZE = 16
        ctx.K_TILE_SIZE = 16
        D = query.shape[-1]
        scale = math.sqrt(D)
        N_QUERIES = query.shape[-2]
        N_KEYS = key.shape[-2]
        lse = None
        output = scaled_dot_product_attention(query=query, key=key, value=value, attn_mask=None)
        
        to_save = [lse, query, key, value, output]
    
    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError

class TritonAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, attn_mask=None):
        raise NotImplementedError
    
    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError