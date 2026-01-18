from einops import einsum, reduce
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

    # K_block_ptr = tl.make_block_ptr(
    #     K_ptr + batch_index * stride_kb,
    #     shape=(N_KEYS, D),
    #     strides=(stride_kk, stride_kd),
    #     offsets=(key_tile_index * K_TILE_SIZE, 0),
    #     block_shape=(K_TILE_SIZE, D),
    #     order=(1, 0),
    # )

    # V_block_ptr = tl.make_block_ptr(
    #     V_ptr + batch_index * stride_vb,
    #     shape=(N_KEYS, D),
    #     strides=(stride_vk, stride_vd),
    #     offsets=(key_tile_index * K_TILE_SIZE, 0),
    #     block_shape=(K_TILE_SIZE, D),
    #     order=(1, 0),
    # )

    output = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    raise NotImplementedError

class TritonAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, attn_mask=None):
        ctx.Q_TILE_SIZE = 16
        ctx.K_TILE_SIZE = 16
        D = query.shape[-1]
        scale = math.sqrt(D)
        N_BATCHES = query.shape[:-2]
        N_QUERIES = query.shape[-2]
        N_KEYS = key.shape[-2]

        N_TILES_Q = math.ceil(N_QUERIES / ctx.Q_TILE_SIZE)
        N_TILES_KV = math.ceil(N_KEYS / ctx.K_TILE_SIZE)
        O = torch.empty_like(query)
        # flash_fwd_kernel[(*N_BATCHES, ctx.N_TILES_Q, ctx.N_TILES_KV)](

        # )
        return O
    
    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError

class PyTorchFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, is_causal: bool=False):
        ctx.Q_TILE_SIZE = 16
        ctx.K_TILE_SIZE = 16
        D = query.shape[-1]
        N_QUERIES = query.shape[-2]
        N_KEYS = key.shape[-2]

        N_TILES_Q = math.ceil(N_QUERIES / ctx.Q_TILE_SIZE)
        N_TILES_KV = math.ceil(N_KEYS / ctx.K_TILE_SIZE)
        O_list = []
        L_list = []
        for i in range(N_TILES_Q):
            # load Q_i from global memory
            lower_ptr_q = i * ctx.Q_TILE_SIZE
            upper_ptr_q = min(N_QUERIES, (i + 1) * ctx.Q_TILE_SIZE)
            Q_i = query[..., lower_ptr_q : upper_ptr_q, :]
            # initialize O_i^{(0)} = 0 \in \R^{B_q \times d}, \ell_i^{(0)} = 0 \in \R^{B_q}, m_{i}^{(0)} = -\infty \in \R^{B_q}
            m_shape = Q_i.shape[:-1]
            m_i = -torch.full(m_shape, float("inf"))
            l_i = torch.zeros(m_shape)
            O_i = torch.zeros(Q_i.shape)
            for j in range(N_TILES_KV):
                # load K^{(j)} and V^{(j)} from global memory
                lower_ptr_k = j * ctx.K_TILE_SIZE
                upper_ptr_k = min(N_KEYS, (j + 1) * ctx.K_TILE_SIZE)
                K_j = key[..., lower_ptr_k : upper_ptr_k, :]
                V_j = value[..., lower_ptr_k : upper_ptr_k, :]
                scores = einsum(Q_i, K_j, "batch ... seq_q d_model, batch ... seq_k d_model -> batch ... seq_q seq_k")
                scores *= (D ** -0.5)
                rowmax = torch.amax(scores, dim=-1)
                m_i_new = torch.max(m_i, rowmax)
                m_i_new_blown = m_i_new.unsqueeze(-1).expand_as(scores)
                P_i = torch.exp(scores - m_i_new_blown)
                P_i_rowsum = reduce(P_i, "batch ... seq_q seq_k -> batch ... seq_q", "sum")
                exp_mi_diff = torch.exp(m_i - m_i_new)
                l_i = einsum(exp_mi_diff, l_i, "batch ... seq_q, batch ... seq_q -> batch ... seq_q") + P_i_rowsum
                O_i = (einsum(torch.diag_embed(exp_mi_diff), O_i, "batch ... seq_q_1 seq_q_2, batch ... seq_q_2 d_model -> batch ... seq_q_1 d_model") + 
                       einsum(P_i, V_j, "batch ... seq_q seq_k, batch ... seq_k d_model -> batch ... seq_q d_model"))
                m_i = m_i_new
            O_i = einsum((torch.diag_embed(l_i ** -1)), O_i, "batch ... seq_q_1 seq_q_2, batch ... seq_q_2 d_model -> batch ... seq_q_1 d_model")
            L_i = m_i + torch.log(l_i)
            O_list.append(O_i)
            L_list.append(L_i)
        O = torch.cat(O_list, dim=1)
        L = torch.cat(L_list, dim=1)
        to_save = [L, query, key, value, O]
        ctx.save_for_backward(*to_save)
        return O

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError