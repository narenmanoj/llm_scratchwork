from collections.abc import Callable
from einops import einsum, rearrange, reduce
from jaxtyping import Float, Int
import math
import numpy as np
import torch
from typing import Optional


def _subtract_max(x: torch.Tensor, dim=0) -> torch.Tensor:
    vals, indices = torch.max(x, dim=dim, keepdim=True)
    return x - vals.expand_as(x)


def _default_init(num_rows, num_cols, device=None, dtype=None) -> torch.Tensor:
    init_var = 2.0 / (num_rows + num_cols)
    trunc_up = 3 * np.sqrt(init_var)
    if num_cols > 0:
        rand_tensor = torch.randn(num_rows, num_cols, device=device, dtype=dtype)
    else:
        rand_tensor = torch.randn(num_rows, device=device, dtype=dtype)
    return torch.clamp(rand_tensor, min=-trunc_up, max=trunc_up)


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def softmax(x: torch.Tensor, dim=0) -> torch.Tensor:
    expx = torch.exp(_subtract_max(x, dim=dim))
    axis_sum = torch.sum(expx, dim=dim, keepdim=True).expand_as(x)
    return expx / axis_sum


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = torch.nn.parameter.Parameter(_default_init(num_rows=out_features,
                                                            num_cols=in_features,
                                                            device=device,
                                                            dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.W, x, "out_features in_features, ... in_features -> ... out_features")


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super(Embedding, self).__init__()
        self.embedding_matrix = torch.nn.parameter.Parameter(_default_init(num_rows=num_embeddings,
                                                                           num_cols=embedding_dim,
                                                                           device=device, dtype=dtype))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding_matrix[token_ids]


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.gains = torch.nn.parameter.Parameter(_default_init(num_rows=d_model,
                                                                num_cols=0,
                                                                device=device,
                                                                dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        sq = torch.square(x)
        rms = torch.sqrt(1.0 / self.d_model * reduce(sq, "... d_model -> ...", "sum") + self.eps)
        scale = 1.0 / rms
        unscaled_result = einsum(x, self.gains, "... d_model, d_model -> ... d_model")
        result = einsum(unscaled_result, scale, "... d_model, ... -> ... d_model")

        return result.to(in_dtype)


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, device=None):
        super(SwiGLU, self).__init__()
        if not d_ff:
            self.d_ff = int(8 / 3 * d_model)
        else:
            self.d_ff = d_ff
        self.d_model = d_model

        self.W_1 = Linear(in_features=self.d_model, out_features=self.d_ff, device=device)
        self.W_2 = Linear(in_features=self.d_ff, out_features=self.d_model, device=device)
        self.W_3 = Linear(in_features=self.d_model, out_features=self.d_ff, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W_2(silu(self.W_1(x)) * self.W_3(x))


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super(RotaryPositionalEmbedding, self).__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len # (i is in [1, max_seq_len])

        # axis 0 : [1, max_seq_len]. axis 1: [0, d_k - 1]
        half_dk = d_k // 2
        # theta_vec_half = (1 + torch.arange(half_dk * max_seq_len)).reshape(max_seq_len, half_dk)
        numerator_vec_half = einsum(torch.arange(max_seq_len, device=device), 
                                    torch.ones((max_seq_len, half_dk), device=device), 
                                    "seq_len, seq_len d_k -> seq_len d_k")
        denominator_vec_half = torch.pow(theta, ((2 * (torch.arange(half_dk, device=device) + 1) - 2) / d_k).repeat(max_seq_len, 1))

        theta_vec_half = numerator_vec_half / denominator_vec_half
        self.theta_vec = torch.stack((theta_vec_half, theta_vec_half), dim=-1).view(*(theta_vec_half.shape[:-1]), -1)
        self.cosines = torch.cos(self.theta_vec)
        self.sines = torch.sin(self.theta_vec)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        assert len(token_positions.shape) <= 2
        # x flipped
        x_flip = torch.stack((-x[..., 1::2], x[..., 0::2]), dim=-1).view(*(x.shape[:-1]), -1)

        # x can be some collection of key or query vectors
        relevant_cosines = self.cosines[token_positions]
        relevant_sines = self.sines[token_positions]

        cosine_result = einsum(x,
                               relevant_cosines,
                               "batch ... seq d_k, batch seq d_k -> batch ... seq d_k")
        sine_result = einsum(x_flip,
                             relevant_sines,
                             "batch ... seq d_k, batch seq d_k -> batch ... seq d_k")
        return cosine_result + sine_result


def scaled_dot_product_attention(query: torch.Tensor,
                                 key: torch.Tensor,
                                 value: torch.Tensor,
                                 attn_mask: torch.Tensor | None = None) -> torch.Tensor:
    qk_t = einsum(query,
                  key,
                  "batch ... seq_q d_k, batch ... seq_k d_k -> batch ... seq_q seq_k")
    d_k = query.shape[-1]
    pre_softmax = qk_t * (d_k ** -0.5)
    if attn_mask is not None:
        pre_softmax_masked = pre_softmax.masked_fill(attn_mask == 0, float("-inf"))
    else:
        pre_softmax_masked = pre_softmax
    softmax_qkt = softmax(pre_softmax_masked, dim=-1)
    result = einsum(softmax_qkt,
                    value,
                    "batch ... seq_q seq_k, batch ... seq_k d_v -> batch ... seq_q d_v")
    return result


class MultiheadAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, rope=None, device=None):
        super(MultiheadAttention, self).__init__()
        self.d_k = int(embed_dim / num_heads)
        self.num_heads = num_heads
        self.out_dim = num_heads * self.d_k
        self.W_Q = Linear(in_features=embed_dim, out_features=self.out_dim, device=device)
        self.W_K = Linear(in_features=embed_dim, out_features=self.out_dim, device=device)
        self.W_V = Linear(in_features=embed_dim, out_features=self.out_dim, device=device)
        self.W_O = Linear(in_features=self.out_dim, out_features=embed_dim, device=device)
        self.rope = rope
    
    def forward(self, x: torch.Tensor, is_causal=False, token_positions=None) -> torch.Tensor:
        batch_size = x.shape[0]
        num_tokens = x.shape[1]
    
        query = self.W_Q(x)
        key = self.W_K(x)
        value = self.W_V(x)
        
        # slice and dice q k v, into another batch dimension, to support multihead
        query = rearrange(query, "batch seq_k (h d_k) -> batch seq_k h d_k", h=self.num_heads, d_k=self.d_k)
        key = rearrange(key, "batch seq_k (h d_k) -> batch seq_k h d_k", h=self.num_heads, d_k=self.d_k)
        value = rearrange(value, "batch seq_k (h d_k) -> batch seq_k h d_k", h=self.num_heads, d_k=self.d_k)

        query = rearrange(query, "batch seq_k h d_k -> batch h seq_k d_k")
        key = rearrange(key, "batch seq_k h d_k -> batch h seq_k d_k")
        value = rearrange(value, "batch seq_k h d_k -> batch h seq_k d_k")

        if token_positions is None:
            token_positions = torch.arange(num_tokens, device=x.device).repeat(batch_size).reshape((batch_size, num_tokens))

        if self.rope is not None:
            query = self.rope(query, token_positions)
            key = self.rope(key, token_positions)
        # call scaled_dot_product_attention on the output with the causal mask
        if is_causal:
            seq_len = x.shape[1]
            mask = torch.tril(
                torch.ones(seq_len, seq_len, device=x.device)
            )
        else:
            mask = None
        attn = scaled_dot_product_attention(query=query, key=key, value=value, attn_mask=mask)
        attn = rearrange(attn, "batch h seq_k d_k -> batch seq_k h d_k")
        attn = rearrange(attn, "batch seq_k h d_k -> batch seq_k (h d_k)")
        result = self.W_O(attn)
        return result


class PreNormTransformer(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, rope_theta, max_seq_len, device=None):
        super(PreNormTransformer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.max_seq_len = max_seq_len

        # layers
        self.rms_1 = RMSNorm(d_model=d_model, device=device)
        self.rms_2 = RMSNorm(d_model=d_model, device=device)
        self.rope = RotaryPositionalEmbedding(theta=rope_theta,
                                              d_k=(d_model // num_heads),
                                              max_seq_len=max_seq_len,
                                              device=device)
        self.mha = MultiheadAttention(embed_dim=d_model,
                                      num_heads=num_heads,rope=self.rope,
                                      device=device)
        self.swiglu = SwiGLU(d_model=d_model, d_ff=d_ff, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.mha(self.rms_1(x), is_causal=True)
        attention_plus_residual = attn + x
        swiglu_val = self.swiglu(self.rms_2(attention_plus_residual))
        result = swiglu_val + attention_plus_residual
        return result


class TransformerLM(torch.nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, device=None):
        super(TransformerLM, self).__init__()
        
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device)
        self.layers = [None] * num_layers
        for i in range(num_layers):
            self.layers[i] = PreNormTransformer(d_model=d_model,
                                                num_heads=num_heads,
                                                d_ff=d_ff,
                                                rope_theta=rope_theta,
                                                max_seq_len=context_length,
                                                device=device)
        self.last_norm = RMSNorm(d_model=d_model, device=device)
        self.last_linear = Linear(in_features=d_model, out_features=vocab_size, device=device)

        self.transformers = torch.nn.ModuleList(self.layers)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        interm = self.embedding(in_indices)
        for layer in self.transformers:
            interm = layer(interm)
        result = self.last_linear(self.last_norm(interm))
        return result


def cross_entropy(
    inputs: Float[torch.Tensor, " batch_size vocab_size"], targets: Int[torch.Tensor, " batch_size"]
) -> Float[torch.Tensor, ""]:
    inputs_demaxed = _subtract_max(inputs, dim=1)
    log_sum_exps = torch.log(torch.exp(inputs_demaxed).sum(dim=1))

    targets_blown = targets.reshape((len(targets), 1))
    sum_vals = inputs_demaxed.gather(dim=1, index=targets_blown)
    return -torch.mean(sum_vals.flatten() - log_sum_exps)


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8):
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps
        }
        super().__init__(params, defaults)
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.eps = eps

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        alpha = self.lr
        beta_1 = self.betas[0]
        beta_2 = self.betas[1]
        eps = self.eps
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                grad = p.grad.data

                # in here, do all the work
                first_moment = state.get("first_moment", torch.zeros(grad.shape, device=p.grad.device))
                second_moment = state.get("second_moment", torch.zeros(grad.shape, device=p.grad.device))

                first_moment = beta_1 * first_moment + (1 - beta_1) * grad
                second_moment = beta_2 * second_moment + (1 - beta_2) * torch.square(grad)
                alpha_t = alpha * math.sqrt(1 - beta_2 ** t) / (1 - beta_1 ** t)

                p.data -= alpha_t * (first_moment / (torch.sqrt(second_moment) + eps))
                p.data *= (1 - alpha * self.weight_decay)

                state["t"] = t + 1
                state["first_moment"] = first_moment
                state["second_moment"] = second_moment
        return loss


def lr_cosine_schedule(t, alpha_max, alpha_min, t_w, t_c):
    if t < t_w:
        return t / t_w * alpha_max
    if t > t_c:
        return alpha_min
    return alpha_min + 0.5 * (1 + math.cos((t - t_w) / (t_c - t_w) * math.pi)) * (alpha_max - alpha_min)


def clip_grad(params, M, eps=1e-6):
    total_norm = 0
    for p in params:
        if p.grad is None:
            continue
        grad = p.grad.data
        total_norm += torch.norm(grad, p=2) ** 2
    total_norm = math.sqrt(total_norm)
    if total_norm > M:
        for p in params:
            if p.grad is None:
                continue
            p.grad.data /= (total_norm + eps)
            p.grad.data *= M