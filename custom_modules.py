from einops import einsum, reduce
import numpy as np
import torch


def _default_init(num_rows, num_cols, device=None, dtype=None):
    init_var = 2.0 / (num_rows + num_cols)
    trunc_up = 3 * np.sqrt(init_var)
    if num_cols > 0:
        rand_tensor = torch.randn(num_rows, num_cols, device=device, dtype=dtype)
    else:
        rand_tensor = torch.randn(num_rows, device=device, dtype=dtype)
    return torch.clamp(rand_tensor, min=-trunc_up,max=trunc_up)


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def softmax(x: torch.Tensor, axis=0) -> torch.Tensor:
    vals, indices = torch.max(x, dim=axis, keepdim=True)
    expx = torch.exp(x - vals.expand_as(x))
    axis_sum = torch.sum(expx, dim=axis, keepdim=True).expand_as(x)
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
    def __init__(self, d_model: int, d_ff: int | None = None):
        super(SwiGLU, self).__init__()
        if not d_ff:
            self.d_ff = int(8 / 3 * d_model)
        else:
            self.d_ff = d_ff
        self.d_model = d_model

        self.W_1 = Linear(in_features=self.d_model, out_features=self.d_ff)
        self.W_2 = Linear(in_features=self.d_ff, out_features=self.d_model)
        self.W_3 = Linear(in_features=self.d_model, out_features=self.d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W_2(silu(self.W_1(x)) * self.W_2(x))


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super(RotaryPositionalEmbedding, self).__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # self.theta_vec = 
        raise NotImplementedError

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x can be k or q. input has shape (..., seq_len, d_k) or (..., seq_len, d_q)

        # return shape: (..., seq_len, d_k)

        # select the 
        raise NotImplementedError
    

def scaled_dot_product_attention(query, key, value, attn_mask=None):
    raise NotImplementedError