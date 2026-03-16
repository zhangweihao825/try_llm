import torch
from transformers import PreTrainedConfig

class MiniMindConfig(PreTrainedConfig):
    model_type = "minimind" # 在加载模型（如使用 AutoConfig.from_pretrained）时，Hugging Face 会根据配置文件 config.json 中的 model_type 字段，自动识别并映射到你定义的 MiniMindConfig 类。

    def __init__(
            self,

    ):
        super().__init__()


# 模型
import math
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) # 可学习的模型参数，梯度下降对其进行更新，dim全为1说明训练开始的时候对归一化没有影响

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) # pow(2)每个元素的平方，在最后一个维度上求平均值

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x) # .type_as(x) 精度还原


def precompute_freqs_cis(dim: int, end: int = int(32*1024), rope_base: float = 1e6, rope_scaling: Optional[dict] = None):
    # 新的位置编码，旋转位置编码RoPE
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 0.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attn_factor", 1.0),
        )

        if end / orig_max > 1:
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.floor(inv_dim(beta_slow)), end)
            ramp = torch.clamp((torch.arang(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)

        t = torch.arange(end, device=freqs.device)
        freqs = torch.outer(t, freqs).float()
        freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
        freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
        return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., x.shape[-1] // 2:]), dim=-1)

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


