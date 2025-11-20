# models/hstu_rec.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .basemodel import BaseModel  # 假设已有 BaseModel 定义了 item_emb 等

#modify from: https://github.com/roman-dusek/GR-HSTU/blob/master/hstu.py
class RelativeAttentionBias(nn.Module):
    """
    T5-style relative attention bias based on relative position.
    可加入时间戳信息扩展为 time-aware 版本。
    """
    def __init__(self, num_heads, relative_attention_num_buckets=32, relative_attention_max_distance=128):
        super().__init__()
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.relative_attention_bias = nn.Embedding(relative_attention_num_buckets, num_heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=False, num_buckets=32, max_distance=128):
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).long() * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_position_if_large = max_exact + (
                torch.log(relative_position.float() / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
        ).long()
        relative_position_if_large = torch.min(relative_position_if_large,
                                              torch.full_like(relative_position_if_large, num_buckets - 1))
        return relative_buckets + torch.where(is_small, relative_position, relative_position_if_large)

    def forward(self, query_length, key_length, device=None):
        if device is None:
            device = self.relative_attention_bias.weight.device
        context_pos = torch.arange(query_length, device=device)[:, None]
        memory_pos = torch.arange(key_length, device=device)[None, :]
        relative_position = memory_pos - context_pos  # [Q, K]
        bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=False,  # 因果性：只能看过去
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance
        )
        bias = self.relative_attention_bias(bucket)  # [Q, K, H]
        return bias.permute(2, 0, 1).unsqueeze(0)  # [1, H, Q, K]


class PointwiseAggregatedAttention(nn.Module):
    """
    PAA: v @ softmax(silu(q @ k^T + R)) 的变体，使用 silu(att_score) @ v
    注意：这里不是 softmax！而是 Silu 加权聚合
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.rab_p = RelativeAttentionBias(num_heads, relative_attention_num_buckets=32, relative_attention_max_distance=128)

    def split_heads(self, x, batch_size):
        # x: [B, L, D] -> [B, L, H, D/H] -> [B, H, L, D/H]
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, v, k, q, mask=None):
        B, L, D = q.shape
        q = self.split_heads(q, B)  # [B, H, L, d]
        k = self.split_heads(k, B)
        v = self.split_heads(v, B)

        # Scaled dot-product
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, L, L]
        rab = self.rab_p(L, L, device=q.device)  # [1, H, L, L]
        att_w_bias = attention_scores + rab

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, L]
            mask = mask.expand(-1, self.num_heads, L, -1)  # [B, H, L, L]
            # ❌ 不要用 -inf
            # att_w_bias = att_w_bias.masked_fill(~mask.bool(), float('-inf'))
            att_w_bias = att_w_bias.masked_fill(~mask.bool(), -1e4)

        weights = F.silu(att_w_bias)  # Now safe

        av = torch.matmul(weights, v)  # [B, H, L, d]

        av = av.permute(0, 2, 1, 3).contiguous().view(B, L, D)
        return av



class HSTUBlock(nn.Module):
    """
    HSTU Block: f1(x) -> u,v,q,k -> PAA -> norm(u * av) -> f2
    Residual connection outside.
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.f1 = nn.Linear(d_model, d_model * 4)  # expand to 4xD
        self.pointwise_attn = PointwiseAggregatedAttention(d_model, num_heads)
        self.f2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def split(self, x):
        """Split into u, v, q, k along feature dim"""
        return x.chunk(4, dim=-1)

    def forward(self, x, mask=None):
        """
        x: [B, L, D]
        mask: [B, L] or None
        """
        # Expand and split
        x_proj = F.silu(self.f1(x))  # [B, L, 4D]
        u, v, q, k = self.split(x_proj)

        # Spatial Aggregation via PAA
        av = self.pointwise_attn(v, k, q, mask=mask)  # [B, L, D]

        # Gating & Transformation
        y = self.f2(self.norm(av * u))
        y = self.dropout(y)

        return y + x  # residual


class GenRec(nn.Module):
    """
    Stack of HSTUBlocks
    """
    def __init__(self, d_model, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            HSTUBlock(d_model, num_heads, dropout) for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x


# ================================
# 推荐主模型：HSTURec
# ================================

class HSTURec(BaseModel):
    def __init__(self, item_num, args, device):
        super(HSTURec, self).__init__(item_num, args, device)

        d_model = args.hidden_units
        num_heads = args.num_heads
        num_layers = args.num_blocks

        # Dropout
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)
        self.ln_final = nn.LayerNorm(d_model)

        # Main Backbone
        self.transformer = GenRec(d_model, num_heads, num_layers, args.dropout_rate)

        # Loss weight
        self.user_l2_lambda = 0.05
        self.item_l2_lambda = 0.5

        self.apply(self._init_weights)

    def get_user_emb(self, user_seq):
        """
        user_seq: list of lists, shape [B, L]
        Returns: user embedding at last valid step [B, D]
        """
        # print("i am in hstu")
        user_seq_tensor = user_seq[:,:,0].to(self.dev)
        seq_mask = (user_seq_tensor != 0).to(torch.bool)  # [B, L]

        # Item Embedding
        item_embs = self.item_emb(user_seq_tensor)  # [B, L, D]

        # Position Embedding
        positions = torch.arange(user_seq_tensor.size(1), device=self.dev).expand_as(user_seq_tensor)
        pos_embs = self.position_emb(positions + 1)  # shift by 1 due to padding_idx=0

        # Combine
        seq_emb = item_embs + pos_embs  # [B, L, D]
        seq_emb = self.emb_dropout(seq_emb)

        # Causal mask (optional, already in RelativeBias with bidirectional=False)
        # But we still want to mask future for safety
        L = seq_emb.size(1)
        causal_mask = torch.tril(torch.ones(L, L, device=self.dev)).bool()  # [L, L]
        full_mask = seq_mask.unsqueeze(1).unsqueeze(2) & causal_mask.unsqueeze(0)  # [B, 1, L, L]

        # Forward through HSTU blocks
        output = self.transformer(seq_emb, mask=full_mask)  # [B, L, D]
        output = self.ln_final(output)

        # Take last valid item representation
        last_idx = seq_mask.sum(dim=1) - 1  # [B]
        user_emb = output[torch.arange(output.size(0)), last_idx]  # [B, D]

        return user_emb
