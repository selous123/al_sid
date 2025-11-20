# models/bert4rec_lite.py
import torch
import torch.nn as nn
from .basemodel import BaseModel
from .modules import PointWiseFeedForward  # 假设已定义


class Bert4RecLite(BaseModel):
    def __init__(self, item_num, args, device):
        super(Bert4RecLite, self).__init__(item_num, args, device)

        self.user_l2_lambda = 0.05
        self.item_l2_lambda = 0.5

        self.emb_dropout = nn.Dropout(p=args.dropout_rate)
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            self.attention_layernorms.append(nn.LayerNorm(args.hidden_units, eps=1e-8))
            self.attention_layers.append(
                nn.MultiheadAttention(
                    args.hidden_units,
                    args.num_heads,
                    dropout=args.dropout_rate,
                    batch_first=False  # expect [L, B, D]
                )
            )
            self.forward_layernorms.append(nn.LayerNorm(args.hidden_units, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(args.hidden_units, args.dropout_rate))

        self.apply(self._init_weights)

    

    def get_user_emb(self, user_seq):
        """
        user_seq: [B, seq_len], 原始行为序列（padding=0）
        返回：用户在最后一个有效位置的隐状态作为 user_emb
        """
        # print("i am in bert4rec")
        user_seq_tensor = user_seq[:,:,0].to(self.dev)
        seq_mask = (user_seq_tensor != 0)  # boolean mask [B, L]
        batch_size, seq_len = user_seq_tensor.shape

        # Item Embedding
        item_embs = self.item_emb(user_seq_tensor)  # [B, L, D]

        # Position Embedding
        positions = torch.arange(1, seq_len + 1, device=self.dev).unsqueeze(0).expand(batch_size, -1)
        pos_embs = self.position_emb(positions)  # [B, L, D]

        # 融合
        seq_emb = item_embs + pos_embs  # [B, L, D]
        seq_emb = self.emb_dropout(seq_emb)

        # 转置为 [L, B, D] 以适配 MultiheadAttention
        seq_emb = seq_emb.transpose(0, 1)  # [L, B, D]

        # 注意力 mask：仅屏蔽 padding 位置，不限制 future（双向）
        key_padding_mask = ~seq_mask  # True 表示要被忽略的位置（PyTorch MHA 要求）

        # Transformer Blocks
        for i in range(len(self.attention_layers)):
            # Pre-LN 结构（推荐）
            norm_x = self.attention_layernorms[i](seq_emb)
            mha_outputs, _ = self.attention_layers[i](
                norm_x, norm_x, norm_x,
                attn_mask=None,           # ← 关键：None 表示允许所有位置相互 attention（双向）
                key_padding_mask=key_padding_mask  # 屏蔽 padding
            )
            seq_emb = seq_emb + mha_outputs

            norm_y = self.forward_layernorms[i](seq_emb)
            ffn_outputs = self.forward_layers[i](norm_y)
            seq_emb = seq_emb + ffn_outputs

        # 转回 [B, L, D]
        seq_emb = seq_emb.transpose(0, 1)  # [B, L, D]
        seq_emb = self.last_layernorm(seq_emb)

        # 获取最后一个有效 item 的位置
        last_valid_idx = seq_mask.sum(dim=1) - 1  # [B]
        user_emb = seq_emb[torch.arange(batch_size), last_valid_idx, :]  # [B, D]

        return user_emb
